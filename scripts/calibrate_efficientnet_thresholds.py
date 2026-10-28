#!/usr/bin/env python3
"""Calibrate EfficientNetV2 77-class Youden thresholds on labelled NPY ECGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import HeartWiseModelFactory  # noqa: E402
from utils.constants import ECG_PATTERNS  # noqa: E402
from utils.credentials import read_api_key  # noqa: E402
from utils.model_manifest import model_entry  # noqa: E402
from utils.threshold_calibration import (  # noqa: E402
    compare_existing_sl_thresholds,
    compute_calibration_metrics,
    infer_minibatches,
    label_column,
    parse_path_maps,
    preprocess_npy_batch,
    probability_column,
    remap_path,
    resolve_label_columns,
)

MODEL_NAME = "efficientnetv2_77_classes"
MODEL_REPOSITORY = f"heartwise/{MODEL_NAME}"
SCHEMA_VERSION = 1


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Labelled training parquet")
    parser.add_argument("--output-dir", required=True, type=Path, help="Private checkpoint/output directory")
    parser.add_argument("--path-column", default="npy_path")
    parser.add_argument(
        "--path-map", action="append", default=[], metavar="SOURCE=DESTINATION",
        help="Replace a stored path prefix; may be supplied more than once",
    )
    parser.add_argument("--preprocessing-batch-size", type=int, default=10000)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--preprocessing-workers", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--max-rows", type=int, help="Pilot-only row limit")
    parser.add_argument(
        "--api-key-file", type=Path,
        help="Optional mounted JSON credential file; public models use anonymous access by default",
    )
    parser.add_argument("--resume", action="store_true", help="Resume matching completed shards")
    parser.add_argument("--skip-final-metrics", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def shard_path(output: Path, batch_index: int) -> Path:
    return output / "predictions" / f"batch_{batch_index:06d}.parquet"


def valid_shard(path: Path, start: int, rows: int) -> bool:
    try:
        metadata = pq.read_metadata(path)
        if metadata.num_rows != rows:
            return False
        indexes = pq.read_table(path, columns=["row_index"])["row_index"]
        return len(indexes) == rows and indexes[0].as_py() == start and indexes[-1].as_py() == start + rows - 1
    except Exception:
        return False


def write_shard(
    path: Path,
    start: int,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    data = {"row_index": pa.array(np.arange(start, start + len(labels)), type=pa.int64())}
    for index in range(len(ECG_PATTERNS)):
        data[label_column(index)] = pa.array(labels[:, index], type=pa.uint8())
        data[probability_column(index)] = pa.array(probabilities[:, index], type=pa.float32())
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.table(data), temporary, compression="zstd")
    temporary.replace(path)


def load_checkpoint_matrices(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    label_names = [label_column(index) for index in range(len(ECG_PATTERNS))]
    probability_names = [probability_column(index) for index in range(len(ECG_PATTERNS))]
    table = dataset.to_table(columns=label_names + probability_names)
    labels = np.column_stack([
        table[name].to_numpy(zero_copy_only=False) for name in label_names
    ]).astype(np.uint8, copy=False)
    probabilities = np.column_stack([
        table[name].to_numpy(zero_copy_only=False) for name in probability_names
    ]).astype(np.float32, copy=False)
    return labels, probabilities


def main() -> int:
    args = arguments()
    if args.preprocessing_batch_size < 1 or args.inference_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    if args.preprocessing_workers < 1 or args.torch_threads < 1:
        raise ValueError("Worker and thread counts must be positive.")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("max_rows must be positive.")

    args.input = args.input.resolve()
    args.output_dir = args.output_dir.resolve()
    predictions_dir = args.output_dir / "predictions"
    manifest_path = args.output_dir / "manifest.json"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(args.input)
    if args.path_column not in parquet.schema_arrow.names:
        raise ValueError(f"Path column {args.path_column!r} is missing from the parquet.")
    label_mapping = resolve_label_columns(parquet.schema_arrow.names)
    path_maps = parse_path_maps(args.path_map)
    rows_requested = min(parquet.metadata.num_rows, args.max_rows or parquet.metadata.num_rows)
    model_info = model_entry(MODEL_REPOSITORY)
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "sha256": sha256(args.input),
            "rows_available": parquet.metadata.num_rows,
            "rows_requested": rows_requested,
            "path_column": args.path_column,
            "label_columns": label_mapping,
        },
        "model": model_info,
        "preprocessing": {
            "implementation": "in_memory_parallel_order_preserving",
            "batch_size": args.preprocessing_batch_size,
            "workers": args.preprocessing_workers,
            "path_maps": [{"source": source, "destination": destination} for source, destination in path_maps],
        },
        "inference": {
            "device": "cpu",
            "batch_size": args.inference_batch_size,
            "torch_threads": args.torch_threads,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        comparable = {key: existing.get(key) for key in run_manifest}
        if comparable != run_manifest or not args.resume:
            raise ValueError("Output directory contains a different or non-resumable calibration run.")
    else:
        if any(predictions_dir.iterdir()):
            raise ValueError("Prediction directory is non-empty but has no matching manifest.")
        atomic_json(manifest_path, run_manifest)

    torch.set_num_threads(args.torch_threads)
    token = (
        read_api_key(args.api_key_file).get("HUGGING_FACE_API_KEY", "")
        if args.api_key_file else ""
    )
    model = HeartWiseModelFactory.create_model({
        "model_name": MODEL_NAME,
        "map_location": torch.device("cpu"),
        "hugging_face_api_key": token,
    })

    selected_columns = [args.path_column] + [label_mapping[label] for label in ECG_PATTERNS]
    processed_rows = successful_rows = failed_rows = 0
    started = time.monotonic()
    completed_paths: list[Path] = []
    for batch_index, record_batch in enumerate(parquet.iter_batches(
        batch_size=args.preprocessing_batch_size,
        columns=selected_columns,
        use_threads=True,
    )):
        if processed_rows >= rows_requested:
            break
        remaining = rows_requested - processed_rows
        if record_batch.num_rows > remaining:
            record_batch = record_batch.slice(0, remaining)
        rows = record_batch.num_rows
        output = shard_path(args.output_dir, batch_index)
        completed_paths.append(output)
        if args.resume and output.exists() and valid_shard(output, processed_rows, rows):
            probabilities = pq.read_table(output, columns=[probability_column(0)])[probability_column(0)]
            succeeded = int(np.isfinite(probabilities.to_numpy(zero_copy_only=False)).sum())
            successful_rows += succeeded
            failed_rows += rows - succeeded
            processed_rows += rows
            print(f"batch={batch_index + 1} rows={processed_rows}/{rows_requested} resumed=true", flush=True)
            continue

        paths = [
            remap_path(str(value), path_maps)
            for value in record_batch.column(0).to_pylist()
        ]
        label_arrays = [record_batch.column(index + 1) for index in range(len(ECG_PATTERNS))]
        null_labels = [ECG_PATTERNS[index] for index, column in enumerate(label_arrays) if column.null_count]
        if null_labels:
            raise ValueError(f"Binary label columns contain nulls: {null_labels}")
        raw_labels = np.column_stack([
            column.to_numpy(zero_copy_only=False) for column in label_arrays
        ])
        if not np.all((raw_labels == 0) | (raw_labels == 1)):
            raise ValueError("The 77 calibration label columns must contain only 0 or 1.")
        labels = raw_labels.astype(np.uint8, copy=False)
        positions, cleaned, report = preprocess_npy_batch(paths, args.preprocessing_workers)
        probabilities = np.full((rows, len(ECG_PATTERNS)), np.nan, dtype=np.float32)
        probabilities[positions] = infer_minibatches(
            model, cleaned, args.inference_batch_size, device="cpu"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_shard(output, processed_rows, labels, probabilities)

        succeeded = len(positions)
        successful_rows += succeeded
        failed_rows += rows - succeeded
        processed_rows += rows
        elapsed = time.monotonic() - started
        rate = processed_rows / elapsed if elapsed else 0
        print(
            f"batch={batch_index + 1} rows={processed_rows}/{rows_requested} "
            f"successful={successful_rows} failed={failed_rows} rows_per_second={rate:.2f}",
            flush=True,
        )
        del labels, probabilities, positions, cleaned, report, record_batch

    run_manifest["result"] = {
        "rows_processed": processed_rows,
        "rows_successful": successful_rows,
        "rows_failed": failed_rows,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(manifest_path, run_manifest)

    if not args.skip_final_metrics:
        labels, probabilities = load_checkpoint_matrices(completed_paths)
        metrics = compute_calibration_metrics(labels, probabilities)
        compare_existing_sl_thresholds(metrics, ROOT / "thresholds" / "77_tasks_final.json")
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "method": "Youden index (maximum sensitivity + specificity - 1)",
            "model": model_info,
            "input_sha256": run_manifest["input"]["sha256"],
            "rows": run_manifest["result"],
            **metrics,
        }
        atomic_json(args.output_dir / "thresholds.json", artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
