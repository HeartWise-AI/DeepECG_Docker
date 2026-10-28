#!/usr/bin/env python3
"""Verify complete calibration partitions and create one aggregate threshold artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_efficientnet_thresholds import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_json,
    load_checkpoint_matrices,
    valid_shard,
)
from utils.threshold_calibration import (  # noqa: E402
    compare_existing_sl_thresholds,
    compute_calibration_metrics,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition-dir", action="append", required=True, type=Path,
        help="Completed partition output directory; supply once per partition",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def collect_partition_shards(partition_dirs: list[Path]) -> tuple[list[Path], dict]:
    """Return verified, globally ordered shards and aggregate-safe run metadata."""
    if not partition_dirs:
        raise ValueError("At least one partition directory is required.")
    partitions = []
    for directory in partition_dirs:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Partition manifest is missing: {manifest_path}")
        partitions.append((directory, json.loads(manifest_path.read_text())))
    partitions.sort(key=lambda item: item[1].get("input", {}).get("start_row", -1))

    first = partitions[0][1]
    comparable_keys = ("model", "preprocessing", "inference")
    expected_row = 0
    successful = failed = 0
    elapsed = 0.0
    shards: list[Path] = []
    for directory, manifest in partitions:
        input_info = manifest.get("input", {})
        result = manifest.get("result")
        if result is None:
            raise ValueError(f"Partition is incomplete: {directory}")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Partition schema version differs: {directory}")
        if input_info.get("sha256") != first["input"]["sha256"]:
            raise ValueError(f"Partition input hash differs: {directory}")
        if input_info.get("rows_available") != first["input"]["rows_available"]:
            raise ValueError(f"Partition source row count differs: {directory}")
        if any(manifest.get(key) != first.get(key) for key in comparable_keys):
            raise ValueError(f"Partition model or runtime configuration differs: {directory}")
        start = input_info.get("start_row")
        end = input_info.get("end_row_exclusive")
        if start != expected_row or end != start + input_info.get("rows_requested", -1):
            raise ValueError(
                f"Partitions must be contiguous; expected row {expected_row}, got {start}:{end}."
            )
        if result.get("rows_processed") != input_info["rows_requested"]:
            raise ValueError(f"Partition did not process every requested row: {directory}")
        if result.get("rows_successful", 0) + result.get("rows_failed", 0) != result["rows_processed"]:
            raise ValueError(f"Partition result counts are inconsistent: {directory}")

        partition_rows = 0
        for path in sorted((directory / "predictions").glob("batch_*.parquet")):
            rows = pq.read_metadata(path).num_rows
            if not valid_shard(path, start + partition_rows, rows):
                raise ValueError(f"Partition shard is invalid or out of order: {path}")
            partition_rows += rows
            shards.append(path)
        if partition_rows != input_info["rows_requested"]:
            raise ValueError(f"Partition shard rows do not match its manifest: {directory}")

        expected_row = end
        successful += result["rows_successful"]
        failed += result["rows_failed"]
        elapsed += float(result.get("elapsed_seconds", 0))

    if expected_row != first["input"]["rows_available"]:
        raise ValueError(
            f"Partitions cover {expected_row} of {first['input']['rows_available']} source rows."
        )
    return shards, {
        "rows_processed": expected_row,
        "rows_successful": successful,
        "rows_failed": failed,
        "partition_count": len(partitions),
        "partition_elapsed_seconds_sum": elapsed,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": first["input"]["sha256"],
        "model": first["model"],
    }


def main() -> int:
    args = arguments()
    shards, summary = collect_partition_shards([path.resolve() for path in args.partition_dir])
    labels, probabilities = load_checkpoint_matrices(shards)
    metrics = compute_calibration_metrics(labels, probabilities)
    compare_existing_sl_thresholds(metrics, ROOT / "thresholds" / "77_tasks_final.json")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "method": "Youden index (maximum sensitivity + specificity - 1)",
        "model": summary.pop("model"),
        "input_sha256": summary.pop("input_sha256"),
        "rows": summary,
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output.resolve(), artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
