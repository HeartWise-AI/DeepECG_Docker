"""Synthetic validation of the in-memory EfficientNet threshold workflow."""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch

from scripts.calibrate_efficientnet_thresholds import (
    load_checkpoint_matrices,
    valid_shard,
    write_shard,
)
from utils.constants import ECG_PATTERNS
from utils.ecg_signal_processor import ECGSignalProcessor
from utils.threshold_calibration import (
    CalibrationECGSignalProcessor,
    compute_calibration_metrics,
    infer_minibatches,
    parse_path_maps,
    preprocess_npy_batch,
    probability_column,
    remap_path,
    resolve_label_columns,
)


def test_label_resolution_accepts_non_breaking_space_only():
    columns = list(ECG_PATTERNS)
    index = columns.index("RV1 + SV6 > 11 mm")
    columns[index] = "RV1 + SV6\xa0> 11 mm"
    resolved = resolve_label_columns(columns)
    assert resolved["RV1 + SV6 > 11 mm"] == "RV1 + SV6\xa0> 11 mm"
    columns.remove("Acute MI")
    with pytest.raises(ValueError, match="Acute MI.*missing"):
        resolve_label_columns(columns)


def test_path_maps_only_replace_a_path_prefix():
    mappings = parse_path_maps(["/media/data1=/mnt/data1"])
    assert remap_path("/media/data1/cohort/a.npy", mappings) == "/mnt/data1/cohort/a.npy"
    assert remap_path("/media/data10/cohort/a.npy", mappings) == "/media/data10/cohort/a.npy"
    with pytest.raises(ValueError, match="SOURCE=DESTINATION"):
        parse_path_maps(["broken"])


def test_inference_minibatch_size_does_not_change_per_sample_results():
    class SyntheticModel:
        model = torch.nn.Identity()

        def __call__(self, tensor):
            value = torch.sigmoid(tensor.mean(dim=(1, 2))).unsqueeze(1)
            return value.repeat(1, len(ECG_PATTERNS))

    signals = [np.full((2500, 12), value, dtype=np.float32) for value in (-2, -1, 0, 1, 2)]
    one = infer_minibatches(SyntheticModel(), signals, 1)
    four = infer_minibatches(SyntheticModel(), signals, 4)
    np.testing.assert_array_equal(one, four)


def test_inference_rejects_invalid_probabilities():
    class InvalidModel:
        model = torch.nn.Identity()

        def __call__(self, tensor):
            return torch.full((len(tensor), len(ECG_PATTERNS)), float("nan"))

    signals = [np.zeros((2500, 12), dtype=np.float32)]
    with pytest.raises(ValueError, match="non-finite"):
        infer_minibatches(InvalidModel(), signals, 1)


def test_parallel_cohort_operations_match_production_implementation():
    rng = np.random.default_rng(20261028)
    arrays = [rng.normal(size=(2500, 12)).astype(np.float64) for _ in range(8)]
    original = pd.DataFrame({
        "ecg_path": [str(index) for index in range(len(arrays))],
        "ecg_signal": arrays,
    })
    production = ECGSignalProcessor()
    calibration = CalibrationECGSignalProcessor(max_workers=4)

    expected_freq, expected_spectrum = production.plot_mean_spectrum(
        np.stack(arrays)[:, :, 0]
    )
    actual_freq, actual_spectrum = calibration.plot_mean_spectrum(
        np.stack(arrays)[:, :, 0]
    )
    np.testing.assert_array_equal(actual_freq, expected_freq)
    np.testing.assert_array_equal(actual_spectrum, expected_spectrum)

    expected_scaled = production.scale_ecg_signals(original.copy(), power_ratio=2.5)
    actual_scaled = calibration.scale_ecg_signals(original.copy(), power_ratio=2.5)
    for expected, actual in zip(expected_scaled["ecg_signal"], actual_scaled["ecg_signal"]):
        np.testing.assert_array_equal(actual, expected)


def test_metrics_are_json_safe_and_leave_single_class_threshold_unavailable():
    labels = np.zeros((4, len(ECG_PATTERNS)), dtype=np.uint8)
    probabilities = np.full((4, len(ECG_PATTERNS)), 0.1, dtype=np.float32)
    labels[:, 0] = [0, 0, 1, 1]
    probabilities[:, 0] = [0.1, 0.4, 0.6, 0.9]
    metrics = compute_calibration_metrics(labels, probabilities)
    assert metrics["labels"][ECG_PATTERNS[0]]["threshold"] == pytest.approx(0.6)
    assert metrics["labels"][ECG_PATTERNS[1]]["threshold"] is None
    json.dumps(metrics, allow_nan=False)


def test_preprocessing_stays_in_memory_without_base64_files(tmp_path, monkeypatch):
    paths = []
    for index in range(2):
        path = tmp_path / f"synthetic_{index}.npy"
        np.save(path, np.full((2500, 12, 1), index + 1, dtype=np.float64))
        paths.append(str(path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ECGSignalProcessor, "scale_ecg_signals", lambda self, frame, **kw: frame)
    monkeypatch.setattr(
        ECGSignalProcessor, "clean_and_process_ecg_leads", lambda self, frame, **kw: frame
    )
    positions, signals, reports = preprocess_npy_batch(paths, preprocessing_workers=2)
    assert positions.tolist() == [0, 1]
    assert len(signals) == len(reports) == 2
    assert not list(tmp_path.rglob("*.base64"))


def test_checkpoint_shard_has_no_source_paths_and_round_trips(tmp_path):
    labels = np.zeros((3, len(ECG_PATTERNS)), dtype=np.uint8)
    probabilities = np.full((3, len(ECG_PATTERNS)), 0.25, dtype=np.float32)
    path = tmp_path / "batch.parquet"
    write_shard(path, start=100, labels=labels, probabilities=probabilities)
    assert valid_shard(path, start=100, rows=3)
    assert "path" not in " ".join(pq.read_schema(path).names)
    actual_labels, actual_probabilities = load_checkpoint_matrices([path])
    np.testing.assert_array_equal(actual_labels, labels)
    np.testing.assert_array_equal(actual_probabilities, probabilities)
    assert probability_column(0) in pq.read_schema(path).names
