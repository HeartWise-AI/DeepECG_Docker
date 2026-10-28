"""CPU calibration helpers for the EfficientNetV2 77-class model.

The production pipeline serializes preprocessed ECGs as base64 before inference.
Calibration does not need that intermediate representation: this module reuses the
same decoder, cohort scaling and cleaning implementations, then passes the cleaned
arrays directly to the model in bounded inference minibatches.
"""

from __future__ import annotations

import json
import logging
import math
import os
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

from utils.constants import ECG_CATEGORIES, ECG_PATTERNS, PTBXL_POWER_RATIO
from utils.ecg_signal_processor import ECGSignalProcessor
from utils.files_handler import NPYProcessor


LABEL_COLUMN_PREFIX = "label__"
PROBABILITY_COLUMN_PREFIX = "probability__"


class CalibrationECGSignalProcessor(ECGSignalProcessor):
    """Exact production preprocessing with cohort-wide NumPy work parallelized.

    ``ECGSignalProcessor`` already parallelizes the final per-signal cleaning, but
    computes every FFT used for cohort scaling and spectral-peak detection in a
    serial loop.  Those FFTs are independent.  ``executor.map`` preserves their
    order, so the subsequent NumPy mean and every downstream operation are the
    same as the production implementation.
    """

    def __init__(self, max_workers: int, fs: int = 250):
        super().__init__(fs=fs)
        self.max_workers = max_workers

    def plot_mean_spectrum(self, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        signals = signals.astype(np.float32)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            spectra = list(executor.map(self.compute_magnitude_spectrum, signals))
        if not spectra:
            raise ValueError("Cannot calculate a spectrum for an empty ECG batch.")
        fft_freq = spectra[-1][0]
        mean_magnitude_spectrum = np.mean([item[1] for item in spectra], axis=0)
        return fft_freq, mean_magnitude_spectrum

    def scale_ecg_signals(self, df: pd.DataFrame, power_ratio: float) -> pd.DataFrame:
        _, mean_magnitude_spectrum = self.plot_mean_spectrum(
            np.array(df["ecg_signal"].tolist())[:, :, 0]
        )
        factor = power_ratio / self.compute_average_spectral_power(mean_magnitude_spectrum)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            scaled = list(executor.map(
                lambda signal: self.adjust_spectral_power(signal, factor),
                df["ecg_signal"].tolist(),
            ))
        result = df.copy()
        result["ecg_signal"] = scaled
        return result


def normalized_name(value: str) -> str:
    """Normalize Unicode spacing without otherwise changing a clinical label."""
    return " ".join(unicodedata.normalize("NFKC", value).replace("\xa0", " ").split())


def resolve_label_columns(schema_names: Iterable[str]) -> dict[str, str]:
    """Map runtime labels to parquet columns, accepting Unicode-space differences."""
    names = list(schema_names)
    normalized: dict[str, list[str]] = {}
    for name in names:
        normalized.setdefault(normalized_name(name), []).append(name)

    resolved = {}
    for label in ECG_PATTERNS:
        if label in names:
            resolved[label] = label
            continue
        matches = normalized.get(normalized_name(label), [])
        if len(matches) != 1:
            detail = "missing" if not matches else f"ambiguous: {matches}"
            raise ValueError(f"Parquet label column {label!r} is {detail}.")
        resolved[label] = matches[0]
    return resolved


def parse_path_maps(values: Sequence[str]) -> list[tuple[str, str]]:
    """Parse repeated ``SOURCE=DESTINATION`` prefix mappings."""
    mappings = []
    for value in values:
        source, separator, destination = value.partition("=")
        if not separator or not source or not destination:
            raise ValueError(f"Invalid path map {value!r}; expected SOURCE=DESTINATION.")
        mappings.append((source.rstrip(os.sep), destination.rstrip(os.sep)))
    return mappings


def remap_path(path: str, mappings: Sequence[tuple[str, str]]) -> str:
    """Apply the first matching path-prefix mapping."""
    for source, destination in mappings:
        if path == source or path.startswith(source + os.sep):
            return destination + path[len(source):]
    return path


def preprocess_npy_batch(
    paths: Sequence[str],
    preprocessing_workers: int,
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[str, str, str]]]:
    """Decode, scale and clean one cohort batch without writing intermediate files.

    Returns successful zero-based positions within ``paths``, cleaned arrays in the
    same order, and the processor's aggregate-safe report entries. Report entries
    are returned for counting only and must not be persisted because file stems can
    be identifying.
    """
    processor = NPYProcessor()
    # NPYProcessor's general-purpose exception log includes the source path. Suppress
    # that logger for this protected cohort; only aggregate failure counts leave the run.
    file_logger = logging.getLogger("utils.files_handler")
    previous_disabled = file_logger.disabled
    file_logger.disabled = True
    try:
        with ThreadPoolExecutor(max_workers=preprocessing_workers) as executor:
            decoded = list(executor.map(processor.process_single_file, paths))
    finally:
        file_logger.disabled = previous_disabled
    processor.report.extend(item[0] for item in decoded)
    successes = [
        (position, item[2])
        for position, item in enumerate(decoded)
        if item[2] is not None
    ]
    if not successes:
        return np.empty(0, dtype=np.int64), [], processor.report

    positions = np.fromiter((item[0] for item in successes), dtype=np.int64)
    # Positional placeholders avoid carrying protected source paths any further.
    signals = pd.DataFrame({
        "ecg_path": positions.astype(str),
        "ecg_signal": [item[1] for item in successes],
    })
    signal_processor = CalibrationECGSignalProcessor(preprocessing_workers)
    scaled = signal_processor.scale_ecg_signals(signals, power_ratio=PTBXL_POWER_RATIO)
    cleaned = signal_processor.clean_and_process_ecg_leads(
        scaled,
        max_workers=preprocessing_workers,
    )
    if len(cleaned) != len(positions):
        raise RuntimeError("Signal cleaning changed the number of successfully decoded ECGs.")
    return positions, cleaned["ecg_signal"].tolist(), processor.report


@torch.no_grad()
def infer_minibatches(
    model,
    cleaned_signals: Sequence[np.ndarray],
    inference_batch_size: int,
    device: str = "cpu",
) -> np.ndarray:
    """Run finalized signals through an eval-mode model in bounded minibatches."""
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive")
    model.model.eval()
    rows = []
    for start in range(0, len(cleaned_signals), inference_batch_size):
        arrays = np.stack(cleaned_signals[start:start + inference_batch_size])
        tensor = torch.from_numpy(arrays).transpose(1, 2).float().to(device)
        probabilities = model(tensor).detach().cpu().numpy().astype(np.float32, copy=False)
        expected_shape = (len(arrays), len(ECG_PATTERNS))
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"EfficientNet returned shape {probabilities.shape}; "
                f"expected {expected_shape}."
            )
        if not np.isfinite(probabilities).all():
            raise ValueError("EfficientNet returned non-finite probabilities.")
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("EfficientNet returned values outside the probability range [0, 1].")
        rows.append(probabilities)
    if not rows:
        return np.empty((0, len(ECG_PATTERNS)), dtype=np.float32)
    return np.concatenate(rows)


def label_column(index: int) -> str:
    return f"{LABEL_COLUMN_PREFIX}{index:02d}"


def probability_column(index: int) -> str:
    return f"{PROBABILITY_COLUMN_PREFIX}{index:02d}"


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    valid = np.isfinite(probabilities)
    labels = labels[valid].astype(np.uint8, copy=False)
    probabilities = probabilities[valid].astype(np.float64, copy=False)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    result = {
        "n": int(len(labels)),
        "n_positive": positives,
        "n_negative": negatives,
        "auc": None,
        "auprc": None,
        "threshold": None,
        "youden_j": None,
        "f1": None,
        "prevalence_gt %": (positives / len(labels) * 100) if len(labels) else None,
        "prevalence_pred %": None,
    }
    if not positives or not negatives:
        return result

    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    best = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best])
    if not math.isfinite(threshold):
        return result
    binary = probabilities >= threshold
    result.update({
        "auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "threshold": threshold,
        "youden_j": float(tpr[best] - fpr[best]),
        "f1": float(f1_score(labels, binary)),
        "prevalence_pred %": float(binary.mean() * 100),
    })
    return result


def compute_calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Compute robust per-label and category Youden metrics."""
    expected = (labels.shape[0], len(ECG_PATTERNS))
    if labels.shape != expected or probabilities.shape != expected:
        raise ValueError(
            f"Expected label and probability matrices shaped {expected}; "
            f"got {labels.shape} and {probabilities.shape}."
        )

    per_label = {
        label: _binary_metrics(labels[:, index], probabilities[:, index])
        for index, label in enumerate(ECG_PATTERNS)
    }
    categories = {}
    label_indexes = {label: index for index, label in enumerate(ECG_PATTERNS)}
    for category, category_labels in ECG_CATEGORIES.items():
        usable = [label for label in category_labels if per_label[label]["threshold"] is not None]
        category_result = {
            "labels_total": len(category_labels),
            "labels_with_threshold": len(usable),
            "macro_auc": None,
            "macro_auprc": None,
            "macro_f1": None,
            "micro_auc": None,
            "micro_auprc": None,
            "micro_f1": None,
            "threshold": None,
            "youden_j": None,
            "prevalence_gt %": None,
            "prevalence_pred %": None,
        }
        if usable:
            category_result.update({
                "macro_auc": float(np.mean([per_label[label]["auc"] for label in usable])),
                "macro_auprc": float(np.mean([per_label[label]["auprc"] for label in usable])),
                "macro_f1": float(np.mean([per_label[label]["f1"] for label in usable])),
            })
            indexes = [label_indexes[label] for label in usable]
            micro = _binary_metrics(labels[:, indexes].ravel(), probabilities[:, indexes].ravel())
            for key in (
                "auc", "auprc", "f1", "threshold", "youden_j",
                "prevalence_gt %", "prevalence_pred %",
            ):
                category_result[("micro_" + key) if key in {"auc", "auprc", "f1"} else key] = micro[key]
        categories[category] = category_result
    return {"categories": categories, "labels": per_label}


def compare_existing_sl_thresholds(metrics: dict, path: Path) -> None:
    """Annotate label metrics with the repository's historical SL values when present."""
    if not path.is_file():
        return
    existing = json.loads(path.read_text())
    for label, result in metrics["labels"].items():
        historical = existing.get(label, {}).get("SL_sens_spec_threshold")
        result["existing_sl_threshold"] = historical
        result["delta_from_existing_sl"] = (
            float(result["threshold"] - historical)
            if result["threshold"] is not None and historical is not None else None
        )
