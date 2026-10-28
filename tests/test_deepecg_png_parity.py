"""Parity of the Docker DeepECG-PNG model with the DeepECG-PNG code it was exported from.

Needs the pinned weights snapshot downloaded from Hugging Face
(weights/deepecg_png/heartwise--deepecg_png/<commit>, models/manifest.json) and the reference
file written by `tools/export_deepecg_png.py --reference-out FILE` ($DEEPECG_PNG_REFERENCE): real
validation PNGs with the features and probabilities computed by the training code. Skipped when
either is missing.

Checks, under this image's torch / transformers:
  * preprocessing: the same input tensor as DeepECG-PNG's process_image;
  * features: cosine >= 0.9999 with the training-time cached features and with the
    reference features;
  * probabilities: within 1e-3 of the reference for every output (SHD, LVEF<=40, LVEF<50,
    iAFib5 and the 75 classes with a head).
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import deepecg_png  # noqa: E402
from utils.image_preprocessing import load_ecg_image  # noqa: E402
from utils.model_manifest import ModelPreparationError, model_entry  # noqa: E402


def _pinned_weights():
    """The pinned snapshot folder of heartwise/deepecg_png, or '' if the manifest has no entry."""
    try:
        revision = model_entry(deepecg_png.HF_REPO_ID)['revision']
    except ModelPreparationError:
        return ''
    return os.path.join(deepecg_png.WEIGHTS_DIR, deepecg_png.HF_REPO_ID.replace('/', '--'), revision)


REFERENCE = os.environ.get("DEEPECG_PNG_REFERENCE", "")
pytestmark = pytest.mark.skipif(
    not (REFERENCE and os.path.exists(REFERENCE)
         and os.path.exists(os.path.join(_pinned_weights(), "encoder.safetensors"))),
    reason="needs the pinned heartwise/deepecg_png snapshot and $DEEPECG_PNG_REFERENCE")


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REFERENCE))


@pytest.fixture(scope="module")
def bank():
    return deepecg_png._get_bank(torch.device("cpu"), "")


@pytest.fixture(scope="module")
def ours(ref, bank):
    files = [str(f) for f in ref["files"]]
    x = torch.stack([load_ecg_image(f, bank.config["preprocessing"]) for f in files])
    feats = torch.cat([bank.features(x[i:i + 32]) for i in range(0, len(x), 32)]).numpy()
    probs = torch.cat([bank.probabilities(x[i:i + 32], [str(o) for o in ref["outputs"]])
                       for i in range(0, len(x), 32)]).numpy()
    return {"pixels": x[:len(ref["pixels"])].numpy(), "features": feats, "probabilities": probs}


def _cosine(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def test_preprocessing_is_the_training_preprocessing(ref, ours):
    assert ours["pixels"].shape == ref["pixels"].shape
    assert np.abs(ours["pixels"] - ref["pixels"]).max() < 1e-6


def test_features_match_training_and_reference(ref, ours):
    assert _cosine(ours["features"], ref["cached_features"]).min() >= 0.9999
    assert _cosine(ours["features"], ref["features"]).min() >= 0.9999


def test_probabilities_match_reference(ref, ours):
    assert ours["probabilities"].shape == ref["probabilities"].shape
    assert np.abs(ours["probabilities"] - ref["probabilities"]).max() < 1e-3
