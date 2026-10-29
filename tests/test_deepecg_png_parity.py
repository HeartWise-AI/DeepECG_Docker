"""Parity of the Docker DeepECG-PNG model with the DeepECG-PNG code it was exported from.

Needs the six pinned SigLIP2 snapshots downloaded from Hugging Face and an approved reference
archive selected with $DEEPECG_PNG_REFERENCE: validation PNGs with the features and probabilities
computed by the DeepECG-PNG training code. Skipped when either is missing.

Checks, under this image's torch / transformers:
  * preprocessing: the same input tensor as DeepECG-PNG's process_image;
  * features: cosine >= 0.9999 with the training-time cached features and with the
    reference features;
  * probabilities: within 1e-3 of the reference for every output (SHD, LVEF<=40, LVEF<50,
    iAFib5 and all 77 diagnostic classes).
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import siglip2_wrapper  # noqa: E402
from utils.image_preprocessing import load_ecg_image  # noqa: E402
from utils.model_manifest import ModelPreparationError, model_entry  # noqa: E402


def _pinned_weights(model_key):
    """The pinned snapshot folder for one SigLIP2 manifest entry, or '' if unavailable."""
    repo_id = siglip2_wrapper.MODEL_REPOSITORIES[model_key]
    try:
        revision = model_entry(repo_id)['revision']
    except ModelPreparationError:
        return ''
    return os.path.join('weights', model_key, repo_id.replace('/', '--'), revision)


HEAD_SPECS = [
    ('siglip2_77_classes', siglip2_wrapper.ALL_77),
    ('siglip2_afib_5y', ('iAFib5',)),
    ('siglip2_lvef_equal_under_40', ('LVEF_le40',)),
    ('siglip2_lvef_under_50', ('LVEF_lt50',)),
    ('siglip2_shd', ('SHD',)),
]


REFERENCE = os.environ.get("DEEPECG_PNG_REFERENCE", "")
pytestmark = pytest.mark.skipif(
    not (REFERENCE and os.path.exists(REFERENCE)
         and os.path.exists(os.path.join(
             _pinned_weights(siglip2_wrapper.BASE_MODEL_KEY), "image_encoder.safetensors"))
         and all(os.path.exists(os.path.join(_pinned_weights(key), "classifier.safetensors"))
                 for key, _ in HEAD_SPECS)),
    reason="needs all pinned SigLIP2 snapshots and $DEEPECG_PNG_REFERENCE")


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REFERENCE))


@pytest.fixture(scope="module")
def bank():
    return siglip2_wrapper._DeepECG_SigLIP2_Bank(
        _pinned_weights(siglip2_wrapper.BASE_MODEL_KEY), torch.device("cpu"))


@pytest.fixture(scope="module")
def heads(bank):
    return [bank.load_head(_pinned_weights(key), key, outputs) for key, outputs in HEAD_SPECS]


@pytest.fixture(scope="module")
def ours(ref, bank, heads):
    files = [str(f) for f in ref["files"]]
    x = torch.stack([load_ecg_image(f, bank.config["preprocessing"]) for f in files])
    feats = torch.cat([bank.features(x[i:i + 32]) for i in range(0, len(x), 32)]).numpy()
    probabilities = []
    for i in range(0, len(x), 32):
        by_output = {}
        for head in heads:
            values = bank.probabilities(x[i:i + 32], head)
            by_output.update({name: values[:, j] for j, name in enumerate(head['outputs'])})
        probabilities.append(torch.stack([by_output[str(name)] for name in ref['outputs']], dim=1))
    probs = torch.cat(probabilities).numpy()
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
