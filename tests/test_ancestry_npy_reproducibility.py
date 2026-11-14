"""Unit checks for the image-to-image Ancestry prediction comparison."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'ancestry_npy_comparison', ROOT / 'scripts' / 'test_ancestry_npy_reproducibility.py')
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def payload(ensemble=0.25, fold=0.2):
    return {
        'schema_version': 1,
        'input_scale': 0.00488,
        'inputs_sha256': {'approved.npy': 'a' * 64},
        'model_files': ['raw/fold_1/bs_128/checkpoint_best.pt'],
        'environment': {'torch': 'test'},
        'predictions': {
            'approved.npy': {'ensemble': ensemble, 'folds': [fold, 0.3]},
        },
    }


def test_identical_predictions_pass_at_zero_tolerance():
    report = comparison.compare(payload(), payload(), 0.0)
    assert report['pass']
    assert report['values_compared'] == 3
    assert report['max_absolute_difference'] == 0.0


def test_prediction_difference_is_reported():
    report = comparison.compare(payload(), payload(ensemble=0.2501), 0.0)
    assert not report['pass']
    assert report['differing_values'] == 1
    assert report['max_absolute_difference'] == pytest.approx(0.0001)


def test_cross_runtime_roundoff_passes_at_documented_tolerance():
    report = comparison.compare(payload(), payload(ensemble=0.2500004), 1e-6)
    assert report['pass']
    assert report['differing_values'] == 0


def test_input_hash_mismatch_is_rejected():
    candidate = payload()
    candidate['inputs_sha256'] = {'approved.npy': 'b' * 64}
    with pytest.raises(ValueError, match='inputs_sha256'):
        comparison.compare(payload(), candidate, 0.0)


def test_image_id_is_trimmed_and_uses_inspect(monkeypatch):
    called = []

    def inspect(command, text):
        called.append((command, text))
        return 'sha256:abc\n'

    monkeypatch.setattr(comparison.subprocess, 'check_output', inspect)
    assert comparison.image_id('candidate:tag') == 'sha256:abc'
    assert called == [(['docker', 'image', 'inspect', 'candidate:tag',
                        '--format', '{{.Id}}'], True)]
