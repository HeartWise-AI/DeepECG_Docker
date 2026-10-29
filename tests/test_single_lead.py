"""Single-lead NPY input: lead expansion, preprocessing on the lead that has signal, model output order."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as pipeline
from utils.analysis_pipeline import AnalysisPipeline, predict_77_classes
from utils.constants import WCR_COLUMN_CONVERSION
from utils.ecg_signal_processor import ECGSignalProcessor
from utils.files_handler import ECGFileHandler, LEAD_ORDER, NPYProcessor, expand_single_lead, single_lead_slots


def wave(samples=2500, amplitude=1.0):
    return (amplitude * np.sin(2 * np.pi * 1.2 * np.arange(samples) / 250)).astype(np.float32)


def other_leads(x, lead):
    """Largest absolute value outside `lead`. The power scaling runs an FFT across the leads, which leaves
    about 1e-17 in leads that were zero, so 'zero' below means below 1e-6."""
    return np.abs(np.delete(x, lead, axis=-1)).max()


def test_single_lead_slots():
    assert single_lead_slots('none') is None and single_lead_slots(None) is None
    assert single_lead_slots('all') == list(range(12))
    assert single_lead_slots('V1') == [LEAD_ORDER.index('V1')] == [6]
    with pytest.raises(ValueError):
        single_lead_slots('V7')


def test_expand_single_lead():
    w = np.stack([wave(), 2 * wave()])
    x = expand_single_lead(w, single_lead_slots('all'))
    assert x.shape == (2, 2500, 12) and all(np.array_equal(x[..., k], w) for k in range(12))
    x = expand_single_lead(w[0], single_lead_slots('II'))
    assert x.shape == (2500, 12) and np.array_equal(x[:, 1], w[0]) and not np.delete(x, 1, axis=1).any()


@pytest.mark.parametrize('shape', [(2500,), (2500, 1), (1, 2500)])
def test_npy_processor_expands_single_lead_files(tmp_path, shape):
    path = tmp_path / 'one.npy'
    np.save(path, wave().reshape(shape))
    report, _, x = NPYProcessor('aVF').process_single_file(str(path))
    assert report[1] == 'Success' and x.shape == (2500, 12)
    assert np.array_equal(x[:, 5], wave()) and not np.delete(x, 5, axis=1).any()
    report, _, x = NPYProcessor().process_single_file(str(path))
    assert x is None and report[2] == 'Single-lead signal: set single_lead'


def test_npy_processor_keeps_12_lead_files_with_single_lead_set(tmp_path):
    path = tmp_path / 'twelve.npy'
    signal = np.repeat(wave()[:, None], 12, axis=1)
    np.save(path, signal)
    _, _, x = NPYProcessor('all').process_single_file(str(path))
    assert np.array_equal(x, signal)


def test_scaling_uses_the_reference_lead():
    x = expand_single_lead(np.stack([wave(amplitude=a) for a in (1, 2, 3)]), single_lead_slots('V1'))

    def scaled(processor):
        df = pd.DataFrame({'ecg_path': ['a', 'b', 'c'], 'ecg_signal': list(x)})
        with np.errstate(all='ignore'):
            return np.stack(processor.scale_ecg_signals(df, power_ratio=3.0).ecg_signal.to_list())
    assert np.isnan(scaled(ECGSignalProcessor())).all()  # lead I is empty: the cohort spectrum is zero
    out = scaled(ECGSignalProcessor(ref_lead=6))
    assert np.isfinite(out).all() and other_leads(out, 6) < 1e-6


def test_preprocessing_writes_single_lead_signals_as_12_lead(tmp_path):
    paths = [tmp_path / f'ecg_{i}.npy' for i in range(3)]
    for i, path in enumerate(paths):
        np.save(path, wave(amplitude=i + 1))
    (tmp_path / 'pre').mkdir()
    errors = []
    result = AnalysisPipeline.save_and_preprocess_data(
        pd.DataFrame({'ecg_path': list(map(str, paths))}), str(tmp_path / 'out'), str(tmp_path / 'pre'), 2,
        errors=errors, single_lead='II')
    assert errors == [] and len(result) == 3
    for path in result.ecg_path:
        x = ECGFileHandler.load_ecg_signal(path)
        assert x.shape == (2500, 12) and np.isfinite(x).all() and x[:, 1].any() and other_leads(x, 1) < 1e-6


def test_predict_77_classes_reorders_wcr_outputs_only():
    def model(x):
        return torch.arange(77, dtype=torch.float32).repeat(len(x), 1)
    x = torch.zeros(2, 12, 2500)
    assert predict_77_classes(model, 'wcr_77_classes', x)[0].tolist() == [float(i) for i in WCR_COLUMN_CONVERSION]
    assert predict_77_classes(model, 'efficientnetv2_77_classes', x)[0].tolist() == list(map(float, range(77)))


def test_main_rejects_a_bad_single_lead(monkeypatch, capsys):
    monkeypatch.setattr(pipeline, 'load_df', lambda path: pd.DataFrame())
    with pytest.raises(SystemExit) as exc:
        pipeline.main(SimpleNamespace(mode='full_run', data_path='unused.csv', single_lead='V7'))
    assert exc.value.code == 1 and 'single_lead must be' in capsys.readouterr().out
