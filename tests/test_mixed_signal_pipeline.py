"""Synthetic signals exercise real decoders, batching, row alignment and output grouping."""
import base64
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as pipeline
from utils.analysis_pipeline import AnalysisPipeline
from utils.ecg_signal_processor import ECGSignalProcessor
from utils.files_handler import ECGFileHandler, NPYProcessor, XMLProcessor


def write_signal(path, amplitude=1, samples=2500):
    wave = (amplitude * (50 + 20 * np.sin(np.arange(samples) * 2 * np.pi / 250))).astype(np.int16)
    signal = np.repeat(wave[:, None], 12, axis=1)
    if path.suffix.lower() == '.npy':
        # Using a stream preserves uppercase .NPY instead of appending .npy.
        with path.open('wb') as stream:
            np.save(stream, signal)
    else:
        root = ET.Element('RestingECG')
        ET.SubElement(root, 'Waveform').text = 'Synthetic median waveform'
        rhythm = ET.SubElement(root, 'Waveform')
        for lead in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
            entry = ET.SubElement(rhythm, 'LeadData')
            ET.SubElement(entry, 'LeadID').text = lead
            ET.SubElement(entry, 'WaveFormData').text = base64.b64encode(wave.tobytes()).decode()
        ET.ElementTree(root).write(path)
    return signal


def preprocess(df, tmp_path, errors=None):
    (tmp_path / 'preprocessed').mkdir(exist_ok=True)
    return AnalysisPipeline.save_and_preprocess_data(
        df, str(tmp_path / 'outputs'), str(tmp_path / 'preprocessed'), 3, errors=errors
    )


@pytest.mark.parametrize('extensions', [
    ['.xml', '.XML'], ['.npy', '.NPY'], ['.xml', '.npy', '.XML', '.NPY'],
    ['.NPY', '.XML', '.npy', '.xml'],
])
def test_decoders_restore_original_order_before_shared_scaling(tmp_path, monkeypatch, extensions):
    paths = [tmp_path / f'ecg_{i}{ext}' for i, ext in enumerate(extensions)]
    expected = [write_signal(path, i + 1) for i, path in enumerate(paths)]
    df = pd.DataFrame({'ecg_path': list(map(str, paths)), 'label': list(range(len(paths)))},
                      index=np.arange(len(paths)) * 7 + 10000)
    scale = ECGSignalProcessor.scale_ecg_signals
    seen = []

    def check_and_scale(self, df, power_ratio):
        seen.append(df.ecg_path.tolist())
        for actual, original in zip(df.ecg_signal, expected):
            np.testing.assert_array_equal(actual, original)
        return scale(self, df, power_ratio)

    monkeypatch.setattr(ECGSignalProcessor, 'scale_ecg_signals', check_and_scale)
    monkeypatch.setattr(ECGSignalProcessor, 'clean_and_process_ecg_leads', lambda self, df, **kw: df)
    errors = []
    result = preprocess(df, tmp_path, errors)
    assert errors == []
    assert len(seen) == 1  # Scaling is shared, never done independently per format.
    assert result.label.tolist() == list(range(len(paths)))
    assert result.ecg_path.tolist() == seen[0]
    assert all(Path(path).exists() for path in result.ecg_path)
    mixed = len({ext.lower() for ext in extensions}) == 2
    for ext in {ext.lower() for ext in extensions}:
        folder = tmp_path / 'outputs/batch_1'
        if mixed:
            folder /= ext[1:]
        report = pd.read_csv(folder / 'ecg_processing_detailed_report.csv')
        assert set(report.file_id) == {path.stem for path in paths if path.suffix.lower() == ext}
        assert set(report.status) == {'Success'}


@pytest.mark.parametrize('processor,extension', [(NPYProcessor, '.npy'), (XMLProcessor, '.xml')])
def test_processors_preserve_nonconsecutive_indexes_and_exclude_failures(tmp_path, processor, extension):
    paths = [tmp_path / f'{name}{extension}' for name in ('first', 'broken', 'last')]
    write_signal(paths[0], 1)
    paths[1].write_text('not an ECG')
    write_signal(paths[2], 2)
    original = pd.DataFrame({'ecg_path': list(map(str, paths)), 'label': [0, 99, 1]}, index=[44, 87, 10001])
    metadata, signals = processor().process_batch(original, num_workers=2, preprocessing_folder=str(tmp_path))
    assert metadata.index.tolist() == signals.index.tolist() == [44, 10001]
    assert metadata.label.tolist() == [0, 1]
    assert metadata.ecg_path.tolist() == signals.ecg_path.tolist()


def test_short_xml_is_reported_as_failed(tmp_path):
    path = tmp_path / 'short.xml'
    write_signal(path, samples=100)
    processor = XMLProcessor()
    metadata, signals = processor.process_batch(pd.DataFrame({'ecg_path': [str(path)]}))
    assert metadata.empty and signals.empty
    assert processor.report[0][2] == 'Failed'


def test_mixed_batches_above_10000_preserve_metadata_and_scaling_order(tmp_path, monkeypatch):
    count = 10003
    failed = 10001
    names = [f'{i}.xml' if i % 2 == 0 else f'{i}.npy' for i in range(count)]
    df = pd.DataFrame({'ecg_path': names, 'label': np.arange(count)}, index=np.arange(count) * 3 + 20000)

    def parse(self, path):
        number = int(Path(path).stem)
        assert Path(path).suffix == ('.xml' if isinstance(self, XMLProcessor) else '.npy')
        status = 'Failed' if number == failed else 'Success'
        report = (str(number), status, '')
        if isinstance(self, XMLProcessor):
            report = (str(number), 'MHI', status, '')
        signal = None if number == failed else np.broadcast_to(np.float32(number), (2500, 12))
        return report, str(number), signal

    batches, saved = [], []

    def scale(self, df, **kwargs):
        batches.append([int(signal[0, 0]) for signal in df.ecg_signal])
        return df

    def save(ecg_signal, filename):
        saved.append((Path(filename).stem, int(ecg_signal[0, 0])))

    monkeypatch.setattr(XMLProcessor, 'process_single_file', parse)
    monkeypatch.setattr(NPYProcessor, 'process_single_file', parse)
    monkeypatch.setattr(ECGSignalProcessor, 'scale_ecg_signals', scale)
    monkeypatch.setattr(ECGSignalProcessor, 'clean_and_process_ecg_leads', lambda self, df, **kw: df)
    monkeypatch.setattr(ECGFileHandler, 'save_ecg_signal', save)
    errors = []
    result = preprocess(df, tmp_path, errors)
    expected = [i for i in range(count) if i != failed]
    assert batches == [list(range(10000)), [10000, 10002]]
    assert result.label.tolist() == expected
    assert [int(Path(p).stem) for p in result.ecg_path] == expected
    assert saved == [(str(i), i) for i in expected]
    assert len(errors) == 1 and '1 .npy file(s) failed' in errors[0]


@pytest.mark.parametrize('paths,message', [
    (['good.xml', 'unsupported.txt'], 'Unsupported signal extension'),
    (['same.xml', 'same.npy'], 'Ambiguous preprocessing filename'),
    (['one/same.xml', 'two/same.xml'], 'Ambiguous preprocessing filename'),
])
def test_invalid_paths_are_rejected_before_any_decode(tmp_path, monkeypatch, paths, message):
    def unexpected(*args, **kwargs):
        pytest.fail('Invalid inputs must be rejected before processing any batch')
    monkeypatch.setattr(XMLProcessor, 'process_batch', unexpected)
    monkeypatch.setattr(NPYProcessor, 'process_batch', unexpected)
    with pytest.raises(ValueError, match=message):
        preprocess(pd.DataFrame({'ecg_path': paths}), tmp_path)


@pytest.mark.parametrize('mode', ['full_run', 'analysis'])
def test_missing_source_cannot_borrow_a_same_stem_output(tmp_path, monkeypatch, capsys, mode):
    write_signal(tmp_path / 'same.npy')
    # same.xml is missing, but both CSV rows would otherwise map to same.base64.
    df = pd.DataFrame({'afib_5y': [0, 1], 'afib_ecg_file_name': ['same.npy', 'same.xml']})
    monkeypatch.setattr(pipeline, 'load_df', lambda path: df)
    with pytest.raises(SystemExit) as exc:
        pipeline.main(SimpleNamespace(mode=mode, data_path='unused.csv', ecg_signals_path=str(tmp_path)))
    assert exc.value.code == 1
    assert 'Ambiguous preprocessing filename' in capsys.readouterr().out


@pytest.mark.parametrize('broken', [False, True])
def test_full_run_combines_outputs_per_model_and_excludes_stale_failed_files(tmp_path, monkeypatch, broken):
    # Exercise decoding, shared scaling, saved signals, real dataloading and metrics.
    # The 50-ECG integration check covers the expensive cleaning and trained models.
    monkeypatch.setattr(ECGSignalProcessor, 'clean_and_process_ecg_leads', lambda self, df, **kw: df)
    names = ['one.xml', 'two.NPY', 'three.XML', 'four.npy']
    for i, name in enumerate(names):
        write_signal(tmp_path / name, i + 1)
    labels = [0, 1, 0, 1]
    preprocessed = tmp_path / 'preprocessed'
    preprocessed.mkdir()
    if broken:
        names.insert(1, 'broken.npy')
        labels.insert(1, 99)
        (tmp_path / 'broken.npy').write_text('not an ECG')
        ECGFileHandler.save_ecg_signal(np.ones((2500, 12)), str(preprocessed / 'broken.base64'))
    pd.DataFrame({'afib_5y': labels, 'afib_ecg_file_name': names}).to_csv(tmp_path / 'input.csv', index=False)
    (tmp_path / 'key.json').write_text(json.dumps({'HUGGING_FACE_API_KEY': ''}))
    created = []

    class SyntheticModel:
        def __init__(self):
            self.model = torch.nn.Identity()

        def __call__(self, signals):
            return torch.sigmoid(signals.mean(dim=(1, 2))).unsqueeze(1)

    def factory(config):
        created.append(config['model_name'])
        return SyntheticModel()

    monkeypatch.setattr('utils.analysis_pipeline.HeartWiseModelFactory.create_model', factory)
    args = SimpleNamespace(
        mode='full_run', data_path=str(tmp_path / 'input.csv'), ecg_signals_path=str(tmp_path),
        output_folder=str(tmp_path / 'outputs'), preprocessing_folder=str(preprocessed),
        preprocessing_n_workers=3, batch_size=2,
        signal_processing_device='cpu', diagnosis_classifier_device='cpu',
        hugging_face_api_key_path=str(tmp_path / 'key.json'),
    )
    if broken:
        with pytest.raises(SystemExit) as exc:
            pipeline.main(args)
        assert exc.value.code == 1
    else:
        pipeline.main(args)
    assert len(created) == 2
    for model in created:
        outputs = list((tmp_path / 'outputs').glob(f'{model}_*probabilities.csv'))
        assert len(outputs) == 1
        probabilities = pd.read_csv(outputs[0])
        assert probabilities.file_name.tolist() == ['one.base64', 'two.base64', 'three.base64', 'four.base64']
        assert probabilities.ground_truth.tolist() == [0, 1, 0, 1]
        assert len(list((tmp_path / 'outputs').glob(f'{model}_*.json'))) == 1
        assert len(list((tmp_path / 'outputs').glob(f'{model}_*.csv'))) == 2
        assert np.isfinite(probabilities.predictions).all()


def test_blank_file_cells_are_not_checked_as_signal_paths(tmp_path, monkeypatch):
    # pandas 3 keeps NaN in DataFrame.stack(): a row without a file must not reach the path checks
    monkeypatch.setattr(ECGSignalProcessor, 'clean_and_process_ecg_leads', lambda self, df, **kw: df)
    write_signal(tmp_path / 'one.npy')
    pd.DataFrame({'afib_5y': [0, 1], 'afib_ecg_file_name': ['one.npy', None]}).to_csv(
        tmp_path / 'input.csv', index=False)
    args = SimpleNamespace(
        mode='preprocessing', data_path=str(tmp_path / 'input.csv'), ecg_signals_path=str(tmp_path),
        output_folder=str(tmp_path / 'outputs'), preprocessing_folder=str(tmp_path / 'preprocessed'),
        preprocessing_n_workers=1,
    )
    pipeline.main(args)
    assert (tmp_path / 'preprocessed' / 'one.base64').exists()
