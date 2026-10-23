"""DeepECG-PNG in the pipeline, on CPU, without real weights or patient data.

A tiny random encoder + probes is written in the exported-weights format, synthetic PNG
pages are drawn, and main() runs end to end (the BERT classifier is replaced by a stub, so
nothing is downloaded):
  * a CSV with only `png_path`: every PNG through the 5 outputs, predictions only, BERT
    never loaded, one combined row per image;
  * the same CSV with labels and report text: metrics for every task;
  * PNG rows never reach the signal preprocessing; a missing PNG is reported and skipped;
  * the preprocessing is the training preprocessing; the 0/1 prediction is prob >= threshold.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, ImageDraw

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO)
_cwd = os.getcwd()
os.chdir(REPO)   # models/wcr_ecg_transformer.py loads ./fairseq-signals at import
import main as pipeline  # noqa: E402
from models import deepecg_png  # noqa: E402
from models.bert_classifier import BertClassifier_En_Fr  # noqa: E402
from utils import analysis_pipeline  # noqa: E402
from utils.constants import DIAGNOSIS_TO_FILE_COLUMNS, ECG_PATTERNS  # noqa: E402
from utils.image_preprocessing import load_ecg_image  # noqa: E402
os.chdir(_cwd)

THRESHOLDS = {'SHD': 0.5, 'LVEF_le40': 0.3, 'LVEF_lt50': 0.4, 'iAFib5': 0.2}
NO_HEAD = ['Lead misplacement', 'Brugada']
PREPROCESSING = {'resize': [32, 32], 'resize_mode': 'stretch', 'color_mode': 'grayscale',
                 'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}


def write_tiny_weights(folder):
    """A random 1-layer CLIP tower (32x32 images, 2x2 patch grid) + probes, exported format."""
    from safetensors.torch import save_file
    from transformers import CLIPVisionConfig, CLIPVisionModel
    torch.manual_seed(0)
    vision = dict(hidden_size=32, intermediate_size=64, projection_dim=32, num_hidden_layers=1,
                  num_attention_heads=2, num_channels=3, image_size=32, patch_size=16,
                  hidden_act='quick_gelu', layer_norm_eps=1e-5, attention_dropout=0.0,
                  initializer_range=0.02, initializer_factor=1.0)
    tower = deepecg_png._vision_tower(CLIPVisionModel(CLIPVisionConfig(**vision)))
    os.makedirs(folder, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tower.state_dict().items() if not k.endswith('position_ids')},
              os.path.join(folder, 'encoder.safetensors'))
    with open(os.path.join(folder, 'config.json'), 'w') as f:
        json.dump({'vision_config': vision, 'rope': {'mode': '2d', 'grid': 2, 'head_dim': 16, 'theta': 10000.0},
                   'preprocessing': PREPROCESSING}, f)
    g = torch.Generator().manual_seed(1)
    labels = [p for p in ECG_PATTERNS if p not in NO_HEAD]
    tensors = {
        'shd_lvef.weight': torch.randn(3, 32, generator=g, dtype=torch.float64),
        'shd_lvef.bias': torch.zeros(3, dtype=torch.float64),
        'iaf5.weight': torch.randn(1, 32, generator=g, dtype=torch.float64),
        'iaf5.bias': torch.zeros(1, dtype=torch.float64),
        'cls77.weight': torch.randn(len(labels), 32, generator=g, dtype=torch.float64),
        'cls77.bias': torch.zeros(len(labels), dtype=torch.float64),
        'cls77.mean': torch.zeros(32, dtype=torch.float64),
        'cls77.scale': torch.ones(32, dtype=torch.float64),
    }
    save_file(tensors, os.path.join(folder, 'heads.safetensors'))
    with open(os.path.join(folder, 'heads.json'), 'w') as f:
        json.dump({'heads': {'shd_lvef': {'outputs': ['SHD', 'LVEF_le40', 'LVEF_lt50'], 'standardize': False},
                             'iaf5': {'outputs': ['iAFib5'], 'standardize': False},
                             'cls77': {'outputs': labels, 'standardize': True}},
                   'labels_77': list(ECG_PATTERNS)}, f)
    with open(os.path.join(folder, 'thresholds.json'), 'w') as f:
        json.dump({'thresholds': THRESHOLDS}, f)


def draw_page(path, k):
    img = Image.new('RGBA', (125, 62), 'white')
    d = ImageDraw.Draw(img)
    d.line([(0, 31), (124, 10 + (7 * k) % 40)], fill=(0, 0, 0, 255), width=2)
    d.line([(0, 5 + k % 20), (124, 50)], fill=(200, 0, 0, 255), width=1)
    img.save(path)


class FakeBert:
    """Stands in for BertClassifier_En_Fr: report text -> 77 probabilities."""
    calls = 0

    def __call__(self, texts):
        FakeBert.calls += 1
        out = torch.zeros(len(texts), len(ECG_PATTERNS))
        for i, t in enumerate(texts):
            for j in range(len(ECG_PATTERNS)):
                out[i, j] = 1.0 if (len(t) + j + i) % 3 == 0 else 0.0
        return out


@pytest.fixture
def world(tmp_path, monkeypatch):
    weights = tmp_path / 'weights'
    write_tiny_weights(str(weights))
    monkeypatch.setenv(deepecg_png.WEIGHTS_DIR_ENV, str(weights))
    deepecg_png._BANKS.clear()
    pngs = tmp_path / 'pngs'
    pngs.mkdir()
    for k in range(12):
        draw_page(str(pngs / f'ecg_{k}.png'), k)
    real_factory = analysis_pipeline.HeartWiseModelFactory
    FakeBert.calls = 0

    class Factory:
        @staticmethod
        def create_model(model_config):
            if model_config['model_name'] == BertClassifier_En_Fr.name:
                return FakeBert()
            return real_factory.create_model(model_config)
    monkeypatch.setattr(analysis_pipeline, 'HeartWiseModelFactory', Factory)
    yield tmp_path
    deepecg_png._BANKS.clear()


def run(tmp_path, df, name, mode='full_run', use_deepecg_png=True, expect_exit=None):
    csv = tmp_path / f'{name}.csv'
    df.to_csv(csv, index=False)
    out = tmp_path / f'out_{name}'
    import argparse
    ns = argparse.Namespace(
        diagnosis_classifier_device='cpu', signal_processing_device='cpu', data_path=str(csv),
        batch_size=5, output_folder=str(out), hugging_face_api_key_path=str(tmp_path / 'key.json'),
        use_wcr=False, use_efficientnet=False, use_deepecg_png=use_deepecg_png,
        ecg_signals_path=str(tmp_path / 'pngs'), mode=mode,
        preprocessing_folder=str(tmp_path / f'prep_{name}'), preprocessing_n_workers=1)
    (tmp_path / 'key.json').write_text(json.dumps({'HUGGING_FACE_API_KEY': ''}))
    if expect_exit is None:
        pipeline.main(ns)
    else:
        with pytest.raises(SystemExit) as e:
            pipeline.main(ns)
        assert e.value.code == expect_exit
    return out


def _one(out, *parts):
    hits = sorted(p for p in os.listdir(out) if all(x in p for x in parts))
    assert len(hits) == 1, (parts, os.listdir(out))
    return os.path.join(out, hits[0])


def test_png_only_csv_gives_predictions_for_every_task(world):
    df = pd.DataFrame({'png_path': [f'ecg_{k}.png' for k in range(12)]})
    out = run(world, df, 'nolabels')
    combined = pd.read_csv(_one(out, '_all_tasks.csv'))
    assert len(combined) == 12 and combined['file_name'].tolist() == df['png_path'].tolist()
    binary = ['SHD', 'LVEF_le40', 'LVEF_lt50', 'iAFib5']
    assert list(combined.columns) == (['file_name'] + [f'{o}_{s}' for o in binary for s in ('prob', 'pred')]
                                      + [f'{p}_prob' for p in ECG_PATTERNS])
    for o in binary:
        assert ((combined[f'{o}_prob'] >= THRESHOLDS[o]).astype(int) == combined[f'{o}_pred']).all()
        assert combined[f'{o}_prob'].between(0, 1).all()
    for p in ECG_PATTERNS:
        assert combined[f'{p}_prob'].isna().all() == (p in NO_HEAD)
    assert FakeBert.calls == 0                         # no report text: BERT never loaded
    shd = pd.read_csv(_one(out, 'deepecg_png_shd_', '_probabilities.csv'))
    assert shd['ground_truth'].isna().all() and set(shd['prediction']) <= {0, 1}
    assert not os.listdir(world / 'prep_nolabels')     # PNGs never went through signal preprocessing


def test_labels_and_reports_give_metrics(world):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'png_path': [f'ecg_{k}.png' for k in range(12)],
                       'shd': rng.integers(0, 2, 12), 'lvef_40': [0, 1] * 6, 'lvef_50': [1, 0, 0] * 4,
                       'afib_5y': [0, 0, 1] * 4,
                       'ecg_machine_diagnosis': [f'report {"x" * k}' for k in range(12)]})
    df.loc[3, 'shd'] = np.nan                           # one unlabelled row: still predicted
    out = run(world, df, 'labels')
    for task in ('shd', 'lvef_40', 'lvef_50', 'afib_5y'):
        with open(_one(out, 'deepecg_png_', f'_{task}.json')) as f:
            metrics = json.load(f)
        assert 0.0 <= metrics['results']['auc'] <= 1.0, task
    shd = pd.read_csv(_one(out, 'deepecg_png_shd_', '_probabilities.csv'))
    assert len(shd) == 12 and shd['ground_truth'].isna().sum() == 1
    assert FakeBert.calls > 0
    with open(_one(out, '_ecg_machine_diagnosis.json')) as f:
        m77 = json.load(f)
    assert all(m77[p]['auc'] is None or np.isnan(m77[p]['auc']) for p in NO_HEAD)   # no probe: no metric
    probs77 = pd.read_csv(_one(out, 'deepecg_png_77_classes_', '_probabilities.csv'))
    assert probs77[[f'{p}_bert_model' for p in ECG_PATTERNS]].notna().all().all()


def test_missing_png_is_reported_and_skipped(world):
    df = pd.DataFrame({'png_path': ['ecg_0.png', 'ecg_1.png', 'not_there.png']})
    out = run(world, df, 'missing')
    combined = pd.read_csv(_one(out, '_all_tasks.csv'))
    assert combined['file_name'].tolist() == ['ecg_0.png', 'ecg_1.png']
    missing = pd.read_csv(_one(out, 'missing_files_'))
    assert missing['file_name'].eq('not_there.png').all() and len(missing) == len(DIAGNOSIS_TO_FILE_COLUMNS)


def test_expand_png_column_keeps_explicit_columns():
    df = pd.DataFrame({'png_path': ['a.png'], 'afib_5y': [1], 'afib_ecg_file_name': ['other.png']})
    out = pipeline.expand_png_column(df)
    assert out['afib_ecg_file_name'].tolist() == ['other.png'] and out['afib_5y'].tolist() == [1]
    for diag, col in DIAGNOSIS_TO_FILE_COLUMNS.items():
        assert col in out and diag in out
    assert out['shd_ecg_file_name'].tolist() == ['a.png'] and out['shd'].isna().all()
    plain = pd.DataFrame({'afib_5y': [1], 'afib_ecg_file_name': ['x.xml']})
    assert pipeline.expand_png_column(plain) is plain   # signal CSVs are untouched


def test_preprocessing_matches_the_training_recipe(tmp_path):
    path = str(tmp_path / 'p.png')
    draw_page(path, 3)
    x = load_ecg_image(path, PREPROCESSING)
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32
    gray = np.asarray(Image.open(path).convert('L').resize((32, 32), Image.BILINEAR), dtype=np.float32)
    want = (torch.from_numpy(gray).mul(1.0 / 255) - PREPROCESSING['mean'][1]) / PREPROCESSING['std'][1]
    assert torch.allclose(x[1], want, atol=1e-6)
    # three identical gray channels, each with its own mean/std
    assert torch.allclose(x[0] * PREPROCESSING['std'][0] + PREPROCESSING['mean'][0],
                          x[2] * PREPROCESSING['std'][2] + PREPROCESSING['mean'][2], atol=1e-6)


def test_missing_weights_say_where_to_mount(tmp_path, monkeypatch):
    monkeypatch.setenv(deepecg_png.WEIGHTS_DIR_ENV, str(tmp_path / 'nowhere'))
    deepecg_png._BANKS.clear()
    with pytest.raises(FileNotFoundError, match='mount the exported folder'):
        deepecg_png.DeepECGPNG_SHD('deepecg_png_shd', torch.device('cpu'), '')


# ---------------------------------------------------------------------------- problems found in review
def test_a_task_without_png_rows_does_not_crash(world):
    """Old per-task format: one task points at a signal file only; with the image model on, the
    other tasks still run and nothing crashes (was: KeyError 'diagnosis', whole run lost)."""
    df = pd.DataFrame({'afib_5y': [0, 1], 'afib_ecg_file_name': ['ecg_0.png', 'ecg_1.png'],
                       'lvef_40': [0, 1], 'lvef_40_ecg_file_name': ['a.npy', 'b.npy']})
    out = run(world, df, 'nopngtask', expect_exit=1)     # a.npy / b.npy do not exist: reported, as before
    combined = pd.read_csv(_one(out, '_all_tasks.csv'))
    assert list(combined.columns) == ['file_name', 'iAFib5_prob', 'iAFib5_pred']
    assert pipeline.create_image_analysis_dataframe(df, 'lvef_40', 'lvef_40_ecg_file_name', str(world)).empty


def test_unreadable_png_is_skipped_and_listed(world):
    (world / 'pngs' / 'corrupt.png').write_bytes(b'not a png at all')
    good = world / 'pngs' / 'ecg_0.png'
    (world / 'pngs' / 'truncated.png').write_bytes(good.read_bytes()[:200])
    pipeline.UNREADABLE_IMAGES.clear()
    df = pd.DataFrame({'png_path': ['ecg_0.png', 'corrupt.png', 'truncated.png', 'ecg_1.png']})
    out = run(world, df, 'corrupt')
    combined = pd.read_csv(_one(out, '_all_tasks.csv'))
    assert combined['file_name'].tolist() == ['ecg_0.png', 'ecg_1.png']
    bad = pd.read_csv(_one(out, 'unreadable_images_'))
    assert sorted(os.path.basename(p) for p in bad['ecg_path']) == ['corrupt.png', 'truncated.png']


def test_unsupported_image_format_does_not_stop_the_pngs(world):
    Image.open(world / 'pngs' / 'ecg_2.png').convert('RGB').save(world / 'pngs' / 'photo.jpg')
    df = pd.DataFrame({'png_path': ['ecg_0.png', 'photo.jpg', 'ecg_1.png']})
    out = run(world, df, 'jpg')
    combined = pd.read_csv(_one(out, '_all_tasks.csv'))
    assert combined['file_name'].tolist() == ['ecg_0.png', 'ecg_1.png']
    assert not os.listdir(world / 'prep_jpg')            # never sent to the signal preprocessing


def test_failed_signal_preprocessing_does_not_stop_the_image_model(world):
    np.save(world / 'pngs' / 'bad.npy', np.full((12, 5000), np.nan, dtype=np.float32))
    df = pd.DataFrame({'afib_5y': [0, 1, 0], 'afib_ecg_file_name': ['ecg_0.png', 'ecg_1.png', 'bad.npy']})
    out = run(world, df, 'badsignal', expect_exit=1)      # the signal failure is still reported
    combined = pd.read_csv(_one(out, '_all_tasks.csv'))
    assert combined['file_name'].tolist() == ['ecg_0.png', 'ecg_1.png']


def test_label_values_are_normalised():
    labels = pd.Series(['1', 0, 'True', 'false', 'yes', 'NO', 2, 0.5, None, ' 1 ', 1.0])
    got = pipeline.normalize_binary_labels(labels, 'shd')
    want = [1, 0, 1, 0, 1, 0, np.nan, np.nan, np.nan, 1, 1]
    assert [g if g == g else 'nan' for g in got] == [w if w == w else 'nan' for w in want]


def test_signal_csv_with_an_unrelated_shd_column_still_validates(world):
    """Image model off: `shd` is not an expected task, as before this feature."""
    df = pd.DataFrame({'afib_5y': [0], 'afib_ecg_file_name': ['x.npy'], 'shd': ['whatever']})
    out = run(world, df, 'signalshd', use_deepecg_png=False, expect_exit=1)   # x.npy is missing
    assert os.path.exists(_one(out, 'missing_files_'))


def test_pngs_with_the_flag_off_warn(world, caplog):
    df = pd.DataFrame({'png_path': ['ecg_0.png']})
    with caplog.at_level('WARNING'):
        run(world, df, 'flagoff', use_deepecg_png=False)
    assert 'use_deepecg_png is False' in caplog.text


def test_preprocessing_then_analysis(world):
    df = pd.DataFrame({'png_path': ['ecg_0.png', 'ecg_1.png']})
    run(world, df, 'twostep', mode='preprocessing')
    out = run(world, df, 'twostep', mode='analysis')
    assert len(pd.read_csv(_one(out, '_all_tasks.csv'))) == 2


def test_parser_defaults_the_flag_to_false(monkeypatch):
    base = ['main.py', '--diagnosis_classifier_device', 'cpu', '--signal_processing_device', 'cpu',
            '--data_path', 'x.csv', '--batch_size', '1', '--output_folder', 'o', '--hugging_face_api_key_path', 'k',
            '--use_wcr', 'False', '--use_efficientnet', 'False', '--ecg_signals_path', 'e', '--mode', 'analysis',
            '--preprocessing_folder', 'p']
    monkeypatch.setattr(sys, 'argv', base)
    assert pipeline.HearWiseArgs.parse_arguments().use_deepecg_png is False
    monkeypatch.setattr(sys, 'argv', base + ['--use_deepecg_png', 'True'])
    assert pipeline.HearWiseArgs.parse_arguments().use_deepecg_png is True


@pytest.mark.parametrize('break_it, message', [
    (lambda d: (d / 'heads.json').write_text('{not json'), 'invalid DeepECG-PNG weights'),
    (lambda d: (d / 'thresholds.json').write_text(json.dumps({'thresholds': {'SHD': 'high'}})),
     'invalid DeepECG-PNG weights'),
    (lambda d: (d / 'encoder.safetensors').write_bytes(b''), 'cannot read'),
])
def test_broken_weights_give_a_clear_error(tmp_path, monkeypatch, break_it, message):
    write_tiny_weights(str(tmp_path / 'w'))
    break_it(tmp_path / 'w')
    monkeypatch.setenv(deepecg_png.WEIGHTS_DIR_ENV, str(tmp_path / 'w'))
    deepecg_png._BANKS.clear()
    with pytest.raises(RuntimeError, match=message):
        deepecg_png.DeepECGPNG_SHD('deepecg_png_shd', torch.device('cpu'), '')


def test_wrong_rope_dtype_or_head_shape_is_refused(tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file
    folder = tmp_path / 'w'
    write_tiny_weights(str(folder))
    monkeypatch.setenv(deepecg_png.WEIGHTS_DIR_ENV, str(folder))
    cfg = json.loads((folder / 'config.json').read_text())
    bad = {**cfg, 'rope': {**cfg['rope'], 'grid': 3}}
    (folder / 'config.json').write_text(json.dumps(bad))
    deepecg_png._BANKS.clear()
    with pytest.raises(RuntimeError, match='RoPE settings'):
        deepecg_png.DeepECGPNG_SHD('deepecg_png_shd', torch.device('cpu'), '')
    (folder / 'config.json').write_text(json.dumps(cfg))
    enc = load_file(str(folder / 'encoder.safetensors'))
    save_file({k: v.half() for k, v in enc.items()}, str(folder / 'encoder.safetensors'))
    deepecg_png._BANKS.clear()
    with pytest.raises(RuntimeError, match='float32'):
        deepecg_png.DeepECGPNG_SHD('deepecg_png_shd', torch.device('cpu'), '')
    save_file(enc, str(folder / 'encoder.safetensors'))
    heads = load_file(str(folder / 'heads.safetensors'))
    heads['shd_lvef.weight'] = heads['shd_lvef.weight'][:2]
    save_file(heads, str(folder / 'heads.safetensors'))
    deepecg_png._BANKS.clear()
    with pytest.raises(RuntimeError, match='head shd_lvef has shapes'):
        deepecg_png.DeepECGPNG_SHD('deepecg_png_shd', torch.device('cpu'), '')
    deepecg_png._BANKS.clear()
    with pytest.raises(RuntimeError, match='base class'):
        deepecg_png.DeepECGPNGWrapper('deepecg_png', torch.device('cpu'), '')


def test_feature_cache_drops_the_oldest_batches(world, monkeypatch):
    bank = deepecg_png._get_bank(torch.device('cpu'))
    monkeypatch.setattr(bank, 'MAX_CACHED_ROWS', 4)
    bank._features.clear()
    bank._cached_rows = 0
    batches = [torch.stack([load_ecg_image(str(world / 'pngs' / f'ecg_{2 * i + j}.png'), PREPROCESSING)
                            for j in range(2)]) for i in range(3)]
    first = [bank.features(b).clone() for b in batches]
    assert bank._cached_rows == 4 and len(bank._features) == 2      # batch 0 dropped
    again = [bank.features(b) for b in batches]
    assert all(torch.equal(a, b) for a, b in zip(first, again))     # recomputed or cached: same values

