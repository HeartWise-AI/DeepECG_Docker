"""DeepECG-PNG in the pipeline, on CPU, without real weights or patient data.

A tiny random encoder + probes is written in the Hugging Face weights format (the download is
replaced by that folder), synthetic PNG pages are drawn, and main() runs end to end (the BERT
classifier is replaced by a stub, so nothing is downloaded):
  * the input type of each file is detected from its extension (no flag): image rows go to the
    image models, signal rows to the signal preprocessing and models, in the same CSV;
  * the usual per-task CSV: predictions for every task, with or without labels; with labels
    and report text, metrics for every task;
  * heartwise.config chooses the models per input type and per task;
  * PNG rows never reach the signal preprocessing; a missing PNG is reported and skipped;
  * the preprocessing is the training preprocessing.
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
from utils.constants import DIAGNOSIS_TO_FILE_COLUMNS, ECG_PATTERNS, IMAGE, SIGNAL  # noqa: E402
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
    monkeypatch.setattr(deepecg_png, 'download_weights', lambda key: str(weights))
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


IMAGE_ONLY = {'signal_models': [], 'image_models': ['deepecg_png']}


def run(tmp_path, df, name, mode='full_run', models=None, expect_exit=None):
    csv = tmp_path / f'{name}.csv'
    df.to_csv(csv, index=False)
    out = tmp_path / f'out_{name}'
    import argparse
    ns = argparse.Namespace(
        diagnosis_classifier_device='cpu', signal_processing_device='cpu', data_path=str(csv),
        batch_size=5, output_folder=str(out), hugging_face_api_key_path=str(tmp_path / 'key.json'),
        models=IMAGE_ONLY if models is None else models, use_wcr=None, use_efficientnet=None,
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


def task_csv(files, **labels):
    """The usual per-task CSV: every task's file column lists `files`; labels optional."""
    df = pd.DataFrame({column: list(files) for column in DIAGNOSIS_TO_FILE_COLUMNS.values()})
    for task in DIAGNOSIS_TO_FILE_COLUMNS:
        df[task] = labels.get(task, pd.Series([None] * len(df), dtype=object))
    return df


def _one(out, *parts):
    hits = sorted(p for p in os.listdir(out) if all(x in p for x in parts))
    assert len(hits) == 1, (parts, os.listdir(out))
    return os.path.join(out, hits[0])


def test_png_rows_give_predictions_for_every_task(world):
    files = [f'ecg_{k}.png' for k in range(12)]
    out = run(world, task_csv(files), 'nolabels')
    for task in ('shd', 'lvef_40', 'lvef_50', 'afib_5y'):
        probs = pd.read_csv(_one(out, f'_{task}_probabilities.csv'))
        # the signal models' columns: no extra column for the image model
        assert list(probs.columns) == ['file_name', 'ground_truth', 'predictions'], task
        assert probs['file_name'].tolist() == files and probs['ground_truth'].isna().all()
        assert probs['predictions'].between(0, 1).all()
    probs77 = pd.read_csv(_one(out, 'deepecg_png_77_classes_', '_probabilities.csv'))
    assert list(probs77.columns) == (['file_name'] + [f'{p}_bert_model' for p in ECG_PATTERNS]
                                     + [f'{p}_sig_model' for p in ECG_PATTERNS])
    for p in ECG_PATTERNS:
        assert probs77[f'{p}_sig_model'].isna().all() == (p in NO_HEAD)
    assert FakeBert.calls == 0                         # no report text: BERT never loaded
    assert not os.listdir(world / 'prep_nolabels')     # PNGs never went through signal preprocessing
    assert not any('_all_tasks' in f for f in os.listdir(out))


def test_labels_and_reports_give_metrics(world):
    rng = np.random.default_rng(0)
    shd = pd.Series(rng.integers(0, 2, 12)).astype(object)
    shd[3] = None                                       # one unlabelled row: still predicted
    df = task_csv([f'ecg_{k}.png' for k in range(12)],
                  shd=shd, lvef_40=pd.Series([0, 1] * 6), lvef_50=pd.Series([1, 0, 0] * 4),
                  afib_5y=pd.Series([0, 0, 1] * 4),
                  ecg_machine_diagnosis=pd.Series([f'report {"x" * k}' for k in range(12)]))
    out = run(world, df, 'labels')
    for task in ('shd', 'lvef_40', 'lvef_50', 'afib_5y'):
        with open(_one(out, 'deepecg_png_', f'_{task}.json')) as f:
            metrics = json.load(f)
        probs = pd.read_csv(_one(out, 'deepecg_png_', f'_{task}_probabilities.csv'))
        labelled = probs[probs['ground_truth'].notna()]
        from sklearn.metrics import roc_auc_score
        want = roc_auc_score(labelled['ground_truth'].astype(int), labelled['predictions'])
        assert metrics['results']['auc'] == pytest.approx(want), task
    shd_probs = pd.read_csv(_one(out, 'deepecg_png_shd_', '_probabilities.csv'))
    assert len(shd_probs) == 12 and shd_probs['ground_truth'].isna().sum() == 1
    assert FakeBert.calls > 0
    with open(_one(out, '_ecg_machine_diagnosis.json')) as f:
        m77 = json.load(f)
    assert all(m77[p]['auc'] is None or np.isnan(m77[p]['auc']) for p in NO_HEAD)   # no probe: no metric
    probs77 = pd.read_csv(_one(out, 'deepecg_png_77_classes_', '_probabilities.csv'))
    assert probs77[[f'{p}_bert_model' for p in ECG_PATTERNS]].notna().all().all()


def test_missing_png_is_reported_and_skipped(world):
    df = task_csv(['ecg_0.png', 'ecg_1.png', 'not_there.png'])
    out = run(world, df, 'missing')
    shd = pd.read_csv(_one(out, '_shd_probabilities.csv'))
    assert shd['file_name'].tolist() == ['ecg_0.png', 'ecg_1.png']
    missing = pd.read_csv(_one(out, 'missing_files_'))
    assert missing['file_name'].eq('not_there.png').all() and len(missing) == len(DIAGNOSIS_TO_FILE_COLUMNS)


def test_input_type_is_detected_from_the_extension():
    assert pipeline.input_type_of('a.png') == IMAGE and pipeline.input_type_of('B.JPEG') == IMAGE
    assert pipeline.input_type_of('photo.jpg') == IMAGE
    for name in ('a.xml', 'a.npy', 'a.XML', 'noextension'):
        assert pipeline.input_type_of(name) == SIGNAL


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


def test_weights_come_from_hugging_face(tmp_path, monkeypatch):
    calls = []

    def get_model(repo_id, local_dir, hugging_face_api_key):
        calls.append((repo_id, local_dir, hugging_face_api_key))
        write_tiny_weights(str(tmp_path / 'hf'))
        return str(tmp_path / 'hf')
    monkeypatch.setattr(deepecg_png.HuggingFaceWrapper, 'get_model', staticmethod(get_model))
    deepecg_png._BANKS.clear()
    deepecg_png.DeepECGPNG_SHD('deepecg_png_shd', torch.device('cpu'), 'token')
    assert calls == [('heartwise/deepecg_png', os.path.join('weights', 'deepecg_png'), 'token')]
    deepecg_png._BANKS.clear()


def test_incomplete_download_says_which_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(deepecg_png, 'download_weights', lambda key: str(tmp_path / 'empty'))
    (tmp_path / 'empty').mkdir()
    deepecg_png._BANKS.clear()
    with pytest.raises(FileNotFoundError, match='heartwise/deepecg_png'):
        deepecg_png.DeepECGPNG_SHD('deepecg_png_shd', torch.device('cpu'), '')


# ---------------------------------------------------------------------------- one CSV, models from the config
def test_a_task_without_png_rows_does_not_crash(world):
    """One task points at signal files only: the other tasks still run on their PNGs."""
    df = pd.DataFrame({'afib_5y': [0, 1], 'afib_ecg_file_name': ['ecg_0.png', 'ecg_1.png'],
                       'lvef_40': [0, 1], 'lvef_40_ecg_file_name': ['a.npy', 'b.npy']})
    out = run(world, df, 'nopngtask', expect_exit=1)     # no signal model selected for the .npy rows: reported
    assert len(pd.read_csv(_one(out, 'deepecg_png_afib_5y_', '_probabilities.csv'))) == 2
    assert not any('lvef_40' in f for f in os.listdir(out))
    assert pipeline.create_image_analysis_dataframe(df, 'lvef_40', 'lvef_40_ecg_file_name', str(world)).empty


def test_jpg_pages_are_analysed_like_pngs(world):
    Image.open(world / 'pngs' / 'ecg_2.png').convert('RGB').save(world / 'pngs' / 'photo.jpg')
    out = run(world, task_csv(['ecg_0.png', 'photo.jpg', 'ecg_1.png']), 'jpg')
    assert pd.read_csv(_one(out, '_shd_probabilities.csv'))['file_name'].tolist() == ['ecg_0.png', 'photo.jpg', 'ecg_1.png']
    assert not os.listdir(world / 'prep_jpg')            # never sent to the signal preprocessing


def test_unreadable_png_is_skipped_and_listed(world):
    (world / 'pngs' / 'corrupt.png').write_bytes(b'not a png at all')
    good = world / 'pngs' / 'ecg_0.png'
    (world / 'pngs' / 'truncated.png').write_bytes(good.read_bytes()[:200])
    pipeline.UNREADABLE_IMAGES.clear()
    out = run(world, task_csv(['ecg_0.png', 'corrupt.png', 'truncated.png', 'ecg_1.png']), 'corrupt')
    assert pd.read_csv(_one(out, '_shd_probabilities.csv'))['file_name'].tolist() == ['ecg_0.png', 'ecg_1.png']
    bad = pd.read_csv(_one(out, 'unreadable_images_'))
    assert sorted(os.path.basename(p) for p in bad['ecg_path']) == ['corrupt.png', 'truncated.png']


def test_failed_signal_preprocessing_does_not_stop_the_image_model(world):
    np.save(world / 'pngs' / 'bad.npy', np.full((12, 5000), np.nan, dtype=np.float32))
    df = pd.DataFrame({'afib_5y': [0, 1, 0], 'afib_ecg_file_name': ['ecg_0.png', 'ecg_1.png', 'bad.npy']})
    out = run(world, df, 'badsignal', models={'image_models': ['deepecg_png'], 'signal_models': ['efficientnet']},
              expect_exit=1)                              # the signal failure is still reported
    probs = pd.read_csv(_one(out, 'deepecg_png_afib_5y_', '_probabilities.csv'))
    assert probs['file_name'].tolist() == ['ecg_0.png', 'ecg_1.png']


def test_png_only_csv_never_runs_the_signal_models(world, monkeypatch):
    """Signal models selected (the default), but no signal row: they are never loaded."""
    loaded = []
    real = analysis_pipeline.HeartWiseModelFactory.create_model

    def spy(model_config):
        loaded.append(model_config['model_name'])
        return real(model_config)
    monkeypatch.setattr(analysis_pipeline.HeartWiseModelFactory, 'create_model', staticmethod(spy))
    run(world, task_csv(['ecg_0.png', 'ecg_1.png']), 'pngonly', models={})
    assert loaded and all(name.startswith('deepecg_png') for name in loaded)


def test_image_rows_without_an_image_model_are_reported(world):
    out = run(world, task_csv(['ecg_0.png', 'ecg_1.png']), 'noimagemodel',
              models={'signal_models': ['efficientnet'], 'image_models': []}, expect_exit=1)
    assert not any(f.endswith('_probabilities.csv') for f in os.listdir(out))


def test_per_task_model_selection_overrides_the_default(world):
    out = run(world, task_csv(['ecg_0.png', 'ecg_1.png']), 'override',
              models={'image_models': ['deepecg_png'], 'lvef_40_image_models': []}, expect_exit=1)
    assert os.path.exists(_one(out, 'deepecg_png_shd_', '_probabilities.csv'))
    assert not any('lvef_40' in f for f in os.listdir(out))


def test_select_models():
    import argparse
    ns = argparse.Namespace(models={}, use_wcr=None, use_efficientnet=None)
    assert pipeline.select_models(ns, 'lvef_40', SIGNAL) == ['efficientnetv2_lvef_equal_under_40', 'wcr_lvef_equal_under_40']
    assert pipeline.select_models(ns, 'lvef_40', IMAGE) == ['deepecg_png_lvef_equal_under_40']
    assert pipeline.select_models(ns, 'shd', SIGNAL) == []           # no signal model for shd
    ns.models = {'signal_models': ['wcr'], 'afib_5y_signal_models': ['efficientnet']}
    assert pipeline.select_models(ns, 'lvef_50', SIGNAL) == ['wcr_lvef_under_50']
    assert pipeline.select_models(ns, 'afib_5y', SIGNAL) == ['efficientnetv2_afib_5y']
    # deprecated flags still work when the config has no signal_models
    ns = argparse.Namespace(models={}, use_wcr=False, use_efficientnet=True)
    assert pipeline.select_models(ns, 'afib_5y', SIGNAL) == ['efficientnetv2_afib_5y']


@pytest.mark.parametrize('models, message', [
    ({'signal_models': ['resnet']}, "Unknown signal model"),
    ({'image_models': ['efficientnet']}, "Unknown image model"),
    ({'lvef_45_image_models': ['deepecg_png']}, "Unknown model selection"),
])
def test_unknown_model_selection_is_refused(world, capsys, models, message):
    run(world, task_csv(['ecg_0.png']), 'badconfig', models=models, expect_exit=1)
    assert message in capsys.readouterr().out


def test_label_values_are_normalised():
    labels = pd.Series(['1', 0, 'True', 'false', 'yes', 'NO', 2, 0.5, None, ' 1 ', 1.0])
    got = pipeline.normalize_binary_labels(labels, 'shd')
    want = [1, 0, 1, 0, 1, 0, np.nan, np.nan, np.nan, 1, 1]
    assert [g if g == g else 'nan' for g in got] == [w if w == w else 'nan' for w in want]


def test_signal_csv_with_an_unrelated_shd_column_still_validates(world):
    """No image model selected: `shd` is not an expected task, as before this feature."""
    df = pd.DataFrame({'afib_5y': [0], 'afib_ecg_file_name': ['x.npy'], 'shd': ['whatever']})
    out = run(world, df, 'signalshd', models={'signal_models': ['efficientnet'], 'image_models': []},
              expect_exit=1)                                          # x.npy is missing
    assert os.path.exists(_one(out, 'missing_files_'))


def test_preprocessing_then_analysis(world):
    df = task_csv(['ecg_0.png', 'ecg_1.png'])
    run(world, df, 'twostep', mode='preprocessing')
    out = run(world, df, 'twostep', mode='analysis')
    assert len(pd.read_csv(_one(out, '_shd_probabilities.csv'))) == 2


def test_parser_reads_the_model_selection(monkeypatch):
    base = ['main.py', '--diagnosis_classifier_device', 'cpu', '--signal_processing_device', 'cpu',
            '--data_path', 'x.csv', '--batch_size', '1', '--output_folder', 'o', '--hugging_face_api_key_path', 'k',
            '--ecg_signals_path', 'e', '--mode', 'analysis', '--preprocessing_folder', 'p']
    monkeypatch.setattr(sys, 'argv', base)
    args = pipeline.HearWiseArgs.parse_arguments()
    assert args.models == {} and args.use_wcr is None and args.use_efficientnet is None
    monkeypatch.setattr(sys, 'argv', base + ['--models', 'signal_models:efficientnet,wcr;image_models:deepecg_png;'
                                                         'lvef_40_signal_models:'])
    assert pipeline.HearWiseArgs.parse_arguments().models == {
        'signal_models': ['efficientnet', 'wcr'], 'image_models': ['deepecg_png'], 'lvef_40_signal_models': []}
    monkeypatch.setattr(sys, 'argv', base + ['--use_wcr', 'False', '--use_efficientnet', 'True'])
    args = pipeline.HearWiseArgs.parse_arguments()
    assert args.use_wcr is False and args.use_efficientnet is True


@pytest.mark.parametrize('break_it, message', [
    (lambda d: (d / 'heads.json').write_text('{not json'), 'invalid DeepECG-PNG weights'),
    (lambda d: (d / 'thresholds.json').write_text(json.dumps({'thresholds': {'SHD': 'high'}})),
     'invalid DeepECG-PNG weights'),
    (lambda d: (d / 'encoder.safetensors').write_bytes(b''), 'cannot read'),
])
def test_broken_weights_give_a_clear_error(tmp_path, monkeypatch, break_it, message):
    write_tiny_weights(str(tmp_path / 'w'))
    break_it(tmp_path / 'w')
    monkeypatch.setattr(deepecg_png, 'download_weights', lambda key: str(tmp_path / 'w'))
    deepecg_png._BANKS.clear()
    with pytest.raises(RuntimeError, match=message):
        deepecg_png.DeepECGPNG_SHD('deepecg_png_shd', torch.device('cpu'), '')


def test_wrong_rope_dtype_or_head_shape_is_refused(tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file
    folder = tmp_path / 'w'
    write_tiny_weights(str(folder))
    monkeypatch.setattr(deepecg_png, 'download_weights', lambda key: str(folder))
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
    bank = deepecg_png._get_bank(torch.device('cpu'), '')
    monkeypatch.setattr(bank, 'MAX_CACHED_ROWS', 4)
    bank._features.clear()
    bank._cached_rows = 0
    batches = [torch.stack([load_ecg_image(str(world / 'pngs' / f'ecg_{2 * i + j}.png'), PREPROCESSING)
                            for j in range(2)]) for i in range(3)]
    first = [bank.features(b).clone() for b in batches]
    assert bank._cached_rows == 4 and len(bank._features) == 2      # batch 0 dropped
    again = [bank.features(b) for b in batches]
    assert all(torch.equal(a, b) for a, b in zip(first, again))     # recomputed or cached: same values

