"""DeepECG DINOv3-SigLIP2 image models, on CPU, without real weights or patient data.

A tiny random DINOv3 encoder and the 18 probe heads are written in the layout of
heartwise/DeepECG-PNG_DINOv3-SigLIP2_epoch030 (base/ + probes/), the pinned encoder hash and
architecture are swapped for the tiny ones, and:
  * main() runs every task with image_models deepecg_dinov3 (clean probes) or
    deepecg_dinov3_aug, including the DINOv3-only tasks hcm and aco;
  * every head check of the loader refuses a broken head;
  * the single-image predictor returns the per-head fields, leaves hcm_3class and acco_vs_aico
    off by default and gates acco_vs_aico on aco;
  * the encoder loads from either transformers block layout;
  * the page fixes (EXIF orientation, transparency) leave opaque upright pages unchanged.
With DINOV3_SNAPSHOT=<local copy of the Hugging Face repository>, the real heads are checked too.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))
_cwd = os.getcwd()
os.chdir(REPO)   # models/wcr_ecg_transformer.py loads ./fairseq-signals at import
import test_deepecg_png_pipeline as png  # noqa: E402  (pages, BERT stub, run(), task_csv())
from models import dinov3_bank as db  # noqa: E402
from models import dinov3_wrapper as dw  # noqa: E402
from utils import analysis_pipeline  # noqa: E402
from utils.constants import ECG_PATTERNS, MODEL_MAPPING  # noqa: E402
from utils.image_preprocessing import preprocess_ecg_image  # noqa: E402
os.chdir(_cwd)

TINY = {**db.DINOV3_VITB16, 'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 2,
        'num_attention_heads': 2}
SIZE = [32, 64]
PREPROCESSING = {'resize': SIZE, 'resize_mode': 'stretch', 'color_mode': 'grayscale',
                 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
THRESHOLDS = {'iAFib5': 0.2, 'LVEF<=40': 0.3, 'LVEF<50': 0.4, 'SHD': 0.5, 'confirmed HCM (vs rest)': 0.1,
              'ACO': 0.5, 'ACCO (vs AICO; occlusion ECGs only)': 0.5}


def _tiny_encoder_state(layout: str = 'model.layer'):
    """A random tiny DeepECG_DINOv3_Encoder state dict, blocks named as `layout` (the checkpoint's
    transformers >= 5.4 layout 'model.layer', or 'layer')."""
    torch.manual_seed(0)
    encoder = db.DeepECG_DINOv3_Encoder()
    state = {}
    for key, value in encoder.state_dict().items():
        for prefix in ('backbone.model.layer.', 'backbone.layer.'):
            if key.startswith(prefix):
                key = f'backbone.{layout}.' + key[len(prefix):]
        state[key] = value.clone()
    return state


def write_snapshot(folder, monkeypatch):
    """base/epoch_030.pt + probes/ for every task and variant; pins the tiny encoder."""
    from safetensors.torch import save_file
    monkeypatch.setattr(db, 'DINOV3_VITB16', TINY)
    monkeypatch.setattr(db, 'PROJECTION_DIM', 8)
    os.makedirs(os.path.join(folder, 'base'), exist_ok=True)
    encoder_path = os.path.join(folder, 'base', 'epoch_030.pt')
    torch.save({'image_encoder': _tiny_encoder_state(), 'epoch': 30}, encoder_path)
    sha = hashlib.sha256(open(encoder_path, 'rb').read()).hexdigest()
    monkeypatch.setattr(db, 'ENCODER_SHA256', sha)
    dim = 2 * TINY['hidden_size']
    config = {'model_name': 'dinov3_vitb16_clip',
              'checkpoint': {'file': 'base/epoch_030.pt', 'sha256': sha, 'state_dict_key': 'image_encoder',
                             'strict': True},
              'vision_config': {'backbone': 'DINOv3 ViT-B/16 (tiny)', 'patch_size': 16, 'image_size': SIZE},
              'preprocessing': PREPROCESSING, 'feature': {'dim': dim}}
    os.makedirs(os.path.join(folder, 'probes'), exist_ok=True)
    with open(os.path.join(folder, 'probes', 'config.json'), 'w') as f:
        json.dump(config, f)
    g = torch.Generator().manual_seed(1)
    for probe_task, outputs in db.PROBE_OUTPUTS.items():
        outputs = list(ECG_PATTERNS) if outputs == db.ALL_77 else list(outputs)
        softmax = probe_task == 'hcm_3class'
        for variant in ('clean', 'aug'):
            head = os.path.join(folder, 'probes', f'{probe_task}_{variant}')
            os.makedirs(head, exist_ok=True)
            save_file({'weight': torch.randn(len(outputs), dim, generator=g) * 0.3,
                       'bias': torch.zeros(len(outputs)),
                       'mean': torch.randn(dim, generator=g) * 0.1,
                       'scale': torch.rand(dim, generator=g) + 0.5}, os.path.join(head, 'classifier.safetensors'))
            with open(os.path.join(head, 'config.json'), 'w') as f:
                json.dump(config, f)
            with open(os.path.join(head, 'head.json'), 'w') as f:
                json.dump({'name': f'{db.REPO_ID}/probes/{probe_task}_{variant}', 'task': probe_task,
                           'outputs': outputs,
                           'encoder': {'repo': db.REPO_ID, 'file': 'base/epoch_030.pt', 'sha256': sha},
                           'classifier': {'file': 'classifier.safetensors', 'type': 'linear',
                                          'tensors': {'weight': [len(outputs), dim], 'bias': [len(outputs)],
                                                      'mean': [dim], 'scale': [dim]},
                                          'dtype': 'float32', 'input_dim': dim, 'standardize': True,
                                          'activation': 'softmax' if softmax else 'sigmoid'}}, f)
            cutoffs = {} if softmax or len(outputs) == 77 else {o: THRESHOLDS[o] for o in outputs}
            with open(os.path.join(head, 'thresholds.json'), 'w') as f:
                json.dump({'thresholds': cutoffs}, f)
    return folder


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    folder = write_snapshot(str(tmp_path / 'snapshot'), monkeypatch)
    monkeypatch.setattr(dw, 'download_weights', lambda token: folder)
    dw._BANKS.clear()
    yield folder
    dw._BANKS.clear()


@pytest.fixture
def world(tmp_path, snapshot, monkeypatch):
    pngs = tmp_path / 'pngs'
    pngs.mkdir()
    for k in range(6):
        png.draw_page(str(pngs / f'ecg_{k}.png'), k)
    real_factory = analysis_pipeline.HeartWiseModelFactory

    class Factory:
        @staticmethod
        def create_model(model_config):
            if model_config['model_name'] == png.BertClassifier_En_Fr.name:
                return png.FakeBert()
            return real_factory.create_model(model_config)
    monkeypatch.setattr(analysis_pipeline, 'HeartWiseModelFactory', Factory)
    return tmp_path


FILES = [f'ecg_{k}.png' for k in range(6)]
BINARY_TASKS = ('afib_5y', 'lvef_40', 'lvef_50', 'shd', 'hcm', 'aco')


@pytest.mark.parametrize('family', ['deepecg_dinov3', 'deepecg_dinov3_aug'])
def test_every_task_runs_with_the_selected_probe_variant(world, family):
    out = png.run(world, png.task_csv(FILES), family, models={'signal_models': [], 'image_models': [family]})
    for task in BINARY_TASKS:
        probs = pd.read_csv(png._one(out, f'{family}_', f'_{task}_probabilities.csv'))
        assert list(probs.columns) == ['file_name', 'ground_truth', 'predictions'], task
        assert probs['file_name'].tolist() == FILES and probs['predictions'].between(0, 1).all()
    probs77 = pd.read_csv(png._one(out, f'{family}_77_classes_', '_probabilities.csv'))
    assert probs77[[f'{p}_sig_model' for p in ECG_PATTERNS]].notna().all().all()
    if family == 'deepecg_dinov3':
        assert not any('_aug_' in name for name in os.listdir(out))


def test_variants_use_their_own_heads(world):
    models = lambda family: {'signal_models': [], 'image_models': [family]}
    clean = png.run(world, png.task_csv(FILES), 'clean', models=models('deepecg_dinov3'))
    aug = png.run(world, png.task_csv(FILES), 'aug', models=models('deepecg_dinov3_aug'))
    a = pd.read_csv(png._one(clean, '_shd_probabilities.csv'))['predictions']
    b = pd.read_csv(png._one(aug, '_shd_probabilities.csv'))['predictions']
    assert not np.allclose(a, b)


def test_siglip2_selection_does_not_expect_the_dinov3_only_tasks():
    assert MODEL_MAPPING['hcm']['image'].keys() == {'deepecg_dinov3', 'deepecg_dinov3_aug'}
    assert 'deepecg_siglip2' in MODEL_MAPPING['shd']['image']
    assert 'acco_vs_aico' not in MODEL_MAPPING and 'hcm_3class' not in MODEL_MAPPING   # off by decision
    names = [cls.name for cls in dw.TASK_MODELS.values()]
    assert len(set(names)) == len(names) == 18


def _bank(snapshot):
    return db._DeepECG_DINOv3_Bank(snapshot, torch.device('cpu'))


def _edit(path, change):
    with open(path) as f:
        document = json.load(f)
    change(document)
    with open(path, 'w') as f:
        json.dump(document, f)


@pytest.mark.parametrize('file,change,message', [
    ('head.json', lambda d: d['encoder'].update(sha256='0' * 64), 'does not match the loaded encoder'),
    ('head.json', lambda d: d['encoder'].update(file='other.pt'), 'does not match the loaded encoder'),
    ('head.json', lambda d: d.update(name='heartwise/other'), 'head name'),
    ('head.json', lambda d: d['classifier'].update(activation='tanh'), 'classifier metadata'),
    ('head.json', lambda d: d['classifier'].update(input_dim=768), 'classifier metadata'),
    ('head.json', lambda d: d.update(outputs=['other']), 'do not match'),
    ('thresholds.json', lambda d: d.update(thresholds={'SHD': 1.5}), 'cutoff'),
    ('config.json', lambda d: d['preprocessing'].update(resize=[224, 224]), 'differs'),
])
def test_broken_heads_are_refused(snapshot, file, change, message):
    _edit(os.path.join(snapshot, 'probes', 'shd_clean', file), change)
    with pytest.raises(RuntimeError, match=message):
        _bank(snapshot).load_head('shd_clean', ('SHD',))


def test_softmax_head_must_not_define_cutoffs(snapshot):
    _edit(os.path.join(snapshot, 'probes', 'hcm_3class_clean', 'thresholds.json'),
          lambda d: d.update(thresholds={'rest': 0.1, 'probable HCM': 0.1, 'confirmed HCM': 0.1}))
    with pytest.raises(RuntimeError, match='probability-only'):
        _bank(snapshot).load_head('hcm_3class_clean', db.PROBE_OUTPUTS['hcm_3class'])


def test_an_encoder_other_than_the_pinned_one_is_refused(snapshot, monkeypatch):
    monkeypatch.setattr(db, 'ENCODER_SHA256', 'f' * 64)
    with pytest.raises(RuntimeError, match='does not match'):
        _bank(snapshot)


def test_probabilities_follow_the_published_formula(snapshot):
    from safetensors.torch import load_file
    bank = _bank(snapshot)
    x = torch.randn(3, 2 * TINY['hidden_size'])
    t = load_file(os.path.join(snapshot, 'probes', 'shd_clean', 'classifier.safetensors'))
    head = bank.load_head('shd_clean', ('SHD',))
    z = (x - t['mean']) / t['scale']
    torch.testing.assert_close(db.head_probabilities(x, head), torch.sigmoid(z @ t['weight'].T + t['bias']))
    head3 = bank.load_head('hcm_3class_clean', db.PROBE_OUTPUTS['hcm_3class'])
    p3 = db.head_probabilities(x, head3)
    torch.testing.assert_close(p3.sum(dim=1), torch.ones(3))


def test_single_image_predictor_fields_and_gating(world, snapshot):
    page = os.path.join(world, 'pngs', 'ecg_0.png')
    predictor = db.DeepECG_DINOv3_Predictor('clean', 'cpu', snapshot_folder=snapshot)
    result = predictor.predict_image(page)
    assert result['variant'] == 'clean' and result['encoder_sha256'] == db.ENCODER_SHA256
    rows = {row['id']: row for row in result['heads']}
    assert all(set(row) == {'id', 'label', 'probability', 'threshold', 'positive', 'applicable'}
               for row in result['heads'])
    assert len([i for i in rows if i.startswith('ecg77:')]) == 77
    assert not any(i.startswith('hcm_3class') for i in rows) and 'acco_vs_aico' not in rows   # off by default
    assert [i for i in rows if ':' not in i] == ['shd', 'lvef_le40', 'lvef_lt50', 'afib_5y', 'hcm', 'aco']
    assert rows['ecg77:Afib']['threshold'] is None and rows['ecg77:Afib']['positive'] is None
    assert rows['shd']['positive'] == (rows['shd']['probability'] >= 0.5)
    same = predictor.predict_image(Image.open(page))
    assert same['heads'] == result['heads']
    only = db.DeepECG_DINOv3_Predictor('aug', 'cpu', snapshot_folder=snapshot, heads=('acco_vs_aico', 'hcm_3class'))
    rows = only.predict_image(page)['heads']
    assert [r['id'] for r in rows] == ['hcm_3class:rest', 'hcm_3class:probable HCM', 'hcm_3class:confirmed HCM',
                                       'acco_vs_aico']
    assert abs(sum(r['probability'] for r in rows[:3]) - 1) < 1e-5
    aco = db.DeepECG_DINOv3_Predictor('aug', 'cpu', snapshot_folder=snapshot, heads=('aco',)).predict_image(page)['heads'][0]
    assert rows[3]['applicable'] == aco['positive']                       # gated on aco, computed although not listed
    if not rows[3]['applicable']:
        assert rows[3]['positive'] is None


def test_cli_prints_one_json_line_per_image(world, snapshot, capsys):
    page = os.path.join(world, 'pngs', 'ecg_1.png')
    assert db.main(['--image', page, page, '--snapshot', snapshot, '--device', 'cpu', '--heads', 'aco']) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line['image'] for line in lines] == ['ecg_1.png'] * 2
    assert [row['id'] for row in lines[0]['heads']] == ['aco']


@pytest.mark.parametrize('layout', ['model.layer', 'layer'])
def test_encoder_loads_from_either_block_layout(tmp_path, monkeypatch, layout):
    monkeypatch.setattr(db, 'DINOV3_VITB16', TINY)
    monkeypatch.setattr(db, 'PROJECTION_DIM', 8)
    state = _tiny_encoder_state(layout)
    path = str(tmp_path / 'epoch_030.pt')
    torch.save({'image_encoder': state}, path)
    encoder = db.DeepECG_DINOv3_Encoder()
    encoder.load_weights(path)
    del state[next(iter(state))]
    torch.save({'image_encoder': state}, path)
    with pytest.raises(RuntimeError, match='missing'):
        db.DeepECG_DINOv3_Encoder().load_weights(path)


def test_page_fixes_leave_opaque_upright_pages_unchanged(tmp_path):
    page = Image.new('RGBA', (125, 62), 'white')
    page.putpixel((10, 10), (0, 0, 0, 255))
    fixed = {**PREPROCESSING, **db.PAGE_FIXES}
    assert torch.equal(preprocess_ecg_image(page, fixed), preprocess_ecg_image(page, PREPROCESSING))
    clear = Image.new('RGBA', (125, 62), (0, 0, 0, 0))                    # transparent page
    white = Image.new('RGB', (125, 62), 'white')
    assert torch.equal(preprocess_ecg_image(clear, fixed), preprocess_ecg_image(white, PREPROCESSING))
    photo = Image.new('RGB', (40, 20), 'white')
    photo.paste((0, 0, 0), (0, 0, 10, 20))                               # a dark left edge
    exif = photo.getexif()
    exif[0x0112] = 6                                                     # displayed rotated 90 degrees clockwise
    path = tmp_path / 'photo.png'
    photo.save(path, exif=exif)
    with Image.open(path) as loaded:
        upright = photo.transpose(Image.Transpose.ROTATE_270)
        assert torch.equal(preprocess_ecg_image(loaded, fixed), preprocess_ecg_image(upright, PREPROCESSING))
        assert not torch.equal(preprocess_ecg_image(loaded, PREPROCESSING), preprocess_ecg_image(upright, PREPROCESSING))


@pytest.mark.skipif(not os.environ.get('DINOV3_SNAPSHOT'), reason='set DINOV3_SNAPSHOT to a local copy of the repository')
def test_published_heads_load_against_the_published_encoder():
    bank = db._DeepECG_DINOv3_Bank(os.environ['DINOV3_SNAPSHOT'], torch.device('cpu'))
    assert bank.encoder_sha256 == db.ENCODER_SHA256
    for (task, variant), cls in dw.TASK_MODELS.items():
        head = bank.load_head(f'{cls.task}_{variant}', cls.expected_outputs)
        outputs = list(ECG_PATTERNS) if cls.expected_outputs == db.ALL_77 else list(cls.expected_outputs)
        assert head['outputs'] == outputs
        assert tuple(head['tensors']['weight'].shape) == (len(outputs), 1536)
        assert head['activation'] == ('softmax' if cls.task == 'hcm_3class' else 'sigmoid')
