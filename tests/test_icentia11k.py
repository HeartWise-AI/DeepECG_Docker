"""Icentia-11k adapter on synthetic WFDB records (no access to the real dataset)."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingestion import mapping, metrics
from ingestion.icentia11k import labels, reader, run
from utils.constants import ECG_PATTERNS

MAPPING = 'ingestion/icentia11k/mapping.csv'


def _ann(events):
    """events: (sample, symbol, aux), in time order."""
    return reader.Annotations(np.asarray([e[0] for e in events], dtype=np.int64),
                              np.asarray([e[1] for e in events]), np.asarray([e[2] for e in events]))


def _write_segment(d, name, n_samples, events, fs=250, truncate=False):
    sig = (np.random.default_rng(0).normal(size=n_samples) * 74).astype(np.int16)  # no line-noise peaks: fast cleaning
    wfdb.wrsamp(name, fs=fs, units=['mV'], sig_name=['ecg'], d_signal=sig[:, None], fmt=['16'],
                adc_gain=[74.36], baseline=[0], write_dir=str(d))
    wfdb.wrann(name, 'atr', sample=np.asarray([e[0] for e in events]), symbol=[e[1] for e in events],
               aux_note=[e[2] for e in events], write_dir=str(d))
    if truncate:
        dat = d / f'{name}.dat'
        dat.write_bytes(dat.read_bytes()[:-10])


def _sinus(n_samples):
    return [(0, '+', '(N')] + [(s, 'N', '') for s in range(100, n_samples, 250)] + [(n_samples, '+', ')')]


def test_read_segment_and_windows(tmp_path):
    _write_segment(tmp_path, 'p00000_s00', 2500 * 3 + 7, _sinus(7500))
    mv, raw, ann = reader.read_segment(str(tmp_path), 'p00000_s00')
    assert mv[1] == pytest.approx(raw[1] / 74.36) and ann.symbols[0] == '+' and ann.aux[0] == '(N'
    raw = raw.copy()
    raw[2500 + 3], raw[5000 + 9] = 32767, -32768
    w, starts, reason, flat = reader.cut_windows(mv, raw)
    assert w.shape == (3, 2500) and starts.tolist() == [0, 2500, 5000]
    assert reason.tolist() == ['', 'saturated', 'invalid'] and not flat.any()


def test_window_labels():
    fs, size = 250, 2500
    ev = [(0, '+', '(N')]
    ev += [(s, 'N', '') for s in range(100, 2500, 250)]                                     # w0: clean sinus, 60 bpm
    ev += [(s, 'N' if k % 2 == 0 else 'V', '') for k, s in enumerate(range(2600, 5000, 150))]  # w1: NVNV..., 100 bpm
    ev += [(5100, '+', ')')]                                                                # rhythm ends inside w2
    ev += [(s, 'Q', '') for s in range(5100, 7500, 300)]                                    # w2: all Q, partly unlabelled
    ev += [(7500, '+', '(AFIB')] + [(s, 'N', '') for s in range(7600, 10000, 200)] + [(10000, '+', ')')]  # w3: AF
    ev += [(s, 'N', '') for s in range(10100, 12500, 180)]                                  # w4: unlabelled, 83 bpm
    ev += [(12500, '+', '(N')] + [(s, 'N', '') for s in range(12600, 15000, 300)]           # w5: clean sinus, 50 bpm
    ev += [(s, 'N' if s != 15700 else 'S', '') for s in range(15100, 17500, 200)]           # w6: sinus with a PAC
    ev += [(17500, '+', ')')]
    lab = labels.label_windows(_ann(sorted(ev)), np.arange(7) * size, size, fs)

    def col(c):
        return [None if np.isnan(v) else v for v in lab[c]]
    assert col('AFIB') == [0, 0, None, 1, None, 0, 0]
    assert col('N_RHYTHM') == [1, 1, None, 0, None, 1, 1]
    assert col('PVC') == [0, 1, None, None, None, 0, None]
    assert col('PAC') == [0, None, None, None, None, 0, 1]
    assert col('BIGEMINY') == [0, 1, None, None, None, 0, None]
    assert col('BRADY') == [0, None, None, None, None, 1, None]
    assert col('TACHY') == [0, None, None, None, None, None, None]
    assert lab.hr_bpm[0] == pytest.approx(60) and lab.q_fraction[2] == 1


@pytest.mark.parametrize('late,whole', [(2, True), (3, False)])
def test_whole_window_rhythm_allows_two_samples(late, whole):
    ann = _ann([(late, '+', '(AFIB'), (2500, '+', ')')])
    lab = labels.label_windows(ann, np.array([0]), 2500, 250)
    assert (lab.rhythm[0] == 'AFIB') == whole


def test_heart_rate_needs_three_beats_and_no_duplicates():
    def hr(beats):
        ev = [(0, '+', '(N')] + [(s, 'N', '') for s in beats] + [(2500, '+', ')')]
        return labels.label_windows(_ann(ev), np.array([0]), 2500, 250).hr_bpm[0]
    assert hr([100, 1000, 1900]) == pytest.approx(60 / 3.6)  # 3 beats, 3.6 s gaps: rate known
    assert np.isnan(hr([100, 1000])) and np.isnan(hr([100, 100, 600, 1100]))


def test_alternating_run():
    assert labels.longest_alternating_nv(list('NVNVNV')) == 6
    assert labels.longest_alternating_nv(list('VNVNV')) == 5
    assert labels.longest_alternating_nv(list('NVNNVNVNVQ')) == 6


def test_rhythm_reopened_and_unclosed_intervals():
    ann = _ann([(0, '+', '(N'), (5000, '+', '(AFIB'), (9000, '+', ')'), (10000, '+', '(AFL')])
    iv = labels.rhythm_intervals(ann)
    assert iv[:2] == [(0, 5000, 'N_RHYTHM'), (5000, 9000, 'AFIB')]
    assert iv[2][0] == 10000 and iv[2][2] == 'AFL' and iv[2][1] > 10 ** 12
    lab = labels.label_windows(ann, np.array([0, 2500, 10000]), 2500, 250)
    assert lab.rhythm.tolist() == ['N_RHYTHM', 'N_RHYTHM', 'AFL']


def test_missing_annotations_are_unknown_not_zero():
    lab = labels.label_windows(None, np.array([0, 2500]), 2500, 250)
    assert lab.n_N.isna().all() and lab.clean_sinus.isna().all() and lab[labels.DATASET_LABELS].isna().all().all()


def test_mapping(tmp_path):
    p = tmp_path / 'm.csv'
    p.write_text('dataset_label,deepecg_class\nAFIB,Afib\nAFL,Afib\nPVC,Premature ventricular complex\n')
    m = mapping.load_mapping(str(p), labels.DATASET_LABELS)
    assert m == {'Afib': ['AFIB', 'AFL'], 'Premature ventricular complex': ['PVC']}
    lab = pd.DataFrame({'AFIB': [1, 0, 0, np.nan], 'AFL': [0, 1, 0, np.nan], 'PVC': [0, 0, 0, 1]})
    gt = mapping.ground_truth(lab, m)
    assert gt.Afib.tolist()[:3] == [1, 1, 0] and np.isnan(gt.Afib[3])
    p.write_text('dataset_label,deepecg_class\nAFIB,Atrial fib\n')
    with pytest.raises(ValueError):
        mapping.load_mapping(str(p), labels.DATASET_LABELS)
    with pytest.raises(FileNotFoundError):
        mapping.load_mapping(str(tmp_path / 'missing.csv'), labels.DATASET_LABELS)


def test_default_mapping_file_is_valid():
    m = mapping.load_mapping(MAPPING, labels.DATASET_LABELS)
    assert m['Afib|Atrial flutter'] == ['AFIB', 'AFL'] and len(m) == 7


def test_combined_task_score():
    probs = np.zeros((2, 77), dtype=np.float32)
    probs[0, ECG_PATTERNS.index('Afib')] = 0.5
    probs[0, ECG_PATTERNS.index('Atrial flutter')] = 0.5
    probs[1, ECG_PATTERNS.index('Atrial flutter')] = 0.2
    assert mapping.task_scores(probs, 'Afib|Atrial flutter') == pytest.approx([0.75, 0.2])


def test_thresholds_use_the_thresholds_json_format(tmp_path, capsys):
    tasks = ['Afib|Atrial flutter', 'Sinusal']
    p = tmp_path / 't.json'
    p.write_text(json.dumps({'Afib|Atrial flutter': {'threshold': 0.2}, 'Bradycardia': {'threshold': 0.4}}))
    assert run.task_thresholds(str(p), tasks) == {'Afib|Atrial flutter': 0.2}
    assert "no threshold for 'Sinusal'" in capsys.readouterr().out
    p.write_text(json.dumps({'Sinusal': {'threshold': 1.5}}))
    with pytest.raises(ValueError, match='between 0 and 1'):
        run.task_thresholds(str(p), tasks)
    assert run.task_thresholds('thresholds/wcr_77_classes_ecg_machine_diagnosis.json', tasks)['Sinusal'] > 0


def test_per_patient_groups_match_naive():
    rng = np.random.default_rng(1)
    patients = rng.choice(['p2', 'p0', 'p1'], 300)
    y = rng.choice([0.0, 1.0, np.nan], 300)
    s = rng.random(300)
    got = metrics.per_patient(y, s, patients, 0.5)
    for _, r in got.iterrows():
        m = (patients == r.patient) & ~np.isnan(y)
        assert r.n_pos == (y[m] == 1).sum() and r.auroc == pytest.approx(metrics.roc_auc_score(y[m], s[m]))
    assert got.patient.tolist() == ['p0', 'p1', 'p2']


def test_parse_patients_only_accepts_patient_ids():
    assert run.parse_patients('p00000-p00002,p00020') == ['p00000', 'p00001', 'p00002', 'p00020']
    assert run.parse_patients('p00000-p00002,p00001') == ['p00000', 'p00001', 'p00002']
    for bad in ['../p00000', 'p0000', 'p00002-p00001', 'p00000-p00001-p00002']:
        with pytest.raises(ValueError):
            run.parse_patients(bad)


class FakeModel:
    """Stands in for the 77-class model: probability = mean of the input, so the input reaches it."""
    def __init__(self):
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x.cpu().numpy())
        return torch.sigmoid(x.mean(dim=(1, 2)))[:, None].repeat(1, 77)


def test_cli_end_to_end(tmp_path, monkeypatch):
    pdir = tmp_path / 'data' / 'p00' / 'p00000'
    pdir.mkdir(parents=True)
    _write_segment(pdir, 'p00000_s00', 2500 * 2, _sinus(5000))
    _write_segment(pdir, 'p00000_s01', 2500 * 2, _sinus(5000), truncate=True)  # unreadable
    _write_segment(pdir, 'p00000_s02', 2500 * 2, _sinus(5000), fs=500)         # wrong sampling rate
    model = FakeModel()
    monkeypatch.setattr(run, 'load_model', lambda *args: model)
    out = tmp_path / 'out'
    run.main(['--data', str(tmp_path / 'data'), '--patients', 'p00000-p00001', '--out', str(out),
              '--single_lead', 'V1', '--device', 'cpu', '--bootstrap', '0',
              '--hugging_face_api_key_path', str(tmp_path / 'key.json')])
    df = pd.read_parquet(out / 'p00000_windows.parquet')
    assert len(df) == 2 and df.segment.unique().tolist() == ['p00000_s00']
    assert df['N_RHYTHM'].tolist() == [1, 1] and df['gt::Sinusal'].tolist() == [1, 1]
    x = np.concatenate(model.inputs)
    assert x.shape == (2, 12, 2500) and np.isfinite(x).all() and x[:, 6].any()
    assert np.abs(np.delete(x, 6, axis=1)).max() < 1e-6  # --single_lead V1: the other leads stay zero
    assert np.isfinite(np.load(out / 'p00000_probs.npy')).all()
    m = json.loads((out / 'run_manifest.json').read_text())
    assert m['missing_patients'] == ['p00001'] and m['lead_slots'] == ['V1']
    assert [b['segment'] for b in m['bad_segments']] == ['p00000_s01', 'p00000_s02']
    assert (out / 'label_summary.csv').exists() and (out / 'metrics.csv').exists()


def test_cli_refuses_bad_out_and_empty_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'load_model', lambda *args: FakeModel())
    args = ['--patients', 'p00000', '--device', 'cpu', '--hugging_face_api_key_path', str(tmp_path / 'key.json')]
    (tmp_path / 'data').mkdir()
    with pytest.raises(SystemExit, match='inside --data'):
        run.main(args + ['--data', str(tmp_path / 'data'), '--out', str(tmp_path / 'data' / 'out')])
    (tmp_path / 'used').mkdir()
    (tmp_path / 'used' / 'x').write_text('')
    with pytest.raises(SystemExit, match='new or empty'):
        run.main(args + ['--data', str(tmp_path / 'data'), '--out', str(tmp_path / 'used')])
    with pytest.raises(SystemExit, match="'all' or a lead name"):
        run.main(args + ['--data', str(tmp_path / 'data'), '--out', str(tmp_path / 'o'), '--single_lead', 'none'])
    with pytest.raises(SystemExit, match='no patients'):
        run.main(args + ['--data', str(tmp_path / 'data'), '--out', str(tmp_path / 'o')])
