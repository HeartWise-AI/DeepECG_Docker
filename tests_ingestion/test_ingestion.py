"""Unit tests on synthetic data only (no access to the real dataset)."""
import numpy as np
import pandas as pd
import pytest

from ingestion import icentia, labels, leads, mapping, windows


def _encode_atr(events):
    """events: list of (sample, code, aux). Writes MIT format, using SKIP for big or negative steps."""
    words, t = [], 0
    for s, code, aux in events:
        d = s - t
        if d < 0 or d > 1023:
            v = d & 0xFFFFFFFF
            words += [59 << 10, v >> 16, v & 0xFFFF]
            d = 0
        words.append((code << 10) | d)
        if aux:
            b = aux.encode()
            words.append((63 << 10) | len(b))
            b += b'\x00' * (len(b) % 2)
            words += list(np.frombuffer(b, dtype='<u2'))
        t = s
    words.append(0)
    return np.asarray(words, dtype='<u2')


def test_atr_signed_skip(tmp_path):
    ev = [(5000, 1, ''), (100, 5, ''), (200, 28, '(N')]  # second step is negative
    p = tmp_path / 'x.atr'
    _encode_atr(ev).tofile(p)
    a = icentia.read_annotations(str(p))
    assert a.samples.tolist() == [5000, 100, 200]
    assert a.codes.tolist() == [1, 5, 28]
    assert a.aux == ['', '', '(N']


def test_header_and_signal(tmp_path):
    (tmp_path / 'r.hea').write_text('r 1 250 4\nr.dat 16 74.36(0)/mV 16 0 -5 1667 0 ecg\n')
    np.asarray([0, 74, -74, 32767], dtype='<i2').tofile(tmp_path / 'r.dat')
    h = icentia.read_header(str(tmp_path / 'r.hea'))
    assert (h.fs, h.n_samples, h.gain, h.baseline) == (250.0, 4, 74.36, 0)
    mv, raw = icentia.read_signal_mv(str(tmp_path / 'r.dat'), h)
    assert mv[1] == pytest.approx(74 / 74.36)


def test_windows_and_skip():
    raw = np.zeros(2500 * 3 + 7, dtype=np.int16)
    raw[2500 + 3] = 32767
    raw[5000 + 9] = -32768
    w, starts, reason, flat = windows.cut_windows(raw / 74.36, raw)
    assert w.shape == (3, 2500) and starts.tolist() == [0, 2500, 5000]
    assert reason.tolist() == ['', 'saturated', 'invalid'] and flat.tolist() == [True, False, False]


def test_lead_options():
    w = np.random.randn(2, 2500).astype(np.float32)
    x = leads.build_12_lead(w, leads.parse_lead('12'))
    assert x.shape == (2, 12, 2500) and np.allclose(x[:, 7], w)
    x = leads.build_12_lead(w, leads.parse_lead('II'))
    assert np.allclose(x[:, 1], w) and not x[:, [0] + list(range(2, 12))].any()
    with pytest.raises(ValueError):
        leads.parse_lead('V7')


def _ann(events):
    return icentia.Annotations(np.asarray([e[0] for e in events]), np.asarray([e[1] for e in events]),
                               [e[2] for e in events])


def test_window_labels():
    fs, size = 250, 2500
    ev = [(0, 28, '(N')]
    ev += [(s, 1, '') for s in range(100, 2500, 250)]                    # w0: clean sinus, 60 bpm
    ev += [(s, 1 if k % 2 == 0 else 5, '') for k, s in enumerate(range(2600, 5000, 150))]  # w1: NVNV..., 100 bpm
    ev += [(5100, 28, ')')]                                             # rhythm ends inside w2
    ev += [(s, 13, '') for s in range(5100, 7500, 300)]                 # w2: all Q, partly unlabelled
    ev += [(7500, 28, '(AFIB')] + [(s, 1, '') for s in range(7600, 10000, 200)] + [(10000, 28, ')')]  # w3: AF
    ev += [(s, 1, '') for s in range(10100, 12500, 180)]                # w4: unlabelled, 83 bpm
    ev += [(12500, 28, '(N')] + [(s, 1, '') for s in range(12600, 15000, 300)]  # w5: clean sinus, 50 bpm
    ev += [(s, 1 if s != 15700 else 9, '') for s in range(15100, 17500, 200)]   # w6: sinus with a PAC
    ev += [(17500, 28, ')')]
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


def test_alternating_run():
    assert labels.longest_alternating_nv(list('NVNVNV')) == 6
    assert labels.longest_alternating_nv(list('VNVNV')) == 5
    assert labels.longest_alternating_nv(list('NVNNVNVNVQ')) == 6


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
    m = mapping.load_mapping('configs/icentia_mapping.csv', labels.DATASET_LABELS)
    assert len(mapping.load_ecg_patterns()) == 77 and m['Afib|Atrial flutter'] == ['AFIB', 'AFL']


def test_combined_task_score():
    classes = mapping.load_ecg_patterns()
    probs = np.zeros((2, 77), dtype=np.float32)
    probs[0, classes.index('Afib')] = 0.5
    probs[0, classes.index('Atrial flutter')] = 0.5
    probs[1, classes.index('Atrial flutter')] = 0.2
    s = mapping.task_scores(probs, 'Afib|Atrial flutter', classes)
    assert s == pytest.approx([0.75, 0.2])


def _write_segment(d, name, n_samples, events, truncate=False, fs=250):
    (d / f'{name}.hea').write_text(f'{name} 1 {fs} {n_samples}\n{name}.dat 16 74.36(0)/mV 16 0 0 0 0 ecg\n')
    sig = (np.sin(np.arange(n_samples) / 10) * 74).astype('<i2')
    (sig[:-5] if truncate else sig).tofile(d / f'{name}.dat')
    _encode_atr(events).tofile(d / f'{name}.atr')


def test_cli_end_to_end(tmp_path):
    import json
    from ingestion import run
    pdir = tmp_path / 'data' / 'p00' / 'p00000'
    pdir.mkdir(parents=True)
    ev = [(0, 28, '(N')] + [(s, 1, '') for s in range(100, 5000, 250)] + [(5000, 28, ')')]
    _write_segment(pdir, 'p00000_s00', 2500 * 2, ev)
    _write_segment(pdir, 'p00000_s01', 2500 * 2, ev, truncate=True)  # unreadable: .dat shorter than header
    _write_segment(pdir, 'p00000_s02', 2500 * 2, ev, fs=500)         # wrong sampling rate
    out = tmp_path / 'out'
    run.main(['--data', str(tmp_path / 'data'), '--lead', '12', '--mapping', 'configs/icentia_mapping.csv',
              '--patients', 'p00000-p00001', '--out', str(out), '--labels-only'])
    df = pd.read_parquet(out / 'p00000_windows.parquet')
    assert len(df) == 2 and df.segment.unique().tolist() == ['p00000_s00']
    assert df['N_RHYTHM'].tolist() == [1, 1] and df['gt::Sinusal'].tolist() == [1, 1]
    m = json.loads((out / 'run_manifest.json').read_text())
    assert m['mode'] == 'labels-only' and m['missing_patients'] == ['p00001']
    assert [b['segment'] for b in m['bad_segments']] == ['p00000_s01', 'p00000_s02']
    assert (out / 'label_summary.csv').exists() and not (out / 'metrics.csv').exists()


def test_cli_refuses_out_inside_data_and_empty_runs(tmp_path):
    from ingestion import run
    args = ['--lead', '12', '--mapping', 'configs/icentia_mapping.csv', '--patients', 'p00000', '--labels-only']
    with pytest.raises(SystemExit, match='inside --data'):
        run.main(args + ['--data', str(tmp_path), '--out', str(tmp_path / 'out')])
    with pytest.raises(SystemExit, match='inside --data'):
        run.main(args + ['--data', str(tmp_path), '--out', str(tmp_path)])
    (tmp_path / 'link').symlink_to(tmp_path / 'data_dir', target_is_directory=True)
    (tmp_path / 'data_dir').mkdir()
    with pytest.raises(SystemExit, match='inside --data'):
        run.main(args + ['--data', str(tmp_path / 'data_dir'), '--out', str(tmp_path / 'link' / 'o')])
    with pytest.raises(SystemExit, match='no patients'):
        run.main(args + ['--data', str(tmp_path / 'data'), '--out', str(tmp_path / 'o')])


def test_unsorted_annotations_give_same_labels():
    ev = [(0, 28, '(N')] + [(s, 1 if s != 3100 else 5, '') for s in range(100, 5000, 250)] + [(5000, 28, ')')]
    starts = np.arange(2) * 2500
    ref = labels.label_windows(_ann(ev), starts, 2500, 250)
    shuffled = [ev[i] for i in np.random.default_rng(0).permutation(len(ev))]
    assert labels.label_windows(_ann(shuffled), starts, 2500, 250).equals(ref)
    assert ref.n_V.tolist() == [0, 1] and ref.PVC.tolist() == [0, 1]


def test_rhythm_reopened_and_unclosed_intervals():
    ann = _ann([(0, 28, '(N'), (5000, 28, '(AFIB'), (9000, 28, ')'), (10000, 28, '(AFL')])
    iv = labels.rhythm_intervals(ann)
    assert iv[:2] == [(0, 5000, 'N_RHYTHM'), (5000, 9000, 'AFIB')]
    assert iv[2][0] == 10000 and iv[2][2] == 'AFL' and iv[2][1] > 10 ** 12
    lab = labels.label_windows(ann, np.array([0, 2500, 10000]), 2500, 250)
    assert lab.rhythm.tolist() == ['N_RHYTHM', 'N_RHYTHM', 'AFL']


def test_duplicate_beat_marks_leave_rate_unknown():
    ev = [(0, 28, '(N')] + [(s, 1, '') for s in [100, 100, 100, 100, 600]] + [(2500, 28, ')')]
    lab = labels.label_windows(_ann(ev), np.array([0]), 2500, 250)
    assert np.isnan(lab.hr_bpm[0]) and np.isnan(lab.TACHY[0]) and np.isnan(lab.BRADY[0])


def test_threshold_file_validated(tmp_path):
    from ingestion import run
    tasks = ['Afib|Atrial flutter', 'Sinusal']
    p = tmp_path / 't.csv'
    p.write_text('task,threshold\n Afib|Atrial flutter ,0.2\n')
    assert run.load_thresholds(str(p), tasks) == {'Afib|Atrial flutter': 0.2}
    p.write_text('task,threshold\nAfib | Atrial flutter,0.2\n')
    with pytest.raises(ValueError, match='not in the mapping'):
        run.load_thresholds(str(p), tasks)
    m = mapping.load_mapping('configs/icentia_mapping.csv', labels.DATASET_LABELS)
    p.write_text('task,threshold\n' + ''.join(f'"{t}",0.5\n' for t in list(m)[:-1]))
    assert set(run.load_thresholds(str(p), m)) == set(list(m)[:-1])  # last task missing: warning only


def test_per_patient_groups_match_naive():
    from ingestion import metrics
    rng = np.random.default_rng(1)
    patients = rng.choice(['p2', 'p0', 'p1'], 300)
    y = rng.choice([0.0, 1.0, np.nan], 300)
    s = rng.random(300)
    got = metrics.per_patient(y, s, patients, 0.5)
    for _, r in got.iterrows():
        m = (patients == r.patient) & ~np.isnan(y)
        assert r.n_pos == (y[m] == 1).sum() and r.auroc == pytest.approx(
            metrics.roc_auc_score(y[m], s[m]))
    assert got.patient.tolist() == ['p0', 'p1', 'p2']


# --- PR #81 review fixes --------------------------------------------------------------------------

def _one_patient(tmp_path, pid='p00000'):
    pdir = tmp_path / 'data' / pid[:3] / pid
    pdir.mkdir(parents=True)
    ev = [(0, 28, '(N')] + [(s, 1, '') for s in range(100, 5000, 250)] + [(5000, 28, ')')]
    _write_segment(pdir, f'{pid}_s00', 2500 * 2, ev)
    return tmp_path / 'data'


def test_parse_patients_only_accepts_patient_ids():
    from ingestion import run
    assert run.parse_patients('p00000-p00002,p00020') == ['p00000', 'p00001', 'p00002', 'p00020']
    for bad in ['../etc', 'p0000', 'p00001-../p00009', 'p00005-p00001', 'p00001-p00002-p00003', 'p00001/x']:
        with pytest.raises(ValueError):
            run.parse_patients(bad)


def test_out_must_be_empty_or_overwrite(tmp_path):
    import json
    from ingestion import run
    data, out = _one_patient(tmp_path), tmp_path / 'out'
    args = ['--data', str(data), '--lead', '12', '--mapping', 'configs/icentia_mapping.csv',
            '--patients', 'p00000', '--out', str(out), '--labels-only']
    run.main(args)
    (out / 'p00099_windows.parquet').write_text('stale output of an earlier run')
    (out / 'notes.txt').write_text('not ours')
    with pytest.raises(SystemExit, match='not empty'):
        run.main(args)
    run.main(args + ['--overwrite'])
    assert not (out / 'p00099_windows.parquet').exists() and (out / 'notes.txt').exists()
    assert json.loads((out / 'run_manifest.json').read_text())['complete'] is True


def test_interrupted_run_leaves_incomplete_manifest(tmp_path, monkeypatch):
    import json
    from ingestion import run
    data, out = _one_patient(tmp_path), tmp_path / 'out'

    def boom(*a, **k):
        raise RuntimeError('interrupted')
    monkeypatch.setattr(run, 'process_patient', boom)
    with pytest.raises(RuntimeError):
        run.main(['--data', str(data), '--lead', '12', '--mapping', 'configs/icentia_mapping.csv',
                  '--patients', 'p00000', '--out', str(out), '--labels-only'])
    assert json.loads((out / 'run_manifest.json').read_text())['complete'] is False


@pytest.mark.parametrize('value', ['', 'abc', '1.5', '-0.1', 'inf', 'nan'])
def test_bad_threshold_values_rejected(tmp_path, value):
    from ingestion import run
    p = tmp_path / 't.csv'
    p.write_text(f'task,threshold\nSinusal,{value}\n')
    with pytest.raises(ValueError, match='between 0 and 1'):
        run.load_thresholds(str(p), ['Sinusal'])


def test_duplicate_threshold_rows_rejected(tmp_path):
    from ingestion import run
    p = tmp_path / 't.csv'
    p.write_text('task,threshold\nSinusal,0.5\nSinusal,0.6\n')
    with pytest.raises(ValueError, match='more than once'):
        run.load_thresholds(str(p), ['Sinusal'])


def test_any_zero_rr_gap_leaves_rate_unknown():
    beats = [100, 400, 700, 700, 1000, 1300, 1600]  # one duplicate mark; median gap is still 300
    ev = [(0, 28, '(N')] + [(s, 1, '') for s in beats] + [(2500, 28, ')')]
    lab = labels.label_windows(_ann(ev), np.array([0]), 2500, 250)
    assert np.isnan(lab.hr_bpm[0]) and np.isnan(lab.BRADY[0]) and np.isnan(lab.TACHY[0])


def test_missing_annotations_are_unknown_not_zero():
    lab = labels.label_windows(None, np.array([0, 2500]), 2500, 250)
    assert lab.n_N.isna().all() and lab.n_V.isna().all() and lab.clean_sinus.isna().all()
    assert lab.PVC.isna().all() and lab.AFIB.isna().all()


def test_inference_path_batch_size_and_strict_json(tmp_path, monkeypatch):
    """Fake model (no torch): --batch-size reaches the scorer, metrics.json has null, not NaN."""
    import json
    import sys
    import types
    from ingestion import run
    seen = []

    class FakeScorer:
        def __init__(self, device):
            pass

        def __call__(self, x, batch_size=512):
            seen.append(batch_size)
            return np.random.default_rng(0).random((x.shape[0], 77)).astype(np.float32)
    monkeypatch.setitem(sys.modules, 'ingestion.infer',
                        types.SimpleNamespace(WCRScorer=FakeScorer, MODEL_NAME='fake'))
    data, out = _one_patient(tmp_path), tmp_path / 'out'
    run.main(['--data', str(data), '--lead', '12', '--mapping', 'configs/icentia_mapping.csv',
              '--patients', 'p00000', '--out', str(out), '--batch-size', '7', '--bootstrap', '0'])
    assert seen == [7]
    text = (out / 'metrics.json').read_text()
    assert 'NaN' not in text
    m = {r['task']: r for r in json.loads(text)}
    assert m['Sinusal']['n_pos'] == 2 and m['Sinusal']['auroc'] is None  # no negatives -> null
    with pytest.raises(SystemExit, match='batch-size'):
        run.main(['--data', str(data), '--lead', '12', '--mapping', 'configs/icentia_mapping.csv',
                  '--patients', 'p00000', '--out', str(tmp_path / 'o2'), '--batch-size', '0'])


# --- second code review of the PR #81 fixes -------------------------------------------------------

def test_overlapping_patient_specs_count_each_patient_once():
    from ingestion import run
    assert run.parse_patients('p00000-p00002,p00001,p00002-p00003') == ['p00000', 'p00001', 'p00002', 'p00003']


def test_overwrite_deletes_only_exact_outputs_even_with_glob_characters(tmp_path):
    from ingestion import run
    data, out = _one_patient(tmp_path), tmp_path / 'run[batch1]'
    args = ['--data', str(data), '--lead', '12', '--mapping', 'configs/icentia_mapping.csv',
            '--patients', 'p00000', '--out', str(out), '--labels-only']
    run.main(args)
    (out / 'p00099_windows.parquet').write_text('stale')
    (out / 'pilot_probs.npy').write_text('a user file shaped like an output')
    run.main(args + ['--overwrite'])
    assert not (out / 'p00099_windows.parquet').exists() and (out / 'pilot_probs.npy').exists()


def test_out_that_is_a_file_is_refused(tmp_path):
    from ingestion import run
    data = _one_patient(tmp_path)
    (tmp_path / 'f').write_text('x')
    with pytest.raises(SystemExit, match='not a folder'):
        run.main(['--data', str(data), '--lead', '12', '--mapping', 'configs/icentia_mapping.csv',
                  '--patients', 'p00000', '--out', str(tmp_path / 'f'), '--labels-only'])


def test_failed_model_load_keeps_previous_results(tmp_path, monkeypatch):
    import sys
    import types
    from ingestion import run
    data, out = _one_patient(tmp_path), tmp_path / 'out'
    args = ['--data', str(data), '--lead', '12', '--mapping', 'configs/icentia_mapping.csv',
            '--patients', 'p00000', '--out', str(out)]
    run.main(args + ['--labels-only'])
    before = sorted(p.name for p in out.iterdir())

    class Broken:
        def __init__(self, device):
            raise RuntimeError('no such device')
    monkeypatch.setitem(sys.modules, 'ingestion.infer', types.SimpleNamespace(WCRScorer=Broken, MODEL_NAME='x'))
    with pytest.raises(RuntimeError):
        run.main(args + ['--overwrite'])
    assert sorted(p.name for p in out.iterdir()) == before


def test_json_safe_handles_numpy_and_pandas_values():
    import json
    from ingestion import run
    x = {'a': np.nan, 'b': pd.NA, 'c': np.bool_(True), 'd': np.arange(2), 'e': (np.float32(0.5), np.inf), 'f': None}
    assert json.loads(json.dumps(run._json_safe(x), allow_nan=False)) == \
        {'a': None, 'b': None, 'c': True, 'd': [0, 1], 'e': [0.5, None], 'f': None}


def test_window_table_column_types_match_with_and_without_annotations():
    ev = [(0, 28, '(N')] + [(s, 1, '') for s in range(100, 2500, 250)] + [(2500, 28, ')')]
    a = labels.label_windows(_ann(ev), np.array([0]), 2500, 250)
    b = labels.label_windows(None, np.array([0]), 2500, 250)
    assert a.dtypes.equals(b.dtypes) and a.n_N[0] == 10 and bool(a.clean_sinus[0])
    both = pd.concat([a, b], ignore_index=True)
    assert both[both.clean_sinus.fillna(False)].shape[0] == 1


# --- PR #81 review: two runs sharing --out ---------------------------------------------------------

def test_out_is_locked_while_a_run_uses_it(tmp_path, monkeypatch):
    from ingestion import run
    data, out = _one_patient(tmp_path), tmp_path / 'out'
    args = ['--data', str(data), '--lead', '12', '--mapping', 'configs/icentia_mapping.csv',
            '--patients', 'p00000', '--out', str(out), '--labels-only']
    run.main(args)
    assert not (out / run.LOCK_FILE).exists()  # released after a normal run

    def second_run(*a, **k):  # a second run starts while the first is still processing
        with pytest.raises(SystemExit, match='in use by another run'):
            run.main(args + ['--overwrite'])
        raise RuntimeError('first run interrupted')
    monkeypatch.setattr(run, 'process_patient', second_run)
    with pytest.raises(RuntimeError):
        run.main(args + ['--overwrite'])
    assert not (out / run.LOCK_FILE).exists()  # released after a failed run too

    (out / run.LOCK_FILE).write_text('left by a killed run')
    with pytest.raises(SystemExit, match='delete that file'):
        run.main(args + ['--overwrite'])
