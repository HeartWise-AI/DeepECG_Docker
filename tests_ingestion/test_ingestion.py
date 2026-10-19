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
