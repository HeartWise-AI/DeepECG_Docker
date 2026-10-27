"""External-validation ingestion: read -> 10 s windows -> labels -> --mapping -> --lead [-> model -> metrics].

By default it runs wcr_77_classes (needs torch, i.e. the DeepECG Docker image; see run_icentia.sh) and
writes metrics. --labels-only stops after the labels (no torch needed). The dataset is only read.
Missing patients and unreadable segments are logged in run_manifest.json and skipped. --out must be
empty (or --overwrite) and is locked while a run uses it, so a second run on it stops; run_manifest.json is
written first with "complete": false and rewritten with "complete": true at the end, so an interrupted run is visible.

    python -m ingestion.run --data <icentia root> --lead 12 --mapping configs/icentia_mapping.csv \
        --patients p00000-p00009 --out <out dir> \
        [--thresholds <thresholds.csv> --device cuda:0] [--labels-only]
"""
import argparse
import fcntl
import json
import math
import os
import re
import subprocess
import time

import numpy as np
import pandas as pd

from ingestion import icentia, labels, leads, mapping, windows


PATIENT_ID = re.compile(r'p\d{5}')
# Files this pipeline writes into --out (removed by --overwrite, nothing else is touched).
OUTPUT_FILE = re.compile(r'p\d{5}_(windows\.parquet|probs\.npy|example_input\.npy)'
                         r'|label_summary\.csv|metrics\.csv|metrics_per_patient\.csv|metrics\.json'
                         r'|run_manifest\.json')


def own_outputs(out):
    """Names of this pipeline's output files in --out (exact matches, no glob)."""
    return sorted(f for f in os.listdir(out) if OUTPUT_FILE.fullmatch(f))


def lock_out(out):
    """Exclusive flock on the --out folder itself, held until the returned fd is closed.

    No lock file: there is nothing to delete by mistake, and the OS releases the lock when the process
    exits (even on kill -9), so it never goes stale. Works on local filesystems and Docker bind mounts;
    on NFS, flock may not be honoured across machines.
    """
    fd = os.open(out, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise SystemExit(f'--out {out} is in use by another run; wait for it to finish or use another folder')
    return fd


def parse_patients(spec):
    """'p00000-p00009,p00020' -> list of IDs. Only pNNNNN IDs are accepted, so paths can't escape --data/--out."""
    out = []
    for part in spec.split(','):
        ends = part.strip().split('-')
        if len(ends) > 2 or not all(PATIENT_ID.fullmatch(e) for e in ends):
            raise ValueError(f'--patients: {part!r} is not a patient ID (pNNNNN) or range (pNNNNN-pNNNNN)')
        a, b = int(ends[0][1:]), int(ends[-1][1:])
        if a > b:
            raise ValueError(f'--patients: range {part!r} is reversed')
        out += [f'p{i:05d}' for i in range(a, b + 1)]
    return list(dict.fromkeys(out))  # overlapping ranges must not count a patient twice


def patient_dir(data, patient):
    return os.path.join(data, patient[:3], patient)  # p00012 -> p00/p00012


def process_patient(data, patient, slots, class_map, scorer=None, fs_expected=250, bad_segments=None,
                    batch_size=512):
    """Window table for one patient and, if scorer is given, probabilities [n_windows, 77] (NaN for skipped).

    Segments that can't be read are appended to bad_segments as (segment, error) and skipped.
    """
    d = patient_dir(data, patient)
    frames, probs, n_built = [], [], 0
    for seg in icentia.list_segments(d):
        try:
            header, sig, raw, ann = icentia.load_segment(d, seg)
            if header.fs != fs_expected:
                raise ValueError(f'fs={header.fs}, expected {fs_expected} (no resampling)')
        except (OSError, ValueError, IndexError) as e:
            if bad_segments is None:
                raise
            bad_segments.append((seg, f'{type(e).__name__}: {e}'))
            print(f'  skipping {seg}: {type(e).__name__}: {e}', flush=True)
            continue
        w, starts, reason, flat = windows.cut_windows(sig, raw)
        lab = labels.label_windows(ann, starts, windows.WINDOW_SAMPLES, header.fs)
        gt = mapping.ground_truth(lab, class_map).add_prefix('gt::')
        meta = pd.DataFrame({'patient': patient, 'segment': seg, 'window_idx': np.arange(starts.size),
                             'start_sample': starts, 'has_atr': ann is not None,
                             'skip_reason': reason, 'flat': flat})
        frames.append(pd.concat([meta, lab, gt], axis=1))
        keep = reason == ''
        x = leads.build_12_lead(w[keep], slots)  # --lead
        assert x.shape[1:] == (12, windows.WINDOW_SAMPLES)
        n_built += x.shape[0]
        if scorer is not None:
            p = np.full((starts.size, 77), np.nan, dtype=np.float32)
            p[keep] = scorer(x, batch_size=batch_size)
            probs.append(p)
    if not frames:
        return None, None, 0
    return pd.concat(frames, ignore_index=True), (np.concatenate(probs) if probs else None), n_built


def load_thresholds(path, tasks):
    """{task: threshold}. Task names must match the mapping's tasks; missing ones get a warning."""
    if not path:
        return {}
    df = pd.read_csv(path)
    names = df.task.astype(str).str.strip()
    values = pd.to_numeric(df.threshold, errors='coerce')
    bad = [n for n, v in zip(names, values) if not (math.isfinite(v) and 0 <= v <= 1)]
    if bad:
        raise ValueError(f'{path}: thresholds must be numbers between 0 and 1; bad rows: {bad}')
    dup = sorted(names[names.duplicated()].unique())
    if dup:
        raise ValueError(f'{path}: tasks listed more than once: {dup}')
    out = dict(zip(names, values.astype(float)))
    unknown = sorted(set(out) - set(tasks))
    if unknown:
        raise ValueError(f'{path}: tasks not in the mapping: {unknown} (mapping tasks: {sorted(tasks)})')
    for t in tasks:
        if t not in out:
            print(f'warning: no threshold for {t!r}; sensitivity/specificity not reported', flush=True)
    return out


def _json_safe(x):
    """NaN/inf -> None (null) so metrics.json is strict JSON."""
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in x]
    if x is None or x is pd.NA:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def _write_manifest(out, manifest):
    with open(os.path.join(out, 'run_manifest.json'), 'w') as f:
        json.dump(_json_safe(manifest), f, indent=2, allow_nan=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--lead', required=True, help="lead name (I, II, ..., V6) or '12' to copy to all")
    p.add_argument('--mapping', required=True, help="CSV: dataset_label,deepecg_class ('A|B' = combined)")
    p.add_argument('--patients', required=True, help='e.g. p00000-p00009 or p00000,p00003')
    p.add_argument('--out', required=True, help='output folder; must be empty unless --overwrite')
    p.add_argument('--overwrite', action='store_true',
                   help="reuse a non-empty --out: delete this pipeline's own output files in it (the previous "
                        "run's results are lost even if this run then fails); other files are kept")
    p.add_argument('--save-example-input', action='store_true',
                   help='save one 12 x 2500 input per patient for inspection')
    p.add_argument('--labels-only', action='store_true',
                   help='stop after the labels: no model, no metrics (no torch needed)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--thresholds', help='CSV: task,threshold (fixed thresholds for sensitivity/specificity)')
    p.add_argument('--bootstrap', type=int, default=1000, help='patient-bootstrap resamples for CIs (0 = none)')
    a = p.parse_args(argv)

    out, data = os.path.realpath(a.out), os.path.realpath(a.data)
    if out == data or out.startswith(data + os.sep):
        raise SystemExit('--out must not be inside --data (the dataset is read-only)')
    try:
        patient_ids = parse_patients(a.patients)
    except ValueError as e:
        raise SystemExit(str(e))
    if a.batch_size < 1:
        raise SystemExit('--batch-size must be at least 1')
    slots = leads.parse_lead(a.lead)
    class_map = mapping.load_mapping(a.mapping, labels.DATASET_LABELS)
    thresholds = load_thresholds(a.thresholds, class_map)
    if os.path.exists(out) and not os.path.isdir(out):
        raise SystemExit(f'--out {out} exists and is not a folder')
    os.makedirs(out, exist_ok=True)
    lock = lock_out(out)  # before the emptiness check, so a second run always stops here
    try:
        if os.listdir(out) and not a.overwrite:
            raise SystemExit(f'--out {out} is not empty; use another folder or --overwrite')
        _run(a, out, patient_ids, slots, class_map, thresholds)
    finally:
        os.close(lock)


def _run(a, out, patient_ids, slots, class_map, thresholds):
    infer = not a.labels_only
    scorer = None
    if infer:  # load the model before touching --out, so a bad --device or missing weights deletes nothing
        from ingestion.infer import WCRScorer, MODEL_NAME
        scorer = WCRScorer(a.device)
    for f in own_outputs(out):  # only reached with --overwrite (or an empty folder)
        os.remove(os.path.join(out, f))
    _write_manifest(out, {'complete': False, 'args': vars(a), 'started': time.strftime('%Y-%m-%dT%H:%M:%S')})

    t0, per_patient, missing, bad_segments = time.time(), [], [], []
    for pat in patient_ids:
        tp = time.time()
        d = patient_dir(a.data, pat)
        if not os.path.isdir(d) or not icentia.list_segments(d):
            missing.append(pat)
            print(f'{pat}: missing or no readable segments in {d}, skipped', flush=True)
            continue
        df, probs, n_built = process_patient(a.data, pat, slots, class_map, scorer, bad_segments=bad_segments,
                                             batch_size=a.batch_size)
        if df is None:
            missing.append(pat)
            print(f'{pat}: no readable segments, skipped', flush=True)
            continue
        df.to_parquet(os.path.join(out, f'{pat}_windows.parquet'), index=False)
        if probs is not None:
            np.save(os.path.join(out, f'{pat}_probs.npy'), probs)  # float32: metrics are computed from it
        if a.save_example_input:
            first = df.index[(df.skip_reason == '') & df['gt::' + next(iter(class_map))].notna()]
            if len(first):
                r = df.loc[first[0]]
                h, sig, raw, _ = icentia.load_segment(patient_dir(a.data, pat), r.segment)
                w = sig[r.start_sample:r.start_sample + windows.WINDOW_SAMPLES][None]
                np.save(os.path.join(out, f'{pat}_example_input.npy'), leads.build_12_lead(w, slots)[0])
        per_patient.append({'patient': pat, 'segments': df.segment.nunique(), 'windows': len(df),
                            'inputs_built': n_built, 'seconds': round(time.time() - tp, 1)})
        print(f'{pat}: {df.segment.nunique()} segments, {len(df)} windows, {n_built} inputs built '
              f'({per_patient[-1]["seconds"]} s)', flush=True)

    pats = [r['patient'] for r in per_patient]
    if not pats:
        raise SystemExit(f'no patients could be read from {a.data} (missing: {missing})')
    all_df = pd.concat([pd.read_parquet(os.path.join(out, f'{p}_windows.parquet')) for p in pats],
                       ignore_index=True)
    summary = []
    for col in labels.DATASET_LABELS + [c for c in all_df.columns if c.startswith('gt::')]:
        v = all_df.loc[all_df.skip_reason == '', col]
        summary.append({'label': col, 'positive': int((v == 1).sum()), 'negative': int((v == 0).sum()),
                        'unknown': int(v.isna().sum())})
    pd.DataFrame(summary).to_csv(os.path.join(out, 'label_summary.csv'), index=False)

    if infer:
        from ingestion import metrics
        classes = mapping.load_ecg_patterns()
        probs = np.concatenate([np.load(os.path.join(out, f'{p}_probs.npy')) for p in pats])
        patients = all_df.patient.to_numpy()
        pooled, per_pat = [], []
        for task in class_map:
            y = all_df['gt::' + task].to_numpy(dtype=float)
            y[all_df.skip_reason.to_numpy() != ''] = np.nan
            s = mapping.task_scores(probs, task, classes)
            thr = thresholds.get(task)
            m = metrics.task_metrics(y, s, patients, thr, n_boot=a.bootstrap)
            pooled.append({'task': task, 'threshold': thr, **m})
            per_pat.append(metrics.per_patient(y, s, patients, thr).assign(task=task))
        pd.DataFrame(pooled).to_csv(os.path.join(out, 'metrics.csv'), index=False)
        pd.concat(per_pat).to_csv(os.path.join(out, 'metrics_per_patient.csv'), index=False)
        with open(os.path.join(out, 'metrics.json'), 'w') as f:
            json.dump(_json_safe(pooled), f, indent=2, allow_nan=False)

    try:
        git = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                      cwd=os.path.dirname(os.path.abspath(__file__)), text=True,
                                      stderr=subprocess.DEVNULL).strip()
    except Exception:
        git = None
    manifest = {
        'complete': True, 'mode': 'infer' if infer else 'labels-only', 'args': vars(a),
        # set by run_icentia.sh: the host paths behind the container's --data / --out
        'host_paths': {k: os.environ[v] for k, v in (('data', 'ICENTIA_HOST_DATA'), ('out', 'ICENTIA_HOST_OUT'))
                       if v in os.environ},
        'lead_slots': [leads.LEAD_ORDER[s] for s in slots],
        'mapping': class_map, 'thresholds': thresholds, 'label_rules': labels.RULES, 'git_commit': git,
        'model': MODEL_NAME if infer else None, 'model_mode': 'eval + no_grad' if infer else None,
        'window_samples': windows.WINDOW_SAMPLES,
        'skipped_windows': all_df.skip_reason[all_df.skip_reason != ''].value_counts().to_dict(),
        'flat_windows': int(all_df.flat.sum()),
        'segments_without_atr': sorted(all_df.loc[~all_df.has_atr, 'segment'].unique().tolist()),
        'missing_patients': missing,
        'bad_segments': [{'segment': s, 'error': e} for s, e in bad_segments],
        'per_patient': per_patient, 'seconds': round(time.time() - t0, 1),
    }
    _write_manifest(out, manifest)
    print(f'done in {manifest["seconds"]} s -> {out}')


if __name__ == '__main__':
    main()
