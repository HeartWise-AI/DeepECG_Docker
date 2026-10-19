"""External-validation ingestion: read -> 10 s windows -> labels -> --mapping -> --lead [-> model -> metrics].

By default it runs wcr_77_classes (needs torch, i.e. the DeepECG Docker image; see run_icentia.sh) and
writes metrics. --labels-only stops after the labels (no torch needed). The dataset is only read.
Missing patients and unreadable segments are logged in run_manifest.json and skipped.

    python -m ingestion.run --data <icentia root> --lead 12 --mapping configs/icentia_mapping.csv \
        --patients p00000-p00009 --out <out dir> \
        [--thresholds <thresholds.csv> --device cuda:0] [--labels-only]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

from ingestion import icentia, labels, leads, mapping, windows


def parse_patients(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += [f'p{i:05d}' for i in range(int(a[1:]), int(b[1:]) + 1)]
        else:
            out.append(part)
    return out


def patient_dir(data, patient):
    return os.path.join(data, patient[:3], patient)  # p00012 -> p00/p00012


def process_patient(data, patient, slots, class_map, scorer=None, fs_expected=250, bad_segments=None):
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
            p[keep] = scorer(x)
            probs.append(p)
    if not frames:
        return None, None, 0
    return pd.concat(frames, ignore_index=True), (np.concatenate(probs) if probs else None), n_built


def load_thresholds(path, tasks):
    """{task: threshold}. Task names must match the mapping's tasks; missing ones get a warning."""
    if not path:
        return {}
    df = pd.read_csv(path)
    out = {str(t).strip(): float(v) for t, v in zip(df.task, df.threshold)}
    unknown = sorted(set(out) - set(tasks))
    if unknown:
        raise ValueError(f'{path}: tasks not in the mapping: {unknown} (mapping tasks: {sorted(tasks)})')
    for t in tasks:
        if t not in out:
            print(f'warning: no threshold for {t!r}; sensitivity/specificity not reported', flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--lead', required=True, help="lead name (I, II, ..., V6) or '12' to copy to all")
    p.add_argument('--mapping', required=True, help="CSV: dataset_label,deepecg_class ('A|B' = combined)")
    p.add_argument('--patients', required=True, help='e.g. p00000-p00009 or p00000,p00003')
    p.add_argument('--out', required=True)
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
    slots = leads.parse_lead(a.lead)
    class_map = mapping.load_mapping(a.mapping, labels.DATASET_LABELS)
    thresholds = load_thresholds(a.thresholds, class_map)
    os.makedirs(out, exist_ok=True)

    infer = not a.labels_only
    scorer = None
    if infer:
        from ingestion.infer import WCRScorer, MODEL_NAME
        scorer = WCRScorer(a.device)

    t0, per_patient, missing, bad_segments = time.time(), [], [], []
    for pat in parse_patients(a.patients):
        tp = time.time()
        d = patient_dir(a.data, pat)
        if not os.path.isdir(d) or not icentia.list_segments(d):
            missing.append(pat)
            print(f'{pat}: missing or no readable segments in {d}, skipped', flush=True)
            continue
        df, probs, n_built = process_patient(a.data, pat, slots, class_map, scorer, bad_segments=bad_segments)
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
            json.dump(pooled, f, indent=2, default=float)

    try:
        git = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                      cwd=os.path.dirname(os.path.abspath(__file__)), text=True,
                                      stderr=subprocess.DEVNULL).strip()
    except Exception:
        git = None
    manifest = {
        'mode': 'infer' if infer else 'labels-only', 'args': vars(a),
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
    with open(os.path.join(out, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'done in {manifest["seconds"]} s -> {out}')


if __name__ == '__main__':
    main()
