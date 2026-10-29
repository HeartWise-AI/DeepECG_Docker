"""Validate a DeepECG 77-class model on Icentia-11k (single-lead Holter) against its beat and rhythm annotations.

Per patient: segments -> 10 s windows -> 12-lead input (--single_lead) -> the pipeline's preprocessing -> model
-> 77 probabilities; window labels from the annotations -> --mapping -> per-task AUROC, AUPRC, sensitivity and
specificity, with patient-bootstrap CIs. BERT is not used. The dataset is only read. Missing patients and
unreadable segments are skipped and listed in run_manifest.json.

    python -m ingestion.icentia11k.run --data <icentia root> --patients p00000-p00009 --out <new folder> \
        [--single_lead all] [--thresholds <thresholds.json>] [--model wcr] [--device cuda:0]
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from ingestion import mapping, metrics
from ingestion.icentia11k import labels, reader
from utils.analysis_pipeline import PREPROCESSING_BATCH_SIZE, AnalysisPipeline, load_model, predict_77_classes
from utils.constants import MODEL_MAPPING, SIGNAL, load_thresholds
from utils.credentials import read_api_key
from utils.files_handler import LEAD_ORDER, expand_single_lead, save_json, single_lead_slots

PATIENT_ID = re.compile(r'p\d{5}')
DEFAULT_MAPPING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mapping.csv')


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


def task_thresholds(path, tasks):
    """{task: threshold} from a JSON file in the thresholds/ format; a task without one gets no sensitivity/specificity."""
    if not path:
        return {}
    entries = load_thresholds(path)
    out = {t: entries[t]['threshold'] for t in tasks if entries.get(t, {}).get('threshold') is not None}
    bad = sorted(t for t, v in out.items() if not 0 <= v <= 1)
    if bad:
        raise ValueError(f'{path}: thresholds must be between 0 and 1; bad tasks: {bad}')
    for t in tasks:
        if t not in out:
            print(f'warning: no threshold for {t!r}; sensitivity/specificity not reported', flush=True)
    return out


@torch.no_grad()
def score_windows(windows_mv, slots, model, model_name, device, batch_size, n_workers):
    """windows [n, 2500] mV -> probabilities [n, 77] float32 in ECG_PATTERNS order.

    The windows get the same preprocessing as the main pipeline, in batches of PREPROCESSING_BATCH_SIZE.
    """
    out = np.empty((len(windows_mv), 77), dtype=np.float32)
    for i in range(0, len(windows_mv), PREPROCESSING_BATCH_SIZE):
        chunk = expand_single_lead(windows_mv[i:i + PREPROCESSING_BATCH_SIZE], slots)  # [m, 2500, 12]
        df = pd.DataFrame({'ecg_path': np.arange(len(chunk)), 'ecg_signal': list(chunk)})
        df = AnalysisPipeline.preprocess_signals(df, n_workers, ref_lead=slots[0])
        x = np.stack(df.ecg_signal.to_list()).astype(np.float32).transpose(0, 2, 1)  # [m, 12, 2500], as in the dataloader
        for j in range(0, len(x), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(x[j:j + batch_size])).to(device)
            out[i + j:i + j + len(xb)] = predict_77_classes(model, model_name, xb).float().cpu().numpy()
    return out


def process_patient(data, patient, class_map, score, bad_segments):
    """Window table and probabilities [n_windows, 77] (NaN for skipped windows) for one patient, or (None, None).

    Segments that can't be read are appended to bad_segments and skipped.
    """
    d = reader.patient_dir(data, patient)
    frames, kept = [], []
    for seg in reader.list_segments(d):
        try:
            sig, raw, ann = reader.read_segment(d, seg)
        except (OSError, ValueError, IndexError) as e:
            bad_segments.append({'segment': seg, 'error': f'{type(e).__name__}: {e}'})
            print(f'  skipping {seg}: {type(e).__name__}: {e}', flush=True)
            continue
        w, starts, reason, flat = reader.cut_windows(sig, raw)
        lab = labels.label_windows(ann, starts, reader.WINDOW_SAMPLES, reader.FS)
        gt = mapping.ground_truth(lab, class_map).add_prefix('gt::')
        meta = pd.DataFrame({'patient': patient, 'segment': seg, 'window_idx': np.arange(starts.size),
                             'start_sample': starts, 'has_atr': ann is not None,
                             'skip_reason': reason, 'flat': flat})
        frames.append(pd.concat([meta, lab, gt], axis=1))
        kept.append(w[reason == ''])
    if not frames:
        return None, None
    df = pd.concat(frames, ignore_index=True)
    probs = np.full((len(df), 77), np.nan, dtype=np.float32)
    probs[(df.skip_reason == '').to_numpy()] = score(np.concatenate(kept))
    return df, probs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='Icentia-11k root (the folder with p00/ ... p10/)')
    p.add_argument('--patients', required=True, help='e.g. p00000-p00009 or p00000,p00003')
    p.add_argument('--out', required=True, help='output folder; must be new or empty')
    p.add_argument('--single_lead', default='all',
                   help="'all' copies the lead to all 12 leads; a lead name (I ... V6) puts it in that lead only")
    p.add_argument('--mapping', default=DEFAULT_MAPPING, help="CSV: dataset_label,deepecg_class ('A|B' = combined)")
    p.add_argument('--thresholds', help='JSON in the thresholds/ format, keyed by mapping task (sensitivity/specificity)')
    p.add_argument('--model', choices=['wcr', 'efficientnet'], default='wcr', help='DeepECG-SSL (wcr) or DeepECG-SL')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--preprocessing_n_workers', type=int, default=16)
    p.add_argument('--bootstrap', type=int, default=1000, help='patient-bootstrap resamples for CIs (0 = none)')
    p.add_argument('--hugging_face_api_key_path', default='api_key.json')
    a = p.parse_args(argv)

    out, data = os.path.realpath(a.out), os.path.realpath(a.data)
    if out == data or out.startswith(data + os.sep):
        raise SystemExit('--out must not be inside --data (the dataset is read-only)')
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        raise SystemExit(f'--out {out} must be a new or empty folder')
    try:
        patient_ids = parse_patients(a.patients)
        slots = single_lead_slots(a.single_lead)
        if slots is None:
            raise ValueError("--single_lead must be 'all' or a lead name")
        class_map = mapping.load_mapping(a.mapping, labels.DATASET_LABELS)
        thresholds = task_thresholds(a.thresholds, class_map)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e))
    if a.batch_size < 1:
        raise SystemExit('--batch_size must be at least 1')

    model_name = MODEL_MAPPING['ecg_machine_diagnosis'][SIGNAL][a.model]
    model = load_model(model_name, a.device, read_api_key(a.hugging_face_api_key_path)['HUGGING_FACE_API_KEY'])
    os.makedirs(out, exist_ok=True)

    def score(windows_mv):
        return score_windows(windows_mv, slots, model, model_name, a.device, a.batch_size, a.preprocessing_n_workers)

    t0, frames, per_patient, missing, bad_segments = time.time(), [], [], [], []
    for pat in patient_ids:
        tp = time.time()
        d = reader.patient_dir(a.data, pat)
        df, probs = process_patient(a.data, pat, class_map, score, bad_segments) if os.path.isdir(d) else (None, None)
        if df is None:
            missing.append(pat)
            print(f'{pat}: missing or no readable segments in {d}, skipped', flush=True)
            continue
        df.to_parquet(os.path.join(out, f'{pat}_windows.parquet'), index=False)
        np.save(os.path.join(out, f'{pat}_probs.npy'), probs)  # float32, ECG_PATTERNS order
        for task in class_map:
            df['score::' + task] = mapping.task_scores(probs, task)
        frames.append(df)
        per_patient.append({'patient': pat, 'segments': df.segment.nunique(), 'windows': len(df),
                            'scored': int((df.skip_reason == '').sum()), 'seconds': round(time.time() - tp, 1)})
        print(f'{pat}: {per_patient[-1]["segments"]} segments, {len(df)} windows, {per_patient[-1]["scored"]} scored '
              f'({per_patient[-1]["seconds"]} s)', flush=True)
    if not frames:
        raise SystemExit(f'no patients could be read from {a.data} (missing: {missing})')

    all_df = pd.concat(frames, ignore_index=True)
    valid = (all_df.skip_reason == '').to_numpy()
    summary = []
    for col in labels.DATASET_LABELS + ['gt::' + t for t in class_map]:
        v = all_df.loc[valid, col]
        summary.append({'label': col, 'positive': int((v == 1).sum()), 'negative': int((v == 0).sum()),
                        'unknown': int(v.isna().sum())})
    pd.DataFrame(summary).to_csv(os.path.join(out, 'label_summary.csv'), index=False)

    patients = all_df.patient.to_numpy()
    pooled, per_pat = [], []
    for task in class_map:
        y = all_df['gt::' + task].to_numpy(dtype=float)
        y[~valid] = np.nan
        s = all_df['score::' + task].to_numpy()
        thr = thresholds.get(task)
        pooled.append({'task': task, 'threshold': thr, **metrics.task_metrics(y, s, patients, thr, n_boot=a.bootstrap)})
        per_pat.append(metrics.per_patient(y, s, patients, thr).assign(task=task))
    pd.DataFrame(pooled).to_csv(os.path.join(out, 'metrics.csv'), index=False)
    pd.concat(per_pat).to_csv(os.path.join(out, 'metrics_per_patient.csv'), index=False)

    save_json({
        'args': vars(a), 'model': model_name, 'model_mode': 'eval + no_grad',
        'lead_slots': [LEAD_ORDER[s] for s in slots],
        'preprocessing': f'main pipeline (PTB-XL power scaling, line-noise cleaning), '
                         f'batches of {PREPROCESSING_BATCH_SIZE} windows per patient',
        'mapping': class_map, 'thresholds': thresholds, 'label_rules': labels.RULES,
        'window_samples': reader.WINDOW_SAMPLES,
        'skipped_windows': {k: int(v) for k, v in all_df.skip_reason[~valid].value_counts().items()},
        'flat_windows': int(all_df.flat.sum()),
        'segments_without_atr': sorted(all_df.loc[~all_df.has_atr, 'segment'].unique().tolist()),
        'missing_patients': missing, 'bad_segments': bad_segments,
        'per_patient': per_patient, 'seconds': round(time.time() - t0, 1),
    }, os.path.join(out, 'run_manifest.json'))
    print(f'done in {round(time.time() - t0, 1)} s -> {out}')


if __name__ == '__main__':
    main()
