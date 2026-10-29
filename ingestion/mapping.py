"""--mapping: dataset label -> DeepECG 77-class name, and the ground truth it implies."""
import os

import numpy as np
import pandas as pd

from utils.constants import ECG_PATTERNS

COMBINE = '|'  # 'Afib|Atrial flutter' = score 1 - (1 - p_Afib)(1 - p_Atrial flutter)


def split_task(task):
    return [c.strip() for c in task.split(COMBINE)]


def load_mapping(path, dataset_labels, classes=ECG_PATTERNS):
    """Return {task: [dataset_label, ...]}, where a task is one DeepECG class or several joined by '|'.

    Errors on unknown labels or classes."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'--mapping file not found: {path}')
    df = pd.read_csv(path, dtype=str).fillna('')
    if list(df.columns) != ['dataset_label', 'deepecg_class']:
        raise ValueError(f'{path}: columns must be dataset_label,deepecg_class, got {list(df.columns)}')
    df = df.apply(lambda c: c.str.strip())
    bad_cls = sorted({c for t in df.deepecg_class for c in split_task(t)} - set(classes))
    if bad_cls:
        raise ValueError(f'{path}: not among the 77 DeepECG classes: {bad_cls}')
    bad_lbl = sorted(set(df.dataset_label) - set(dataset_labels))
    if bad_lbl:
        raise ValueError(f'{path}: unknown dataset labels {bad_lbl}; known: {list(dataset_labels)}')
    out = {}
    for lbl, cls in zip(df.dataset_label, df.deepecg_class):
        out.setdefault(cls, []).append(lbl)
    return out


def ground_truth(labels, mapping):
    """Per DeepECG class: 1 if any mapped label is 1, 0 if all are 0, NaN otherwise."""
    gt = {}
    for cls, lbls in mapping.items():
        m = labels[lbls].to_numpy(dtype=float)
        any_pos = np.nan_to_num(m, nan=0.0).max(axis=1) == 1
        all_neg = (m == 0).all(axis=1)
        gt[cls] = np.where(any_pos, 1.0, np.where(all_neg, 0.0, np.nan))
    return pd.DataFrame(gt, index=labels.index)


def task_scores(probs, task, classes=ECG_PATTERNS):
    """probs [n, 77] in ECG_PATTERNS order -> [n] score; combined classes use 1 - prod(1 - p)."""
    idx = [classes.index(c) for c in split_task(task)]
    p = probs[:, idx].astype(np.float64)
    return 1.0 - np.prod(1.0 - p, axis=1)
