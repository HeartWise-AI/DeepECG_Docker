"""Window-level metrics per task: AUROC, AUPRC, sensitivity/specificity at fixed thresholds, patient bootstrap CIs."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _point(y, s, thr):
    out = {'n_pos': int((y == 1).sum()), 'n_neg': int((y == 0).sum())}
    both = out['n_pos'] > 0 and out['n_neg'] > 0
    out['auroc'] = roc_auc_score(y, s) if both else np.nan
    out['auprc'] = average_precision_score(y, s) if both else np.nan
    if thr is not None:
        out['sensitivity'] = float((s[y == 1] >= thr).mean()) if out['n_pos'] else np.nan
        out['specificity'] = float((s[y == 0] < thr).mean()) if out['n_neg'] else np.nan
    return out


def task_metrics(y, s, patients, thr=None, n_boot=1000, seed=0):
    """Pooled metrics plus 95% CIs from a bootstrap that resamples whole patients."""
    keep = ~np.isnan(y)
    y, s, patients = y[keep], s[keep], patients[keep]
    res = _point(y, s, thr)
    uniq = np.unique(patients)
    if n_boot and len(uniq) > 1:
        rng = np.random.default_rng(seed)
        groups = _groups(patients)
        boots = {'auroc': [], 'auprc': []}
        for _ in range(n_boot):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(uniq), len(uniq))])
            yb = y[idx]
            if (yb == 1).any() and (yb == 0).any():
                boots['auroc'].append(roc_auc_score(yb, s[idx]))
                boots['auprc'].append(average_precision_score(yb, s[idx]))
        for k, v in boots.items():
            res[f'{k}_ci'] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else None
    return res


def _groups(patients):
    """Window indices per patient, in np.unique order, without one full pass per patient."""
    _, inv = np.unique(patients, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    return np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)


def per_patient(y, s, patients, thr=None):
    keep = np.flatnonzero(~np.isnan(y))
    rows = []
    for idx in _groups(patients[keep]):
        m = keep[idx]
        rows.append({'patient': patients[m[0]], **_point(y[m], s[m], thr)})
    return pd.DataFrame(rows)
