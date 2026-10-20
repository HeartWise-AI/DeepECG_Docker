"""Window labels for Icentia-11k, in Icentia's terms (before --mapping renames them).

Each label is 1 (positive window), 0 (negative window) or NaN (neither -> excluded from scoring).
Rules follow the task table of the reference Icentia-11k validation. Where the table is
silent (how heart rate is measured, what "no AF" means for unlabelled time), the choice is marked
in RULES and recorded in every run manifest.
"""
import numpy as np
import pandas as pd

from ingestion.icentia import BEAT_CODES, RHYTHM_CODE, Annotations

DATASET_LABELS = ['AFIB', 'AFL', 'N_RHYTHM', 'PVC', 'PAC', 'BRADY', 'TACHY', 'BIGEMINY']
RHYTHM_LABELS = {'(AFIB': 'AFIB', '(AFL': 'AFL', '(N': 'N_RHYTHM'}

RULES = {
    'source': 'task table of the reference Icentia-11k validation (batch 1)',
    # AF/flutter: + whole window AF or flutter; - whole window normal rhythm.
    # Sinus:      + whole window normal rhythm; - whole window AF or flutter.
    # "Clean sinus" (interpretation): whole window in normal rhythm, no PVC (V) and no PAC (S).
    #   Unreviewed beats (Q) are allowed unless "every beat reviewed" is also required.
    # Bradycardia: + clean sinus, rate < 60;  - clean sinus, rate 60-100.
    # Tachycardia: + clean sinus, rate > 100; - clean sinus, rate 60-100.
    'brady_bpm_below': 60.0,
    'tachy_bpm_above': 100.0,
    # PVC: + at least one V, no AF;           - clean sinus, every beat reviewed (no Q).
    # PAC: + at least one S, no V, no AF;     - clean sinus, every beat reviewed (no Q).
    # "No AF": no part of the window overlaps an AF or flutter interval, so windows in unlabelled
    #   time can be positive. This reproduces the reference PVC/PAC counts; requiring normal rhythm doesn't.
    # Bigeminy: + 6 or more consecutive beats alternating N and V; - clean sinus.
    'bigeminy_min_beats': 6,
    # Heart rate (interpretation): 60 / median gap between consecutive beats (any type),
    # needs this many beats and no longer gap, else the rate is unknown.
    'rate_min_beats': 4,
    'rate_max_rr_s': 3.0,
    # Whole-window rhythm allows the interval to start/end this many samples inside the window.
    # 2 reproduces the reference per-patient counts exactly (reason unknown: a boundary detail of that code).
    'rhythm_tolerance_samples': 2,
}


def rhythm_intervals(ann):
    """List of (start, end, label) from '(X' ... ')' markers; '(N)' zero-length markers are ignored.

    Annotations must be in time order. A new '(X' while an interval is open closes it there, and an
    interval still open at the end of the segment runs to the end (neither occurs in the Icentia
    segments checked so far).
    """
    out, open_start, open_lbl = [], None, None
    for t, code, aux in zip(ann.samples, ann.codes, ann.aux):
        if code != RHYTHM_CODE:
            continue
        if aux in RHYTHM_LABELS:
            if open_start is not None:
                out.append((open_start, int(t), open_lbl))
            open_start, open_lbl = int(t), RHYTHM_LABELS[aux]
        elif aux == ')' and open_start is not None:
            out.append((open_start, int(t), open_lbl))
            open_start = None
    if open_start is not None:
        out.append((open_start, np.iinfo(np.int64).max, open_lbl))
    return out


def _sorted(ann):
    """Annotations in time order (the .atr format allows negative steps)."""
    order = np.argsort(ann.samples, kind='stable')
    if (order == np.arange(order.size)).all():
        return ann
    return Annotations(ann.samples[order], ann.codes[order], [ann.aux[i] for i in order])


def longest_alternating_nv(seq):
    """Length of the longest run of beats alternating N and V (either may come first)."""
    best = run = 0
    prev = None
    for b in seq:
        if b in 'NV' and prev is not None and b != prev and prev in 'NV':
            run += 1
        else:
            run = 1 if b in 'NV' else 0
        best = max(best, run)
        prev = b
    return best


def label_windows(ann, starts, size, fs, rules=RULES):
    """Return a DataFrame (one row per window) of beat counts, heart rate, rhythm and DATASET_LABELS."""
    n = starts.size
    rows = {k: np.full(n, np.nan) for k in DATASET_LABELS}
    counts = {f'n_{b}': np.zeros(n, dtype=np.int32) for b in 'NSVQ'}
    hr = np.full(n, np.nan)
    rhythm = np.full(n, '', dtype=object)  # whole-window rhythm: 'N_RHYTHM', 'AFIB', 'AFL' or ''

    if ann is None:  # no .atr: every label, beat count and clean-sinus flag unknown (NA, not 0 / False)
        return _frame(counts, np.full(n, np.nan), hr, rhythm, np.zeros(n, dtype=bool), rows, known=False)

    ann = _sorted(ann)
    # Rhythm: whole-window coverage, and any overlap with AF or flutter.
    touches_af = np.zeros(n, dtype=bool)
    tol = rules['rhythm_tolerance_samples']
    for s, e, lbl in rhythm_intervals(ann):
        rhythm[(starts >= s - tol) & (starts + size <= e + tol)] = lbl
        if lbl in ('AFIB', 'AFL'):
            touches_af |= (starts < e) & (starts + size > s)
    whole_n = rhythm == 'N_RHYTHM'
    whole_af = (rhythm == 'AFIB') | (rhythm == 'AFL')
    rows['AFIB'][whole_n] = 0.0
    rows['AFIB'][rhythm == 'AFIB'] = 1.0
    rows['AFL'][whole_n] = 0.0
    rows['AFL'][rhythm == 'AFL'] = 1.0
    rows['N_RHYTHM'][whole_af] = 0.0
    rows['N_RHYTHM'][whole_n] = 1.0

    # Beats
    is_beat = np.isin(ann.codes, list(BEAT_CODES))
    bt, bc = ann.samples[is_beat], ann.codes[is_beat]
    lo = np.searchsorted(bt, starts, side='left')
    hi = np.searchsorted(bt, starts + size, side='left')
    clean = np.zeros(n, dtype=bool)
    for i in range(n):
        seq = [BEAT_CODES[c] for c in bc[lo[i]:hi[i]]]
        for b in 'NSVQ':
            counts[f'n_{b}'][i] = seq.count(b)
        has_v, has_s, has_q = 'V' in seq, 'S' in seq, 'Q' in seq

        if len(seq) >= rules['rate_min_beats']:
            rr = np.diff(bt[lo[i]:hi[i]]) / fs
            if rr.max() <= rules['rate_max_rr_s'] and rr.min() > 0:  # a 0 gap = duplicate beat marks
                hr[i] = 60.0 / np.median(rr)

        clean[i] = whole_n[i] and len(seq) > 0 and not has_v and not has_s
        reviewed_clean = clean[i] and not has_q

        if has_v and not touches_af[i]:
            rows['PVC'][i] = 1.0
        elif reviewed_clean:
            rows['PVC'][i] = 0.0
        if has_s and not has_v and not touches_af[i]:
            rows['PAC'][i] = 1.0
        elif reviewed_clean:
            rows['PAC'][i] = 0.0
        if longest_alternating_nv(seq) >= rules['bigeminy_min_beats']:
            rows['BIGEMINY'][i] = 1.0
        elif clean[i]:
            rows['BIGEMINY'][i] = 0.0

        if clean[i] and not np.isnan(hr[i]):
            if hr[i] < rules['brady_bpm_below']:
                rows['BRADY'][i] = 1.0
            elif hr[i] > rules['tachy_bpm_above']:
                rows['TACHY'][i] = 1.0
            else:  # 60-100 bpm: negative for both
                rows['BRADY'][i] = 0.0
                rows['TACHY'][i] = 0.0

    total = sum(counts.values())
    q_frac = np.where(total > 0, counts['n_Q'] / np.maximum(total, 1), np.nan)
    return _frame(counts, q_frac, hr, rhythm, clean, rows, known=True)


def _frame(counts, q_frac, hr, rhythm, clean, rows, known):
    """Window table with one set of column types for every segment: beat counts as nullable Int32 and
    clean_sinus as nullable boolean, all NA when the segment has no annotations (known=False)."""
    mask = np.full(len(hr), not known)
    typed = {k: pd.arrays.IntegerArray(v.astype(np.int32), mask.copy()) for k, v in counts.items()}
    return pd.DataFrame({**typed, 'q_fraction': q_frac, 'hr_bpm': hr, 'rhythm': rhythm,
                         'clean_sinus': pd.arrays.BooleanArray(clean.astype(bool), mask.copy()), **rows})
