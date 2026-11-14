"""ECG trait measurements from the raw waveforms, with NeuroKit2 (step 1).

Figure 2f's claim rests on a positive control: each trait PGS must predict ITS OWN measured ECG
trait before it is allowed to predict anything about ancestry. In the discovery cohorts those
traits come from the recording device's own measurement panel. Lifelines does not return one, so
they are measured here from the same waveforms the ancestry model is scored on -- one delineation
per ECG, fiducials taken on lead II and read out in all twelve leads, every per-beat quantity
aggregated by median.

This is a DIFFERENT instrument from the device panel, and the comparison with discovery has to be
read with that in mind: NeuroKit2 and a GE/Philips measurement module disagree on PR and on wave
amplitudes by more than they disagree on QRS or QT. The control is therefore judged on sign and
on whether the effect clears t = 3, not on matching the discovery percentage.

Lead order assumed: I, II, III, aVR, aVL, aVF, V1..V6 -- the container's own order.

    python 01_measure_neurokit.py \
        --input /data/prepared/all_genotyped --ids /data/prepared/all_genotyped/ids.csv \
        --output /data/local/measurements_local.csv --jobs 16

Needs neurokit2 (`pip install neurokit2`); it does NOT need the container or a GPU, and it reads
only X.npy, so it can run on any machine that already holds the prepared directory.

Writes participant_id + 60 columns. THIS FILE CARRIES IDS and stays on the analysis machine;
02_validate.py consumes it there and emits aggregates only.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
L_I, L_II, L_III = 0, 1, 2
WAVES = {'Pamp': 'ECG_P_Peaks', 'Ramp': 'ECG_R_Peaks',
         'Samp': 'ECG_S_Peaks', 'Tamp': 'ECG_T_Peaks'}
FID = ['ECG_P_Onsets', 'ECG_P_Offsets', 'ECG_R_Onsets', 'ECG_R_Offsets',
       'ECG_T_Onsets', 'ECG_T_Offsets', 'ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks',
       'ECG_T_Peaks']
COLS = (['n_rpeaks', 'HR_bpm', 'RR_ms', 'PR_ms', 'QRS_ms', 'QT_ms', 'JT_ms',
         'QTc_bazett_ms', 'QRSTangleF_deg']
        + [f'{w}_{L}' for w in WAVES for L in LEADS])


def _med(a):
    a = np.asarray([v for v in a if np.isfinite(v)], dtype=float)
    return float(np.median(a)) if a.size else np.nan


def _pair(anchor, marks, lo, hi, fs):
    """Median of (mark - anchor) in ms, over beats where the two belong together.

    Each anchor is matched to the mark CLOSEST TO IT inside a physiological window; a mark that
    lands outside the window is a mis-detection on that beat, not a long interval, so it is
    dropped rather than pulled into the median. The window may be negative, which is how the P
    onset -- the one fiducial that precedes the anchor -- is matched.
    """
    if anchor.size == 0 or marks.size == 0:
        return np.nan
    out = []
    for a in anchor:
        d = (marks - a) * 1000.0 / fs
        d = d[(d >= lo) & (d <= hi)]
        if d.size:
            out.append(float(d[np.argmin(np.abs(d))]))
    return _med(out)


def _frontal(x_i, y_avf):
    """Frontal-plane axis in degrees from a lead-I and an aVF value -- the orthogonal pair the
    axis is actually defined on."""
    return float(np.degrees(np.arctan2(y_avf, x_i)))


def _extremum(B, s, e, j):
    """The value of largest magnitude in B[s:e, j], sign kept."""
    if not (0 <= s < e <= len(B)):
        return np.nan
    w = B[s:e, j]
    return float(w[int(np.argmax(np.abs(w)))])


def _angle_between(a, b):
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    d = abs(a - b) % 360.0
    return float(min(d, 360.0 - d))


def median_beat(ecg, rp, fs, pre_ms=450.0, post_ms=600.0, up=4):
    """Average beat, R-aligned, baseline-corrected, upsampled.

    Two reasons this is the right surface to measure on. Noise: one beat at 250 Hz is a poor
    place to look for a QRS onset, and the median over ten beats is not. Timing: the device
    panel this has to agree with measures a representative beat, not beat i. The upsampling is
    interpolation, not information -- it buys sub-sample fiducial resolution, and at 250 Hz the
    4 ms quantisation is otherwise the single largest term in the disagreement.

    The 450 ms before R is not generous: the P search needs 320 ms of it and the baseline
    estimate the first 40 ms.
    """
    from scipy.signal import resample_poly
    pre, post = int(round(pre_ms * fs / 1000)), int(round(post_ms * fs / 1000))
    seg = [ecg[r - pre:r + post] for r in rp if r - pre >= 0 and r + post <= len(ecg)]
    if len(seg) < 2:
        return None, fs * up, pre * up
    B = resample_poly(np.median(np.stack(seg), axis=0), up, 1, axis=0)
    B = B - np.median(B[:int(0.04 * fs * up)], axis=0, keepdims=True)
    return B, fs * up, pre * up


def _vm(B):
    """Vector magnitude over the eight independent leads -- the global waveform on which a wave
    onset and offset are defined, and what a device means by QRS duration or QT. The augmented
    leads are linear combinations of I and II and would only re-weight what is already there."""
    return np.sqrt((B[:, [0, 1, 6, 7, 8, 9, 10, 11]] ** 2).sum(axis=1))


def _qrs_bounds(v, r0, base, frac=0.15, win_on=130, win_off=170):
    """QRS onset and offset: where the vector magnitude falls to `frac` of its peak, searched
    outward from R and BOUNDED.

    The bound is the point. An unbounded search walks the offset into the T wave whenever the T
    is tall, which turns a 90 ms QRS into a 400 ms one; against the device those outliers, not
    the typical beat, were most of the disagreement. If the threshold is never crossed inside
    the window the local minimum of the window is used instead, which is the J point on any beat
    whose ST segment does not return to baseline.
    """
    a0, a1 = max(0, r0 - win_on), min(len(v), r0 + win_off)
    thr = base + frac * (float(v[a0:a1].max()) - base)
    on = r0
    while on > a0 and v[on] >= thr:
        on -= 1
    if v[on] >= thr:
        on = a0 + int(np.argmin(v[a0:r0]))
    off = r0
    while off < a1 - 1 and v[off] >= thr:
        off += 1
    if v[off] >= thr:
        off = r0 + int(np.argmin(v[r0:a1]))
    return (on, off) if off > on else (None, None)


def _t_offset(v, t_pk, base, fs, lim_ms=250):
    """T offset by the tangent method: the steepest downslope after the T peak, extended to
    baseline. This is what measurement modules do, and it is far steadier than asking a wavelet
    where a low, broad wave ends."""
    lim = int(lim_ms * fs / 1000)
    e = min(len(v), t_pk + lim)
    if e - t_pk < 6:
        return None
    d = np.diff(v[t_pk:e])
    k = int(np.argmin(d))
    if d[k] >= -1e-9:
        return None
    x = t_pk + k - (v[t_pk + k] - base) / d[k]
    return int(round(x)) if t_pk < x < t_pk + lim else None


def measure_one(ecg, fs=250):
    """ecg: (T, 12) in mV. Returns a dict over COLS; NaN wherever a step failed.

    Everything is measured on the MEDIAN BEAT. Wave boundaries come from the vector magnitude,
    as a device defines them; the P wave from the largest of leads I, II and V1, where it is
    visible at all. Agreement with the device panel, measured on 2,400 of our own ECGs that have
    both, is in ref/neurokit_vs_device.csv -- read it before reading any control built on this.

    The T wave is located TWICE, on purpose. Its amplitude is taken from a peak searched from
    150 ms after R, which is the discovery window and keeps the ST shoulder out of it; the
    offset is taken from a peak searched from the J point, because a tangent dropped from a peak
    found 150 ms out lands too late on a beat with a long QRS. Tuned against the device, those
    two choices are worth r = 0.80 and r = 0.59 respectively, against 0.71 and 0.53 for either
    one used for both.
    """
    import neurokit2 as nk
    rec = {c: np.nan for c in COLS}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sig = nk.ecg_clean(ecg[:, L_II], sampling_rate=fs)
            _, info = nk.ecg_peaks(sig, sampling_rate=fs)
            rp = np.asarray(info.get('ECG_R_Peaks', []), dtype=int)
    except Exception:
        return rec
    rec['n_rpeaks'] = int(rp.size)
    if rp.size < 3:
        return rec
    rr = np.diff(rp) * 1000.0 / fs
    rr = rr[(rr > 250) & (rr < 2500)]
    rec['RR_ms'] = _med(rr)
    if np.isfinite(rec['RR_ms']) and rec['RR_ms'] > 0:
        rec['HR_bpm'] = 60000.0 / rec['RR_ms']

    B, f2, r0 = median_beat(ecg, rp, fs)
    if B is None:
        return rec
    v = _vm(B)
    base = float(np.median(v[:int(0.04 * f2)]))
    ms = lambda n: n * 1000.0 / f2
    sm = lambda t: int(round(t * f2 / 1000.0))

    on, off = _qrs_bounds(v, r0, base)
    if on is None:
        return rec
    q = ms(off - on)
    if 50 <= q <= 220:
        rec['QRS_ms'] = q
    for j, L in enumerate(LEADS):
        seg = B[on:off, j]
        rec[f'Ramp_{L}'] = float(seg.max())
        rec[f'Samp_{L}'] = float(-seg.min())      # a DEPTH, positive, as the device reports it

    # the T search never runs past 55% of the RR interval, or it finds the next P wave
    lim = sm(min(450.0, 0.55 * rec['RR_ms'] if np.isfinite(rec['RR_ms']) else 450.0))
    s0, s1 = off + sm(30), min(len(v), off + sm(30) + lim)
    if s1 - s0 > sm(20):
        t_off_pk = s0 + int(np.argmax(v[s0:s1]))
        t_off = _t_offset(v, t_off_pk, base, f2)
        if t_off is not None:
            qt = ms(t_off - on)
            if 250 <= qt <= 600:
                rec['QT_ms'] = qt
                if np.isfinite(rec['QRS_ms']):
                    rec['JT_ms'] = qt - rec['QRS_ms']
                if np.isfinite(rec['RR_ms']) and rec['RR_ms'] > 0:
                    rec['QTc_bazett_ms'] = qt / np.sqrt(rec['RR_ms'] / 1000.0)
    a0, a1 = r0 + sm(150), min(len(v), r0 + sm(150) + lim)
    t_amp_pk = a0 + int(np.argmax(v[a0:a1])) if a1 - a0 > sm(20) else None
    if t_amp_pk is not None:
        for j, L in enumerate(LEADS):
            rec[f'Tamp_{L}'] = float(B[t_amp_pk, j])

    # P wave: the 40-320 ms before the QRS onset, on whichever of I, II and V1 shows it best.
    # The vector magnitude is the wrong surface here -- the P wave is small enough that at that
    # scale the VM is mostly noise, and it measured a third as well in testing.
    p0, p1 = on - sm(320), on - sm(40)
    if p0 > 0 and p1 - p0 > sm(20):
        w = np.abs(B[p0:p1][:, [0, 1, 6]]).max(axis=1)
        k = int(np.argmax(w))
        thr = 0.20 * float(w[k])
        pon = k
        while pon > 0 and w[pon] > thr:
            pon -= 1
        pr = ms(on - (p0 + pon))
        if 80 <= pr <= 300:
            rec['PR_ms'] = pr
        for j, L in enumerate(LEADS):
            rec[f'Pamp_{L}'] = float(B[p0 + k, j])

    # frontal QRS-T angle, from the PEAK amplitudes in leads I and aVF -- the QRS peak for one
    # axis, the value at the T peak for the other. Net area over each window, the textbook
    # construction, measured half as well against the device panel (r 0.40 against 0.68): the
    # area of a biphasic QRS is near zero and its sign is then set by noise.
    if t_amp_pk is not None:
        rec['QRSTangleF_deg'] = _angle_between(
            _frontal(_extremum(B, on, off, 0), _extremum(B, on, off, 5)),
            _frontal(float(B[t_amp_pk, 0]), float(B[t_amp_pk, 5])))
    return rec


def mad_clean(M, cols=None, k=5.0):
    """Drop values more than k robust SDs from the cohort median, per column.

    Spearman and Pearson against the device disagree sharply before this step (QT 0.70 against
    0.46), which is the signature of a handful of fiducials landing on the wrong wave rather
    than of a biased measurement. One bad beat should cost its own row, not the correlation. The
    rule is applied per column on the whole cohort, after measurement and before anything is
    regressed, and the count it removes is printed.
    """
    cols = cols or [c for c in M.columns if c not in ('participant_id', 'n_rpeaks')]
    n = 0
    for c in cols:
        x = M[c].astype(float)
        med = x.median()
        mad = (x - med).abs().median() * 1.4826
        if not np.isfinite(mad) or mad <= 0:
            continue
        bad = (x - med).abs() > k * mad
        n += int(bad.sum())
        M.loc[bad, c] = np.nan
    return M, n


def _chunk(args):
    X, fs, i0 = args
    return i0, [measure_one(X[i], fs) for i in range(len(X))]


def participant_ids(path):
    """Read the row-aligned identifier column without dropping leading zeros."""
    return pd.read_csv(path, dtype=str).iloc[:, 0].astype(str).values


def merge_inputs(patterns):
    """Expand quoted scheduler globs and refuse a merge with a missing shard pattern."""
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            sys.exit(f'--merge pattern matched no files: {pattern}')
        files.extend(matches)
    # A file named explicitly and through a glob must not be counted twice.
    return list(dict.fromkeys(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='the prepared directory holding X.npy')
    ap.add_argument('--ids', required=True, help='one column, same row order as X.npy')
    ap.add_argument('--output', required=True)
    ap.add_argument('--fs', type=int, default=250)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--units', default='auto', choices=['auto', 'mV', 'uV'])
    ap.add_argument('--limit', type=int, default=0, help='measure only the first N ECGs (a dry run)')
    ap.add_argument('--shard', default=None, metavar='I/N',
                    help='measure only slice I of N (contiguous rows) and write the RAW per-ECG '
                         'measurements, so the cohort can be split over N jobs. The outlier rule '
                         'and the per-participant median are deliberately NOT applied here: both '
                         'are whole-cohort operations and are done by --merge.')
    ap.add_argument('--merge', nargs='+', default=None,
                    help='combine --shard outputs: the same global outlier rule and the same '
                         'per-participant median the one-process path applies, so the result is '
                         'identical to running unsharded.')
    a = ap.parse_args()

    if a.merge:
        parts = []
        files = merge_inputs(a.merge)
        for f in files:
            d = pd.read_csv(f, dtype={'participant_id': str})
            parts.append(d)
            print(f'  {Path(f).name}: {len(d):,} ECGs')
        M = pd.concat(parts, ignore_index=True)
        if M.row.duplicated().any():
            sys.exit(f'{int(M.row.duplicated().sum())} ECG rows appear in more than one shard; '
                     'the shards overlap and the median would be weighted wrongly')
        M = M.sort_values('row').reset_index(drop=True)
        pid = M.pop('participant_id'); M.pop('row')
        M, n_drop = mad_clean(M)
        print(f'robust outlier rule removed {n_drop:,} individual values '
              f'({100 * n_drop / max(len(M) * (len(COLS) - 1), 1):.2f}% of all measurements)')
        M.insert(0, 'participant_id', pid.values)
        M = M.groupby('participant_id', as_index=False).median(numeric_only=True)
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        M.to_csv(a.output, index=False)
        print(f'merged {len(files)} shards -> {len(M):,} participants; wrote {a.output}')
        return

    try:
        import neurokit2                                                   # noqa: F401
    except ImportError:
        sys.exit('neurokit2 is not installed in this interpreter: pip install neurokit2')

    X = np.load(Path(a.input) / 'X.npy', mmap_mode='r')
    if X.shape[1] == 12 and X.shape[2] != 12:
        X = np.asarray(X).transpose(0, 2, 1)
    ids = participant_ids(a.ids)
    if len(ids) != len(X):
        sys.exit(f'--ids has {len(ids)} rows but X.npy has {len(X)}; they must be row-aligned')
    if a.limit:
        X, ids = X[:a.limit], ids[:a.limit]
    # Units are decided from the WHOLE array, before any slicing: a shard that looked at its own
    # first 200 rows could call them mV while its neighbour called them uV, and the two halves of
    # the cohort would then differ by a factor of a thousand in every amplitude.
    amp_all = float(np.nanpercentile(np.abs(np.asarray(X[:min(len(X), 200)])), 99))
    row0 = 0
    if a.shard:
        i, n = (int(v) for v in a.shard.split('/'))
        if not 1 <= i <= n:
            sys.exit(f'--shard {a.shard}: need 1 <= I <= N')
        lo, hi = len(X) * (i - 1) // n, len(X) * i // n
        X, ids, row0 = X[lo:hi], ids[lo:hi], lo
        print(f'--shard {a.shard}: rows {lo:,}-{hi - 1:,} ({hi - lo:,} ECGs)')

    # a microvolt array measured as millivolts gives amplitudes a thousand times too large and
    # intervals that are perfectly fine, so the error survives every interval check
    amp = amp_all
    units = a.units
    if units == 'auto':
        units = 'uV' if amp > 50 else 'mV'
    print(f'{len(X):,} ECGs, {X.shape[1]} samples x {X.shape[2]} leads at {a.fs} Hz | '
          f'99th pct |signal| = {amp:.3g} -> treating as {units}')
    scale = 1e-3 if units == 'uV' else 1.0

    t0 = time.time()
    size = max(16, len(X) // (a.jobs * 8) + 1)
    jobs = [(np.asarray(X[i:i + size]) * scale, a.fs, i) for i in range(0, len(X), size)]
    out = [None] * len(X)
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(a.jobs) as pool:
            done = 0
            for i0, recs in pool.imap_unordered(_chunk, jobs):
                out[i0:i0 + len(recs)] = recs
                done += len(recs)
                el = time.time() - t0
                print(f'  {done:,}/{len(X):,} ECGs  {el / 60:.1f} min elapsed, '
                      f'{(el / max(done, 1)) * (len(X) - done) / 60:.1f} min left', flush=True)
    else:
        for j, job in enumerate(jobs, 1):
            i0, recs = _chunk(job)
            out[i0:i0 + len(recs)] = recs
            print(f'  chunk {j}/{len(jobs)}', flush=True)

    M = pd.DataFrame(out, columns=COLS)
    if a.shard:
        M.insert(0, 'participant_id', ids)
        M.insert(0, 'row', np.arange(row0, row0 + len(M)))
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        M.to_csv(a.output, index=False)
        print(f'wrote {a.output}: {len(M):,} RAW per-ECG measurements '
              '(no outlier rule, no median -- --merge does both)')
        return
    M, n_drop = mad_clean(M)
    print(f'robust outlier rule removed {n_drop:,} individual values '
          f'({100 * n_drop / (len(M) * (len(COLS) - 1)):.2f}% of all measurements)')
    M.insert(0, 'participant_id', ids)
    # one row per ECG; a person with several gets the median of their ECGs, the same way the
    # score averages them
    M = M.groupby('participant_id', as_index=False).median(numeric_only=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    M.to_csv(a.output, index=False)

    key = ['HR_bpm', 'PR_ms', 'QRS_ms', 'QT_ms', 'JT_ms', 'QRSTangleF_deg',
           'Ramp_I', 'Samp_I', 'Tamp_V6']
    print(f'\nwrote {a.output}: {len(M):,} participants, {time.time() - t0:.0f}s')
    print(M[key].describe().T[['count', 'mean', 'std', '50%']].round(2).to_string())
    bad = [c for c in key if M[c].notna().mean() < 0.8]
    if bad:
        print('WARNING: under 80% of participants have a value for: ' + ', '.join(bad))
    print('THIS FILE CARRIES IDS. It stays on the analysis machine.')


if __name__ == '__main__':
    main()
