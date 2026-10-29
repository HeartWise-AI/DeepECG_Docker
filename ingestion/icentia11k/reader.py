"""Read-only access to Icentia-11k WFDB records (.hea / .dat / .atr) and 10 s windows.

Layout: <root>/p00/p00000/p00000_s00.{hea,dat,atr}; one lead at 250 Hz per segment. Nothing here writes to the dataset.
"""
import os
from dataclasses import dataclass

import numpy as np
import wfdb

FS = 250
WINDOW_SAMPLES = 2500  # 10 s at 250 Hz, the model's input length
INVALID_SAMPLE = -32768  # WFDB format 16: invalid / missing sample
SATURATED_SAMPLE = 32767
FLAT_STD_MV = 0.01     # a window with less variation than this is flagged as flat


@dataclass
class Annotations:
    samples: np.ndarray  # int64 sample index of each annotation, in time order
    symbols: np.ndarray  # beat ('N', 'S', 'V', 'Q') or rhythm change ('+')
    aux: np.ndarray      # on '+': '(N', '(AFIB', '(AFL' or ')'


def patient_dir(data, patient):
    return os.path.join(data, patient[:3], patient)  # p00012 -> p00/p00012


def list_segments(patient_dir):
    """Sorted segment names (e.g. 'p00000_s00') that have a .hea and .dat file."""
    names = sorted({f[:-4] for f in os.listdir(patient_dir) if f.endswith('.hea')})
    return [s for s in names if os.path.exists(os.path.join(patient_dir, s + '.dat'))]


def read_segment(patient_dir, segment):
    """Return (signal in mV float32, raw ADC samples, Annotations or None when there is no .atr file)."""
    path = os.path.join(patient_dir, segment)
    rec = wfdb.rdrecord(path, physical=False)
    if rec.n_sig != 1 or rec.fs != FS:
        raise ValueError(f'{segment}: expected 1 signal at {FS} Hz, got {rec.n_sig} at {rec.fs} Hz (no resampling)')
    raw = rec.d_signal[:, 0]
    mv = ((raw - rec.baseline[0]) / rec.adc_gain[0]).astype(np.float32)
    ann = None
    if os.path.exists(path + '.atr'):
        a = wfdb.rdann(path, 'atr')
        order = np.argsort(a.sample, kind='stable')  # the .atr format allows steps back in time
        ann = Annotations(np.asarray(a.sample, dtype=np.int64)[order], np.asarray(a.symbol)[order],
                          np.asarray([x.strip() for x in a.aux_note])[order])
    return mv, raw, ann


def cut_windows(signal_mv, raw, size=WINDOW_SAMPLES):
    """Return (windows [n, size] float32 mV, starts, skip_reason per window, flat flag per window).

    A trailing partial window is dropped. A window is skipped (reason != '') if it contains an invalid (-32768)
    or saturated (32767) sample.
    """
    n = signal_mv.size // size
    starts = np.arange(n, dtype=np.int64) * size
    w = signal_mv[:n * size].reshape(n, size)
    r = raw[:n * size].reshape(n, size)
    invalid = (r == INVALID_SAMPLE).any(axis=1)
    sat = (r == SATURATED_SAMPLE).any(axis=1)
    reason = np.where(invalid, 'invalid', np.where(sat, 'saturated', ''))
    flat = np.nanstd(w, axis=1) < FLAT_STD_MV
    return w, starts, reason, flat
