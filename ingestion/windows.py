"""Cut a long single-lead recording into non-overlapping 10 s windows."""
import numpy as np

WINDOW_SAMPLES = 2500  # 10 s at 250 Hz, the model's input length
INVALID_SAMPLE = -32768  # WFDB format 16: invalid / missing sample
SATURATED_SAMPLE = 32767
FLAT_STD_MV = 0.01     # a window with less variation than this is flagged as flat


def window_starts(n_samples, size=WINDOW_SAMPLES):
    """Start sample of every full window; a trailing partial window is dropped."""
    return np.arange(n_samples // size, dtype=np.int64) * size


def cut_windows(signal_mv, raw, size=WINDOW_SAMPLES):
    """Return (windows [n, size] float32 mV, starts, skip_reason per window, flat flag per window).

    A window is skipped (reason != '') if it contains an invalid (-32768) or saturated (32767) sample.
    """
    starts = window_starts(signal_mv.size, size)
    n = starts.size
    w = signal_mv[:n * size].reshape(n, size)
    r = raw[:n * size].reshape(n, size)
    invalid = (r == INVALID_SAMPLE).any(axis=1)
    sat = (r == SATURATED_SAMPLE).any(axis=1)
    reason = np.where(invalid, 'invalid', np.where(sat, 'saturated', ''))
    flat = np.nanstd(w, axis=1) < FLAT_STD_MV
    return w, starts, reason, flat
