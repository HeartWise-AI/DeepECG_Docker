"""Read-only reader for Icentia-11k WFDB records (.hea / .dat / .atr).

Plain numpy, no wfdb dependency. Nothing here writes to the dataset.
"""
import os
from dataclasses import dataclass

import numpy as np

# WFDB annotation codes used by Icentia-11k.
BEAT_CODES = {1: 'N', 9: 'S', 5: 'V', 13: 'Q'}
RHYTHM_CODE = 28  # '+', rhythm change; the aux string carries '(N', '(AFIB', '(AFL' or ')'

# Pseudo-codes in the MIT annotation format.
_SKIP, _NUM, _SUB, _CHN, _AUX = 59, 60, 61, 62, 63


@dataclass
class Header:
    record: str
    n_signals: int
    fs: float
    n_samples: int
    fmt: int
    gain: float
    baseline: int


@dataclass
class Annotations:
    samples: np.ndarray  # int64 sample index of each annotation
    codes: np.ndarray    # int16 WFDB annotation code
    aux: list            # aux string per annotation ('' when absent)


def read_header(path_hea):
    with open(path_hea) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    rec = lines[0].split()
    sig = lines[1].split()
    gain_field = sig[2]  # e.g. '74.36(0)/mV'
    gain = float(gain_field.split('(')[0].split('/')[0])
    baseline = int(gain_field.split('(')[1].split(')')[0]) if '(' in gain_field else int(sig[4])
    return Header(record=rec[0], n_signals=int(rec[1]), fs=float(rec[2]), n_samples=int(rec[3]),
                  fmt=int(sig[1]), gain=gain, baseline=baseline)


def read_signal_mv(path_dat, header):
    """Return the single-lead signal in mV as float32. Format 16 = little-endian int16."""
    if header.n_signals != 1 or header.fmt != 16:
        raise ValueError(f'{path_dat}: expected 1 signal in format 16, got {header.n_signals} / {header.fmt}')
    raw = np.fromfile(path_dat, dtype='<i2')
    if raw.size != header.n_samples:
        raise ValueError(f'{path_dat}: {raw.size} samples, header says {header.n_samples}')
    return ((raw.astype(np.float32) - header.baseline) / header.gain), raw


def read_annotations(path_atr):
    """Parse a WFDB MIT-format annotation file.

    The SKIP interval is a *signed* 32-bit value (high 16-bit word first);
    reading it unsigned shifts every following annotation by 2**32 samples.
    """
    words = np.fromfile(path_atr, dtype='<u2')
    samples, codes, aux = [], [], []
    t = 0
    i = 0
    n = words.size
    while i < n:
        w = int(words[i])
        code, interval = w >> 10, w & 0x3FF
        i += 1
        if code == 0 and interval == 0:
            break
        if code == _SKIP:
            hi, lo = int(words[i]), int(words[i + 1])
            i += 2
            skip = (hi << 16) | lo
            if skip >= 2 ** 31:
                skip -= 2 ** 32
            t += skip
        elif code == _AUX:
            nbytes = interval
            nwords = (nbytes + 1) // 2
            text = words[i:i + nwords].tobytes()[:nbytes].split(b'\x00')[0].decode('ascii', 'replace')
            i += nwords
            if aux:
                aux[-1] = '' if text == 'None' else text  # Icentia writes 'None' on every beat
        elif code in (_NUM, _SUB, _CHN):
            continue
        elif code == 0:
            t += interval  # code 0 carries only a time step
        else:
            t += interval
            samples.append(t)
            codes.append(code)
            aux.append('')
    return Annotations(np.asarray(samples, dtype=np.int64), np.asarray(codes, dtype=np.int16), aux)


def list_segments(patient_dir):
    """Sorted segment names (e.g. 'p00000_s00') that have a .hea and .dat file."""
    names = sorted({f[:-4] for f in os.listdir(patient_dir) if f.endswith('.hea')})
    return [s for s in names if os.path.exists(os.path.join(patient_dir, s + '.dat'))]


def load_segment(patient_dir, segment):
    base = os.path.join(patient_dir, segment)
    header = read_header(base + '.hea')
    signal_mv, raw = read_signal_mv(base + '.dat', header)
    ann = read_annotations(base + '.atr') if os.path.exists(base + '.atr') else None
    return header, signal_mv, raw, ann
