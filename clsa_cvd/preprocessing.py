"""The final CVD model's source-specific mV conversion; no batch normalization."""
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly

LEADS = ('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6')


def prepare_waveform(waveform, source_format, lead_order=LEADS):
    if tuple(lead_order) != LEADS:
        raise ValueError('Expected lead order I, II, III, aVR, aVL, aVF, V1-V6')
    if source_format not in ('clsa','muse','mv250'):
        raise ValueError('format must be clsa (500-Hz ADC), muse (250-Hz ADC), or mv250 (250-Hz mV)')
    x = np.asarray(waveform)
    if x.dtype.kind not in 'iuf':
        raise ValueError('Waveform must be real numeric data')
    x = np.squeeze(x).astype(np.float64)
    if x.ndim != 2:
        raise ValueError('Expected a two-dimensional 12-lead waveform')
    if x.shape[0] != 12:
        x = x.T
    expected = (12,5000 if source_format == 'clsa' else 2500)
    if x.shape != expected:
        raise ValueError(f'Expected shape {expected} or its transpose')
    if not np.isfinite(x).all():
        raise ValueError('Waveform contains nonfinite values')
    if source_format == 'clsa':
        x = resample_poly(x * .005,1,2,axis=-1)
    elif source_format == 'muse':
        x = x * .00488
    x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Waveform exceeds float32 range')
    return x


def load_waveform(path, source_format):
    return prepare_waveform(np.load(Path(path),allow_pickle=False),source_format)


def validate_clinical(age, female, hypertension=None):
    try:
        age,female = float(age),float(female)
        htn = None if hypertension is None else float(hypertension)
    except (TypeError, ValueError):
        raise ValueError('Clinical inputs must be numeric') from None
    if not np.isfinite(age) or not 18 <= age <= 100:
        raise ValueError('age must be finite years in [18,100]; CLSA development ages were 45-85')
    if female not in (0.,1.):
        raise ValueError('female must be 0 or 1')
    if htn is not None and htn not in (0.,1.):
        raise ValueError('hypertension must be 0 or 1')
    return age,female,htn
