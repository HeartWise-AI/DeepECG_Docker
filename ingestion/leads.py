"""--lead: build the model's 12 × 2500 input from one lead."""
import numpy as np

LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
ALL_LEADS = '12'


def parse_lead(spec):
    """Return the list of slot indices that receive the signal; the others are zero."""
    if spec == ALL_LEADS:
        return list(range(12))
    if spec in LEAD_ORDER:
        return [LEAD_ORDER.index(spec)]
    raise ValueError(f"--lead must be one of {LEAD_ORDER} or '12', got {spec!r}")


def build_12_lead(windows, slots):
    """windows [n, 2500] -> [n, 12, 2500]; the signal goes into `slots`, other leads are zero."""
    out = np.zeros((windows.shape[0], 12, windows.shape[1]), dtype=np.float32)
    out[:, slots, :] = windows[:, None, :]
    return out
