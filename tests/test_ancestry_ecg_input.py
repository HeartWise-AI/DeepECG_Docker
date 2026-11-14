"""The ancestry pipeline must read the ECG files the main pipeline already produces."""
from pathlib import Path
import base64
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ancestry.prepare_external import load_ecg  # noqa: E402

needs_repo_python = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason='utils/files_handler imports utils/credentials, which needs Python >= 3.10')


def write_base64(path, signal):
    """Exactly what ECGFileHandler.save_ecg_signal writes: raw float32, sample-major."""
    path.write_text(base64.b64encode(signal.astype(np.float32).tobytes()).decode())


def test_a_base64_file_round_trips(tmp_path):
    sig = (np.random.default_rng(0).standard_normal((2500, 12)) * 0.5).astype(np.float32)
    f = tmp_path / 'a.base64'
    write_base64(f, sig)
    assert np.array_equal(load_ecg(str(f)), sig)


@needs_repo_python
def test_our_decoding_is_the_repositorys_own(tmp_path):
    """The format is defined by utils/files_handler; this asserts we have not drifted from it.

    The decode is duplicated rather than imported because that module needs Python >= 3.10
    while the ancestry package is 3.8-clean, so the agreement has to be tested, not assumed.
    """
    from utils.files_handler import ECGFileHandler
    sig = (np.random.default_rng(1).standard_normal((2500, 12)) * 0.5).astype(np.float32)
    f = tmp_path / 'b.base64'
    ECGFileHandler.save_ecg_signal(sig, str(f))             # upstream writes it
    assert np.array_equal(load_ecg(str(f)), ECGFileHandler.load_ecg_signal(str(f)))
    assert np.array_equal(load_ecg(str(f)), sig)


def test_a_longer_recording_is_decimated_the_way_the_main_pipeline_does_it(tmp_path):
    """10 s at 500 Hz -> 2500 samples, by the same integer step as utils/files_handler."""
    long = (np.random.default_rng(2).standard_normal((5000, 12)) * 0.5).astype(np.float32)
    f = tmp_path / 'c.npy'
    np.save(f, long)
    got = load_ecg(str(f))
    assert got.shape == (2500, 12)
    assert np.array_equal(got, long[::2][:2500])


@pytest.mark.parametrize('array,shape', [
    (np.zeros((12, 2500), np.float32), 'lead-major'),
    (np.zeros((2500, 12, 1), np.float32), 'trailing singleton'),
])
def test_the_shapes_the_main_pipeline_tolerates_are_tolerated(tmp_path, array, shape):
    f = tmp_path / 'd.npy'
    np.save(f, array)
    assert load_ecg(str(f)).shape == (2500, 12), shape


@pytest.mark.parametrize('samples', [3000, 4096, 2501, 6000])
def test_a_length_that_is_not_a_multiple_of_2500_is_refused(tmp_path, samples):
    """Truncating to the first 2500 samples would shorten the window, not the sampling rate.

    The signal would still score -- it would score a different duration of heartbeat -- which
    is the failure this pipeline cannot have. 2500 / 5000 / 7500 are 250, 500 and 750 Hz.
    """
    f = tmp_path / 'odd.npy'
    np.save(f, np.zeros((samples, 12), np.float32))
    with pytest.raises(ValueError) as excinfo:
        load_ecg(str(f))
    assert 'not a multiple of 2500' in str(excinfo.value)


@pytest.mark.parametrize('samples,step', [(2500, 1), (5000, 2), (7500, 3)])
def test_the_rates_a_cart_actually_produces_are_decimated(tmp_path, samples, step):
    sig = (np.random.default_rng(3).standard_normal((samples, 12)) * 0.5).astype(np.float32)
    f = tmp_path / 'ok.npy'
    np.save(f, sig)
    got = load_ecg(str(f))
    assert got.shape == (2500, 12)
    assert np.array_equal(got, sig[::step])


@pytest.mark.parametrize('array,message', [
    (np.zeros((1000, 12), np.float32), 'too short'),
    (np.zeros((2500, 8), np.float32), '12 leads'),
    (np.zeros(2500, np.float32), 'unexpected ECG shape'),
])
def test_what_cannot_be_read_is_refused_rather_than_padded(tmp_path, array, message):
    """A padded or part-filled waveform would score; it would just score something else."""
    f = tmp_path / 'e.npy'
    np.save(f, array)
    with pytest.raises(ValueError) as excinfo:
        load_ecg(str(f))
    assert message in str(excinfo.value)


def test_xml_is_refused_by_name_and_says_what_to_do():
    """Upstream parses CLSA and MHI XML, but its cohort scaling happens further along; an
    unverified scale would silently change every score instead of failing."""
    with pytest.raises(ValueError) as excinfo:
        load_ecg('/some/where/ecg.xml')
    assert 'base64' in str(excinfo.value)
