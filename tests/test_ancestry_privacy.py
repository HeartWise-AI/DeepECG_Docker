"""The disclosure rules, which are the reason this pipeline may run inside a hospital at all.

Nothing per-patient leaves: every exported table goes through `privacy.write_table`, every file
is re-audited at packaging time, and small counts are masked. These tests check the rules
themselves rather than any analysis that uses them.
"""
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ancestry import privacy as PV  # noqa: E402


def test_small_counts_are_masked_and_large_ones_are_not():
    assert PV.cell(0) == 0                       # a true zero is not a small cell
    assert PV.cell(1) == f'<{PV.MIN_CELL}'
    assert PV.cell(PV.MIN_CELL - 1) == f'<{PV.MIN_CELL}'
    assert PV.cell(PV.MIN_CELL) == PV.MIN_CELL


def test_masking_reaches_into_nested_structures():
    got = PV.mask({'n': 3, 'inner': [{'n_pos': 2, 'auroc': 0.81}]})
    assert got['n'] == f'<{PV.MIN_CELL}'
    assert got['inner'][0]['n_pos'] == f'<{PV.MIN_CELL}'
    assert got['inner'][0]['auroc'] == 0.81      # a statistic is not a count


@pytest.mark.parametrize('column', ['participant_id', 'IID', 'MRN', 'ecg_path', 'eid'])
def test_an_identifier_column_cannot_be_written(tmp_path, column):
    df = pd.DataFrame({column: ['a', 'b'], 'value': [1.0, 2.0]})
    with pytest.raises(ValueError, match='identifier column'):
        PV.write_table(df, tmp_path / 'out.csv', where='test')
    assert not (tmp_path / 'out.csv').exists(), 'refused, so nothing should be on disk'


def test_a_per_person_sized_table_cannot_be_written(tmp_path):
    df = pd.DataFrame({'stratum': ['EUR'] * (PV.MAX_TABLE_ROWS + 1)})
    with pytest.raises(ValueError, match='per-person'):
        PV.write_table(df, tmp_path / 'out.csv', where='test')


def test_write_table_masks_counts_on_the_way_out(tmp_path):
    PV.write_table(pd.DataFrame({'stratum': ['EUR', 'AFR'], 'n': [5000, 3]}),
                   tmp_path / 'out.csv', where='test')
    got = pd.read_csv(tmp_path / 'out.csv', dtype=str)
    assert got.loc[1, 'n'] == f'<{PV.MIN_CELL}'
    assert got.loc[0, 'n'] == '5000'


def test_the_audit_finds_a_planted_identifier(tmp_path):
    (tmp_path / 'summary.md').write_text('the worst case was patient STPID00042\n')
    problems = PV.audit_files([tmp_path / 'summary.md'], ids=['STPID00042'])
    assert problems and 'summary.md' in problems[0]


def test_the_audit_passes_a_clean_aggregate(tmp_path):
    PV.write_table(pd.DataFrame({'stratum': ['EUR', 'AFR'], 'n': [5000, 400],
                                 'auroc': [0.80, 0.88]}),
                   tmp_path / 'clean.csv', where='test')
    assert PV.audit_files([tmp_path / 'clean.csv'], ids=['STPID00042']) == []


def test_a_stratum_below_the_floor_is_not_analysed():
    assert not PV.stratum_ok(PV.MIN_STRATUM - 1)
    assert PV.stratum_ok(PV.MIN_STRATUM)


# --- the deprivation cut is tested at the median AND the mean -------------------------------
#
# A skewed index makes these two different splits, so a verdict that holds at one and not the
# other is a statement about the threshold. Both must be reachable and both must be reported.

def test_the_cut_can_be_the_mean_as_well_as_the_median():
    import numpy as np
    from ancestry.ses import pick_cut
    rng = np.random.default_rng(2000)
    v = np.clip(rng.beta(2, 5, 4000) * 100, 1, 100)        # right-skewed, 1-100 like SDI
    med, med_lab, _ = pick_cut(v, 'cohort-median')
    mean, mean_lab, _ = pick_cut(v, 'cohort-mean')
    assert abs(med - float(np.median(v))) < 1e-9
    assert abs(mean - float(np.mean(v))) < 1e-9
    assert mean > med, 'for a right-skewed index the mean must sit above the median'
    assert 'median' in med_lab and 'mean' in mean_lab


def test_a_bad_cut_spec_is_refused_by_name():
    import numpy as np
    import pytest as _pytest
    from ancestry.ses import pick_cut
    with _pytest.raises(SystemExit) as e:
        pick_cut(np.linspace(1, 100, 500), 'cohort-mode')
    assert 'cohort-mean' in str(e.value), 'the refusal should list the specs that do exist'


def test_every_named_cut_including_the_mean_is_in_the_grid():
    import numpy as np
    from ancestry.ses import named_cuts
    rng = np.random.default_rng(7)
    v = np.clip(rng.beta(2, 5, 3000) * 100, 1, 100)
    specs = [spec for _lab, spec, _c in named_cuts(v)]
    for want in ('national-median', 'cohort-median', 'cohort-mean', 'q80'):
        assert want in specs, f'{want} is missing from the named cuts the grid reports'
    # and a non-percentile index has no national anchor to offer
    specs2 = [spec for _l, spec, _c in named_cuts(rng.normal(0, 1, 3000))]
    assert 'national-median' not in specs2 and 'cohort-mean' in specs2
