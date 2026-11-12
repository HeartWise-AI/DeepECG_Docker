"""Regressions for identity, privacy, shard merge, and liftover review findings."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ancestry import genotypes as GE  # noqa: E402
from ancestry import prepare as PR  # noqa: E402
from ancestry import prepare_external as PE  # noqa: E402
from ancestry import privacy as PV  # noqa: E402
from ancestry import score as SC  # noqa: E402


def _alignment_fixture(tmp_path, monkeypatch, source_rows=True):
    gdir = tmp_path / 'all_genotyped'
    gdir.mkdir()
    np.save(gdir / 'X.npy', np.zeros((1, 2500, 12), dtype=np.float32))
    cov = pd.DataFrame({'batch': [1], 'round': [1], 'sex': ['MALE'], 'age': [50.0],
                        'AFR': [0.1], 'EAS': [0.0], 'SAS': [0.0], 'EUR': [0.9]})
    cov.to_csv(gdir / 'covariates.csv', index=False)
    if source_rows:
        np.save(gdir / 'source_rows.npy', np.array([1], dtype=np.int64))
    people = pd.DataFrame({
        'project_pseudo_id': ['person-first', 'person-second'],
        'Sample': ['geno-first', 'geno-second'],
        'batch': [1, 1], 'round': [1, 1], 'gender': ['MALE', 'MALE'],
        'age': [50.0, 50.0], 'AFR': [0.1, 0.1], 'EAS': [0.0, 0.0],
        'SAS': [0.0, 0.0], 'EUR': [0.9, 0.9],
    })
    monkeypatch.setattr(PR.PI, 'build_rfmix_table', lambda *args, **kwargs: people)
    monkeypatch.setattr(PR.PI, 'load_ecg_metadata', lambda *args, **kwargs: people)
    monkeypatch.setattr(PR.PI, 'link', lambda *args, **kwargs: people.copy())
    monkeypatch.setattr(PR.PI, 'normalise_sex', lambda s: s)
    monkeypatch.setattr(PR.PI, 'person_table', lambda *args, **kwargs: people.copy())
    args = SimpleNamespace(rfmix=['r'], ecg_meta=['e'], sample_col='Sample',
                           ancestries=['AFR', 'EAS', 'SAS', 'EUR'], threshold=.7,
                           rfmix_sep='auto', ecg_id_col='ecg_id', gender_col='gender',
                           age_col='age', path_col='path', ecg_prefix='', patient_id_col=None,
                           patient_id_prefix='', date_col=None, linkage='link',
                           barcode_col='Barcode', pseudo_col='project_pseudo_id')
    return args


def test_exact_source_rows_select_the_right_duplicate_identity(tmp_path, monkeypatch):
    args = _alignment_fixture(tmp_path, monkeypatch, source_rows=True)
    rows, _cov, _anc = PR.recover_rows(args, tmp_path)
    assert rows.project_pseudo_id.tolist() == ['person-second']
    assert rows.Sample.tolist() == ['geno-second']


def test_legacy_duplicate_alignment_is_refused_when_a_waveform_was_dropped(tmp_path, monkeypatch):
    args = _alignment_fixture(tmp_path, monkeypatch, source_rows=False)
    with pytest.raises(SystemExit, match='identity cannot be recovered safely'):
        PR.recover_rows(args, tmp_path)


def test_prepare_records_the_exact_waveform_rows_that_survived(tmp_path, monkeypatch):
    people = pd.DataFrame({
        'full_ecg_path': ['bad', 'good'], 'project_pseudo_id': ['p1', 'p2'],
        'batch': [1, 1], 'round': [1, 1], 'gender': ['MALE', 'FEMALE'],
        'age': [50, 51], 'ancestry': ['EUR', 'AFR'], 'AFR': [0.0, 1.0],
        'EAS': [0.0, 0.0], 'SAS': [0.0, 0.0], 'EUR': [1.0, 0.0],
    })

    def fake_save(_paths, _ids, path):
        np.save(path, np.zeros((1, 2500, 12), dtype=np.float32))
        return np.array([False, True])

    monkeypatch.setattr(PE, '_save_ecgs', fake_save)
    PE.build_all_genotyped(people, None, tmp_path, ['AFR', 'EAS', 'SAS', 'EUR'], 'EUR', .7)
    assert np.load(tmp_path / 'source_rows.npy').tolist() == [1]


def test_default_genotype_ids_are_rfmix_samples_not_participant_ids():
    linked = pd.DataFrame({'project_pseudo_id': ['person-1'], 'Sample': ['genotype-9']})
    pid, gid = PR.roster_ids(linked)
    assert pid.tolist() == ['person-1']
    assert gid.tolist() == ['genotype-9']


def test_explicit_ancestry_count_columns_and_provenance_are_masked(tmp_path):
    out = tmp_path / 'strata_by_batch.csv'
    PV.write_table(pd.DataFrame({'batch': [1], 'EUR': [500], 'AFR': [3]}), out,
                   count_cols=['EUR', 'AFR'])
    got = pd.read_csv(out, dtype=str)
    assert got.loc[0, 'EUR'] == '500'
    assert got.loc[0, 'AFR'] == f'<{PV.MIN_CELL}'
    masked = PV.mask({'strata': {'EUR': 500, 'AFR': 3}})
    assert masked['strata']['AFR'] == f'<{PV.MIN_CELL}'
    assert PV.small_count_cells({'strata': {'AFR': 3}}) == ['.strata.AFR']

    planted = tmp_path / 'unmasked' / 'strata_by_batch.csv'
    planted.parent.mkdir()
    pd.DataFrame({'batch': [1], 'EUR': [500], 'AFR': [3]}).to_csv(planted, index=False)
    assert any('AFR=3' in problem for problem in PV.audit_files([planted]))


def test_finding_shards_merge_in_score_order(tmp_path):
    score_files = [tmp_path / 'scores_1of2.csv', tmp_path / 'scores_2of2.csv']
    for path in score_files:
        path.touch()
    pd.DataFrame({'participant_id': ['p2'], 'out0': [.2]}).to_parquet(
        tmp_path / 'findings_1of2.parquet', index=False)
    pd.DataFrame({'participant_id': ['p1'], 'out0': [.1]}).to_parquet(
        tmp_path / 'findings_2of2.parquet', index=False)
    SC.merge_finding_shards(tmp_path, score_files, ['p1', 'p2'])
    got = pd.read_parquet(tmp_path / 'findings.parquet')
    assert got.participant_id.tolist() == ['p1', 'p2']
    assert got.out0.tolist() == [.1, .2]


def test_requested_grch37_liftover_happens_before_panel_discovery(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(GE, 'lift_panel',
                        lambda ref, work, build, chain: calls.append(('lift', chain)) or tmp_path)
    monkeypatch.setattr(GE, 'wanted_variants',
                        lambda ref, build, work=None: calls.append(('wanted', work)) or 'variants')
    assert GE.resolve_wanted_variants(tmp_path, tmp_path / 'work', 'grch37', 'chain.gz') == 'variants'
    assert calls == [('lift', 'chain.gz'), ('wanted', tmp_path / 'work')]
