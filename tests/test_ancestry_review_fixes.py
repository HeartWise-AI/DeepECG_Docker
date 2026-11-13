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
from ancestry import geno_ops as GO  # noqa: E402
from ancestry import fairness as FA  # noqa: E402
from ancestry import loci as LO  # noqa: E402
from ancestry import pgs as PG  # noqa: E402
from ancestry import prepare as PR  # noqa: E402
from ancestry import prepare_external as PE  # noqa: E402
from ancestry import privacy as PV  # noqa: E402
from ancestry import score as SC  # noqa: E402
from ancestry import submit as SU  # noqa: E402
from ancestry.vendor import gwas_aggregate as GA  # noqa: E402


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


def test_fairness_suppresses_only_metrics_whose_split_denominator_is_too_small():
    # Inferred group 1 has 40 positives but only one negative.  Its FNR has enough denominator;
    # its FPR and AUROC do not.  Genetic groups remain balanced so both splits are exercised.
    y = np.array([1] * 80 + [0] * 40, dtype=float)
    inferred = np.array([0] * 40 + [1] * 40 + [1] + [0] * 39)
    genetic = np.array([0] * 40 + [1] * 40 + [0] * 20 + [1] * 20)
    d = pd.DataFrame({'finding_score': np.linspace(.01, .99, 120),
                      'group_genetic': genetic,
                      'group_inferred': inferred})
    rows = FA.audit_finding(d, 'finding_score', y, .5, n_boot=5, seed=1)
    inferred_metrics = {row['metric'] for row in rows if row['split'] == 'inferred'}
    assert inferred_metrics == {'fnr'}
    assert {row['metric'] for row in rows if row['split'] == 'genetic'} == {
        'fpr', 'fnr', 'auroc'}


def test_matching_skips_a_sex_stratum_with_no_european_controls():
    d = pd.DataFrame({'sex_num': [0, 0], 'group_genetic': [1, 1], 'age': [40, 41]})
    assert FA.match_pairs(d) is None


def test_plink_reader_preserves_leading_zero_identifiers(tmp_path):
    path = tmp_path / 'plink.sscore'
    path.write_text('#FID\tIID\tSCORE1_SUM\n001\t0007\t1.25\n')
    got = GO.read_plink_output(path, sep='\t')
    assert got.loc[0, '#FID'] == '001'
    assert got.loc[0, 'IID'] == '0007'
    assert got.loc[0, 'SCORE1_SUM'] == pytest.approx(1.25)


def test_aggregate_locus_z_is_oriented_to_the_discovery_effect():
    disc = np.array([.2, -.2, .2, -.2, .2, -.2])
    frame = pd.DataFrame({'status': ['tested'] * 6, 'disc_beta': disc,
                          'site_beta': disc, 'SE': [.1] * 6,
                          'P': [0.04] * 6, 'N_SITE': [1000] * 6})
    summary, _ = LO.summarise(frame, 'EUR', 'everyone')
    assert summary['aggregate_z'] > 4.8
    assert summary['aggregate_p'] < 1e-5


def test_pgs_frame_is_restricted_to_the_unrelated_keep_file(tmp_path):
    keep = tmp_path / 'keep_unrelated.txt'
    keep.write_text('#FID\tIID\nfamily\t0002\n')
    d = pd.DataFrame({'geno_id': ['0001', '0002'], 'score_logit': [0.1, 0.2]})
    got = PG.restrict_to_keep(d, keep)
    assert got.geno_id.tolist() == ['0002']


def test_lifted_panel_promotes_mapped_coordinates_and_drops_gaps(tmp_path):
    panel = tmp_path / 'panel.tsv'
    pd.DataFrame({'CHR': ['1', '2'], 'POS': [100, 200], 'A1': ['A', 'C'],
                  'CHR37': ['1', 'NA'], 'POS37': ['90', 'NA']}).to_csv(
                      panel, sep='\t', index=False)
    assert GE.promote_lifted_panel(panel) == panel
    got = pd.read_csv(panel, sep='\t', dtype={'CHR': str})
    assert got[['CHR', 'POS']].to_dict('records') == [{'CHR': '1', 'POS': 90}]
    assert 'CHR37' not in got and 'POS37' not in got


def test_lowercase_rfmix_headers_become_canonical_roster_proportions():
    cov = pd.DataFrame({'afr': [.8], 'eas': [.1], 'sas': [.05], 'eur': [.05]})
    got = PR.canonical_proportions(cov, ['afr', 'eas', 'sas', 'eur'])
    assert list(got.columns) == ['AFR', 'EAS', 'SAS', 'EUR']
    assert got.loc[0, 'AFR'] == pytest.approx(.8)


def test_admixed_exclusion_counts_are_masked_and_audited(tmp_path):
    raw = {'n_admixed_excluded': 3,
           'coverage': {'n_admixed_excluded_from_genetic_split': 3}}
    masked = PV.mask(raw)
    assert masked['n_admixed_excluded'] == f'<{PV.MIN_CELL}'
    assert masked['coverage']['n_admixed_excluded_from_genetic_split'] == f'<{PV.MIN_CELL}'
    path = tmp_path / 'auditability.json'
    path.write_text(__import__('json').dumps(raw))
    problems = PV.audit_files([path])
    assert len(problems) == 2


def test_slurm_geno_stage_builds_shared_panel_once_and_keeps_all_strata():
    args = SimpleNamespace(
        work='/work', results='/results', geno='/data/chr@', no_convert=True,
        rfmix=['/data/rfmix.tsv'], ecg_meta=['/data/meta.csv'], reuse_prepared=None,
        score_shards=2, save_embeddings=False, device='auto', batch_size=32,
        measurements='/data/measurements.csv', trait_shards=2, jobs=2,
        build='grch38', chain=None, threads=4, with_scan=False, scan_threads=2,
        labels=None, statements=None, ses=None, ses_col=None, ses_name=None,
        ses_geography=None, run_name='TEST')
    plan = SU.build_plan(args, ['EUR', 'AFR'], ['1'], {'gpu': False})
    geno = next(stage for stage in plan if stage.name == 'geno')
    assert geno.array is None
    assert geno.args[-3:] == ['--strata', 'EUR', 'AFR']


def test_scan_aggregate_reads_the_pheno_suffix_written_by_plink(tmp_path):
    path = tmp_path / 'chr1.pheno.glm.linear'
    pd.DataFrame({'#CHROM': ['1'], 'POS': [123], 'ID': ['v1'], 'A1': ['A'],
                  'TEST': ['ADD'], 'BETA': [.2], 'SE': [.1], 'P': [.04]}).to_csv(
                      path, sep='\t', index=False)
    got = GA.read_glm(str(tmp_path / 'chr'), ['1'])
    assert got[['CHROM', 'POS', 'ID']].to_dict('records') == [
        {'CHROM': 1, 'POS': 123, 'ID': 'v1'}]


def test_permuted_pheno_preserves_leading_zero_sample_ids(tmp_path):
    source = tmp_path / 'pheno.tsv'
    destination = tmp_path / 'pheno_permuted.tsv'
    source.write_text('#FID\tIID\tpheno\n0001\t0007\t0.1\n0002\t0008\t0.2\n')
    LO.write_permuted_pheno(source, destination)
    got = pd.read_csv(destination, sep='\t', dtype=str)
    assert got['#FID'].tolist() == ['0001', '0002']
    assert got.IID.tolist() == ['0007', '0008']
    assert set(got.pheno) == {'0.1', '0.2'}
