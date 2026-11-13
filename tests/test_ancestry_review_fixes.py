"""Regressions for identity, privacy, shard merge, and liftover review findings."""
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ancestry import genotypes as GE  # noqa: E402
from ancestry import geno_ops as GO  # noqa: E402
from ancestry import convert as CV  # noqa: E402
from ancestry import fairness as FA  # noqa: E402
from ancestry import __main__ as AM  # noqa: E402
from ancestry import loci as LO  # noqa: E402
from ancestry import lib as L  # noqa: E402
from ancestry import metrics as ME  # noqa: E402
from ancestry import package as PA  # noqa: E402
from ancestry import pgs as PG  # noqa: E402
from ancestry import prepare as PR  # noqa: E402
from ancestry import prepare_external as PE  # noqa: E402
from ancestry import privacy as PV  # noqa: E402
from ancestry import score as SC  # noqa: E402
from ancestry import scan as SN  # noqa: E402
from ancestry import ses as SES  # noqa: E402
from ancestry import submit as SU  # noqa: E402
from ancestry.vendor import gwas_aggregate as GA  # noqa: E402
from ancestry.vendor import liftover_min as LM  # noqa: E402
from ancestry.vendor import measure_neurokit as MN  # noqa: E402


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


def test_embedding_merge_uses_only_selected_shards_and_score_order(tmp_path):
    score_files = [tmp_path / 'scores_1of2.csv', tmp_path / 'scores_2of2.csv']
    for path in score_files:
        path.touch()
    np.save(tmp_path / 'embeddings_1of2.npy', np.array([[[2.0]]]))
    pd.DataFrame({'participant_id': ['p2']}).to_csv(
        tmp_path / 'embeddings_1of2_ids.csv', index=False)
    np.save(tmp_path / 'embeddings_2of2.npy', np.array([[[1.0]]]))
    pd.DataFrame({'participant_id': ['p1']}).to_csv(
        tmp_path / 'embeddings_2of2_ids.csv', index=False)
    # A complete but unrelated older run must not be included.
    np.save(tmp_path / 'embeddings_1of3.npy', np.array([[[99.0]]]))
    pd.DataFrame({'participant_id': ['old']}).to_csv(
        tmp_path / 'embeddings_1of3_ids.csv', index=False)

    SC.merge_embedding_shards(tmp_path, score_files, ['p1', 'p2'])

    assert np.load(tmp_path / 'embeddings.npy')[:, 0, 0].tolist() == [1.0, 2.0]
    got = pd.read_csv(tmp_path / 'embeddings_ids.csv', dtype=str)
    assert got.participant_id.tolist() == ['p1', 'p2']


def test_skip_kinship_removes_a_stale_unrelated_filter(tmp_path, monkeypatch):
    keep = tmp_path / 'keep_unrelated.txt'
    keep.write_text('old filter')
    monkeypatch.setattr(GE.G, 'unrelated',
                        lambda *args, **kwargs: pytest.fail('kinship must be skipped'))
    assert GE.refresh_unrelated_keep('pruned', keep, True, 2, 'geno') is None
    assert not keep.exists()


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
    args.with_scan = True
    scan_plan = SU.build_plan(args, ['EUR', 'AFR'], ['1', '2'], {'gpu': False})
    scan = next(stage for stage in scan_plan if stage.name == 'scan')
    assert '--scan-only' in scan.args
    aggregate = next(stage for stage in scan_plan if stage.name == 'scan_aggregate')
    assert aggregate.after == ['scan']
    assert aggregate.args[-3:] == ['--chroms', '1', '2']


def test_ses_waits_for_trait_fallback_when_embeddings_and_measurements_are_absent():
    args = SimpleNamespace(
        work='/work', results='/results', geno='/data/chr@', no_convert=True,
        rfmix=['/data/rfmix.tsv'], ecg_meta=['/data/meta.csv'], reuse_prepared=None,
        score_shards=2, save_embeddings=False, device='auto', batch_size=32,
        measurements=None, trait_shards=2, jobs=2, build='grch38', chain=None,
        threads=4, with_scan=False, scan_threads=2, labels=None, statements=None,
        ses='/data/ses.csv', ses_col=None, ses_name=None, ses_geography=None,
        run_name='TEST')
    plan = SU.build_plan(args, ['EUR', 'AFR'], ['1'], {'gpu': False})
    assert next(stage for stage in plan if stage.name == 'ses').after == [
        'score_merge', 'traits_merge']


def test_ses_receives_a_supplied_measurement_panel_without_waiting_for_traits():
    args = SimpleNamespace(
        work='/work', results='/results', geno='/data/chr@', no_convert=True,
        rfmix=['/data/rfmix.tsv'], ecg_meta=['/data/meta.csv'], reuse_prepared=None,
        score_shards=2, save_embeddings=False, device='auto', batch_size=32,
        measurements='/data/device.csv', trait_shards=2, jobs=2, build='grch38', chain=None,
        threads=4, with_scan=False, scan_threads=2, labels=None, statements=None,
        ses='/data/ses.csv', ses_col=None, ses_name=None, ses_geography=None,
        run_name='TEST')
    plan = SU.build_plan(args, ['EUR'], ['1'], {'gpu': False})
    stage = next(stage for stage in plan if stage.name == 'ses')
    assert stage.after == ['score_merge']
    assert stage.args[-2:] == ['--measurements', '/data/device.csv']


def test_ses_supplied_measurements_align_alias_ids_and_keep_leading_zeros(tmp_path):
    panel = tmp_path / 'device.csv'
    panel.write_text('Participant_ID,HR_bpm\n0008,70\n0007,60\n')
    roster = pd.DataFrame({'participant_id': ['0007', '0008']})
    features, names = SES.measurement_features(roster, panel)
    assert names == ['HR_bpm']
    assert features[:, 0].tolist() == [60.0, 70.0]


def test_single_machine_ses_receives_the_supplied_measurement_panel():
    args = SimpleNamespace(
        ses='/data/ses.csv', ses_col=None, ses_name=None, ses_geography=None,
        measurements='/data/device.csv')
    assert AM.ses_mode_args(args)[-2:] == ['--measurements', '/data/device.csv']


def test_failed_forced_conversion_cannot_reuse_stale_outputs(tmp_path, monkeypatch):
    out = tmp_path / 'pgen'
    out.mkdir()
    for suffix in ('.pgen', '.pvar', '.psam'):
        (out / f'chr1{suffix}').write_text('stale\n')
    monkeypatch.setattr(L, 'plink2', lambda: '/bin/false')
    ok, _variants, _samples, _error = CV.convert_one(
        tmp_path / 'source.vcf', '1', out, 1, False, False, [])
    assert ok is False
    assert not any((out / f'chr1{suffix}').exists()
                   for suffix in ('.pgen', '.pvar', '.psam'))


def test_cached_conversion_refuses_to_relabel_a_different_source(tmp_path, monkeypatch):
    old = tmp_path / 'old_chr1.vcf.gz'
    new = tmp_path / 'new_chr1.vcf.gz'
    old.touch()
    new.touch()
    outdir = tmp_path / 'work' / 'geno' / 'pgen'
    outdir.mkdir(parents=True)
    for suffix in ('.pgen', '.pvar', '.psam'):
        (outdir / f'chr1{suffix}').write_text('cached')
    provenance = tmp_path / 'work' / 'geno' / 'converted.json'
    provenance.write_text(__import__('json').dumps({
        'source': str(old), 'pattern': str(outdir / 'chr@'), 'dosage': 'auto',
        'keep_multiallelic': False, 'chroms': ['1']}))
    monkeypatch.setattr(sys, 'argv', [
        'ancestry.convert', '--work', str(tmp_path / 'work'), '--vcf', str(new),
        '--chroms', '1'])

    with pytest.raises(SystemExit, match='different source'):
        CV.main()

    assert __import__('json').loads(provenance.read_text())['source'] == str(old)
    assert (outdir / 'chr1.pgen').exists()


@pytest.mark.parametrize('mode', ['traits', 'convert'])
def test_shell_wrapper_routes_only_supported_common_paths(mode, tmp_path):
    stub = tmp_path / 'python'
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o755)
    env = os.environ.copy()
    env['PATH'] = f'{tmp_path}:{env.get("PATH", "/usr/bin:/bin")}'
    result = subprocess.run(
        ['bash', str(ROOT / 'run_ancestry.bash'), '--mode', mode], cwd=ROOT,
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    forwarded = result.stdout.splitlines()
    assert '--work' in forwarded
    assert '--results' not in forwarded


def test_matching_ignores_controls_with_missing_age():
    roster = pd.DataFrame({
        'sex_num': [0, 0, 0],
        'group_genetic': [1, 0, 0],
        'age': [50.0, np.nan, 51.0],
    }, index=['case', 'missing-control', 'valid-control'])
    matched = FA.match_pairs(roster, caliper=2.0)
    assert matched is not None
    assert matched.index.tolist() == ['case', 'valid-control']


def test_scan_normalizes_provenance_build_alias_for_aggregation(tmp_path, monkeypatch):
    work, results = tmp_path / 'work', tmp_path / 'results'
    (results / 'geno').mkdir(parents=True)
    (results / 'geno' / 'geno_provenance.json').write_text(__import__('json').dumps({
        'build': 'hg19', 'strata': ['EUR'], 'geno': None}))
    (work / 'geno' / 'EUR').mkdir(parents=True)
    (work / 'geno' / 'EUR' / 'pheno.tsv').write_text('#FID\tIID\tpheno\n1\t1\t0.1\n')
    (work / 'scan' / 'EUR').mkdir(parents=True)
    (work / 'scan' / 'EUR' / 'chr1.pheno.glm.linear').write_text('fixture\n')
    seen = {}
    monkeypatch.setattr(__import__('runpy'), 'run_path',
                        lambda *_args, **_kwargs: seen.update(argv=list(sys.argv)))
    monkeypatch.setattr(sys, 'argv', [
        'scan', '--work', str(work), '--results', str(results),
        '--aggregate-only', '--chroms', '1'])
    SN.main()
    assert seen['argv'][seen['argv'].index('--build') + 1] == 'grch37'


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


def test_allele_matching_requires_the_complete_reference_pair():
    got = L.allele_sign(['A', 'A', 'C'], ['C', 'G', 'A'],
                        ['A', 'A', 'A'], ['C', 'C', 'C'])
    assert got[0] == 1
    assert np.isnan(got[1])
    assert got[2] == -1


def test_partial_regression_ignores_redundant_covariate_columns():
    rng = np.random.default_rng(42)
    covariate = rng.normal(size=200)
    x = rng.normal(size=200)
    y = .4 * x + .8 * covariate + rng.normal(size=200)
    base = L.partial(y, x, covariate)
    redundant = L.partial(y, x, np.column_stack([
        covariate, covariate, np.ones(len(covariate))]))
    for key in ('beta_sd', 'se', 'p', 'partial_r', 'delta_r2', 'r2_base'):
        assert redundant[key] == pytest.approx(base[key])


def test_converted_genotypes_replace_only_the_recorded_source(tmp_path):
    converted = tmp_path / 'geno' / 'pgen' / 'chr@'
    (tmp_path / 'geno').mkdir()
    (tmp_path / 'geno' / 'converted.json').write_text(__import__('json').dumps({
        'source': '/imputed/chr@.vcf.gz', 'pattern': str(converted), 'chroms': [1]}))
    assert L.resolve_geno(tmp_path, '/imputed/chr@.vcf.gz', quiet=True) == str(converted)
    assert L.resolve_geno(tmp_path, '/array/chr@', quiet=True) == '/array/chr@'


def test_ses_matching_verdict_requires_a_two_sided_equivalence_bound():
    # This is the review counterexample: the upper 95% endpoint is inside +0.02, while the
    # lower endpoint admits a much larger loss.  The achieved TOST bound therefore exceeds the
    # margin and must not support a BOUNDED verdict.
    estimate, se, margin = -.015, .015, .02
    achieved_bound = abs(estimate) + 1.645 * se
    assert achieved_bound > margin
    assert SES.matching_verdict(estimate, achieved_bound, margin).startswith(
        'SMALL BUT NOT PROVABLY BOUNDED')
    assert SES.matching_verdict(.002, .01, margin).startswith('BOUNDED')


def test_mediation_uses_complete_cases_and_a_common_predictor_scale():
    rng = np.random.default_rng(7)
    n = 300
    c = rng.normal(size=n)
    x = rng.normal(size=n)
    measurement = .9 * x + rng.normal(scale=.2, size=n)
    y = .4 * x + .3 * c + rng.normal(size=n)
    measurement[5] = np.nan
    base, adjusted, _drop = PG.mediation_effects(y, x, c, measurement)
    ok = np.isfinite(measurement)
    design = np.column_stack([np.ones(ok.sum()), c[ok]])
    x_scale = np.std(x[ok] - design @ np.linalg.lstsq(design, x[ok], rcond=None)[0])
    raw_base = np.linalg.lstsq(np.column_stack([design, x[ok]]), y[ok], rcond=None)[0][-1]
    raw_adjusted = np.linalg.lstsq(
        np.column_stack([design, measurement[ok], x[ok]]), y[ok], rcond=None)[0][-1]
    assert base['n'] == adjusted['n'] == n - 1
    assert base['beta_sd'] == pytest.approx(raw_base * x_scale)
    assert adjusted['beta_sd'] == pytest.approx(raw_adjusted * x_scale)


def test_ses_probe_predictions_remain_aligned_when_a_measurement_is_missing():
    rng = np.random.default_rng(8)
    y = np.array([0] * 25 + [1] * 25)
    features = np.column_stack([y + rng.normal(size=50), rng.normal(size=50)])
    features[7, 0] = np.nan
    _auroc, predictions, n = SES.cv_probe(features, y)
    assert n == 49
    assert len(predictions) == len(y)
    assert np.isnan(predictions[7])
    assert np.isfinite(np.delete(predictions, 7)).all()


def test_per_cohort_calibration_enforces_the_privacy_bin_floor(tmp_path, monkeypatch):
    y = np.tile(np.array([0] * 20 + [1] * 20), 2)
    scores = np.linspace(.01, .99, len(y))
    cohort = np.array(['A'] * 40 + ['B'] * 40)
    seen = []
    original = ME._calibration_bins

    def recording_bins(labels, predictions, n_bins=10, min_bin=0):
        seen.append(min_bin)
        return original(labels, predictions, n_bins=n_bins, min_bin=min_bin)

    monkeypatch.setattr(ME, '_calibration_bins', recording_bins)
    output = tmp_path / 'per_cohort.png'
    assert ME.per_cohort_figure(y, scores, cohort, {'curves': {}}, output,
                                min_bin=1) == output
    assert seen == [PV.MIN_PER_CLASS, PV.MIN_PER_CLASS]


def test_package_summary_uses_fixed_panel_as_the_pgs_headline(tmp_path):
    res, ref = tmp_path / 'results', tmp_path / 'reference'
    (res / 'pgs').mkdir(parents=True)
    ref.mkdir()
    (ref / 'discovery_reference.json').write_text(__import__('json').dumps({
        'external_prior': {},
        'ancestry_pgs': {'loo_beta_per_sd': .1, 'loo_delta_r2_pct': {}}}))
    pd.DataFrame([{'arm': 'scored'}]).to_csv(res / 'pgs' / 'trait_pgs.csv', index=False)
    pd.DataFrame([
        {'arm': 'fixed_panel', 'stratum': 'EUR', 'n': 100, 'n_variants': 8,
         'beta_sd': .123, 'se': .01, 'p': .001, 'delta_r2': .02},
        {'arm': 'clumped', 'stratum': 'EUR', 'p_threshold': 1.0, 'n': 100,
         'n_variants': 7, 'beta_sd': 9.999, 'se': .02, 'p': .002, 'delta_r2': .03},
    ]).to_csv(res / 'pgs' / 'ancestry_pgs.csv', index=False)
    summary = PA.summary_md(res, ref, 'TEST')
    headline = summary.split('### 3b. The ancestry-score PGS', 1)[1].split(
        '### 3c. Site-clumped sensitivity analysis', 1)[0]
    sensitivity = summary.split('### 3c. Site-clumped sensitivity analysis', 1)[1]
    assert '0.123' in headline and '9.999' not in headline
    assert '9.999' in sensitivity


def test_trait_merge_expands_scheduler_globs_and_preserves_leading_zero_ids(
        tmp_path, monkeypatch):
    for shard, row, participant in ((1, 0, '0007'), (2, 1, '0012')):
        pd.DataFrame([{'row': row, 'participant_id': participant, 'HR': 60 + row,
                       'n_rpeaks': 10}]).to_csv(
                           tmp_path / f'traits_{shard}of2.csv', index=False)
    output = tmp_path / 'measurements.csv'
    monkeypatch.setattr(sys, 'argv', [
        'measure_neurokit.py', '--input', str(tmp_path), '--ids', str(tmp_path / 'ids.csv'),
        '--output', str(output), '--merge', str(tmp_path / 'traits_*.csv')])
    MN.main()
    got = pd.read_csv(output, dtype={'participant_id': str})
    assert got.participant_id.tolist() == ['0007', '0012']
    ids = tmp_path / 'ids.csv'
    ids.write_text('participant_id\n0007\n')
    assert MN.participant_ids(ids).tolist() == ['0007']


def test_scan_failure_removes_stale_output_and_propagates(tmp_path, monkeypatch):
    d = tmp_path / 'geno' / 'EUR'
    d.mkdir(parents=True)
    for name in ('keep.txt', 'pheno.tsv', 'covar.tsv'):
        (d / name).write_text('fixture\n')
    out = tmp_path / 'scan' / 'EUR'
    out.mkdir(parents=True)
    stale = out / 'chr1.pheno.glm.linear'
    stale.write_text('old result\n')
    # chr1 lookup must not accidentally accept a successful chr10 file.
    (out / 'chr10.pheno.glm.linear').write_text('other chromosome\n')
    monkeypatch.setattr(L, 'geno_args', lambda *_args, **_kwargs: [])
    monkeypatch.setattr(L, 'plink2', lambda: '/bin/false')
    with pytest.raises(SystemExit, match='command failed'):
        SN.scan_one('EUR', '1', tmp_path, '/geno/chr@', 1, .01, 20, .1)
    assert not stale.exists()
    assert SN.chromosome_outputs(out, '1') == []


def test_chain_common_arguments_match_each_mode_parser():
    assert AM.common_args('traits', '/work', '/results') == ['--work', '/work']
    assert AM.common_args('convert', '/work', '/results') == ['--work', '/work']
    assert AM.common_args('score', '/work', '/results') == [
        '--work', '/work', '--results', '/results']
    assert AM.common_args('figures', '/work', '/results') == ['--results', '/results']


def test_zero_tost_p_is_bounded_and_zero_estimate_keeps_its_bound():
    strong = SES.row('B', 'equivalence', .001, se=.001, margin=.02)
    assert strong['tost_p'] == 0.0
    assert strong['bounded_below_margin'] is True
    exact_zero = SES.row('B', 'equivalence', 0.0, se=.001, margin=.02)
    assert exact_zero['achieved_bound'] == pytest.approx(.0016)


def test_measurement_identifier_alias_is_renamed_and_keeps_leading_zeros(tmp_path):
    panel = tmp_path / 'measurements.csv'
    panel.write_text('Participant_ID,HR\n0007,62\n')
    got = PG.read_measurement_table(panel, 'device')
    assert got.participant_id.tolist() == ['0007']
    assert 'Participant_ID' not in got


def test_fairness_label_alias_keeps_leading_zero_identifiers(tmp_path):
    labels = tmp_path / 'labels.csv'
    labels.write_text('Participant_ID,Finding A\n0007,1\n')
    got, _description = FA.labels_from_table(labels, ['Finding A'])
    assert got.participant_id.tolist() == ['0007']
    assert got['Finding A'].tolist() == [1.0]


def test_overall_model_check_suppresses_a_class_below_the_privacy_floor(tmp_path):
    ref, results = tmp_path / 'ref', tmp_path / 'results'
    ref.mkdir()
    (results / 'score').mkdir(parents=True)
    (ref / 'discovery_reference.json').write_text(__import__('json').dumps({
        'external_prior': {'site': {'auroc': .8, 'ece': .1}},
        'subpopulation_auroc_discovery': {'all non-EUR': {'auroc': .8}},
    }))
    cohort = pd.DataFrame({
        'stratum': ['EUR'] * (PV.MIN_PER_CLASS + 5) + ['AFR'],
        'p_nonEUR': np.linspace(.1, .9, PV.MIN_PER_CLASS + 6),
    })
    table, _verdict = SC.model_check(cohort, results, ref)
    overall = table[table.comparison == 'everyone (admixed counted non-European)'].iloc[0]
    assert overall.n_pos == 1
    assert pd.isna(overall.auroc)
    assert overall.note == PV.SUPPRESSED


def test_pgs_reuses_genotype_covariates_with_rfmix_adjustment(tmp_path):
    covar = tmp_path / 'geno' / 'AFR'
    covar.mkdir(parents=True)
    (covar / 'covar.tsv').write_text(
        '#FID\tIID\tage\tsex\tPC1\trf_AFR\trf_EUR\n'
        'f2\t0008\t60\t0\t.2\t.75\t.25\n'
        'f1\t0007\t50\t1\t.1\t.80\t.20\n')
    analysis = pd.DataFrame({'geno_id': ['0007', '0008'], 'pheno_rint': [.1, .2]})
    got, names = PG.attach_configured_covariates(tmp_path, 'AFR', analysis)
    assert len(names) == 5
    assert got[names].iloc[0].tolist() == pytest.approx([50, 1, .1, .8, .2])
    assert got[names].iloc[1].tolist() == pytest.approx([60, 0, .2, .75, .25])


def test_finding_auroc_requires_both_label_classes_to_clear_privacy_floor():
    too_few_positive = np.array([0] * (PV.MIN_PER_CLASS + 10) + [1])
    assert FA.reportable_label_classes(too_few_positive) is False
    enough = np.array([0] * PV.MIN_PER_CLASS + [1] * PV.MIN_PER_CLASS)
    assert FA.reportable_label_classes(enough) is True


def test_negative_strand_liftover_reverse_complements_panel_alleles(tmp_path, monkeypatch):
    chain = tmp_path / 'negative.chain'
    chain.write_text('chain 100 chr1 1000 + 100 110 chr1 1000 - 200 210 1\n10\n')
    source = tmp_path / 'panel.tsv'
    source.write_text('CHR\tPOS\tA1\tA0\n1\t101\tA\tC\n')
    output = tmp_path / 'lifted.tsv'
    monkeypatch.setattr(sys, 'argv', [
        'liftover_min.py', str(chain), str(source), str(output),
        '--allele-cols', 'A1', 'A0'])
    LM.main()
    got = pd.read_csv(output, sep='\t', dtype=str)
    assert got[['CHR37', 'POS37', 'A1', 'A0']].iloc[0].to_dict() == {
        'CHR37': '1', 'POS37': '800', 'A1': 'T', 'A0': 'G'}


def test_trait_weight_builder_uses_position_not_grouped_series_index(tmp_path):
    idx = [4, 8]
    variants = pd.DataFrame({'CHR': ['1', '1'], 'POS': [100, 200],
                             'vid': ['v1', 'v2'], 'a1': ['A', 'G'], 'a2': ['C', 'T']})
    output = tmp_path / 'weights.tsv'
    kept, published = PG.site_weights(
        variants, pd.Series(['1', '1'], index=idx), pd.Series([100, 200], index=idx),
        pd.Series(['A', 'G'], index=idx), np.array(['C', 'T']),
        pd.Series([.1, .2], index=idx), 'trait', output)
    assert (kept, published) == (2, 2)
    assert pd.read_csv(output, sep='\t').ID.tolist() == ['v1', 'v2']
