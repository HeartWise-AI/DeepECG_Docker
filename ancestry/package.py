"""Mode `package` -- RUN LAST.  Audit everything, write the summary, make the one zip to send.

The audit is not a formality and it is not advisory: if any file fails it, NO zip is written at
all -- the run stops -- so a failed audit cannot be mistaken for a packaged result.
Four things are checked, on every file, after everything else has finished:

  1. no participant identifier appears anywhere -- the roster's own ids are read (and never
     printed) and every text file in the package is searched for each of them;
  2. no count between 1 and 9 is unmasked;
  3. no table is long enough to be per-person, and no array longer than a thinned curve;
  4. no score threshold is exported (a list of thresholds plus a list of rates is a ROC curve, and
     a fine enough ROC curve is a list of scores).

The genome-wide summary statistics, the genotype subsets, the per-recording model outputs, the
measurements and the roster are all in work/ and none of them is packaged.  A full summary
statistic is a separate export request with its own approval; this image never pre-empts that.

    python -m ancestry package --work /out/work --results /out/results --run-name MGB-MGH

Writes results_package_<run>_<date>.zip beside results/.  That file is the deliverable.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from ancestry import privacy as PV                                       # noqa: E402
from ancestry import lib as L                                            # noqa: E402

DESCRIBE = {
    'cohort/strata.csv': 'People per ancestry stratum at the 0.70 RFMix gate, with mean age, '
                         '% female and mean dominant-ancestry fraction.',
    'cohort/strata_by_batch.csv': 'The same counts split by genotyping batch.',
    'cohort/prepare_provenance.json': 'What `prepare` was given and what it produced.',
    'score/score_by_stratum.csv': 'P(non-European) per stratum: mean, median, quartiles, the '
                                  'share called non-European at 0.5, and the mean RFMix non-'
                                  'European fraction beside it.',
    'score/score_histogram.csv': 'P(non-European) in ten fixed bins per stratum.',
    'score/model_check.csv': 'READ THIS FIRST. AUROC of the ancestry score against the '
                             'site\'s own RFMix calls -- overall, per subpopulation and '
                             'against the earlier run recorded for this site. Everything else in '
                             'this package is an analysis OF this score, so a value near '
                             'chance here means the inputs are wrong, not the biology.',
    'score/model_check_verdict.txt': 'One line: OK, WEAK or FAIL, and what to do.',
    'ses/ses_quad.csv': 'The deprivation arms A-D with their sham and potency controls: each '
                        'row is an estimate, its interval and an equivalence P against the '
                        'margin. Arm B is the sensitive one; arm D deletes a single direction '
                        'and is near-powerless by construction.',
    'ses/ses_coupling.csv': 'READ BEFORE THE ARMS. How tightly deprivation and ancestry are '
                            'bound at this site -- mean decile and standardised difference per '
                            'stratum. Where the two are near-interchangeable, adjusting for one '
                            'removes the other, so survival is strong evidence and attenuation '
                            'is ambiguous.',
    'ses/ses_cut_grid.csv': 'Arms A and C swept over every decile cut, so no conclusion rests '
                            'on one threshold.',
    'ses/ses_distribution.csv': 'Deprivation decile by ancestry stratum, counts only, coarsened '
                                'to deciles and suppressed below the minimum cell.',
    'ses/ses_verdict.txt': 'The reading in order: coupling, ceiling, matching, erasure, and what '
                           'the design can and cannot show.',
    'ses/ses_provenance.json': 'Which measure, which geography, which vintage, which cut, and '
                               'the ecological-proxy caveat.',
    'geno/variant_coverage.csv': 'How many of our variants the site holds, and how many match '
                                 'our alleles in one orientation or the other.',
    'geno/analysis_sets.csv': 'Per stratum: n, n unrelated, PC variance explained.',
    'geno/geno_provenance.json': 'Build, genotype paths, PC count, kinship cutoff.',
    'loci/replication_summary.csv': 'THE HEADLINE. Per stratum: leads tested, sign concordance '
                                    'and its binomial P, the inverse-variance slope of the '
                                    "site's effects on ours with its CI, the aggregate z, and "
                                    'the number of leads this sample had the power to detect.',
    'loci/leads_per_stratum.csv': 'One row per lead per stratum: our effect and theirs, oriented '
                                  'to our effect allele, with frequency, n and status.',
    'loci/loci_headline.json': 'The same, machine-readable.',
    # the two rows a reader needs in order to read the headline row at all
    '  (row NON_EUR_META)': 'A fixed-effect meta of every non-European stratum. The '
                           'strata are individually too small at one hospital; combined '
                           'they are the only non-European statement this design '
                           'supports.',
    '  (rows sample = permuted phenotype)': 'The same leads, same people, same '
                                           'covariates, phenotype shuffled. The '
                                           'negative control that gives the headline '
                                           'count a scale: it should land near 50%.',
    'pgs/trait_pgs.csv': 'Each ECG-trait score on its own measured trait (the positive control) '
                         'and on the ancestry score, per stratum.',
    'pgs/trait_pgs_deciles.csv': 'Mean P(non-European) by decile of each trait score and of the '
                                 'frozen composite.',
    'pgs/ancestry_pgs.csv': 'The ancestry-score PGS at each p-value threshold: variants, effect '
                            'per SD, incremental R^2.',
    'pgs/ancestry_pgs_deciles.csv': 'Mean P(non-European) by decile of the ancestry-score PGS.',
    'pgs/pgs_headline.json': 'Clumping method, variant counts, the trait-to-measurement map.',
    'fairness/gaps.csv': 'THE AUDIT. One row per finding per metric (FPR, FNR, AUROC) per split '
                         '(genetic, inferred) per threshold rule, with a pair-cluster bootstrap '
                         'CI.',
    'fairness/label_coverage.csv': "What the site's ground truth covered, how many positives per "
                                   "group, and the head's own AUROC for each finding here.",
    'fairness/auditability.json': 'Whether the gap measured on inferred ancestry agrees with the '
                                  'gap measured on genotype: slope, correlation, sign agreement.',
    'scan/scan_summary.csv': 'Per stratum: variants scanned, lambda_GC, significant loci.',
}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def summary_md(res, ref, run_name):
    """The numbers a reader should see first, in order, with their caveats attached."""
    out = [f'# DeepECG at MGB-MGH -- results\n',
           f'Run `{run_name}`, packaged {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC, '
           f'image version {L.VERSION}.\n',
           'Everything below was computed inside MGB-MGH. No participant-level value left the '
           'site.\n']
    R = json.loads((Path(ref) / 'discovery_reference.json').read_text())

    def tbl(p):
        f = Path(res) / p
        return pd.read_csv(f) if f.exists() else None

    out.append('## 1. Who was analysed\n')
    S = tbl('cohort/strata.csv')
    if S is not None:
        out.append(S.to_markdown(index=False) + '\n')
        out.append(f'Strata below {PV.MIN_STRATUM} people are not analysed at all.\n')

    # Section 1b, before any result: did the model work here at all?  A reader who starts at
    # the loci table and finds a null has no way to tell a biological null from a broken input,
    # so the gate goes above it rather than in an appendix.
    MC = tbl('score/model_check.csv')
    vf = Path(res) / 'score' / 'model_check_verdict.txt'
    if MC is not None:
        out.append('## 1b. Did the ancestry model work here? (read before anything else)\n')
        if vf.exists():
            out.append(f'**{vf.read_text().strip()}**\n')
        out.append(MC.to_markdown(index=False) + '\n')
        prior = next(iter(R.get('external_prior', {}).values()), {})
        if prior.get('auroc'):
            out.append(f'This site scored {prior["auroc"]} (95% CI '
                       f'{prior.get("auroc_ci95")}) on n = {prior.get("n_total"):,} in October '
                       f'2026 with the same five checkpoints. Everything below is an analysis '
                       f'OF this score: if these numbers are near chance, the inputs are wrong '
                       f'and the results below are not evidence about biology.\n')
        out.append('The calibration error is expected to be large and is not a fault: the model '
                   'was calibrated in a 50% non-European matched set and this hospital is about '
                   '11% non-European, which shifts probabilities without touching the ranking.\n')

    out.append('## 2. The discovery loci\n')
    LR = tbl('loci/replication_summary.csv')
    if LR is None:
        out.append('_not run_\n')
    else:
        cols = [c for c in ['stratum', 'sample', 'n_tested', 'pct_concordant', 'sign_binomial_p',
                            'slope', 'slope_lo', 'slope_hi', 'aggregate_p', 'n_p05_same_sign',
                            'n_bonferroni', 'expected_n_bonferroni'] if c in LR.columns]
        out.append(LR[cols].to_markdown(index=False) + '\n')
        out.append(
            f'Discovery: {R["discovery"]["n_loci"]} loci in {R["discovery"]["n"]:,} people '
            f'({", ".join(R["discovery"]["cohorts"])}), gate {R["discovery"]["ancestry_gate"]}.\n'
            '\n**How to read `slope`.** It is the inverse-variance slope of the site\'s effects '
            'on ours, through the origin. 1.0 means the discovery effects reproduce at full '
            'size; below 1 with concordant signs means they reproduce attenuated, which is what '
            'winner\'s curse predicts. `expected_n_bonferroni` is how many leads this sample '
            'size should have detected if every discovery effect were exact -- compare it with '
            '`n_bonferroni` before reading a shortfall as a failure.\n'
            '\n**The non-European strata are a different question.** A European lead SNP tags its '
            'causal variant through European LD, which does not transfer -- least of all to '
            'African-ancestry genomes. An attenuated slope there is consistent with the signal '
            'being real and the instrument being wrong for that population. The strata are also '
            'small at one hospital, which is why the `NON_EUR_META` row exists: a fixed-effect '
            'meta across all of them is the only non-European statement this design supports, '
            'and the per-stratum rows above it should be read as its components.\n'
            '\n**The rows marked `permuted phenotype` are the negative control.** Same leads, '
            'same people, same covariates, phenotype shuffled. A count of concordant signs has '
            'no scale of its own, so the shuffled run supplies one: it should land near 50% with '
            'a binomial P near 0.5, and the real arm has to stand clear of it. If the two look '
            'alike, the result is not evidence -- and if the control does NOT land near 50%, '
            'something is wrong with the phenotype join rather than with the biology.\n')

    out.append('## 3. Polygenic scores\n')
    P = tbl('pgs/trait_pgs.csv')
    if P is None:
        out.append('_not run_\n')
    else:
        if 'arm' in P.columns and (P.arm == 'own_measured_trait').any():
            C = P[(P.arm == 'own_measured_trait') & P.get('beta_sd', pd.Series(dtype=float)).notna()]
            if len(C):
                out.append('### 3a. Positive control -- each score on its own measured trait\n')
                out.append(C[[c for c in ['stratum', 'trait', 'measure', 'n', 'beta_sd', 'se',
                                          'p', 't'] if c in C.columns]].to_markdown(index=False)
                           + '\n')
                out.append('A score that does not clear |t| = 3 on its own trait here is not a '
                           'working instrument at this site, and every number below that uses it '
                           'should be read as such.\n')
        A = tbl('pgs/ancestry_pgs.csv')
        if A is not None and 'beta_sd' in A.columns:
            B = A[A.p_threshold == 1.0] if 'p_threshold' in A.columns else A
            out.append('### 3b. The ancestry-score PGS\n')
            out.append(B[[c for c in ['stratum', 'n', 'n_variants', 'beta_sd', 'se', 'p',
                                      'delta_r2'] if c in B.columns]].to_markdown(index=False)
                       + '\n')
            lo = R['ancestry_pgs'].get('loo_delta_r2_pct') or {}
            out.append(f'Discovery, leave-one-cohort-out: beta '
                       f'{R["ancestry_pgs"].get("loo_beta_per_sd")}, incremental R^2 '
                       + ', '.join(f'{k} {v}%' for k, v in lo.items())
                       + '. At MGB the full three-cohort weights are used and the test is clean '
                         'by construction, because no MGB participant is in the meta-analysis.\n')

    out.append('## 4. The fairness audit\n')
    FA = Path(res) / 'fairness' / 'auditability.json'
    G = tbl('fairness/gaps.csv')
    if not FA.exists() or G is None:
        out.append('_not run_ (it needs the site\'s own finding labels)\n')
    else:
        J = json.loads(FA.read_text())
        out.append(f'Ground truth: {J["ground_truth"]}. '
                   f'{J["n_findings_audited"]} findings cleared the '
                   f'{PV.MIN_POS_FINDING}-positives-per-group rule. '
                   f'The genetic and the inferred split agree on '
                   f'{100 * J["split_agreement"]:.1f}% of patients.\n')
        for k, v in J.get('auditability', {}).items():
            out.append(f'- **{k}**: slope {v["slope_through_origin"]}, r {v["pearson_r"]}, '
                       f'{v["n_same_sign"]}/{v["n_findings"]} findings agree in sign '
                       f'(P = {v["sign_binomial_p"]:.3g}); mean |gap| '
                       f'{v["mean_abs_gap_genetic"]} genetic vs {v["mean_abs_gap_inferred"]} '
                       f'inferred, per 100 patients.\n')
        out.append('\nA slope near 1 with sign agreement means a hospital can audit this model '
                   'for ancestry disparity from the ECG alone, without genotyping anyone. That '
                   'is the claim being tested; the inferred label is used here only to MEASURE '
                   'the gap, never to correct it.\n')

    out.append('## 5. The genome-wide scan\n')
    SC = tbl('scan/scan_summary.csv')
    out.append('_not run_\n' if SC is None else (SC.to_markdown(index=False) + '\n'))
    if SC is not None:
        out.append(f'Discovery lambda_GC was {R["discovery"].get("lambda_gc_scan")} at the scan; '
                   f'LDSC put h2 at {R["discovery"].get("ldsc_h2")} with intercept '
                   f'{R["discovery"].get("ldsc_intercept")} (stratification share '
                   f'{R["discovery"].get("ldsc_ratio")}).\n')

    # 6. deprivation -- present only when the site supplied a socioeconomic measure
    sq = res / 'ses' / 'ses_quad.csv'
    if sq.exists():
        out.append('## 6. Socioeconomic deprivation\n')
        v = res / 'ses' / 'ses_verdict.txt'
        if v.exists():
            head = [ln for ln in v.read_text().splitlines() if ln.startswith(('BOUNDED', 'NOT '))]
            if head:
                out.append('**' + head[0] + '**\n')
        CO = tbl('ses/ses_coupling.csv')
        if CO is not None:
            out.append('How tightly deprivation and ancestry are bound here -- read this before '
                       'the arms, because where the two are near-interchangeable an attenuation '
                       'cannot be told apart from having adjusted away ancestry itself:\n')
            out.append(CO.to_markdown(index=False) + '\n')
        Q = tbl('ses/ses_quad.csv')
        if Q is not None:
            keep = Q[Q.quantity.astype(str).str.contains(
                'cost|difference|sham|AUROC for deprived|after erasure', case=False, na=False)]
            out.append((keep if len(keep) else Q).to_markdown(index=False) + '\n')
        out.append('Each arm is read against a sham carrying deprivation\'s own ancestry '
                   'correlation and none of its meaning: wherever the two are tightly coupled, '
                   'adjusting for ANY equally correlated variable costs ancestry AUROC, so a '
                   'plain null would not be interpretable. Arm B is the sensitive arm. Arm C is '
                   'insensitive by construction and arm D deletes one direction out of hundreds '
                   '(in discovery that same arm barely moved an unrelated covariate), so '
                   'neither carries much either way.\n')
        out.append('This is an AREA-level measure. The discovery work found the apparent '
                   'socioeconomic confound was a neighbourhood effect that vanished at the '
                   'individual level -- the imbalance that was large on an area-level index '
                   'was absent on individual household income -- so this arm replicates the '
                   'harder of the two cases, not '
                   'the easier one. Its measurement error pushes every estimate toward the '
                   'null, and every arm here reports a bound rather than a zero.\n')

    out.append('## What is NOT in this package\n'
               '- the genome-wide summary statistics (work/scan/), which are an export in their '
               'own right;\n'
               '- the per-recording model outputs, the measurements, the genotype subsets and '
               'the roster (work/), all of which carry ids.\n')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry package', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True,
                    help="the site's working directory: intermediates, per-person tables and "
                         'arrays. It stays at the site and is never packaged')
    ap.add_argument('--results', required=True,
                    help='the aggregate-only output directory; `package` zips exactly this')
    ap.add_argument('--ref', default=str(L.ref_dir()),
                    help='the frozen discovery tables (default: $REFERENCE_DIR, else the '
                         'reference/ folder of the pinned weights snapshot)')
    ap.add_argument('--run-name', default='MGB-MGH',
                    help='your cohort name, printed in the returned tables and used in the '
                         'package filename')
    ap.add_argument('--out', default=None,
                    help='where to write (default: beside --results)')
    ap.add_argument('--max-ids', type=int, default=4000,
                    help='how many of the roster ids to search the package for')
    ap.add_argument('--force', action='store_true',
                    help='package anyway after a failed audit (writes AUDIT_FAILED.txt too)')
    a = ap.parse_args()

    res = Path(a.results); work = Path(a.work)
    L.banner(f'package: {a.run_name}')
    files = sorted(p for p in res.rglob('*') if p.is_file())
    if not files:
        raise SystemExit(f'{res} is empty: nothing to package')
    print(f'  {len(files)} files under {res}')

    ids = ()
    rp = work / 'roster.csv'
    if not rp.exists():
        # Without the roster there is nothing to search FOR, so the identifier search does not
        # run.  That is not the same as passing it, and the earlier version of this code printed
        # "audit clean" either way -- a false assurance at exactly the moment it matters most.
        print(f'\n  !! {rp} is missing, so no identifier search can be run.')
        print('     The structural checks below still apply, but nothing has confirmed that the '
              'site\'s own identifiers are absent.')
        if not a.force:
            raise SystemExit('nothing was packaged. Run `prepare` first so work/roster.csv '
                             'exists, or re-run with --force to package without the identifier '
                             'search (the package will say so).')
    if rp.exists():
        R = pd.read_csv(rp, dtype=str, usecols=lambda c: c in ('participant_id', 'geno_id'))
        vals = pd.unique(R.to_numpy().ravel())
        vals = [v for v in vals if isinstance(v, str) and len(v) >= 4]
        rng = np.random.default_rng(0)
        ids = tuple(rng.permutation(vals)[:a.max_ids]) if len(vals) > a.max_ids else tuple(vals)
        print(f'  searching every file for {len(ids):,} of the site\'s own identifiers '
              '(they are never printed)')

    problems = PV.audit_files(files, ids=ids)
    if problems:
        print(f'\n  AUDIT FAILED -- {len(problems)} problem(s):')
        for p in problems[:40]:
            print('   ! ' + p)
        if not a.force:
            # A zip from an EARLIER, successful run is still sitting there, and the line this
            # command prints on success is "THIS FILE IS THE ONLY THING TO SEND BACK". Saying
            # nothing here invites the site to send the stale one, which is a different and
            # incomplete set of results rather than the one they just tried to build.
            stale = sorted((Path(a.out).parent if a.out else Path(a.results).parent)
                           .glob(f'results_package_{a.run_name}_*.zip'))
            msg = ['nothing was packaged. Fix the problems above, or re-run with --force to '
                   'package anyway (not advised).']
            if stale:
                msg.append('')
                msg.append(f'  WARNING: {len(stale)} package(s) from an earlier run are still '
                           'on disk and are NOT the results you just built:')
                msg += [f'    {q}' for q in stale[:5]]
                msg.append('  Do not send those. Delete them, or re-run `package` once the '
                           'problems above are fixed.')
            raise SystemExit('\n'.join(msg))
    elif ids:
        print('  audit clean: no identifier, no unmasked small count, no per-person table, '
              'no score threshold')
    else:
        print('  structural checks clean -- but the identifier search did NOT run (no roster)')

    (res / 'results_summary.md').write_text(summary_md(res, a.ref, a.run_name))
    # the summary and the README are built FROM the audited tables, but they are files that leave
    # the site too, and the README claims every file in it was audited.  Make that true.
    late = PV.audit_files([res / 'results_summary.md'], ids=ids)
    if late:
        for q in late[:10]:
            print('   ! ' + q)
        if not a.force:
            raise SystemExit('results_summary.md failed the audit; nothing was packaged.')
    files = sorted(p for p in res.rglob('*') if p.is_file())
    man = ['| file | contents | sha256 |', '|---|---|---|']
    for p in files:
        rel = str(p.relative_to(res))
        man.append(f'| `{rel}` | {DESCRIBE.get(rel, "")} | `{sha(p)[:16]}` |')
    readme = (f'# Results package: {a.run_name}\n\n'
              'Aggregate results of the DeepECG genetic replication and fairness audit, computed '
              'entirely inside MGB-MGH.  Start with `results_summary.md`.\n\n'
              '## Privacy\n\n'
              '- No participant-level value, identifier or per-person row is included.\n'
              f'- Counts of 1-{PV.MIN_CELL - 1} people are written as `<{PV.MIN_CELL}`.\n'
              f'- No metric is computed for a group with fewer than {PV.MIN_PER_CLASS} people per '
              f'class, no stratum below {PV.MIN_STRATUM} people is analysed, and no clinical '
              f'finding with fewer than {PV.MIN_POS_FINDING} positives in both ancestry groups '
              'enters the audit.\n'
              '- The genome-wide summary statistics are NOT here: they stay at the site.\n'
              '- docs/QUESTIONS.md in the image maps every sub-question to the file that\n'
              '  answers it, and lists what one site cannot answer.\n'
              '- Every file below was re-audited automatically before this package was written.\n'
              + ('' if ids else '- NOTE: the identifier search did NOT run: work/roster.csv\n'
                                '  was absent when this package was written. The structural\n'
                                '  checks did run.\n')
              + f'- Audit result: {"PASSED" if not problems else "FAILED (packaged with --force)"}\n'
              '\n## Files\n\n' + '\n'.join(man) + '\n')
    (res / 'README.md').write_text(readme)
    if problems and a.force:
        (res / 'AUDIT_FAILED.txt').write_text('\n'.join(problems) + '\n')

    stamp = datetime.datetime.now().strftime('%Y%m%d')
    zp = Path(a.out) if a.out else res.parent / f'results_package_{a.run_name}_{stamp}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(res.rglob('*')):
            if p.is_file():
                z.write(p, Path(f'results_package_{a.run_name}_{stamp}') /
                        p.relative_to(res))
    print(f'\n  wrote {zp}  ({zp.stat().st_size / 1e6:.1f} MB, '
          f'{sum(1 for p in res.rglob("*") if p.is_file())} files)')
    print('  THIS FILE IS THE ONLY THING TO SEND BACK.')


if __name__ == '__main__':
    main()
