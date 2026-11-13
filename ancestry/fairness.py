"""Mode `fairness` -- audit the released 77-finding head at MGB, twice: on genotype, then on ECG.

THE QUESTION.  A hospital that deploys a vendor ECG model wants to know whether it fails unequally
across ancestry groups.  It can ask that question two ways, and only one of them is available to a
hospital that has not genotyped its patients:

  genetic   split the patients by their RFMix ancestry -- the ground truth, and what a research
            cohort can do;
  inferred  split them by what DeepECG-Pop predicts from the ECG itself, called at 0.5 -- which
            needs nothing but the waveform the model was already given.

If the two give the same answer, a site can audit a vendor model for ancestry disparity WITHOUT
genotypes.  That is the claim this mode tests at MGB-MGH, and it is the external validation of the
auditability panel of the paper.  Note what the inferred label is used FOR here: only to MEASURE
the gap, never to fix it.

THE METRIC IS THE FALSE-POSITIVE GAP AT A SHARED OPERATING POINT, per 100 patients -- the rate at
which the model flags a finding in people who do not have it, in the non-European group minus the
European group.  The false-negative gap is computed identically and written beside it.  The
threshold is 0.5: the head emits sigmoid probabilities, publishes no operating point, and a
deployed model carries ONE threshold, so giving each group its own would be a correction applied
before the measurement.  Everything is also computed at the pooled 90%-specificity point, and a
conclusion that holds at one and not the other is a conclusion about the threshold.

THE DESIGN IS MATCHED, AND THE BOOTSTRAP RESAMPLES PAIRS.  Ancestry groups in a hospital cohort
differ in age and sex, and both change the prevalence of half the findings, so an unmatched gap
mixes a demographic difference into an ancestry one.  Patients are matched 1:1 on sex and age
(within `--caliper` years), and a bootstrap draw resamples PAIRS and takes both members, so every
draw keeps the balance exactly.  The unmatched analysis is computed too and reported beside it.

GROUND TRUTH.  The site supplies it, one of two ways:
  --labels FILE      a table with participant_id and one 0/1 column per finding, named as in
                     ref/findings77.tsv.  Any subset of the 77 is fine.
  --statements FILE  participant_id plus free-text cardiologist/MUSE statements, which are
                     matched against the regular expressions in ref/findings77.tsv.  This is a
                     weaker instrument and the package labels every number produced from it.
A finding enters the audit only with at least 15 positives IN BOTH ancestry groups, and the head's
own AUROC for it at MGB is reported first: a gap in a finding the model cannot detect at all here
is a statement about the label, not about fairness.

    python -m ancestry fairness --work /out/work --results /out/results \\
        --labels /path/mgb_findings.csv [--caliper 2] [--boot 1000] [--remap]

Writes results/fairness/.  One row per finding per metric per split; no person appears.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
from ancestry import privacy as PV                                       # noqa: E402
from ancestry import lib as L                                            # noqa: E402

SHARED_THR = 0.5
SPEC_Q = 0.90


# --------------------------------------------------------------------------------- labels
def labels_from_table(path, findings):
    # Read the complete table as text so every accepted identifier alias keeps its exact
    # spelling (notably leading zeroes) before we know which column is the identifier.
    # Finding values are converted explicitly by to_binary below.
    D = L.read_table(path, what='--labels', dtype=str)
    idc = L.pick_column(D, 'participant_id',
                        ('participant_id', 'subject_id', 'empi', 'patient_id', 'mrn', 'id'),
                        what=f'--labels {path}', kind='participant_id column',
                        flag='--labels (rename the column to participant_id)')
    if idc != 'participant_id':
        D = D.rename(columns={idc: 'participant_id'})
    have = [f for f in findings if f in D.columns]
    if not have:
        # a near-miss on spelling is the usual cause, so say which ones nearly matched
        import difflib
        near = []
        for c in D.columns[:60]:
            m = difflib.get_close_matches(str(c), list(findings), n=1, cutoff=0.75)
            if m:
                near.append(f'{c!r} -> {m[0]!r}')
        raise SystemExit(f'{path}: none of its columns is one of the 77 findings.\n'
                         f'  columns: {list(D.columns)[:12]} ...\n'
                         + (f'  did you mean:\n    ' + '\n    '.join(near[:8]) + '\n'
                            if near else '')
                         + '  the exact spellings are the `finding` column of ref/findings77.tsv')
    out = D[['participant_id'] + have].copy()
    how_seen, unusable = set(), []
    for f in have:
        v, how, n_bad = L.to_binary(out[f])
        out[f] = v
        how_seen.add(how)
        if v.notna().sum() == 0:
            unusable.append(f)
        elif n_bad:
            print(f'    {f}: {n_bad:,} of {len(v):,} values not understood, left blank')
    if unusable:
        raise SystemExit(
            f'{path}: {len(unusable)} finding column(s) hold no value this image can read as '
            f'0/1 -- e.g. {unusable[:3]}.\n'
            f'  first few values: {list(D[unusable[0]].head(5))}\n'
            '  accepted: 0/1, yes/no, y/n, true/false, positive/negative, present/absent.')
    return out, ('adjudicated label table' +
                 (' (yes/no values mapped to 0/1)' if 'yes/no words' in how_seen else ''))


def labels_from_statements(path, F):
    D = L.read_table(path, what='--statements', dtype=str)
    idc = L.pick_column(D, 'participant_id',
                        ('participant_id', 'id', 'subject_id', 'empi', 'mrn', 'patient_id'),
                        what='--statements', kind='participant_id column',
                        flag='(rename the column to participant_id)')
    if idc != 'participant_id':
        D = D.rename(columns={idc: 'participant_id'})
    txt = [c for c in D.columns if c != 'participant_id']
    if not txt:
        raise SystemExit(f'{path}: needs participant_id and at least one text column')
    S = D[txt].astype(str).agg(' | '.join, axis=1).str.lower()
    out = pd.DataFrame({'participant_id': D.participant_id.astype(str)})
    n = 0
    for _, r in F.iterrows():
        if not isinstance(r.muse_regex, str) or not r.muse_regex:
            continue
        out[r.finding] = S.str.contains(r.muse_regex, regex=True, na=False).astype(float)
        n += 1
    print(f'  text matching resolved {n} of 77 findings from {len(txt)} text column(s)')
    return out, 'free-text statements matched by regular expression (weaker than adjudication)'


# --------------------------------------------------------------------------------- matching
def match_pairs(D, caliper=2.0, seed=2000):
    """1:1 on sex and age within `caliper` years, greedy nearest, non-European first.

    Non-European first because it is the smaller group everywhere; matching the larger group
    first would leave the smaller one unmatchable at the ends of the age distribution.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    for sx, g in D.groupby('sex_num'):
        # Missing age is retained upstream for coverage accounting, but it cannot participate in
        # age matching.  Leaving it in the control pool makes its NaN distance win np.argmin and
        # can prevent every otherwise-matchable case in the stratum from finding a control.
        age = pd.to_numeric(g.age, errors='coerce')
        g = g.loc[np.isfinite(age)].copy()
        a = g[g.group_genetic == 1].sample(frac=1, random_state=seed)
        b = g[g.group_genetic == 0].copy()
        if a.empty or b.empty:
            continue
        used = np.zeros(len(b), dtype=bool)
        bage = b.age.to_numpy(float)
        for i, (_, row) in enumerate(a.iterrows()):
            d = np.abs(bage - float(row.age))
            d[used] = np.inf
            j = int(np.argmin(d))
            if not np.isfinite(d[j]) or d[j] > caliper:
                continue
            used[j] = True
            pairs.append((row.name, b.index[j], len(pairs)))
    if not pairs:
        return None
    idx = [i for p in pairs for i in p[:2]]
    pid = [p[2] for p in pairs for _ in range(2)]
    M = D.loc[idx].copy()
    M['pair_id'] = pid
    return M


# --------------------------------------------------------------------------------- metrics
# one definition, in _lib, so the gate in `score` and the audit here cannot drift apart
_auroc = L.auroc


def rates(y, p, grp, thr):
    """(fpr_eur, fpr_non, fnr_eur, fnr_non) per 100, at a shared threshold."""
    f = p >= thr
    out = {}
    for g, name in ((0, 'eur'), (1, 'non')):
        m = grp == g
        neg, pos = m & (y == 0), m & (y == 1)
        out[f'fpr_{name}'] = 100 * float(f[neg].mean()) if neg.sum() else np.nan
        out[f'fnr_{name}'] = 100 * float((~f[pos]).mean()) if pos.sum() else np.nan
    return out


def audit_finding(d, col, y, thr, n_boot, seed, min_per_class=PV.MIN_PER_CLASS):
    """Both metrics, both splits, with a pair-cluster bootstrap. Returns a list of rows."""
    p = d[col].to_numpy(float)
    rows = []
    for split, gcol in (('genetic', 'group_genetic'), ('inferred', 'group_inferred')):
        g = d[gcol].to_numpy()
        counts = {(group, cls): int(((g == group) & (y == cls)).sum())
                  for group in (0, 1) for cls in (0, 1)}
        base = rates(y, p, g, thr)
        # AUROC is here beside the two error rates on purpose.  the result to read is a PAIR of
        # statements -- the ranking is equally good in both groups (no AUROC gap) while the
        # errors at a SHARED operating point are not -- and only the pair supports the
        # conclusion, which is that the unfairness is in the threshold, not the model's
        # ability to rank this finding.  Reporting the error gap alone reads as the older,
        # weaker claim.  AUROC does not depend on `thr`, so the same value is written against
        # both operating points rather than recomputed.
        base['auroc_eur'] = _auroc(y[g == 0], p[g == 0])
        base['auroc_non'] = _auroc(y[g == 1], p[g == 1])
        for metric in ('fpr', 'fnr', 'auroc'):
            needed_classes = (0,) if metric == 'fpr' else (1,) if metric == 'fnr' else (0, 1)
            if any(counts[group, cls] < min_per_class
                   for group in (0, 1) for cls in needed_classes):
                continue
            e, n = base[f'{metric}_eur'], base[f'{metric}_non']
            gap = n - e
            cl = d['pair_id'].to_numpy() if 'pair_id' in d.columns else None

            def stat(ix, metric=metric, g=g):
                if metric == 'auroc':
                    gi, yi, pi = g[ix], y[ix], p[ix]
                    return _auroc(yi[gi == 1], pi[gi == 1]) - _auroc(yi[gi == 0], pi[gi == 0])
                r = rates(y[ix], p[ix], g[ix], thr)
                return r[f'{metric}_non'] - r[f'{metric}_eur']

            lo, hi = L.boot_ci(stat, len(d), n_boot=n_boot, seed=seed, clusters=cl)
            rows.append(dict(metric=metric,
                             units='per 100 patients' if metric != 'auroc' else 'AUROC (0-1)',
                             split=split, eur=round(e, 4), non=round(n, 4),
                             gap=round(gap, 4), gap_lo=round(lo, 4) if lo == lo else np.nan,
                             gap_hi=round(hi, 4) if hi == hi else np.nan,
                             significant=int((lo == lo) and (lo > 0 or hi < 0)),
                             n_pos_eur=counts[0, 1], n_pos_non=counts[1, 1],
                             n_neg_eur=counts[0, 0], n_neg_non=counts[1, 0]))
    return rows


def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry fairness', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True,
                    help="the site's working directory: intermediates, per-person tables and "
                         'arrays. It stays at the site and is never packaged')
    ap.add_argument('--results', required=True,
                    help='the aggregate-only output directory; `package` zips exactly this')
    ap.add_argument('--ref', default=str(L.ref_dir()),
                    help='the frozen discovery tables (default: $REFERENCE_DIR, else the '
                         'reference/ folder of the pinned weights snapshot)')
    ap.add_argument('--labels', default=None,
                    help='adjudicated 0/1 labels for the 77 findings, one row per person '
                         '(yes/no, Y/N, true/false are all read)')
    ap.add_argument('--statements', default=None,
                    help='free-text ECG statements instead of labels; mapped to the 77 '
                         'findings by the shipped rules')
    ap.add_argument('--findings', default=None, help='work/findings.parquet by default')
    ap.add_argument('--caliper', type=float, default=2.0,
                    help='age caliper in years for the 1:1 sex-and-age matched audit (default 2)')
    ap.add_argument('--boot', type=int, default=1000,
                    help='bootstrap resamples behind every gap and its confidence interval')
    ap.add_argument('--min-pos', type=int, default=PV.MIN_POS_FINDING,
                    help=f'a finding is audited only where BOTH ancestry groups have at least '
                         f'this many positives (default {PV.MIN_POS_FINDING}); below it the row '
                         'is reported as not evaluable rather than estimated')
    ap.add_argument('--seed', type=int, default=2000,
                    help='random seed for the matching draw and the bootstrap')
    ap.add_argument('--remap', action='store_true',
                    help="re-derive output->finding on the site's own labels instead of using "
                         'the shipped discovery mapping (makes the two audits different runs)')
    ap.add_argument('--unmatched-only', action='store_true',
                    help='skip the matched arm and audit the whole cohort only')
    a = ap.parse_args()

    work = Path(a.work); res = Path(a.results) / 'fairness'
    L.require('fairness', [
        (work / 'scores.csv', 'the ancestry scores and the ancestry strata', 'score'),
        (Path(a.findings) if a.findings else work / 'findings.parquet',
         "the 77-finding head's outputs", 'score --head findings  (or --head both)')])
    res.mkdir(parents=True, exist_ok=True)
    ref = Path(a.ref)
    F = pd.read_csv(ref / 'findings77.tsv', sep='\t').fillna({'muse_regex': '', 'output': ''})
    L.banner('fairness: the released 77-finding head, audited on genotype and on the ECG')

    W = pd.read_parquet(Path(a.findings) if a.findings else work / 'findings.parquet')
    W['participant_id'] = W.participant_id.astype(str)
    S = pd.read_csv(work / 'scores.csv', dtype={'participant_id': str})
    if a.labels:
        LB, how = labels_from_table(a.labels, list(F.finding))
    elif a.statements:
        LB, how = labels_from_statements(a.statements, F)
    else:
        raise SystemExit('--labels or --statements is required: the audit needs ground truth')
    print(f'  ground truth: {how}; {LB.shape[1] - 1} findings, {len(LB):,} recordings')

    D = S.merge(W, on='participant_id', how='inner').merge(LB, on='participant_id', how='inner')
    if len(D) < 0.2 * len(S):
        D2, hw = L.join_or_canon(S.merge(W, on='participant_id'), LB, 'participant_id',
                                 what='labels')
        print(f'  id matching: {hw}')
        D = D2
    D['sex_num'] = D.sex.astype(str).str.upper().map(
        {'M': 1, 'MALE': 1, '1': 1, 'F': 0, 'FEMALE': 0, '0': 0})
    # the two splits.  ADMIXED is excluded from the primary -- it is neither group, and the
    # discovery audit defined its two groups the same way -- and counted in the report.
    D['group_genetic'] = np.where(D.stratum == 'EUR', 0,
                                  np.where(D.stratum.isin(['AFR', 'EAS', 'SAS']), 1, -1))
    n_adm = int((D.group_genetic == -1).sum())
    D = D[D.group_genetic >= 0].copy()
    D['group_inferred'] = (D.p_nonEUR >= 0.5).astype(int)
    print(f'  audit pool: {len(D):,} people '
          f'({int((D.group_genetic == 0).sum()):,} European, '
          f'{int((D.group_genetic == 1).sum()):,} non-European; {n_adm:,} admixed excluded)')
    agree = float((D.group_genetic == D.group_inferred).mean())
    print(f'  the two splits agree on {100 * agree:.1f}% of patients')

    arms = [('unmatched', D)]
    if not a.unmatched_only:
        M = match_pairs(D, caliper=a.caliper, seed=a.seed)
        if M is not None and len(M) >= 2 * PV.MIN_PER_CLASS:
            arms.insert(0, ('matched', M))
            print(f'  matched 1:1 on sex and age (+/- {a.caliper:g} y): '
                  f'{len(M) // 2:,} pairs, {len(M):,} people')
        else:
            print('  matching produced too few pairs; the unmatched arm is the only one')

    # ---------------------------------------------------------------- the audit
    rows, gate = [], []
    outcols = [c for c in W.columns if c != 'participant_id']
    for arm, d in arms:
        thr90 = {}
        for _, f in F.iterrows():
            fin = f.finding
            if fin not in d.columns:
                continue
            col = f.output
            if a.remap or not isinstance(col, str) or col not in outcols:
                y_all = d[fin].to_numpy(float)
                ok = np.isfinite(y_all)
                if ok.sum() < 50 or len(np.unique(y_all[ok])) < 2:
                    continue
                aur = {c: _auroc(y_all[ok], d[c].to_numpy()[ok]) for c in outcols}
                col = max(aur, key=lambda k: (aur[k] if aur[k] == aur[k] else -1))
            sub = d[np.isfinite(d[fin].to_numpy(float))]
            y = sub[fin].to_numpy(float)
            g = sub.group_genetic.to_numpy()
            npos = [int(((y == 1) & (g == k)).sum()) for k in (0, 1)]
            auroc = _auroc(y, sub[col].to_numpy(float))
            gate.append(dict(arm=arm, finding=fin, output=col, kind=f.kind,
                             plate_eligible=int(f.plate_eligible), n=len(sub),
                             n_pos_eur=npos[0], n_pos_non=npos[1], auroc_site=auroc,
                             auroc_discovery=f.discovery_auroc,
                             evaluable=int(min(npos) >= a.min_pos)))
            if min(npos) < a.min_pos:
                continue
            thr90[fin] = float(np.quantile(sub.loc[y == 0, col], SPEC_Q)) if (y == 0).any() else np.nan
            for thr, rule in ((SHARED_THR, 'shared 0.5'),
                              (thr90[fin], 'pooled 90% specificity')):
                if not np.isfinite(thr):
                    continue
                for r in audit_finding(sub, col, y, thr, a.boot, a.seed):
                    r.update(arm=arm, finding=fin, output=col, kind=f.kind,
                             plate_eligible=int(f.plate_eligible), threshold_rule=rule,
                             auroc_site=auroc, n=len(sub))
                    rows.append(r)

    if not rows:
        raise SystemExit('no finding had enough positives in both groups to be audited; '
                         'see fairness/label_coverage.csv for what the labels contained')
    R = pd.DataFrame(rows)
    GA = pd.DataFrame(gate)
    PV.write_table(R, res / 'gaps.csv', where='gaps.csv')
    PV.write_table(GA, res / 'label_coverage.csv', where='label_coverage.csv')

    # ---------------------------------------------------------------- auditability
    head = {}
    for arm in R.arm.unique():
        for metric in ('fpr', 'fnr', 'auroc'):
            q = R[(R.arm == arm) & (R.metric == metric) &
                  (R.threshold_rule == 'shared 0.5')]
            g = q[q.split == 'genetic'].set_index('finding').gap
            i = q[q.split == 'inferred'].set_index('finding').gap
            j = pd.concat([g.rename('genetic'), i.rename('inferred')], axis=1).dropna()
            if len(j) < 4:
                continue
            slope = float((j.genetic * j.inferred).sum() / (j.genetic ** 2).sum())
            r = float(np.corrcoef(j.genetic, j.inferred)[0, 1])
            rho = float(stats.spearmanr(j.genetic, j.inferred).statistic)
            same = int((np.sign(j.genetic) == np.sign(j.inferred)).sum())
            head[f'{arm}_{metric}'] = dict(
                n_findings=len(j), slope_through_origin=round(slope, 4),
                pearson_r=round(r, 4), spearman_rho=round(rho, 4),
                n_same_sign=same, pct_same_sign=round(100 * same / len(j), 1),
                sign_binomial_p=float(stats.binomtest(same, len(j), 0.5,
                                                      alternative='greater').pvalue),
                mean_abs_gap_genetic=round(float(j.genetic.abs().mean()), 3),
                mean_abs_gap_inferred=round(float(j.inferred.abs().mean()), 3))
    # THE ALIGNMENT VERDICT.  The released 77-finding head publishes no output order, so the
    # mapping shipped in ref/findings77.tsv was resolved in OUR data.  If it is wrong here, every
    # finding is scored by the wrong output, every AUROC sits at chance, and the audit returns
    # "no gap" -- a false reassurance that looks exactly like a clean result.  This compares the
    # site's own per-finding AUROC against ours and says so in one line.
    GG = GA[(GA.arm == arms[0][0]) & np.isfinite(GA.auroc_site.astype(float))]
    med = float(GG.auroc_site.median()) if len(GG) else float('nan')
    medd = float(pd.to_numeric(GG.auroc_discovery, errors='coerce').median()) if len(GG) else float('nan')
    if not np.isfinite(med):
        align = 'UNKNOWN -- no finding had enough labels to check the output mapping'
    elif med < 0.60:
        align = (f'FAIL -- median AUROC {med:.3f} across {len(GG)} findings, against {medd:.3f} '
                 'in discovery. The 77 outputs are almost certainly in a different order here: '
                 're-run `fairness --remap` before reading any gap.')
    elif med < medd - 0.10:
        align = (f'WEAK -- median AUROC {med:.3f} against {medd:.3f} in discovery; the mapping '
                 'may be partly wrong. Compare label_coverage.csv finding by finding.')
    else:
        align = (f'OK -- median AUROC {med:.3f} across {len(GG)} findings against {medd:.3f} in '
                 'discovery; the shipped output mapping holds at this site.')
    print(f'\n  OUTPUT MAPPING: {align}')

    # WHO WAS ACTUALLY AUDITED.  "Does this cover everyone?" is the first question anyone asks
    # of a fairness result and it should be answerable from the package without reading code.
    # The chain is: people with an ECG AND an ancestry call -> minus admixed (no unambiguous
    # group for the genetic split) -> the `unmatched` arm, which IS everyone; the `matched` arm
    # is a 1:1 age/sex-matched subset of it, reported beside it rather than instead of it.
    cover = dict(
        n_scored_with_ancestry=int(len(D)),
        n_european=int((D.group_genetic == 0).sum()),
        n_non_european=int((D.group_genetic == 1).sum()),
        n_admixed_excluded_from_genetic_split=n_adm,
        arms={nm: dict(n_people=int(len(dd)),
                       n_pairs=int(len(dd) // 2) if nm == 'matched' else None,
                       covers=('every scored person with an unambiguous ancestry call'
                               if nm == 'unmatched' else
                               '1:1 matched on sex and age within the caliper'))
              for nm, dd in arms},
        per_finding_rule=f'a finding is audited only where both groups have >= {a.min_pos} '
                         f'positives; see label_coverage.csv for which did',
        not_covered='ECGs from people this site has not genotyped. The inferred-ancestry arm '
                    'needs no genotype and could in principle run on the whole ECG archive; '
                    'this container scores the genotyped roster so that the genetic and '
                    'inferred arms are measured in the SAME people, which is what makes them '
                    'comparable.')

    auditability = dict(
        version=L.VERSION, ground_truth=how, remapped=bool(a.remap), coverage=cover,
        output_mapping_verdict=align,
        median_auroc_site=None if not np.isfinite(med) else round(med, 4),
        median_auroc_discovery=None if not np.isfinite(medd) else round(medd, 4),
        n_admixed_excluded=n_adm, split_agreement=round(agree, 4),
        n_findings_audited=int(R.finding.nunique()), arms=list(R.arm.unique()),
        auditability=head)
    (res / 'auditability.json').write_text(
        json.dumps(PV.mask(auditability), indent=2, default=float))

    # a compact echo of the plate that will be drawn
    pl = R[(R.arm == arms[0][0]) & (R.metric == 'fpr') & (R.threshold_rule == 'shared 0.5') &
           (R.plate_eligible == 1)]
    if len(pl):
        piv = pl.pivot_table(index='finding', columns='split',
                             values=['gap', 'gap_lo', 'gap_hi'])
        print(f'\n  false-positive gap per 100 patients ({arms[0][0]}, threshold 0.5):')
        print(piv.round(2).to_string())
    for k, v in head.items():
        print(f'\n  auditability [{k}]: slope {v["slope_through_origin"]}, r {v["pearson_r"]}, '
              f'{v["n_same_sign"]}/{v["n_findings"]} same sign (P {v["sign_binomial_p"]:.3g})')
    print(f'\n  wrote {res}/')


if __name__ == '__main__':
    main()
