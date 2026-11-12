"""Mode `ses` -- is the ECG ancestry signal explained by area-level socioeconomic deprivation?

WHAT THIS ARM IS FOR, AND WHY IT IS NOT A REPEAT OF THE PAPER'S NULL

The discovery work asked this twice and got two different-looking answers, and the difference is
the whole point of running it here:

  FIRST, on an AREA-LEVEL census deprivation index.  Matching on it cost the ancestry AUROC a
        small but non-zero amount -- the one arm in the entire confound audit that was not a
        clean null -- and the EUR/non-EUR imbalance on that index was the largest of any
        covariate in the analysis.
  SECOND, on INDIVIDUAL household income.  All four tests were null and the imbalance collapsed
        to an interval spanning zero.

So the paper's own conclusion is that the apparent socioeconomic confound was a NEIGHBOURHOOD
effect, not a property of the patients.  The Social Deprivation Index is a neighbourhood measure.
Running it here therefore replicates the arm that was NOT null, under conditions harder than the
UK Biobank's: in a US hospital catchment, census-tract deprivation is far more tightly coupled to
genetic ancestry than Townsend is in Britain.  This is the adversarial case, and it is worth more
than a third confirmation of the easy one.

That tight coupling is also the one way this analysis can mislead, so it is measured and printed
BEFORE any arm is interpreted (`ses_coupling.csv`).  If SDI separates the ancestry groups nearly
as well as the ECG does, then "adjust for SDI" and "adjust for ancestry" are close to the same
operation, and an attenuation cannot be read as socioeconomics explaining the signal -- it is
equally consistent with the adjustment having removed ancestry itself.  Survival under that
adjustment is strong evidence; attenuation under it is ambiguous evidence.  The verdict file says
which of the two happened.

THE FOUR ARMS, named as in the discovery deprivation analyses so the numbers drop into the existing table

  A  can the ECG see deprivation at all?      a probe fitted here, out of fold, pooled and
                                              WITHIN Europeans (pooled alone could be reading
                                              ancestry, which is the thing being controlled for)
  B  does ancestry survive matching on it?     1:1, sex-exact, 0.2 SD caliper
                                              + SHAM: the same matcher on a permuted SDI
                                              + the same-size random-subset band
  C  does ancestry work in both halves?        split at the cut, AUROC inside each
  D  does removing its direction cost anything? SDI projected out of the embedding
                                              + SHAM: 200 random directions
                                              + POTENCY: can an SDI probe still read SDI off the
                                                residual?  An erasure that removed nothing makes
                                                D's null uninformative, which was missing from the
                                                first version of this arm.

EVERY ARM REPORTS A BOUND, NOT A NULL.  At this n an AUROC difference is resolvable to about
+/-0.02, which is the pre-specified equivalence margin and is larger than the cost the
discovery area-level arm measured.  So a
non-significant arm here does NOT show the effect is zero -- it shows it is smaller than the
margin.  Each arm is therefore reported with a two-one-sided-test equivalence P against
`--margin`, and the verdict is phrased as "bounded below X", never as "no effect".

THE CUT.  `--cut` defaults to the NATIONAL median of the index (50, since SDI is a 1-100
percentile rank), not the cohort median, and the cohort median is reported beside it.  This is
deliberate: the paper was already bitten once by exactly this (UK Biobank's Townsend is z-scored
against the national census, so the cohort mean is -1.02 and a split at zero is 29/71, not a
median split).  An academic hospital catchment is not a national sample, so a within-MGB median
split can mean "the more deprived half of an affluent catchment" and is comparable to nothing.
MEDIAN AND MEAN ARE BOTH TESTED, because for a skewed index they are different numbers and
picking one is otherwise an unexamined choice.  A right-skewed deprivation distribution puts the
mean ABOVE the median, so a mean split calls fewer people deprived and loads the deprived half
with the tail; a left-skewed one does the reverse.  `ses_cut_grid.csv` carries the national
median, the cohort median, the cohort mean and the 80th percentile as named rows beside the full
decile sweep, each with the ancestry AUROC in both halves, so a conclusion that holds at the
median and not at the mean is visible as a statement about the threshold rather than about
deprivation.  The chosen `--cut` steers arms B and C; the grid is what shows whether that choice
mattered.

    python -m ancestry ses --work /out/work --results /out/results --ses /data/sdi_by_patient.csv

Input: a csv with participant_id and one deprivation column (--ses-col, autodetected when it is
called sdi / sdi_score / SDI). Higher = more deprived. Arms A and D additionally need the
embeddings, which `score --save-embeddings` writes; without them A falls back to the measured
ECG trait panel and D to residualising the score itself, both labelled as the weaker test.

Writes results/ses/. Rows are per arm or per decile; no person appears, and the exported
deprivation axis is coarsened to deciles because a fine area-level score crossed with a small
ancestry stratum is a neighbourhood identifier.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
from ancestry import privacy as PV                                       # noqa: E402
from ancestry import lib as L                                            # noqa: E402

SES_COLS = ('sdi', 'sdi_score', 'sdi_pct', 'social_deprivation_index', 'ses', 'deprivation')
TRAIT_COLS = ['HR_bpm', 'PR_ms', 'QRS_ms', 'QT_ms', 'JT_ms', 'Tamp_V6', 'Ramp_I']
CALIPER_SD = 0.2          # the discovery caliper, kept so arm B is the same estimator
SEED = 2000
NFOLD = 5


# ----------------------------------------------------------------------------- input
def load_ses(path, col=None, id_col='participant_id'):
    """participant_id + one numeric deprivation column, from whatever the site exported."""
    D = L.read_table(path, what='--ses', dtype=str)
    idc = L.pick_column(D, id_col,
                        ('participant_id', 'subject_id', 'empi', 'patient_id', 'mrn',
                         'sample', 'sample_id', 'iid', 'id'),
                        what=f'--ses {path}', kind='participant id column',
                        flag='--ses-id-col')
    lower = {c.lower().strip(): c for c in D.columns}
    if col:
        if col not in D.columns:
            raise SystemExit(f'--ses-col {col!r} is not a column of {path}\n'
                             f'  columns present: {list(D.columns)[:15]}')
        vc = col
    else:
        vc = next((lower[c] for c in SES_COLS if c in lower), None)
        if vc is None:
            num = [c for c in D.columns if c != idc
                   and pd.to_numeric(D[c], errors='coerce').notna().mean() > 0.8]
            if len(num) == 1:
                vc = num[0]
            else:
                raise SystemExit(
                    f'--ses {path}: cannot tell which column holds the deprivation score.\n'
                    f'  numeric candidates: {num or "none"}\n'
                    f'  all columns: {list(D.columns)[:15]}\n'
                    f'  pass --ses-col <name>  (it is found automatically when called "sdi")')
    v = pd.to_numeric(D[vc], errors='coerce')
    if v.notna().sum() == 0:
        raise SystemExit(f'--ses {path}: column {vc!r} has no numeric values.\n'
                         f'  first few: {list(D[vc].head(5))}\n'
                         '  if the decimal separator is a comma, re-export with a point.')
    n_bad = int(v.isna().sum())
    if n_bad:
        print(f'  note: {n_bad:,} of {len(v):,} rows have no usable {vc!r} value and are dropped')
    out = pd.DataFrame({'participant_id': D[idc].astype(str), 'sdi': v})
    out = out[out.sdi.notna()].drop_duplicates('participant_id')
    if len(out) == 0:
        raise SystemExit(f'--ses {path}: no row survived -- the file has no usable rows')
    return out, vc


SPECS = ('national-median', 'national-mean', 'cohort-median', 'cohort-mean', 'q80')


def _looks_percentile(v):
    """Is this index a 1-100 national percentile rank, so that 50 means something off-site?"""
    return bool((np.nanmin(v) >= 0) and (np.nanmax(v) <= 100) and (np.nanmax(v) > 1.5))


def named_cuts(v):
    """The cuts a reader asks for by name -> [(label, spec, value), ...], median and mean both.

    Returned together rather than one at a time: the only way to show that a verdict does not
    turn on the choice is to report it at each of them.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    out = []
    if _looks_percentile(v):
        out.append(('national median (SDI 50)', 'national-median', 50.0))
    out += [('cohort median', 'cohort-median', float(np.median(v))),
            ('cohort mean', 'cohort-mean', float(np.mean(v))),
            ('cohort 80th pct', 'q80', float(np.percentile(v, 80)))]
    return out


def pick_cut(v, spec):
    """-> (cut, label, note).  `spec` is a number or one of SPECS (median and mean, cohort or
    national).  The mean is offered because a skewed index makes it a different split from the
    median, and which one is used changes who counts as deprived."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    looks_pct = _looks_percentile(v)
    if spec not in SPECS:
        try:
            return float(spec), f'{float(spec):g}', 'cut supplied on the command line'
        except ValueError:
            raise SystemExit(f'--cut {spec!r}: give a number, or one of '
                             + ' / '.join(SPECS))
    if spec == 'cohort-median':
        return float(np.median(v)), 'cohort median', \
            'the median OF THIS COHORT; not comparable across sites'
    if spec == 'cohort-mean':
        return float(np.mean(v)), 'cohort mean', \
            'the MEAN of this cohort, which sits above the median for a right-skewed index ' \
            'and so calls fewer people deprived; not comparable across sites'
    if spec == 'national-mean':
        if looks_pct:
            return 50.5, 'national mean (SDI 50.5)', \
                'the index is a 1-100 percentile rank, whose national mean is 50.5 -- within ' \
                'half a percentile of the national median, by construction'
        return float(np.mean(v)), 'cohort mean (fallback)', \
            'the values are not a national percentile rank, so no national mean exists and ' \
            'the cohort mean is used; say so in any caption'
    if spec == 'q80':
        return float(np.percentile(v, 80)), 'cohort 80th pct', \
            "the SDI literature's 'most deprived quintile', realised in this cohort"
    if looks_pct:
        return 50.0, 'national median (SDI 50)', \
            'the index is a national percentile rank, so 50 is the national median and means ' \
            'the same thing at every site'
    return float(np.median(v)), 'cohort median (fallback)', \
        'the values do not look like a 1-100 percentile rank, so no national anchor exists ' \
        'and the cohort median is used; say so in any caption'


# ----------------------------------------------------------------------------- statistics
def smd(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / s) if s > 0 else np.nan


def auroc_ci(y, p, n_boot=2000, rng=None, alpha=0.05):
    """AUROC with a stratified bootstrap interval (resample cases and controls separately)."""
    rng = rng or np.random.default_rng(SEED)
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    a = L.auroc(y, p)
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(i1) < PV.MIN_PER_CLASS or len(i0) < PV.MIN_PER_CLASS:
        return a, np.nan, np.nan, np.nan
    bs = np.empty(n_boot)
    for b in range(n_boot):
        j = np.concatenate([rng.choice(i1, len(i1), True), rng.choice(i0, len(i0), True)])
        bs[b] = L.auroc(y[j], p[j])
    lo, hi = np.nanpercentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return a, float(lo), float(hi), float(np.nanstd(bs, ddof=1))


def fold_blocks(E):
    """Yield (label, E_k) for a (N, k, D) stack or a bare (N, D) matrix."""
    if E is None:
        return
    if E.ndim == 2:
        yield 'fold1', E
        return
    for k in range(E.shape[1]):
        yield f'fold{k + 1}', np.ascontiguousarray(E[:, k, :])


def rank_avg(cols):
    """Mean of per-fold scores after ranking each to [0, 1].

    The five folds are different fine-tuned models, so their probe outputs are not on one scale
    and a plain mean is not meaningful; their RANKINGS are comparable, and AUROC only reads
    ranks. This is the same ensembling the ancestry score itself uses, applied one level down.
    """
    R = []
    for c in cols:
        c = np.asarray(c, float)
        r = np.full(len(c), np.nan)
        ok = np.isfinite(c)
        if ok.sum() == 0:
            continue
        r[ok] = stats.rankdata(c[ok]) / ok.sum()
        R.append(r)
    return np.nanmean(np.column_stack(R), axis=1) if R else None


def paired_delta_ci(y, p_a, p_b, n_boot=2000, rng=None, alpha=0.05):
    """AUROC(p_b) - AUROC(p_a) with a PAIRED stratified bootstrap (same people both times)."""
    rng = rng or np.random.default_rng(SEED)
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.isfinite(p_a) & np.isfinite(p_b)
    y, p_a, p_b = y[ok], np.asarray(p_a)[ok], np.asarray(p_b)[ok]
    d = L.auroc(y, p_b) - L.auroc(y, p_a)
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if min(len(i1), len(i0)) < PV.MIN_PER_CLASS:
        return d, np.nan, np.nan, np.nan
    bs = np.empty(n_boot)
    for b in range(n_boot):
        j = np.concatenate([rng.choice(i1, len(i1), True), rng.choice(i0, len(i0), True)])
        bs[b] = L.auroc(y[j], p_b[j]) - L.auroc(y[j], p_a[j])
    lo, hi = np.nanpercentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d), float(lo), float(hi), float(np.nanstd(bs, ddof=1))


def coupled_sham(grp, target, rng):
    """A variable with the SAME correlation to ancestry as deprivation has, and nothing else.

    This is the control that matters wherever deprivation and ancestry are tightly bound, which
    at a US hospital they are. Permuting the deprivation score destroys that coupling, so a
    permutation sham measures only what the MECHANICS cost and will sit near zero however
    confounded the real variable is. Matching on -- or erasing -- any variable this correlated
    with ancestry necessarily removes ancestry information too. So the question each arm actually
    has to answer is not "does adjusting for deprivation cost anything" but "does it cost more
    than adjusting for an equally ancestry-correlated variable that is socioeconomically
    meaningless". That is this.
    """
    g = np.asarray(grp, float)
    v = float(np.var(g))
    r = float(abs(target))
    if not np.isfinite(r) or r >= 0.999 or v <= 0:
        return rng.normal(size=len(g))
    a = r / np.sqrt(v * (1 - r ** 2))
    z = a * g + rng.normal(size=len(g))
    return z if target >= 0 else -z


def tost(delta, se, margin):
    """Equivalence P for H1: |delta| < margin. Small P = the effect is bounded by the margin."""
    if not np.isfinite(delta) or not np.isfinite(se) or se <= 0:
        return np.nan
    return float(max(stats.norm.cdf((delta - margin) / se),
                     stats.norm.cdf(-(delta + margin) / se)))


def row(arm, what, est, lo=np.nan, hi=np.nan, se=np.nan, n=None, margin=None, note=''):
    r = dict(arm=arm, quantity=what, estimate=_r(est), ci_lo=_r(lo), ci_hi=_r(hi), se=_r(se),
             n=n if n is None else int(n))
    if margin is not None:
        r['tost_margin'] = margin
        r['tost_p'] = _r(tost(est, se, margin), 4)
        r['bounded_below_margin'] = bool(np.isfinite(r['tost_p'] or np.nan)
                                         and (r['tost_p'] or 1) < 0.05)
        # The smallest margin this arm COULD have cleared at this n. Reporting it stops a
        # failure to clear 0.02 from being read as an effect: most of the time it means the
        # stratum was small, not that deprivation did something.
        r['achieved_bound'] = _r(abs(est) + 1.645 * se, 4) \
            if np.isfinite(est or np.nan) and np.isfinite(se or np.nan) else None
    r['note'] = note
    return r


def _r(x, k=5):
    try:
        return None if x is None or not np.isfinite(float(x)) else round(float(x), k)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- probe
def cv_probe(X, y, groups=None, n_fold=NFOLD, seed=SEED):
    """Out-of-fold AUROC of a standardised logistic probe. Returns (auroc, oof_scores)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    X = np.asarray(X, float); y = np.asarray(y, int)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if min(int((y == 1).sum()), int((y == 0).sum())) < max(PV.MIN_PER_CLASS, n_fold):
        return np.nan, None, int(len(y))
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(n_fold, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit((X[tr] - mu) / sd, y[tr])
        oof[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    return L.auroc(y, oof), oof, int(len(y))


# ----------------------------------------------------------------------------- matching
def greedy_match(D, cov, caliper_sd=CALIPER_SD, seed=SEED):
    """1:1 non-European to European, sex-exact, within `caliper_sd` SDs of `cov`.

    Non-European first because it is the smaller group; matching the larger one first strands the
    smaller group at the ends of the distribution. Identical in mechanics to the fairness arm's
    matcher and to the discovery matcher, so arm B here is the same estimator as arm B there.
    """
    s = np.nanstd(D[cov].to_numpy(float))
    if not np.isfinite(s) or s == 0:
        return None
    cal = caliper_sd * s
    keep = []
    for _sx, g in D.groupby('sex_num', dropna=True):
        a = g[g.nonEUR == 1].sample(frac=1, random_state=seed)
        b = g[g.nonEUR == 0]
        if len(a) == 0 or len(b) == 0:
            continue
        bv = b[cov].to_numpy(float)
        used = np.zeros(len(b), bool)
        for _, r in a.iterrows():
            d = np.abs(bv - float(r[cov]))
            d[used | ~np.isfinite(d)] = np.inf
            j = int(np.argmin(d))
            if not np.isfinite(d[j]) or d[j] > cal:
                continue
            used[j] = True
            keep += [r.name, b.index[j]]
    if len(keep) < 4 * PV.MIN_PER_CLASS:
        return None
    return D.loc[keep]


# ----------------------------------------------------------------------------- the arms
def arm_coupling(D, cut, res, rng):
    """How tightly is deprivation bound to ancestry here? Read this before anything else."""
    out = []
    eur = D.loc[D.nonEUR == 0, 'sdi'].to_numpy(float)
    for st in L.STRATA:
        g = D[D.stratum == st]
        if len(g) < PV.MIN_PER_CLASS:
            out.append(dict(stratum=st, n=len(g), note=PV.SUPPRESSED))
            continue
        v = g.sdi.to_numpy(float)
        out.append(dict(stratum=st, n=len(g),
                        mean_decile=_r(np.nanmean(g.sdi_decile.to_numpy(float)), 2),
                        median_decile=_r(np.nanmedian(g.sdi_decile.to_numpy(float)), 1),
                        pct_deprived=_r(100 * np.nanmean(v >= cut), 1),
                        smd_vs_eur=_r(smd(v, eur), 3) if st != 'EUR' else 0.0, note=''))
    C = pd.DataFrame(out)
    PV.write_table(C, res / 'ses_coupling.csv', where='ses_coupling')

    # the one number that decides how arms B-D may be read
    a, lo, hi, _se = auroc_ci(D.nonEUR.to_numpy(), D.sdi.to_numpy(float), 1000, rng)
    sp = stats.spearmanr(D.sdi.to_numpy(float), D.p_nonEUR.to_numpy(float),
                         nan_policy='omit')
    pr = L.partial(L.logit(D.p_nonEUR.to_numpy(float)), D.sdi.to_numpy(float),
                   np.column_stack([D.age.to_numpy(float), D.sex_num.to_numpy(float)]))
    link = dict(auroc_sdi_for_ancestry=_r(a), ci=[_r(lo), _r(hi)],
                smd_eur_vs_non=_r(smd(D.loc[D.nonEUR == 1, 'sdi'], eur), 3),
                spearman_sdi_vs_score=_r(sp.statistic, 4), spearman_p=_r(sp.pvalue, 6),
                beta_logit_score_per_sd_sdi=_r(pr['beta_sd'], 4), beta_p=_r(pr['p'], 6),
                delta_r2_pct=_r(100 * pr['delta_r2'], 3))
    return C, link


def arm_a(D, cut, work, use_emb, E, rng, margin):
    """A: can anything in the ECG see deprivation? Pooled, and inside Europeans only."""
    rows = []
    y = (D.sdi.to_numpy(float) >= cut).astype(int)
    feats, what = None, ''
    if use_emb and E is not None:
        feats, what = E, 'SSL embedding (768-d, mean over folds)'
    else:
        M = work / 'measurements.csv'
        if M.exists():
            T = pd.read_csv(M, dtype={'participant_id': str})
            T = D[['participant_id']].merge(T, on='participant_id', how='left')
            have = [c for c in TRAIT_COLS if c in T.columns]
            if have:
                feats, what = T[have].to_numpy(float), \
                    f'{len(have)} measured ECG traits ({", ".join(have)}) -- the weaker probe; ' \
                    'run `score --save-embeddings` for the embedding version'
    if feats is None:
        rows.append(row('A_visibility', 'AUROC for deprived', np.nan,
                        note='no features available: run `traits` or `score --save-embeddings`'))
        return rows, what
    blocks = list(fold_blocks(feats)) if isinstance(feats, np.ndarray) and feats.ndim == 3 \
        else [('', np.asarray(feats))]
    for scope, m in (('pooled', np.ones(len(D), bool)),
                     ('within EUR', (D.nonEUR == 0).to_numpy())):
        per, oofs, n = [], [], 0
        for _lab, Fk in blocks:
            a_k, oof, n = cv_probe(Fk[m], y[m])
            if oof is not None:
                per.append(a_k); oofs.append(oof)
        if not oofs:
            rows.append(row('A_visibility', f'AUROC for deprived ({scope})', np.nan, n=n,
                            note=PV.SUPPRESSED))
            continue
        ens = rank_avg(oofs)
        keep = np.isfinite(ens)
        a, lo, hi, se = auroc_ci(y[m][keep], ens[keep], 1000, rng)
        spread = (f'; per-fold {min(per):.3f}-{max(per):.3f}' if len(per) > 1 else '')
        rows.append(row('A_visibility', f'AUROC for deprived ({scope})', a, lo, hi, se, n,
                        note=(f'out-of-fold, {NFOLD}-fold, {len(oofs)} model(s) rank-averaged; '
                              + what + spread) if scope == 'pooled' else
                             'out-of-fold inside Europeans only -- the pooled probe can reach '
                             'this by reading ancestry, which is the thing being controlled '
                             'for, so THIS is the honest version' + spread))
    return rows, what


def arm_b(D, n_sham, rng, margin):
    """B: does the ancestry AUROC survive matching the groups on deprivation?"""
    rows = []
    full, flo, fhi, fse = auroc_ci(D.nonEUR.to_numpy(), D.p_nonEUR.to_numpy(float), 2000, rng)
    rows.append(row('B_matched', 'AUROC, full sample', full, flo, fhi, fse, len(D),
                    note='the reference the matched arm is compared with'))
    M = greedy_match(D, 'sdi', seed=SEED)
    if M is None:
        rows.append(row('B_matched', 'AUROC, matched on deprivation', np.nan,
                        note='matching produced too few pairs to report'))
        return rows, None
    m_a, mlo, mhi, mse = auroc_ci(M.nonEUR.to_numpy(), M.p_nonEUR.to_numpy(float), 2000, rng)
    bal_before = smd(D.loc[D.nonEUR == 1, 'sdi'], D.loc[D.nonEUR == 0, 'sdi'])
    bal_after = smd(M.loc[M.nonEUR == 1, 'sdi'], M.loc[M.nonEUR == 0, 'sdi'])
    rows.append(row('B_matched', 'AUROC, matched on deprivation', m_a, mlo, mhi, mse, len(M),
                    note=f'1:1 sex-exact, {CALIPER_SD} SD caliper; deprivation SMD '
                         f'{bal_before:+.3f} -> {bal_after:+.3f}'))
    delta = m_a - full
    dse = float(np.sqrt(mse ** 2 + fse ** 2)) if np.isfinite(mse) and np.isfinite(fse) else np.nan
    rows.append(row('B_matched', 'cost of matching (matched - full)', delta,
                    delta - 1.96 * dse, delta + 1.96 * dse, dse, len(M), margin=margin,
                    note='the matched set is a SUBSET of the full one, so this interval is indicative; the '
                         'sham distribution below is the inference. The discovery area-level '
                         'arm found a small negative cost of the same kind'))

    # SHAM: the same matcher, the same sample reduction, a permuted deprivation score.  If the
    # real drop sits inside this distribution it is the price of any 1:1 match of that size.
    sh = []
    for i in range(n_sham):
        Dp = D.copy()
        Dp['sdi'] = rng.permutation(Dp.sdi.to_numpy(float))
        Ms = greedy_match(Dp, 'sdi', seed=SEED + 1 + i)
        if Ms is not None and len(Ms) >= 4 * PV.MIN_PER_CLASS:
            sh.append(L.auroc(Ms.nonEUR.to_numpy(), Ms.p_nonEUR.to_numpy(float)) - full)
    sh = np.asarray([x for x in sh if np.isfinite(x)])
    if len(sh) >= 20:
        pct = float((sh <= delta).mean())
        rows.append(row('B_matched', 'sham matching, mean cost', float(sh.mean()),
                        float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5)),
                        float(sh.std(ddof=1)), None,
                        note=f'{len(sh)} permutations of the deprivation score through the same '
                             f'matcher; the real cost sits at the {100 * pct:.0f}th percentile '
                             f'of this distribution'
                             + (' -- i.e. indistinguishable from what any match of this size '
                                'costs' if 0.05 < pct < 0.95 else
                                ' -- i.e. outside what the mechanics alone explain')))
    # SHAM 2, the one that bites here: a variable with deprivation's ancestry correlation and
    # none of its meaning. If the real cost matches this, the cost is the ancestry correlation.
    r_couple = float(np.corrcoef(D.sdi.to_numpy(float), D.nonEUR.to_numpy(float))[0, 1])
    sh2 = []
    for i in range(n_sham):
        Dp = D.copy()
        Dp['sdi'] = coupled_sham(Dp.nonEUR.to_numpy(), r_couple, rng)
        Ms = greedy_match(Dp, 'sdi', seed=SEED + 10_000 + i)
        if Ms is not None and len(Ms) >= 4 * PV.MIN_PER_CLASS:
            sh2.append(L.auroc(Ms.nonEUR.to_numpy(), Ms.p_nonEUR.to_numpy(float)) - full)
    sh2 = np.asarray([x for x in sh2 if np.isfinite(x)])
    if len(sh2) >= 20:
        pct2 = float((sh2 <= delta).mean())
        rows.append(row('B_matched', 'coupling-matched sham, mean cost', float(sh2.mean()),
                        float(np.percentile(sh2, 2.5)), float(np.percentile(sh2, 97.5)),
                        float(sh2.std(ddof=1)), None,
                        note=f'{len(sh2)} draws of a meaningless variable carrying deprivation\'s '
                             f'own ancestry correlation (r = {r_couple:+.3f}); the real cost is '
                             f'at its {100 * pct2:.0f}th percentile'
                             + (' -- i.e. fully explained by that correlation, with nothing left '
                                'for socioeconomics' if 0.05 < pct2 < 0.95 else
                                ' -- i.e. deprivation costs MORE than its ancestry correlation '
                                'alone would')))

    # the same-size random subset band (R-39b's sample-loss control)
    n_eur = int((M.nonEUR == 0).sum()); n_non = int((M.nonEUR == 1).sum())
    rs = []
    ieur = np.flatnonzero((D.nonEUR == 0).to_numpy()); inon = np.flatnonzero((D.nonEUR == 1).to_numpy())
    for _ in range(min(n_sham, 200)):
        j = np.concatenate([rng.choice(ieur, min(n_eur, len(ieur)), False),
                            rng.choice(inon, min(n_non, len(inon)), False)])
        rs.append(L.auroc(D.nonEUR.to_numpy()[j], D.p_nonEUR.to_numpy(float)[j]))
    rs = np.asarray(rs)
    rows.append(row('B_matched', 'random same-size subset, AUROC', float(rs.mean()),
                    float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)),
                    float(rs.std(ddof=1)), len(M),
                    note='deprivation ignored, only the sample size matched -- separates "the '
                         'groups are now balanced" from "the sample is now smaller"'))
    return rows, M


def arm_c(D, cut, label, rng, margin):
    """C: does the score discriminate ancestry inside each half of the deprivation split?"""
    rows, halves = [], {}
    dep = D.sdi.to_numpy(float) >= cut
    for name, m in (('deprived', dep), ('not deprived', ~dep)):
        g = D[m]
        npos, nneg = int((g.nonEUR == 1).sum()), int((g.nonEUR == 0).sum())
        if min(npos, nneg) < PV.MIN_PER_CLASS:
            rows.append(row('C_halves', f'AUROC, {name}', np.nan, n=len(g), note=PV.SUPPRESSED))
            continue
        a, lo, hi, se = auroc_ci(g.nonEUR.to_numpy(), g.p_nonEUR.to_numpy(float), 2000, rng)
        halves[name] = (a, se)
        rows.append(row('C_halves', f'AUROC, {name}', a, lo, hi, se, len(g),
                        note=f'cut at {label}; {npos:,} non-European, {nneg:,} European'))
    if len(halves) == 2:
        (a1, s1), (a0, s0) = halves['deprived'], halves['not deprived']
        d = a1 - a0
        ds = float(np.sqrt(s1 ** 2 + s0 ** 2))
        rows.append(row('C_halves', 'difference (deprived - not)', d, d - 1.96 * ds,
                        d + 1.96 * ds, ds, len(D), margin=margin,
                        note='unpaired: the two halves are different people. This arm is '
                             'insensitive by construction -- deprivation still varies widely '
                             'INSIDE each half, so a real socioeconomic contribution survives '
                             'the split and shows up here as near zero. A null is expected '
                             'whatever the truth; arm B is what decides. It is unbiased but '
                             'noisy: simulated at strata of this size under a true null its '
                             'sampling spread is wide enough that a small difference here '
                             'should not be read as anything'))
    return rows


def arm_d(D, E, n_sham, rng, margin):
    """D: project deprivation out and re-measure ancestry. With a sham and a potency check."""
    rows = []
    y = D.nonEUR.to_numpy(int)
    cov = D.sdi.to_numpy(float)
    if E is None:
        # score-level fallback: residualise the score's logit on deprivation. A 1-d version of
        # the same idea and a genuinely weaker test -- it can only remove the part of the score
        # that is linear in SDI -- so it is labelled as such rather than called arm D.
        lg = L.logit(D.p_nonEUR.to_numpy(float))
        ok = np.isfinite(lg) & np.isfinite(cov)
        X = np.column_stack([np.ones(ok.sum()), cov[ok]])
        r = lg[ok] - X @ np.linalg.lstsq(X, lg[ok], rcond=None)[0]
        base, blo, bhi, bse = auroc_ci(y[ok], lg[ok], 2000, rng)
        a, lo, hi, se = auroc_ci(y[ok], r, 2000, rng)
        rows.append(row('D_adjust_score', 'AUROC, score as is', base, blo, bhi, bse, int(ok.sum())))
        rows.append(row('D_adjust_score', 'AUROC, score residualised on deprivation', a, lo, hi,
                        se, int(ok.sum())))
        d = a - base
        ds = float(np.sqrt(se ** 2 + bse ** 2))
        rows.append(row('D_adjust_score', 'cost of removing deprivation', d, d - 1.96 * ds,
                        d + 1.96 * ds, ds, int(ok.sum()), margin=margin,
                        note='ONE dimension removed from the score, not a direction removed from '
                             'the embedding: weaker than the paper\'s arm D. Run '
                             '`score --save-embeddings` for the real thing. This arm is BIASED '
                             'downward wherever deprivation is ancestry-correlated -- compare it '
                             'with the coupling-matched sham below, never with zero'))
        # The same coupled sham as the embedding path. Without it this arm is unreadable: in a
        # simulation at this scale with NO socioeconomic path at all, residualising the score
        # on an ancestry-correlated deprivation measure still cost a non-trivial amount of
        # AUROC. All of that is the correlation. A site reading the raw number against zero would report an effect that is
        # not there.
        r_c = float(np.corrcoef(cov[ok], y[ok].astype(float))[0, 1])
        shs = []
        for _ in range(max(20, min(n_sham, 200))):
            z = coupled_sham(y[ok], r_c, rng)
            Xz = np.column_stack([np.ones(ok.sum()), z])
            rz = lg[ok] - Xz @ np.linalg.lstsq(Xz, lg[ok], rcond=None)[0]
            shs.append(L.auroc(y[ok], rz) - base)
        shs = np.asarray([x for x in shs if np.isfinite(x)])
        if len(shs) >= 10:
            pct = float((shs <= d).mean())
            rows.append(row('D_adjust_score', 'coupling-matched sham, mean cost',
                            float(shs.mean()), float(np.percentile(shs, 2.5)),
                            float(np.percentile(shs, 97.5)), float(shs.std(ddof=1)), None,
                            note=f'{len(shs)} meaningless variables with deprivation\'s own '
                                 f'ancestry correlation (r = {r_c:+.3f}); the real adjustment is '
                                 f'at their {100 * pct:.0f}th percentile'
                                 + (' -- i.e. the whole cost is that correlation'
                                    if 0.05 < pct < 0.95 else
                                    ' -- i.e. more than the correlation alone')))
        return rows
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    def oof(Ek, resid=None, target=None):
        """One fold-model's embedding -> out-of-fold probe scores for `target` (default ancestry).

        `resid`, if given, is the covariate whose direction is projected out. The projection is
        fitted on the TRAINING rows only and applied to both, so neither the erasure nor the
        probe ever sees a held-out person. Fitting it on everything -- which is the obvious way
        to write it -- leaks, and leaks in the direction of over-removal: it can push the
        potency check BELOW chance, which then reads as an error rather than as a result.
        """
        t = y if target is None else np.asarray(target, int)
        out = np.full(len(t), np.nan)
        for tr, te in StratifiedKFold(NFOLD, shuffle=True, random_state=SEED).split(Ek, t):
            Etr, Ete = Ek[tr], Ek[te]
            if resid is not None:
                X = np.column_stack([np.ones(len(tr)), resid[tr]])
                b = np.linalg.lstsq(X, Etr, rcond=None)[0]
                Etr = Etr - X @ b
                Ete = Ete - np.column_stack([np.ones(len(te)), resid[te]]) @ b
            mu, sd = Etr.mean(0), Etr.std(0)
            sd[sd == 0] = 1.0
            m = LogisticRegression(max_iter=1000, C=1.0)
            m.fit((Etr - mu) / sd, t[tr])
            out[te] = m.predict_proba((Ete - mu) / sd)[:, 1]
        return out

    blocks = list(fold_blocks(E))
    base_o = rank_avg([oof(Ek) for _l, Ek in blocks])
    eras_o = rank_avg([oof(Ek, cov) for _l, Ek in blocks])
    b_a, blo, bhi, bse = auroc_ci(y, base_o, 2000, rng)
    e_a, elo, ehi, ese = auroc_ci(y, eras_o, 2000, rng)
    rows.append(row('D_erasure', 'AUROC, embedding as is', b_a, blo, bhi, bse, len(y),
                    note=f'ancestry probe on the SSL embedding, out of fold, '
                         f'{len(blocks)} fold-model(s) rank-averaged'))
    rows.append(row('D_erasure', 'AUROC, deprivation projected out', e_a, elo, ehi, ese, len(y)))
    d, dlo, dhi, dse = paired_delta_ci(y, base_o, eras_o, 2000, rng)
    rows.append(row('D_erasure', 'cost of the erasure', d, dlo, dhi, dse, len(y),
                    margin=margin,
                    note='paired bootstrap, same people before and after. READ THE LIMIT: this '
                         'deletes ONE direction from a representation that also carries an '
                         'independent ancestry axis, so it can only show whether the ancestry '
                         'information LIVES IN the deprivation direction -- never whether '
                         'deprivation contributes to the score. The discovery control found '
                         'the same arm barely moved an unrelated covariate in a far larger '
                         'sample. Arm B is the sensitive arm'))

    # SHAM: random directions, the same rank-1 deletion. Without this, "we deleted one of 768
    # directions and nothing changed" is not a finding -- it is what deleting any direction does.
    n_s = max(20, min(n_sham, 200))
    sh = []
    for _ in range(n_s):
        r = rng.normal(size=len(y))
        sh.append(L.auroc(y, rank_avg([oof(Ek, r) for _l, Ek in blocks])) - b_a)
    sh = np.asarray([x for x in sh if np.isfinite(x)])
    if len(sh) >= 10:
        pct = float((sh <= d).mean())
        rows.append(row('D_erasure', 'sham erasure, mean cost', float(sh.mean()),
                        float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5)),
                        float(sh.std(ddof=1)), None,
                        note=f'{len(sh)} random directions through the same machinery; the real '
                             f'erasure sits at the {100 * pct:.0f}th percentile of them'
                             + (' -- indistinguishable from deleting a random direction'
                                if 0.05 < pct < 0.95 else
                                ' -- outside what a random deletion explains')))

    # SHAM 2: a direction with deprivation's own ancestry correlation and none of its meaning.
    # Under tight coupling this, not the random direction, is the null that decides arm D:
    # erasing ANY variable this correlated with ancestry must cost ancestry AUROC.
    r_couple = float(np.corrcoef(cov, y.astype(float))[0, 1])
    sh2 = []
    for _ in range(n_s):
        z = coupled_sham(y, r_couple, rng)
        sh2.append(L.auroc(y, rank_avg([oof(Ek, z) for _l, Ek in blocks])) - b_a)
    sh2 = np.asarray([x for x in sh2 if np.isfinite(x)])
    if len(sh2) >= 10:
        pct2 = float((sh2 <= d).mean())
        rows.append(row('D_erasure', 'coupling-matched sham, mean cost', float(sh2.mean()),
                        float(np.percentile(sh2, 2.5)), float(np.percentile(sh2, 97.5)),
                        float(sh2.std(ddof=1)), None,
                        note=f'{len(sh2)} meaningless variables carrying deprivation\'s own '
                             f'ancestry correlation (r = {r_couple:+.3f}); the real erasure is at '
                             f'their {100 * pct2:.0f}th percentile'
                             + (' -- i.e. the attenuation IS that correlation, and nothing is '
                                'left for socioeconomics' if 0.05 < pct2 < 0.95 else
                                ' -- i.e. erasing deprivation costs more than its ancestry '
                                'correlation alone')))

    # POTENCY: did the erasure actually remove deprivation? If an SDI probe still reads
    # deprivation off the residual embedding, the null above is uninformative rather than
    # reassuring. This is the control the paper's own arm D was missing.
    yd = (cov >= np.median(cov)).astype(int)
    if min(int(yd.sum()), int((1 - yd).sum())) >= PV.MIN_PER_CLASS:
        p0 = rank_avg([oof(Ek, None, yd) for _l, Ek in blocks])
        p1 = rank_avg([oof(Ek, cov, yd) for _l, Ek in blocks])
        a_before, lo0, hi0, _s0 = auroc_ci(yd, p0, 1000, rng)
        a_after, lo1, hi1, _s1 = auroc_ci(yd, p1, 1000, rng)
        rows.append(row('D_potency', 'deprivation AUROC before erasure', a_before, lo0, hi0,
                        n=int(len(yd)),
                        note='how much deprivation the embedding carried to begin with -- the '
                             'ceiling on what any of these arms could have found'))
        rows.append(row('D_potency', 'deprivation AUROC after erasure', a_after, lo1, hi1,
                        n=int(len(yd)),
                        note='an erasure that worked drives this to 0.5. If it stays well above, '
                             'little was removed and the arm-D null says little'))

    # POSITIVE CONTROL: erase AGE, which the score is known to track at this site, and show the
    # machinery CAN move the ancestry number. A pipeline that cannot move it has not been shown
    # to be sensitive to anything.
    agev = D.age.to_numpy(float)
    if np.isfinite(agev).sum() > 0.8 * len(agev):
        age_o = rank_avg([oof(Ek, np.nan_to_num(agev, nan=float(np.nanmedian(agev))))
                          for _l, Ek in blocks])
        da, dalo, dahi, dase = paired_delta_ci(y, base_o, age_o, 1000, rng)
        rows.append(row('D_control', 'cost of erasing AGE instead', da, dalo, dahi, dase, len(y),
                        note='a covariate the score demonstrably tracks; this is what a '
                             'consequential erasure looks like in these numbers'))
    return rows


def cut_grid(D, res, rng):
    """Arm C swept over every decile cut AND evaluated at each named cut.

    `basis` says which a row is: `decile sweep` for the nine deciles, or the spec name for the
    national median, the cohort median, the cohort MEAN and the 80th percentile. The median and
    the mean are both here deliberately -- they are the same number only for a symmetric index.
    """
    v = D.sdi.to_numpy(float)
    rows = [(f'{q}th pct', 'decile sweep', float(np.percentile(v, q))) for q in range(10, 100, 10)]
    for lab, spec, c in named_cuts(v):
        # the percentile this named cut lands on, so it can be placed on the swept axis
        rows.append((lab, spec, c))
    out = []
    for lab, basis, c in rows:
        dep = v >= c
        q = float(100.0 * np.mean(v < c))
        r = dict(cut=lab, basis=basis, pctile=_r(q, 1), cut_value=_r(c, 3),
                 n_deprived=int(dep.sum()), pct_deprived=_r(100 * float(np.mean(dep)), 1))
        for name, m in (('deprived', dep), ('not_deprived', ~dep)):
            g = D[m]
            if min(int((g.nonEUR == 1).sum()), int((g.nonEUR == 0).sum())) < PV.MIN_PER_CLASS:
                r[f'auroc_{name}'] = None
            else:
                r[f'auroc_{name}'] = _r(L.auroc(g.nonEUR.to_numpy(),
                                                g.p_nonEUR.to_numpy(float)), 4)
        if r['auroc_deprived'] and r['auroc_not_deprived']:
            r['difference'] = _r(r['auroc_deprived'] - r['auroc_not_deprived'], 4)
        out.append(r)
    G = pd.DataFrame(out)
    PV.write_table(G, res / 'ses_cut_grid.csv', where='ses_cut_grid')
    return G


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry ses', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True,
                    help="the site's working directory: intermediates, per-person tables and "
                         'arrays. It stays at the site and is never packaged')
    ap.add_argument('--results', required=True,
                    help='the aggregate-only output directory; `package` zips exactly this')
    ap.add_argument('--ses', required=True,
                    help='csv/tsv with participant_id and a deprivation score (higher = more '
                         'deprived). The Social Deprivation Index is the intended input')
    ap.add_argument('--ses-col', default=None, help='the deprivation column (autodetected)')
    ap.add_argument('--ses-id-col', default='participant_id',
                    help='the id column of --ses, to join it to the roster')
    ap.add_argument('--ses-name', default='Social Deprivation Index',
                    help='what the measure is called, for the report')
    ap.add_argument('--ses-geography', default='census tract',
                    help='the unit the score is assigned at (census tract / ZCTA / county). '
                         'Recorded in the provenance: an area-level measure is not an '
                         'individual one and the interpretation depends on which it is')
    ap.add_argument('--ses-vintage', default='', help='e.g. "ACS 2015-2019"')
    ap.add_argument('--cut', default='national-median', metavar='SPEC',
                    help='national-median (default) | national-mean | cohort-median | '
                         'cohort-mean | q80 | a number. Whichever is chosen, ses_cut_grid.csv '
                         'reports every one of them, so median and mean are always both tested')
    ap.add_argument('--margin', type=float, default=0.02,
                    help='equivalence margin in AUROC (default 0.02, the Fig. 3 margin)')
    ap.add_argument('--n-sham', type=int, default=200,
                    help='coupling-matched sham variables per arm: the null this test is read '
                         'against (default 200)')
    ap.add_argument('--embeddings', default=None,
                    help='default work/embeddings.npy from `score --save-embeddings`')
    ap.add_argument('--no-embeddings', action='store_true',
                    help='ignore the embeddings even if present (fast arms B and C only)')
    ap.add_argument('--seed', type=int, default=SEED,
                    help='random seed for the shams and the bootstrap')
    a = ap.parse_args()

    work = Path(a.work); res = Path(a.results) / 'ses'
    L.require('ses', [(work / 'scores.csv', 'the ancestry scores and the ancestry strata',
                       'score')])
    res.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    L.banner(f'ses: is the ancestry signal explained by {a.ses_name}?')

    S = pd.read_csv(work / 'scores.csv', dtype={'participant_id': str})
    V, vcol = load_ses(a.ses, a.ses_col, a.ses_id_col)
    print(f'  deprivation: column {vcol!r} from {a.ses}, {len(V):,} people '
          f'({a.ses_geography}{", " + a.ses_vintage if a.ses_vintage else ""})')
    D, how = L.join_or_canon(S, V, 'participant_id', what='--ses')
    print(f'  joined to the roster on {how} ids: {len(D):,} of {len(S):,} scored people have a '
          f'deprivation score')
    D['sex_num'] = D.sex.astype(str).str.upper().map(
        {'M': 1, 'MALE': 1, '1': 1, 'F': 0, 'FEMALE': 0, '0': 0})
    D = D[D.sdi.notna() & D.p_nonEUR.notna()].copy()
    # ADMIXED is neither group, exactly as in the fairness audit, and is counted not analysed
    n_adm = int((D.stratum == 'ADMIXED').sum())
    D['nonEUR'] = np.where(D.stratum == 'EUR', 0,
                           np.where(D.stratum.isin(['AFR', 'EAS', 'SAS']), 1, -1))
    D = D[D.nonEUR >= 0].copy()
    if len(D) < PV.MIN_STRATUM or min(int((D.nonEUR == 1).sum()),
                                      int((D.nonEUR == 0).sum())) < PV.MIN_PER_CLASS:
        raise SystemExit(f'only {len(D):,} people have both a score and a deprivation value, '
                         f'with too few in one ancestry group: nothing can be reported')
    cut, label, cutnote = pick_cut(D.sdi.to_numpy(float), a.cut)
    D['sdi_decile'] = pd.qcut(D.sdi.rank(method='first'), 10, labels=False,
                              duplicates='drop') + 1
    print(f'  {len(D):,} analysed ({int((D.nonEUR == 1).sum()):,} non-European, '
          f'{int((D.nonEUR == 0).sum()):,} European; {n_adm:,} admixed excluded)')
    print(f'  cut: {label} = {cut:g}  ({int((D.sdi >= cut).sum()):,} deprived)\n    {cutnote}')
    # median and mean side by side: if these differ, so does who counts as deprived, and
    # ses_cut_grid.csv is where the verdict is re-read at each of them.
    for lab, _spec, c in named_cuts(D.sdi.to_numpy(float)):
        mark = '  <- in use' if abs(c - cut) < 1e-9 else ''
        print(f'    {lab:26} {c:8.3g}   {int((D.sdi >= c).sum()):>7,} deprived'
              f'  ({100 * float((D.sdi >= c).mean()):4.1f}%){mark}')

    E = None
    if not a.no_embeddings:
        ep = Path(a.embeddings) if a.embeddings else work / 'embeddings.npy'
        ei = ep.with_name('embeddings_ids.csv')
        if ep.exists() and ei.exists():
            Eall = np.load(ep, mmap_mode='r')
            ids = pd.read_csv(ei, dtype={'participant_id': str})
            ids['_i'] = np.arange(len(ids))
            j = D[['participant_id']].merge(ids, on='participant_id', how='left')
            m = j._i.notna().to_numpy()
            if m.mean() > 0.8:
                D = D[m].reset_index(drop=True)
                E = np.asarray(Eall[j._i[m].astype(int).to_numpy()], dtype=np.float32)
                nf = 1 if E.ndim == 2 else E.shape[1]
                print(f'  embeddings: {len(E):,} people x {nf} fold-model(s) x {E.shape[-1]} '
                      '-- arms A and D run on the embedding, per fold')
            else:
                print(f'  embeddings found but only {100 * m.mean():.0f}% of people matched; '
                      'falling back to the trait probe and the score-level adjustment')
        else:
            print(f'  no embeddings at {ep} -- arm A falls back to the measured trait panel and '
                  "arm D to residualising the score (run `score --save-embeddings` for the "
                  'embedding versions)')

    # ------------------------------------------------------------------ arms
    C, link = arm_coupling(D, cut, res, rng)
    print('\n  --- how tightly is deprivation bound to ancestry here? read this first ---')
    print(C.to_string(index=False))
    print(f"\n  deprivation alone separates the ancestry groups at AUROC "
          f"{link['auroc_sdi_for_ancestry']}, SMD {link['smd_eur_vs_non']}")

    rows = []
    ra, what = arm_a(D, cut, work, E is not None, E, rng, a.margin)
    rows += ra
    rb, M = arm_b(D, a.n_sham, rng, a.margin)
    rows += rb
    rows += arm_c(D, cut, label, rng, a.margin)
    rows += arm_d(D, E, a.n_sham, rng, a.margin)
    Q = pd.DataFrame(rows)
    Q['n'] = Q.n.astype('Int64')          # sham rows have no n; keep the column integral
    PV.write_table(Q, res / 'ses_quad.csv', where='ses_quad')
    print('\n' + Q[['arm', 'quantity', 'estimate', 'ci_lo', 'ci_hi', 'n']].to_string(index=False))

    # distribution, coarsened to deciles because a fine area score in a small stratum is a place
    dist = (D.groupby(['stratum', 'sdi_decile']).size().rename('n').reset_index())
    dist.loc[dist.n < PV.MIN_PER_CLASS, 'n'] = np.nan
    PV.write_table(dist.rename(columns={'sdi_decile': 'decile'}),
                   res / 'ses_distribution.csv', where='ses_distribution')
    G = cut_grid(D, res, rng)

    # ------------------------------------------------------------------ the verdict
    def get(arm, startswith):
        q = Q[(Q.arm == arm) & (Q.quantity.str.startswith(startswith))]
        if not len(q):
            return None, None, None
        r = q.iloc[0]
        return (None if r.estimate is None else float(r.estimate),
                None if r.ci_hi is None else float(r.ci_hi), r.note)

    def bound_of(arm, startswith):
        q = Q[(Q.arm == arm) & (Q.quantity.str.startswith(startswith))]
        if not len(q) or 'achieved_bound' not in q or q.achieved_bound.iloc[0] is None:
            return None
        return float(q.achieved_bound.iloc[0])

    coup = link['auroc_sdi_for_ancestry'] or 0.5
    vis, _vh, _vn = get('A_visibility', 'AUROC for deprived (within EUR)')
    bcost, bhi, bnote = get('B_matched', 'cost of matching')
    bsham, _x, _y = get('B_matched', 'coupling-matched sham')
    dcost, _dh, _dn = get('D_erasure', 'cost of the erasure')
    dsham, _x2, _y2 = get('D_erasure', 'coupling-matched sham')
    if dcost is None:
        dcost, _dh, _dn = get('D_adjust_score', 'cost of removing deprivation')
        dsham, _x2, _y2 = get('D_adjust_score', 'coupling-matched sham')
    pot_b, _x3, _y3 = get('D_potency', 'deprivation AUROC before')
    pot_a, _x4, _y4 = get('D_potency', 'deprivation AUROC after')

    head = 'INCONCLUSIVE -- arm B did not run'
    ab = bound_of('B_matched', 'cost of matching')
    if bcost is not None:
        if abs(bcost) < a.margin and bhi is not None and abs(bhi) < a.margin:
            head = (f'BOUNDED: balancing the ancestry groups on deprivation moves the signal by '
                    f'{bcost:+.4f} AUROC, bounded below {a.margin}. Deprivation does not '
                    f'explain the ECG ancestry signal here.')
        elif abs(bcost) < a.margin:
            head = (f'SMALL BUT NOT PROVABLY BOUNDED at {a.margin}: the estimate is '
                    f'{bcost:+.4f} AUROC -- well inside the margin -- but this sample can only '
                    f'bound the effect at {ab:.4f}. That is a POWER limit, not an effect. '
                    f'Report the estimate and the bound together; do not report a null.')
        else:
            head = (f'NOT BOUNDED: balancing on deprivation moves the signal by {bcost:+.4f} '
                    f'AUROC. This is NOT a null -- report the estimate, and read arm B\'s sham '
                    f'rows before attributing it to socioeconomics.')
    lines = [f'{a.ses_name} ({a.ses_geography}) vs the ECG ancestry signal at this site', '',
             head, '']
    tight = coup >= 0.70
    if tight:
        lines += [f'1. COUPLING. Deprivation alone separates the ancestry groups at AUROC '
                  f'{coup:.3f} (SMD {link["smd_eur_vs_non"]}). It is nearly an ancestry proxy '
                  f'here, so adjusting for it necessarily removes ancestry information. Every '
                  f'arm below is therefore read against a sham carrying the SAME ancestry '
                  f'correlation and no socioeconomic meaning -- a plain null would be '
                  f'uninterpretable.']
    else:
        lines += [f'1. COUPLING. Deprivation and ancestry are only loosely bound here (AUROC '
                  f'{coup:.3f}), so the arms below are not at risk of over-adjustment.']
    if vis is not None:
        lines += ['', f'2. CEILING. Inside Europeans, the ECG predicts deprivation at AUROC '
                      f'{vis:.3f}. That is the most socioeconomic signal there is to find in '
                      f'these waveforms' + (' -- barely above chance, so there is little for '
                                            'the arms below to remove.' if vis < 0.60 else '.')]
    if bcost is not None:
        lines += ['', f'3. MATCHING (arm B). Balancing the groups on deprivation moves the '
                      f'ancestry AUROC by {bcost:+.4f}'
                      + (f', against {bsham:+.4f} for the coupling-matched sham.'
                         if bsham is not None else '.')]
        if bsham is not None and abs(bcost - bsham) < a.margin:
            lines += [f'   -> the cost is what ANY equally ancestry-correlated variable costs. '
                      f'Nothing is left for socioeconomics.']
        if ab is not None:
            lines += [f'   this sample bounds the socioeconomic contribution at '
                      f'{ab:.4f} AUROC (the smallest margin it could have cleared).']
        if np.isfinite(bhi or np.nan) and abs(bcost) < a.margin and abs(bhi) < a.margin:
            lines += [f'   -> BOUNDED below the {a.margin} margin.']
        elif abs(bcost) < a.margin:
            lines += [f'   -> the estimate is inside {a.margin} but the interval is not: this '
                      f'is a power limit, not an effect.']
        else:
            lines += [f'   -> NOT bounded at {a.margin}: report the estimate and its interval.']
    if dcost is not None:
        lines += ['', f'4. ERASURE (arm D). Projecting deprivation out of the embedding moves '
                      f'the ancestry AUROC by {dcost:+.4f}'
                      + (f', against {dsham:+.4f} for the coupling-matched sham.'
                         if dsham is not None else '.')]
        if dsham is not None and abs(dcost - dsham) < a.margin:
            lines += ['   -> indistinguishable from erasing any equally ancestry-correlated '
                      'direction: the attenuation is the correlation, not socioeconomics.']
        elif dsham is not None and dcost < dsham - a.margin:
            lines += ['   -> deprivation costs MORE than its ancestry correlation alone. This is '
                      'the one pattern here that would support a socioeconomic contribution; '
                      'report it as such.']
        if dsham is not None and abs(dcost) < a.margin and abs(dsham) < a.margin:
            lines += ['   -> but note this arm deletes one direction from a representation that '
                      'carries an independent ancestry axis, so a null here is close to '
                      'guaranteed and carries little information either way. It is reported for '
                      "comparability with the paper's arm D, not as evidence."]
        if pot_b is not None and pot_a is not None:
            lines += [f'   potency: the embedding read deprivation at {pot_b:.3f} before the '
                      f'erasure and {pot_a:.3f} after'
                      + ('. The erasure worked, so the arm is informative.'
                         if pot_a < 0.60 else
                         '. Little was removed, so this arm says little either way.')]
    lines += ['', 'WHAT THIS CAN AND CANNOT SHOW. Every arm reports a bound, not a zero: at this '
                  f'n an AUROC difference is resolvable to roughly the {a.margin} margin, '
                  'which is larger than the cost the discovery area-level arm measured. And an '
                  'area-level index is an ecological proxy: its measurement error pushes every '
                  'arm toward the null, so a null here is weaker evidence than a null from an '
                  'individual measure -- in discovery, the imbalance that was large on the '
                  'area-level index was absent on individual household income. Read the two '
                  'together.',
              '', 'Tables: ses_quad.csv (all arms), ses_coupling.csv (the interpretive key), '
                  'ses_cut_grid.csv (does anything depend on the threshold), '
                  'ses_distribution.csv.']
    (res / 'ses_verdict.txt').write_text('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))

    prov = dict(version=L.VERSION, measure=a.ses_name, geography=a.ses_geography,
                vintage=a.ses_vintage, column=vcol, source=str(a.ses), id_join=how,
                n_joined=int(len(D)), n_admixed_excluded=n_adm,
                cut=dict(value=cut, label=label, note=cutnote, spec=a.cut),
                cohort_median=_r(float(np.median(D.sdi.to_numpy(float))), 3),
                margin=a.margin, n_sham=a.n_sham, seed=a.seed,
                features_arm_a=what, embeddings=(None if E is None else list(E.shape)),
                coupling=link,
                caveat='an area-level score is an ecological proxy for a person\'s '
                       'circumstances; its measurement error attenuates every arm toward the '
                       'null, so a null here is weaker evidence than a null from an individual '
                       'measure (in discovery the imbalance that was large on the area-level '
                       'index was absent on individual household income)')
    (res / 'ses_provenance.json').write_text(json.dumps(PV.mask(prov), indent=2))
    print(f'\n  wrote {res}/  (5 tables + the verdict)')


if __name__ == '__main__':
    main()
