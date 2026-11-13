"""Mode `pgs` -- the polygenic scores: the positive control first, then the ancestry gradient.

THREE ARMS, IN THE ORDER THEY HAVE TO BE READ

  1. POSITIVE CONTROL.  Each of the nine ECG-trait polygenic scores must predict ITS OWN measured
     ECG trait here before it is allowed to say anything about ancestry.  A QT score that does not
     predict QT at MGB is not a QT score at MGB -- it is a genotyping, liftover or allele-matching
     failure wearing a trait's name, and every downstream number built on it would be noise with a
     label.  This arm is the gate, and the package reports it first whatever it says.

     The measurements come from NeuroKit2 on the site's own waveforms (mode `traits`) and/or the
     site's own cart panel via --measurements.  These are DIFFERENT instruments: NeuroKit2 and a
     GE/Philips measurement module agree well on QRS, QT and heart rate and much less well on PR
     and on wave amplitudes (the per-measurement agreement we observed ships as
     ref/neurokit_vs_device.csv).  So the control is judged on SIGN and on clearing t = 3, not on
     reproducing a discovery effect size -- and WHICH ruler it used is reported with every row.

     --measurement-source picks between them, and `both` is the default and the recommendation:
       both      one control row per ruler per trait, printed side by side with a verdict.  This
                 is the only setting that separates the two failures that look alike -- a trait
                 that clears t = 3 on one ruler and not the other is the instruments disagreeing,
                 while a trait that fails on BOTH is genotyping, liftover or allele matching and
                 does invalidate that trait's downstream numbers.
       device    the cart panel only.  A trait the panel does not carry is NOT dropped: it is
                 filled from NeuroKit, `source` reads 'neurokit (fill)' and `note` says why.
       neurokit  the measurements taken here, with the panel as the same per-trait fill.
     --measurement-map remaps the site's own column names and may pin a trait to one ruler.  It
     describes the site's panel only; the NeuroKit table is this image's output and carries this
     image's names.

  2. THE FIGURE-2e ANALOGUE.  Mean predicted P(non-European) by decile of each trait PGS, inside
     the European stratum, on the ORIENTED decile -- the decile of w x PGS, where w is the frozen
     composite weight.  A trait whose score predicts a LOWER ancestry logit enters the composite
     with a negative weight, so its descent on its own raw decile IS its ascending contribution;
     plotting the oriented decile makes the composite the envelope of the fan instead of making
     four of the nine look like they disagree.  The composite is NOT refitted here: it was fitted
     in five folds inside the discovery cohorts and the mean weight is applied as a constant,
     because a refitted composite would be a different estimator from the published one.

  3. THE ANCESTRY-SCORE PGS.  The discovery work built a polygenic score FOR the model's output and showed it
     predicts that output out of sample within Europeans.  In discovery that test had to be
     leave-one-cohort-out, because every phenotyped person is inside the meta-analysis and
     an in-sample score would be circular.  MGB-MGH is in no arm of that meta, so here the
     full-strength three-cohort weights can be used and the test is clean by construction.  This
     is the one arm of this image that is a genuinely stronger test than the one in the paper.

     Clumping happens HERE, in the site's own genotypes (r2 < 0.1 within 500 kb, by our p-values),
     because LD is a property of the target cohort.  It uses no phenotype, so it leaks nothing.

    python -m ancestry pgs --work /out/work --results /out/results \\
        [--measurements /path/device_panel.csv] [--strata EUR AFR EAS SAS] [--threads 8]

Writes results/pgs/.  Rows are per stratum, per trait or per decile; no person appears.
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
from ancestry import geno_ops as G                                       # noqa: E402

# Which measured column is each trait's own trait.  Two of the nine have no local measurement and
# say so rather than being matched to the nearest thing: the spatial QRS-T angle needs a vectorcar-
# diogram, and "QRS amplitude" in the discovery panel is the device's R amplitude in lead I, which
# NeuroKit2 reproduces well (r = 0.95) and is used here under that name.
TRAIT_MEASURE = {
    'HR': 'HR_bpm', 'PR': 'PR_ms', 'QRS': 'QRS_ms', 'QT': 'QT_ms', 'JT': 'JT_ms',
    'QRSTangleF': 'QRSTangleF_deg', 'Tamp': 'Tamp_V6', 'QRSamp': 'Ramp_I',
    'QRSTangle': None,
}
THRESH = [5e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0]
NDEC = 10


# ------------------------------------------------------------------------------ weight files
def site_weights(V, chrom, pos, allele, other_allele, beta, name, out):
    """A plink2 --score weight file keyed on the SITE's own variant ids.

    Our tables are keyed on position; the site's bim is keyed on whatever its imputation server
    wrote. `variant_map.csv` holds the join, so the weight file is built against ids that exist in
    the bim rather than against ids we hope exist.
    """
    # Convert every grouped Series to a positional array first.  Group indices are inherited
    # from the full weight table and are non-contiguous; mixing those indices with a freshly
    # derived NumPy `other_allele` array makes pandas align unrelated rows and inject NaNs.
    W = pd.DataFrame({
        'CHR': pd.Series(np.asarray(chrom)).astype(str).str.replace('chr', '', regex=False),
        'POS': pd.to_numeric(pd.Series(np.asarray(pos)), errors='coerce').astype('Int64'),
        'A1': pd.Series(np.asarray(allele)).astype(str).str.upper(),
        'A0': pd.Series(np.asarray(other_allele)).astype(str).str.upper(),
        'BETA': pd.to_numeric(pd.Series(np.asarray(beta)), errors='coerce')})
    M = W.merge(V[['CHR', 'POS', 'vid', 'a1', 'a2']], on=['CHR', 'POS'], how='inner')
    ok = L.allele_sign(M.a1, M.a2, M.A1, M.A0)
    M = M[np.isfinite(ok)]
    M[['vid', 'A1', 'BETA']].rename(columns={'vid': 'ID'}).to_csv(out, sep='\t', index=False)
    return len(M), len(W)


def zs(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else np.zeros_like(x)


def deciles(x, k=NDEC):
    """Equal-count bins; ties keep the bins monotone and may make them unequal, which is correct."""
    r = pd.Series(x).rank(method='first')
    return np.ceil(k * r / (len(r) + 1e-9)).astype(int).clip(1, k).to_numpy()


def decile_table(d, score_col, label, stratum, trait=None, weight=None):
    out = []
    dec = deciles(d[score_col].to_numpy())
    orient = dec if (weight is None or weight >= 0) else (NDEC + 1 - dec)
    for k in range(1, NDEC + 1):
        g = d[dec == k]
        if len(g) < PV.MIN_PER_CLASS:
            out.append(dict(stratum=stratum, score=label, trait=trait, decile=k,
                            oriented_decile=int(orient[dec == k][0]) if (dec == k).any() else k,
                            n=len(g), note=PV.SUPPRESSED))
            continue
        p = g.p_nonEUR.to_numpy()
        out.append(dict(stratum=stratum, score=label, trait=trait, decile=k,
                        oriented_decile=int(orient[dec == k][0]),
                        n=len(g), mean_p=round(float(p.mean()), 5),
                        se=round(float(p.std(ddof=1) / np.sqrt(len(p))), 5),
                        raw=round(float((p >= 0.5).mean()), 5),
                        mean_logit=round(float(np.mean(L.logit(p))), 5), note=''))
    return out


def restrict_to_keep(d, keep):
    """Restrict an analysis frame to the genotype samples PLINK was told to retain."""
    K = pd.read_csv(keep, sep=r'\s+', dtype=str)
    cols = {str(c).lstrip('#').upper(): c for c in K.columns}
    if 'IID' not in cols:
        raise SystemExit(f'{keep}: no IID column in genotype keep file')
    retained = set(K[cols['IID']].astype(str))
    out = d[d.geno_id.astype(str).isin(retained)].copy()
    if not len(out):
        raise SystemExit(f'{keep}: none of the analysis_set genotype ids were retained')
    return out


def attach_configured_covariates(work, stratum, d):
    """Attach the exact covariate design selected by `geno`, aligned by genotype IID."""
    path = Path(work) / 'geno' / stratum / 'covar.tsv'
    C = pd.read_csv(path, sep=r'\s+', dtype=str)
    cols = {str(c).lstrip('#').upper(): c for c in C.columns}
    if 'IID' not in cols:
        raise SystemExit(f'{path}: no IID column in configured genotype covariates')
    iid = cols['IID']
    if C[iid].astype(str).duplicated().any():
        raise SystemExit(f'{path}: duplicate IID in configured genotype covariates')
    source = [c for c in C.columns if str(c).lstrip('#').upper() not in ('FID', 'IID')]
    if not source:
        raise SystemExit(f'{path}: no configured covariate columns')
    lookup = C.assign(**{iid: C[iid].astype(str)}).set_index(iid)
    ids = d.geno_id.astype(str)
    missing = ids[~ids.isin(lookup.index)]
    if len(missing):
        raise SystemExit(f'{path}: missing configured covariates for {len(missing)} retained '
                         'genotype sample(s)')
    out = d.copy()
    names = []
    for i, col in enumerate(source):
        name = f'__geno_covar_{i}'
        out[name] = pd.to_numeric(lookup[col].reindex(ids.to_numpy()).to_numpy(), errors='coerce')
        names.append(name)
    return out, names


def mediation_effects(y, x, covariates, measurement):
    """Unadjusted/adjusted PGS effects on identical rows and one predictor scale."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    C = np.asarray(covariates, float)
    if C.ndim == 1:
        C = C[:, None]
    m = np.asarray(measurement, float)
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(m) & np.isfinite(C).all(axis=1)
    base = L.partial(y[ok], x[ok], C[ok])
    adjusted = L.partial(y[ok], x[ok], np.column_stack([C[ok], m[ok]]),
                         x_scale=base['x_resid_sd'])
    b0, b1 = base['beta_sd'], adjusted['beta_sd']
    attenuation = (100 * (1 - b1 / b0)
                   if np.isfinite(b0) and np.isfinite(b1) and b0 != 0 else np.nan)
    return base, adjusted, attenuation


def read_measurement_table(path, source):
    """Read a site measurement panel with one canonical, string-valued participant id."""
    t = L.read_table(path, what=f'--measurements ({source})', dtype=str)
    id_col = L.pick_column(t, 'participant_id',
                           ('participant_id', 'id', 'subject_id', 'empi'),
                           what=f'--measurements ({source})', kind='participant_id column',
                           flag='(rename the column to participant_id)')
    if id_col != 'participant_id':
        t = t.rename(columns={id_col: 'participant_id'})
    return t


# ------------------------------------------------------------------------------ the arms
def trait_arm(st, d, V, ref, work, keep, meas, rows, dec_rows, threads, covar_cols,
              build='grch38',
              sources=(), want='both', tmaps=None, pinned=None):
    # build-specific: the weights are keyed on POSITION, and a GRCh38 position looked up in a
    # GRCh37 variant map matches a scatter of unrelated variants rather than failing cleanly
    PW = pd.read_csv(L.ref_table(ref, work, 'pgs_weights', build), sep='\t', dtype={'CHR': str})
    CW = pd.read_csv(ref / 'composite_weights.tsv', sep='\t').set_index('trait').weight
    wdir = Path(work) / 'geno' / st / 'pgs'; wdir.mkdir(exist_ok=True, parents=True)
    scores = {}
    for tr, g in PW.groupby('trait'):
        f = wdir / f'w_{tr}.tsv'
        other = np.where(g.PANEL_A1.str.upper() == g.RISK_ALLELE.str.upper(),
                         g.PANEL_A2, g.PANEL_A1)
        n_ok, n_all = site_weights(V, g.CHR, g.POS, g.RISK_ALLELE, other,
                                   g.BETA, tr, f)
        if n_ok < 5:
            rows.append(dict(stratum=st, trait=tr, arm='scored', n_variants=n_ok,
                             n_variants_published=n_all, note='too few variants present'))
            continue
        S = G.score(Path(work) / 'geno' / 'panel', f, wdir / f's_{tr}', keep=keep, threads=threads)
        col = 'SCORE1_AVG' if 'SCORE1_AVG' in S.columns else S.columns[-1]
        scores[tr] = S[['IID', col]].rename(columns={col: f'PGS_{tr}'})
        rows.append(dict(stratum=st, trait=tr, arm='scored', n_variants=n_ok,
                         n_variants_published=n_all, note=''))
    if not scores:
        return None
    D = d.copy()
    for tr, S in scores.items():
        D = D.merge(S.rename(columns={'IID': 'geno_id'}), on='geno_id', how='inner')
    C = D[covar_cols].to_numpy(float)

    for tr in scores:
        x = zs(D[f'PGS_{tr}'].to_numpy())
        # the column name is a property of the INSTRUMENT: --measurement-map renames the site's
        # own panel, while the NeuroKit table is written by mode `traits` with fixed names
        def _mcol(src, _tr=tr):
            return ((tmaps or {}).get(src) or TRAIT_MEASURE).get(_tr)
        mcol = TRAIT_MEASURE.get(tr)
        # WHICH RULER(S) SERVE THIS TRAIT.
        #   both      -- every instrument that carries it, so a disagreement is visible
        #   device    -- the precomputed panel, and where the panel does not carry the trait,
        #                NeuroKit fills it in and the row is labelled '(fill)'
        #   neurokit  -- the same, the other way round
        # A trait pinned in --measurement-map overrides all of it.
        if (pinned or {}).get(tr):
            order, single = [pinned[tr]], True
        elif want == 'both':
            order, single = list(sources), False
        else:
            order = [want] + [s for s in sources if s != want]
            single = True
        use = []
        for src in order:
            if src not in sources:
                continue
            mc = _mcol(src)
            col = f'{mc}@{src}' if mc else None
            if not col or col not in D.columns:
                continue
            if int(np.isfinite(D[col].to_numpy(float)).sum()) < PV.MIN_PER_CLASS:
                continue
            use.append((src, col, mc,
                        single and src != want and not (pinned or {}).get(tr)))
            if single:
                break
        # one control row per RULER: `source` is 'device' (the cart's precomputed panel) or
        # 'neurokit' (measured here off the same waveforms by mode `traits`).
        for src, col, mc, fill in use:
            r = L.partial(D[col].to_numpy(float), x, C)
            rows.append(dict(stratum=st, trait=tr, arm='own_measured_trait',
                             source=f'{src} (fill)' if fill else src, measure=mc,
                             **{k: r[k] for k in ('n', 'beta_sd', 'se', 'p', 'partial_r',
                                                  'delta_r2')},
                             t=round(r['beta_sd'] / r['se'], 3) if r['se'] else np.nan,
                             note=f'the {want} panel does not carry this trait, so it was '
                                  f'measured by {src}' if fill else ''))
        if not use:
            rows.append(dict(stratum=st, trait=tr, arm='own_measured_trait', source='none',
                             measure=mcol or 'none',
                             note='no local measurement of this trait' if not mcol
                                  else 'measurements not supplied'))
        r2 = L.partial(D.score_logit.to_numpy(float), x, C)
        rows.append(dict(stratum=st, trait=tr, arm='ancestry_score',
                         **{k: r2[k] for k in ('n', 'beta_sd', 'se', 'p', 'partial_r',
                                               'delta_r2')}, note=''))
        # THE MEDIATION QUESTION, which is the point of the trait-PGS arm and not a footnote.
        # A trait PGS moving the ancestry score is only evidence that the score reads cardiac
        # conduction if the effect runs THROUGH the conduction trait.  With the site's own
        # measurement in hand the test is free: repeat the regression with the measured trait in
        # the covariates and report both betas side by side.  A beta that collapses means the
        # genotype acts on the score via the trait; a beta that survives means the genotype
        # indexes something the measurement does not capture, which is the more interesting and
        # the more fragile claim.  Only computable where the site supplied the measurement.
        # run per ruler too: how much of the effect the measurement accounts for is a property
        # of the measurement, so two rulers can disagree about it and the reader should see both.
        for src, col, mc, fill in use:
            m = D[col].to_numpy(float)
            r_base, r3, drop = mediation_effects(D.score_logit.to_numpy(float), x, C, m)
            if r_base['n'] >= PV.MIN_PER_CLASS:
                rows.append(dict(stratum=st, trait=tr, arm='ancestry_score_adj_measured_trait',
                                 source=f'{src} (fill)' if fill else src, measure=mc,
                                 **{k: r3[k] for k in ('n', 'beta_sd', 'se', 'p', 'partial_r',
                                                       'delta_r2')},
                                 n_unadjusted_same_rows=r_base['n'],
                                 beta_sd_unadjusted_same_rows=r_base['beta_sd'],
                                 se_unadjusted_same_rows=r_base['se'],
                                 pct_attenuation=round(float(drop), 1),
                                 note='nested regressions on the same complete-case rows and '
                                      'the unadjusted residual PGS-SD scale; pct_attenuation is '
                                      'how much of that unadjusted effect the measurement '
                                      'accounts for'))
        dec_rows += decile_table(D.assign(**{f'_{tr}': D[f'PGS_{tr}']}), f'_{tr}',
                                 'trait_pgs', st, trait=tr, weight=float(CW.get(tr, 1.0)))

    # the frozen composite, applied as a constant
    have = [t for t in scores if t in CW.index]
    comp = np.zeros(len(D))
    for t in have:
        comp += float(CW[t]) * zs(D[f'PGS_{t}'].to_numpy())
    D['_composite'] = comp
    r = L.partial(D.score_logit.to_numpy(float), zs(comp), C)
    rows.append(dict(stratum=st, trait='COMPOSITE', arm='ancestry_score', n_variants=len(have),
                     **{k: r[k] for k in ('n', 'beta_sd', 'se', 'p', 'partial_r', 'delta_r2')},
                     note=f'frozen discovery weights over {len(have)} traits'))
    dec_rows += decile_table(D, '_composite', 'composite', st, trait='COMPOSITE', weight=1.0)
    return D


def ancestry_arm(st, d, V, ref, work, keep, threads, clump_r2, clump_kb, covar_cols,
                 build='grch38'):
    """The ancestry-score PGS: our meta weights, clumped in their LD, at a ladder of p-thresholds.

    The weight file is build-specific and the panel ships in GRCh38 only, so it is resolved
    through L.ref_table rather than named. Reading the GRCh38 panel against a GRCh37 release
    would not fail -- it would match a few hundred positions by coincidence and report a score
    built from them, which is worse than not running the arm at all.
    """
    f = L.ref_table(ref, work, 'ancestry_pgs_weights', build, required=False)
    if f is None:
        return [], [], dict(note=f'the ancestry-PGS panel ships in GRCh38 only and this run is '
                                 f'{L.build_suffix(build)}; re-run `geno --chain <ucsc chain>` '
                                 f'to lift it, or leave this arm out')
    AW = pd.read_csv(f, sep='\t', dtype={'CHR': str})
    wdir = Path(work) / 'geno' / st / 'pgs'; wdir.mkdir(exist_ok=True, parents=True)
    M = AW.merge(V[['CHR', 'POS', 'vid', 'a1', 'a2']], on=['CHR', 'POS'], how='inner')
    sgn = L.allele_sign(M.a1, M.a2, M.A1, M.A0)
    M = M[np.isfinite(sgn)].copy()
    if len(M) < 100:
        return [], [], dict(note=f'only {len(M)} ancestry-PGS variants usable here')

    rows, dec_rows = [], []

    def score_set(S, label, tag, how_note, p_threshold=None):
        """Apply the SHIPPED betas to one variant set. Nothing here estimates a weight.

        The only regression is pheno ~ score + covariates, which is the validation itself:
        the score is the fixed predictor on the right-hand side, never something being fitted.
        """
        wf = wdir / f'anc_w_{tag}.tsv'
        S[['vid', 'A1', 'BETA']].rename(columns={'vid': 'ID'}).to_csv(wf, sep='\t', index=False)
        SC = G.score(Path(work) / 'geno' / 'panel', wf, wdir / f'anc_s_{tag}', keep=keep,
                     threads=threads)
        col = 'SCORE1_AVG' if 'SCORE1_AVG' in SC.columns else SC.columns[-1]
        D = d.merge(SC[['IID', col]].rename(columns={'IID': 'geno_id', col: 'APGS'}),
                    on='geno_id', how='inner')
        covariates = D[covar_cols].to_numpy(float)
        r = L.partial(D.pheno_rint.to_numpy(float), zs(D.APGS.to_numpy()), covariates)
        rows.append(dict(stratum=st, arm=label, p_threshold=p_threshold, n_variants=len(S),
                         clumping=how_note,
                         **{k: r[k] for k in ('n', 'beta_sd', 'se', 'p', 'partial_r',
                                              'delta_r2')}, note=''))
        return D, rows[-1]

    # ---- PRIMARY: the fixed panel. Every usable variant, the shipped weight on each, no
    # selection of any kind done here. This is the arm that is comparable across sites and the
    # one the headline reports, precisely because nothing about it depends on MGB's own LD or
    # on MGB's own phenotype.
    fixed_D, fixed_row = score_set(M, 'fixed_panel', 'fixed', 'none (fixed panel)')
    print(f'    ancestry PGS, FIXED panel: {len(M):,} variants, shipped weights, no selection')
    dec_rows += decile_table(fixed_D, 'APGS', 'ancestry_pgs', st, trait='ANCESTRY_PGS',
                             weight=1.0)

    # ---- SENSITIVITY: the C+T ladder. Clumping uses the SITE's LD, so which variants
    # survive differs between sites; the weights are still ours and are still not refitted, but
    # the instrument's composition is site-dependent, which is why this is reported beside the
    # fixed panel rather than as the result.
    assoc = wdir / 'anc_assoc.txt'
    M.assign(SNP=M.vid, CHR=M.CHR, BP=M.POS, P=M.P)[['SNP', 'CHR', 'BP', 'P']].to_csv(
        assoc, sep=' ', index=False)
    kept, how = G.clump(Path(work) / 'geno' / 'panel', assoc, wdir / 'anc_clump',
                        p1=1.0, r2=clump_r2, kb=clump_kb, threads=threads)
    K = M[M.vid.isin(set(kept))] if kept else M
    print(f'    ancestry PGS, clumped sensitivity: {len(M):,} usable -> {len(K):,} '
          f'after clumping ({how})')
    for t in THRESH:
        S = K[K.P < t]
        if len(S) < 20:
            rows.append(dict(stratum=st, arm='clumped', p_threshold=t, n_variants=len(S),
                             clumping=how, note='too few'))
            continue
        score_set(S, 'clumped', f'{t:g}', how, p_threshold=t)

    # The headline is the FIXED panel, stated as such, so that a reader never has to work out
    # whether the number in front of them came from a site-selected instrument.
    head = dict(primary_arm='fixed_panel', n_usable=int(len(M)), n_fixed=int(len(M)),
                n_clumped=int(len(K)), clumping='none (fixed panel)',
                clumped_sensitivity_clumping=how,
                weights='shipped ref/ancestry_pgs_weights_*.tsv, applied unchanged; '
                        'no weight is estimated at the site')
    head.update({k: fixed_row.get(k) for k in ('beta_sd', 'se', 'p', 'delta_r2', 'n')})
    return rows, dec_rows, head


def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry pgs', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True,
                    help="the site's working directory: intermediates, per-person tables and "
                         'arrays. It stays at the site and is never packaged')
    ap.add_argument('--results', required=True,
                    help='the aggregate-only output directory; `package` zips exactly this')
    ap.add_argument('--ref', default=str(L.ref_dir()),
                    help='the frozen discovery tables (default: $REFERENCE_DIR, else the '
                         'reference/ folder of the pinned weights snapshot)')
    ap.add_argument('--measurements', default=None,
                    help="the site's own measurement panel (csv with participant_id + columns "
                         'named as in TRAIT_MEASURE). Used BESIDE work/measurements.csv from '
                         '`traits` when both exist, so the positive control runs on each ruler '
                         'and the `source` column says which')
    ap.add_argument('--measurement-source', choices=['both', 'device', 'neurokit'],
                    default='both',
                    help='which ruler the positive control uses. both (default, RECOMMENDED): '
                         'every trait on each instrument that has it, so a disagreement is '
                         'visible. device: the precomputed panel from --measurements, falling '
                         'back to NeuroKit for any trait the panel does not carry. neurokit: '
                         'the measurements taken here by `traits`, same fallback the other way')
    ap.add_argument('--measurement-map', metavar='JSON', default=None,
                    help='optional json naming the column that holds each trait, when the '
                         'panel does not use our names: {"QT": "QTc_ms", "Tamp": "T_V6"}. A '
                         'value may also be {"column": "QTc_ms", "source": "device"} to pin '
                         'one trait to one ruler. Traits left out keep their default column')
    ap.add_argument('--strata', nargs='+', default=None,
                    help='which ancestry strata to run (default: every stratum `geno` built)')
    ap.add_argument('--threads', type=int, default=L.default_threads(),
                    help='threads for plink (default: this machine\'s cores, less two, capped at 16)')
    ap.add_argument('--clump-r2', type=float, default=0.1,
                    help='LD threshold for clumping the ancestry-score PGS in YOUR genotypes '
                         '(default 0.1); LD is a property of your cohort, so it is done here')
    ap.add_argument('--clump-kb', type=int, default=500,
                    help='clumping window in kb (default 500)')
    ap.add_argument('--skip-ancestry-pgs', action='store_true',
                    help='run the trait arms only and leave out the ancestry-score PGS')
    ap.add_argument('--build', default=None,
                    help='normally inherited from `geno`; override only to re-run one arm')
    a = ap.parse_args()

    work = Path(a.work); res = Path(a.results) / 'pgs'; res.mkdir(parents=True, exist_ok=True)
    ref = Path(a.ref)
    L.require('pgs', [
        (Path(a.results) / 'geno' / 'geno_provenance.json',
         'the genotype panel, the within-stratum PCs and the phenotype files', 'geno'),
        (work / 'geno' / 'variant_map.csv',
         "the join between our variant positions and the site's own variant ids", 'geno')])
    prov = json.loads((Path(a.results) / 'geno' / 'geno_provenance.json').read_text())
    strata = a.strata or prov.get('strata', [])
    build = L.build_suffix(a.build or prov.get('build', 'grch38'))
    print(f'  build: {build} (from `geno`)' if not a.build else f'  build: {build} (overridden)')
    V = pd.read_csv(work / 'geno' / 'variant_map.csv', dtype={'CHR': str})
    V = V[V.vid.notna()].copy()
    V['POS'] = pd.to_numeric(V.POS, errors='coerce').astype('Int64')

    # TWO INSTRUMENTS, NOT ONE, WHEN THE SITE HAS BOTH.
    #
    # The panel the recording cart computed and the NeuroKit2 measurements taken from the same
    # waveforms are different rulers. They agree well on QRS, QT and heart rate and much less
    # well on PR and on the wave amplitudes (see ref/neurokit_vs_device.csv for the agreement
    # we measured; on the ancestry task itself the two rulers performed about the same).
    #
    # That is exactly why the positive control is run on BOTH where both exist. With one ruler a
    # trait that fails the control is ambiguous: a genotyping, liftover or allele-matching
    # failure looks the same as an instrument that simply does not measure that trait well. With
    # two, the two cases separate -- failing on one ruler and clearing on the other is a
    # MEASUREMENT disagreement and leaves the trait-PGS result standing, while failing on both
    # is the real thing and invalidates it. `--measurements` used to REPLACE the NeuroKit file;
    # it now sits beside it, and `source` says which ruler each row came from.
    srcs = []
    dev = Path(a.measurements) if a.measurements else None
    nk = work / 'measurements.csv'
    if dev is not None and dev.exists():
        srcs.append(('device', dev))
    if nk.exists() and (dev is None or nk.resolve() != dev.resolve()):
        srcs.append(('neurokit', nk))
    if dev is not None and not dev.exists():
        raise SystemExit(f'--measurements {dev} does not exist')

    # A site whose panel spells the columns differently should not have to rewrite its export:
    # the json remaps trait -> column, and may pin a trait to one instrument.
    tmap, pinned = dict(TRAIT_MEASURE), {}
    if a.measurement_map:
        q = Path(a.measurement_map)
        if not q.exists():
            raise SystemExit(f'--measurement-map {q} does not exist')
        try:
            jm = json.loads(q.read_text())
        except Exception as e:                                            # noqa: BLE001
            raise SystemExit(f'--measurement-map {q} is not valid json: {e}')
        unknown = [k for k in jm if k not in TRAIT_MEASURE]
        if unknown:
            raise SystemExit(f'--measurement-map names traits this analysis does not have: '
                             f'{", ".join(sorted(unknown))}. The nine are: '
                             f'{", ".join(sorted(TRAIT_MEASURE))}')
        for tr, v in jm.items():
            if isinstance(v, dict):
                tmap[tr] = v.get('column') or TRAIT_MEASURE[tr]
                if v.get('source'):
                    if v['source'] not in ('device', 'neurokit'):
                        raise SystemExit(
                            f'--measurement-map pins {tr} to "{v["source"]}"; the only two '
                            f'instruments are "device" (the panel given to --measurements) and '
                            f'"neurokit" (measured here by mode `traits`)')
                    pinned[tr] = v['source']
            else:
                tmap[tr] = v
        print(f'  measurement map: {len(jm)} trait(s) remapped' +
              (f', {len(pinned)} pinned to one ruler' if pinned else ''))

    # the remap applies to the panel the site exported (--measurements); the NeuroKit table is
    # this image's own output and always carries this image's own column names
    TMAPS = {'device': tmap, 'neurokit': dict(TRAIT_MEASURE)}
    MEAS = []
    for src, q in srcs:
        _m = TMAPS[src]
        n_measurable = sum(v is not None for v in _m.values())
        t = read_measurement_table(q, src)
        have = [c for c in _m.values() if c and c in t.columns]
        MEAS.append((src, t))
        print(f'  measurements [{src}]: {len(t):,} recordings, {len(have)} of the '
              f'{n_measurable} measurable traits present  ({q})')
    meas = MEAS[0][1] if MEAS else None
    want = a.measurement_source
    names = [src for src, _ in srcs]
    if want != 'both' and want not in names:
        print(f'  NOTE --measurement-source {want} was asked for but no {want} measurements are '
              f'here' + (f'; every trait falls back to {names[0]}' if names else
                         ', and neither ruler is here'))
    if want == 'both' and len(MEAS) == 2:
        print('  the positive control runs on BOTH rulers: a trait that clears t = 3 on one and '
              'not the other is an instrument disagreement, not a broken score')
    elif want != 'both' and len(MEAS) == 2:
        other = [x for x in names if x != want]
        print(f'  the positive control runs on the {want} ruler only; any trait the {want} panel '
              f'does not carry falls back to {other[0] if other else "the other ruler"} and is '
              f'labelled "(fill)" in `source`.  Re-run with --measurement-source both to see the '
              f'two instruments side by side -- that is the recommended first pass')
    if pinned:
        print('  pinned: ' + ', '.join(f'{k} -> {v}' for k, v in sorted(pinned.items())))
    elif len(MEAS) == 1:
        print(f'  only the {MEAS[0][0]} ruler is available. The control still runs, but a '
              'failure cannot be told apart from an instrument that measures that trait poorly '
              '-- supply the other with --measurements, or run `traits`, to separate them')
    else:
        print(f'  no measurements at {nk}: the positive control is SKIPPED and every '
              'trait-PGS result below is unvalidated. Run `traits` first.')

    L.banner(f'pgs: {", ".join(strata)}')
    rows, dec_rows, anc_rows, anc_dec, heads = [], [], [], [], {}
    for st in strata:
        d = Path(work) / 'geno' / st / 'analysis_set.csv'
        if not d.exists():
            print(f'  {st}: no analysis set, skipped')
            continue
        D = pd.read_csv(d, dtype={'participant_id': str, 'geno_id': str})
        D['sex_num'] = D.sex.astype(str).str.upper().map(
            {'M': 1, 'MALE': 1, '1': 1, 'F': 0, 'FEMALE': 0, '0': 0})
        for _src, _t in MEAS:
            _cols = [c for c in TMAPS[_src].values() if c and c in _t.columns]
            D = D.merge(_t[['participant_id'] + _cols].rename(
                columns={c: f'{c}@{_src}' for c in _cols}),
                on='participant_id', how='left')
        keep = Path(work) / 'geno' / st / 'keep_unrelated.txt'
        keep = str(keep) if keep.exists() else str(Path(work) / 'geno' / st / 'keep.txt')
        before = len(D)
        D = restrict_to_keep(D, keep)
        D, covar_cols = attach_configured_covariates(work, st, D)
        print(f'\n  --- {st}: {len(D):,} retained people'
              + (f' ({before - len(D):,} related/excluded)' if before != len(D) else ''))
        trait_arm(st, D, V, ref, work, keep, meas, rows, dec_rows, a.threads,
                  covar_cols, build=build, sources=[s for s, _ in MEAS],
                  want=a.measurement_source, tmaps=TMAPS, pinned=pinned)
        if not a.skip_ancestry_pgs:
            r, dr, h = ancestry_arm(st, D, V, ref, work, keep, a.threads, a.clump_r2,
                                    a.clump_kb, covar_cols, build=build)
            anc_rows += r; anc_dec += dr; heads[st] = h

    R = pd.DataFrame(rows)
    PV.write_table(R, res / 'trait_pgs.csv', where='trait_pgs.csv')
    PV.write_table(pd.DataFrame(dec_rows), res / 'trait_pgs_deciles.csv',
                   where='trait_pgs_deciles.csv')
    if anc_rows:
        PV.write_table(pd.DataFrame(anc_rows), res / 'ancestry_pgs.csv', where='ancestry_pgs.csv')
        PV.write_table(pd.DataFrame(anc_dec), res / 'ancestry_pgs_deciles.csv',
                       where='ancestry_pgs_deciles.csv')
    (res / 'pgs_headline.json').write_text(json.dumps(
        dict(version=L.VERSION, strata=strata, ancestry_pgs=heads,
             # every ruler used, by name, so a reader knows which instruments the
             # positive control was run on and from which file
             measurements={src: str(q) for src, q in srcs} or None,
             measurement_sources=[src for src, _ in srcs],
             trait_measure_map={k: v for k, v in TRAIT_MEASURE.items()}), indent=2, default=float))

    ctl = R[R.arm == 'own_measured_trait'].dropna(subset=['beta_sd']) if 'beta_sd' in R else \
        pd.DataFrame()
    if len(ctl):
        cols = ['stratum', 'trait'] + (['source'] if 'source' in ctl.columns else []) + \
               ['measure', 'n', 'beta_sd', 'se', 'p', 't']
        print('\n  POSITIVE CONTROL (each score on its own measured trait):')
        print(ctl[cols].to_string(index=False))
        bad = ctl[(ctl.stratum == L.PRIMARY) & (ctl.t.abs() < 3)]
        if len(bad):
            lab = [f'{r.trait}' + (f' [{r.source}]' if 'source' in ctl.columns else '')
                   for r in bad.itertuples()]
            print(f'\n  {len(bad)} of {int((ctl.stratum == L.PRIMARY).sum())} score-ruler pairs '
                  f'do not clear t = 3 on their own trait in {L.PRIMARY}: {", ".join(lab)}')

        # THE TWO RULERS AGAINST EACH OTHER. This is the reason for running both: it separates a
        # score that is broken from a measurement that disagrees, and only the first invalidates
        # anything downstream.
        if 'source' in ctl.columns and ctl.source.nunique() > 1:
            e = ctl[ctl.stratum == L.PRIMARY]
            piv = e.pivot_table(index='trait', columns='source', values='t', aggfunc='first')
            if {'device', 'neurokit'} <= set(piv.columns):
                cd = piv.device.abs() >= 3
                cn = piv.neurokit.abs() >= 3
                print(f'\n  THE TWO RULERS in {L.PRIMARY} (t on each):')
                print(piv[['device', 'neurokit']].round(2).to_string())
                dis = piv.index[cd != cn].tolist()
                both = piv.index[~cd & ~cn].tolist()
                if dis:
                    print(f'\n  {len(dis)} trait(s) clear t = 3 on one ruler and not the other: '
                          f'{", ".join(dis)}.\n    That is a MEASUREMENT disagreement, not a '
                          'broken score -- the two instruments are known to diverge on PR and '
                          'on the amplitudes (ref/neurokit_vs_device.csv).')
                if both:
                    print(f'\n  {len(both)} trait(s) fail on BOTH rulers: {", ".join(both)}.\n'
                          '    That is not an instrument problem. Check genotyping, liftover and '
                          'allele matching before using those trait-PGS results.')
    print(f'\n  wrote {res}/')


if __name__ == '__main__':
    main()
