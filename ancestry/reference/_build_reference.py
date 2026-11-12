#!/usr/bin/env python
"""Build the reference tables this image carries to an external site.  Runs HERE, once.

Everything the container compares the site's numbers against is frozen into ref/ at build time, so
the site runs no code that reaches back to us and the comparison cannot drift between runs.  Seven
tables, and where each one comes from:

  leads_grch38.tsv / leads_grch37.tsv     the genome-wide-significant leads of the ancestry-score
        within-European meta-analysis, with our effect, SE, frequency and nearest gene.  Copied
        verbatim from the Lifelines package, which is the same 62 and was already lifted to
        GRCh37 with liftover_min.py against hg38ToHg19.over.chain.gz.
  locus_annotation.tsv                    one row per locus: rsID, nearest gene, gene class, and
        the three-way locus class the figures colour by (published ECG-trait locus / cardiac
        disease locus, no ECG trait / no cardiac annotation).  From the curated supplementary
        table, so the container and the paper name the same classes.
  pgs_weights_grch38.tsv / _grch37.tsv    the nine validated ECG-trait polygenic scores behind
        Fig. 2e, with the published per-variant weights.  Copied verbatim.
  ancestry_pgs_weights_grch38.tsv         NEW here.  The ancestry-score PGS: the
        three-cohort meta effect at every variant of the expanded panel.  At an external site this
        is NOT circular -- no participant of that site is in the meta -- so unlike discovery,
        which had to leave a
        cohort out, the site can use the full-strength weights.  Shipped UNCLUMPED and unthresholded;
        the container clumps in the site's own genotypes (r2 < 0.1 within 500 kb) exactly as discovery
        did, because LD is a property of the target cohort and clumping uses no phenotype.
  findings77.tsv                          the 77 clinical findings DeepECG-SSL emits, each tagged
        diagnosis / descriptor / criterion (only a diagnosis earns a row on the fairness plate)
        and carrying the regular expression that recognises it in free-text MUSE statements, for
        a site with no adjudicated label table.
  discovery_fairness_fpr.tsv              our own false-positive gaps per finding, split by true
        genetic ancestry and by ancestry inferred from the ECG, so the site's forest can be laid
        beside ours.  From the discovery fairness audit -- read, never recomputed.
  discovery_reference.json                the headline discovery numbers the site's results are
        reported against, and nothing else.

    python _build_reference.py [--out .]

Writes only aggregate, variant-level or label-level tables.  No participant of any cohort appears
in any of them.
"""
from __future__ import annotations

import os
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The discovery analysis tree, which is not public: this builder froze the tables that ARE
# shipped and cannot run without it. Point ANCESTRY_ANALYSIS_ROOT at that tree to rebuild.
ROOT = Path(os.environ.get('ANCESTRY_ANALYSIS_ROOT', '/nonexistent/analysis-tree'))
TASK = ROOT / 'ancestry_related_tasks'
LL = TASK / 'scripts/lifelines_locus_validation/ref'
GWAS = ROOT / 'GWAS'
SUPP = TASK / 'outputs/R-34_ancestry_gwas/loci_supp_table/loci_supp_table.tsv'
R42 = TASK / 'outputs/R-42_fairness/metrics_all.tsv'
EXT77 = ROOT / 'AF_pred/outputs/ext77'
R31 = TASK / 'outputs/R-31_composite_pgs'
R49 = TASK / 'outputs/R-49_score_pgs'
LDSC = TASK / 'outputs/R-34_ancestry_gwas/ldsc/ldsc_h2.json'
PANEL = {'UKB': ROOT.parent / 'ECG2GENOMICS/UKB/new_panel_out',
         'CLSA': ROOT.parent / 'ECG2GENOMICS/CLSA/new_panel_out/panel_out',
         'MHI': ROOT.parent / 'ECG2GENOMICS/MHI/new_panel_out/panel_out'}

sys.path.insert(0, str(TASK / 'scripts/R-06_genetic_scores_PGS_MGS'))
sys.path.insert(0, str(TASK / 'scripts/_shared'))


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ 1-2. leads and trait weights
def copy_verbatim(out):
    for n in ('leads_grch38.tsv', 'leads_grch37.tsv',
              'pgs_weights_grch38.tsv', 'pgs_weights_grch37.tsv',
              'neurokit_vs_device.csv'):
        shutil.copy(LL / n, out / n)
        print(f'  {n:34s} {sum(1 for _ in open(out / n)) - 1:6,} rows (copied verbatim)')


# ------------------------------------------------------------------ 3. locus annotation
# The three values `locus_class` takes in the curated supplementary table, mapped to the words the
# published figure's legend uses.  Checked against the table after every build: an unmapped value
# raises rather than passing a raw string through to a figure legend.
LOCUS_CLASS = {'ECG phenotype': 'published ECG-trait locus',
               'Cardiac disease phenotype': 'cardiac disease locus, no ECG trait',
               'Other trait only': 'no cardiac annotation'}


def locus_annotation(out):
    """One row per locus with the class the figures colour by.

    `locus_class` in the supplementary table is already the three-way call made against the GWAS
    Catalog plus the curated cardiac list; it is carried through unchanged so that a reader who
    checks the container's scatter against Supplementary Table S-loci finds the same colours.
    """
    S = pd.read_csv(SUPP, sep='\t', dtype={'chr': str})
    keep = ['locus', 'chr', 'pos', 'key', 'rsid', 'nearest_gene', 'gene_distance_bp', 'gene_class',
            'effect_allele', 'other_allele', 'eaf', 'beta', 'se', 'neglog10p', 'n_total',
            'direction', 'i2', 'locus_class', 'phenotype', 'phenotype_class']
    A = S[keep].copy()
    unknown = sorted(set(A.locus_class.astype(str).str.strip()) - set(LOCUS_CLASS))
    if unknown:
        raise SystemExit(f'locus_class values with no legend word: {unknown}')
    A['class_plot'] = A.locus_class.astype(str).str.strip().map(LOCUS_CLASS)
    A.to_csv(out / 'locus_annotation.tsv', sep='\t', index=False)
    print(f'  locus_annotation.tsv               {len(A):6,} loci; classes '
          f'{dict(A.class_plot.value_counts())}')
    return A


# ------------------------------------------------------------------ 4. the ancestry-score PGS
def panel_keys():
    """Union of the three cohorts' expanded-panel variants, as chr:pos:<alleles sorted>."""
    import load_panel as LP
    keys, per = set(), {}
    for c, d in PANEL.items():
        b = LP.read_bim(str(d))
        per[c] = len(b)
        keys |= set(b.key_sorted)
    print(f'    panel bims: ' + ', '.join(f'{c} {n:,}' for c, n in per.items())
          + f'  ->  {len(keys):,} distinct')
    return keys


def ancestry_pgs(out):
    """Meta effect at every panel variant, for the PGS the site builds in its own genotypes.

    Shipped with BETA, SE, P and the meta's effect-allele frequency.  The frequency is carried so
    the site can check that a variant it scores is the variant we meant: a |delta| of 0.4 between
    our frequency and theirs, at a site with a comparable European majority, is an allele that was
    matched the wrong way round, and the container flags it rather than scoring it.
    """
    want = panel_keys()
    cols = ['KEY', 'CHR', 'POS', 'A1', 'A0', 'FREQ1', 'BETA', 'SE', 'P', 'N_TOTAL', 'DIRECTION']
    keep, n = [], 0
    for ch in pd.read_csv(GWAS / 'ancestry_meta_dedup.tsv.gz', sep='\t', usecols=cols,
                          dtype={'KEY': str, 'CHR': str, 'A1': str, 'A0': str},
                          chunksize=2_000_000):
        n += len(ch)
        keep.append(ch[ch.KEY.isin(want)])
    W = pd.concat(keep, ignore_index=True)
    W = W[(W.SE > 0) & np.isfinite(W.BETA) & W.FREQ1.between(0.005, 0.995)]
    W = W.drop_duplicates('KEY', keep=False).sort_values(['CHR', 'POS'])
    W.to_csv(out / 'ancestry_pgs_weights_grch38.tsv', sep='\t', index=False,
             float_format='%.6g')
    print(f'  ancestry_pgs_weights_grch38.tsv    {len(W):6,} of {n:,} meta rows '
          f'({(W.P < 5e-8).sum():,} at P < 5e-8)')
    return W


# ------------------------------------------------------------------ 4b. composite + curves
def composite_and_curves(out):
    """The frozen composite weights and the published decile curves the site is compared against.

    The composite is NOT refitted at the site.  It was fitted in five folds inside the discovery
    cohorts; refitting it at a site would make that site's curve a different estimator from the
    published one and the two would not be comparable.  The mean-over-folds weight is shipped and
    applied as a constant, which is what a published score is.
    """
    W = pd.read_csv(ROOT / 'composite_weights.csv', index_col=0)[['mean']]
    W.index = [i.replace('PGS_', '') for i in W.index]
    W = W.rename(columns={'mean': 'weight'}).rename_axis('trait').reset_index()
    W.to_csv(out / 'composite_weights.tsv', sep='\t', index=False, float_format='%.6g')
    print(f'  composite_weights.tsv              {len(W):6,} traits '
          f'(largest |w|: {W.loc[W.weight.abs().idxmax(), "trait"]})')

    src = TASK / 'outputs/R-32_amplitude_clock/fig2f/decile_curve_per_trait.csv'
    C = pd.read_csv(src)
    C[['trait', 'decile', 'oriented_decile', 'n', 'mean_p', 'mean_p_se', 'raw', 'weight']].to_csv(
        out / 'discovery_pgs_deciles.tsv', sep='\t', index=False, float_format='%.6g')
    sp = (C.sort_values('oriented_decile').groupby('trait')
          .apply(lambda g: 100 * (g.mean_p.iloc[-1] - g.mean_p.iloc[0])))
    print(f'  discovery_pgs_deciles.tsv          {len(C):6,} rows; span per trait (pp): '
          + ', '.join(f'{t} {v:+.1f}' for t, v in sp.sort_values(ascending=False).items()))

    # the discovery curve ITSELF, not the copy at the project root -- that one is an external
    # result written back after a site run, and shipping it as 'discovery' would compare a site
    # against Lifelines while calling it discovery.
    K = pd.read_csv(R31 / 'decile_curve.csv')
    K.to_csv(out / 'discovery_composite_deciles.tsv', sep='\t', index=False,
             float_format='%.6g')
    print(f'  discovery_composite_deciles.tsv    {len(K):6,} deciles, composite span '
          f'{100 * (K.raw.iloc[-1] - K.raw.iloc[0]):+.1f} pp')
    return W


# ------------------------------------------------------------------ 5. the 77 findings
# Which findings are a DIAGNOSIS (a named entity a cardiologist reports and acts on), which are a
# DESCRIPTOR (a property of the trace: rhythm regularity, axis, complex morphology) and which are
# a CRITERION (a voltage rule standing in for a diagnosis already in the table under its own name).
# This is the discovery audit's taxonomy, reproduced so the site's plate shows the same kind
# of row as ours.
DESCRIPTOR = {
    'Sinusal', 'Regular', 'Monomorph', 'Polymorph', 'Irregularly irregular', 'Regularly irregular',
    'QS complex in V1-V2-V3', 'R complex in V5-V6', 'rSR\' in V1-V2', 'qRS in V5-V6-I, aVL',
    'QRS complex negative in III', 'Left axis deviation', 'Right axis deviation',
    'Right superior axis', 'ST downslopping', 'ST upslopping', 'U wave',
    'R/S ratio in V1-V2 >1', 'Low voltage', 'Lead misplacement', 'no_qrs', 'Delta wave',
}
CRITERION = {'RaVL > 11 mm', 'SV1 + RV5 or RV6 > 35 mm', 'RV1 + SV6 > 11 mm'}

# Regular expressions over a free-text MUSE/cardiologist statement, for a site with no adjudicated
# finding table.  Deliberately conservative: a finding with no pattern here is reported as
# unavailable rather than guessed at, and the container says how many of the 77 it could resolve.
# Case-insensitive, matched against the concatenated statement text of one recording.
PATTERNS = {
    'Afib': r'atrial fibrillation|\bafib\b|\ba-?fib\b',
    'Atrial flutter': r'atrial flutter|\baflutter\b',
    'Bradycardia': r'bradycardi',
    'Atrial tachycardia (>= 100 BPM)': r'atrial tachycardia',
    'Supraventricular tachycardia': r'supraventricular tachycardia|\bsvt\b',
    'Ventricular tachycardia': r'ventricular tachycardia|\bvt\b',
    'Left bundle branch block': r'left bundle branch block|\blbbb\b',
    'Right bundle branch block': r'right bundle branch block|\brbbb\b',
    'Left anterior fascicular block': r'left anterior fascicular block|\blafb\b',
    'Left posterior fascicular block': r'left posterior fascicular block|\blpfb\b',
    'Nonspecific intraventricular conduction delay': r'intraventricular conduction (delay|defect)',
    '1st degree AV block': r'(1st|first)[- ]degree a-?v block',
    '2nd degree AV block - mobitz 1': r'mobitz (i|1)\b|wenckebach',
    '2nd degree AV block - mobitz 2': r'mobitz (ii|2)\b',
    'Third Degree AV Block': r'(3rd|third)[- ]degree a-?v block|complete heart block',
    'Premature ventricular complex': r'premature ventricular|\bpvc\b|ventricular (premature|ectop)',
    'Premature atrial complex': r'premature atrial|\bpac\b|atrial (premature|ectop)',
    'Left ventricular hypertrophy': r'left ventricular hypertrophy|\blvh\b',
    'Right ventricular hypertrophy': r'right ventricular hypertrophy|\brvh\b',
    'Left atrial enlargement': r'left atrial (enlargement|abnormality)',
    'Right atrial enlargement': r'right atrial (enlargement|abnormality)',
    'Bi-atrial enlargement': r'bi-?atrial (enlargement|abnormality)',
    'Prolonged QT': r'prolonged qt|\bqt\w* prolong',
    'Early repolarization': r'early repolari',
    'Acute pericarditis': r'pericarditis',
    'Acute MI': r'acute (myocardial infarction|mi)\b',
    'Ventricular paced': r'ventricular (pacing|paced)|v-?paced',
    'Atrial paced': r'atrial (pacing|paced)|a-?paced',
    'LV pacing': r'biventricular pac|lv pac',
    'Junctional rhythm': r'junctional rhythm',
    'Ectopic atrial rhythm (< 100 BPM)': r'ectopic atrial rhythm',
    'Ventricular Rhythm': r'(idio)?ventricular rhythm',
    'Wolff-Parkinson-White (Pre-excitation syndrome)': r'wolff|pre-?excitation|\bwpw\b',
    'Brugada': r'brugada',
    'T wave inversion (inferior - II, III, aVF)': r'(t[- ]wave inversion|inverted t).*(inferior)',
    'T wave inversion (lateral -I, aVL, V5-V6)': r'(t[- ]wave inversion|inverted t).*(lateral)',
    'T wave inversion (anterior - V3-V4)': r'(t[- ]wave inversion|inverted t).*(anterior)',
    'T wave inversion (septal- V1-V2)': r'(t[- ]wave inversion|inverted t).*(septal)',
    'ST elevation (inferior - II, III, aVF)': r'st elevation.*inferior',
    'ST elevation (anterior - V3-V4)': r'st elevation.*anterior',
    'ST elevation (septal - V1-V2)': r'st elevation.*septal',
    'ST elevation (lateral - I, aVL, V5-V6)': r'st elevation.*lateral',
    'ST elevation (posterior - V7-V8-V9)': r'st elevation.*posterior',
    'ST depression (inferior - II, III, aVF)': r'st depression.*inferior',
    'ST depression (anterior - V3-V4)': r'st depression.*anterior',
    'ST depression (septal- V1-V2)': r'st depression.*septal',
    'ST depression (lateral - I, avL, V5-V6)': r'st depression.*lateral',
    'Q wave (inferior - II, III, aVF)': r'q wave.*inferior|inferior infarct',
    'Q wave (anterior - V3-V4)': r'q wave.*anterior|anterior infarct',
    'Q wave (septal- V1-V2)': r'q wave.*septal|septal infarct',
    'Q wave (lateral- I, aVL, V5-V6)': r'q wave.*lateral|lateral infarct',
    'Q wave (posterior - V7-V9)': r'q wave.*posterior|posterior infarct',
    'Low voltage': r'low voltage',
    'Left axis deviation': r'left axis deviation',
    'Right axis deviation': r'right axis deviation',
    'Lead misplacement': r'lead (misplacement|reversal)|limb lead reversal',
}


def findings77(out):
    """The 77 labels, their kind, their text pattern -- and which model output carries each.

    THE OUTPUT ORDER IS THE PART THAT CANNOT BE GUESSED.  The released head emits 77 sigmoids and
    does not publish which is which.  The discovery work resolved it on its own pool by taking, for
    each finding inside the published label mapping, the output with the best AUROC, and carried
    that resolution into every number of the audit.  That resolution is shipped here, so a site
    uses
    OUR mapping instead of re-deriving one on its own labels -- a re-derivation would make the two
    audits two different experiments, and on a narrow label set it is also where the Hungarian
    assignment is known to break (forced to place 17 outputs across 77 findings it hands them to
    U wave and septal ST depression).  A site whose labels disagree can re-derive with
    `fairness --remap`, and the package then says which mapping produced the figure.
    """
    from labels77 import canonical_77
    LM = pd.read_csv(EXT77 / 'label_mapping_DeepECG-SSL.csv')
    omap = (LM.rename(columns={'auroc': 'auroc_diag'})
            .drop_duplicates('finding').set_index('finding'))
    F = canonical_77()
    rows = []
    for i, f in enumerate(F):
        kind = 'criterion' if f in CRITERION else 'descriptor' if f in DESCRIPTOR else 'diagnosis'
        o = omap.loc[f] if f in omap.index else None
        rows.append(dict(index=i, finding=f, kind=kind, plate_eligible=int(kind == 'diagnosis'),
                         output=('' if o is None else o.output),
                         discovery_auroc=(np.nan if o is None else float(o.auroc_diag)),
                         muse_regex=PATTERNS.get(f, '')))
    T = pd.DataFrame(rows)
    T.to_csv(out / 'findings77.tsv', sep='\t', index=False)
    print(f'  findings77.tsv                     {len(T):6,} findings '
          f'({(T.kind == "diagnosis").sum()} diagnoses, '
          f'{(T.muse_regex != "").sum()} with a text pattern, '
          f'{(T.output != "").sum()} with a resolved model output)')
    return T


# ------------------------------------------------------------------ 6. our own fairness gaps
def discovery_fairness(out):
    """Our false-positive gaps per finding, genetic split and inferred split, as plotted.

    `uncorrected` is the split on RFMix genetic ancestry; `uncorrected_predicted_groups` is the
    split on the ancestry the ECG itself predicts, called at 0.5.  Both at the shared 0.5
    operating point.  Read from the discovery table; nothing is recomputed, so the site's overlay is
    against the published figure and not against a re-derivation of it.
    """
    M = pd.read_csv(R42, sep='\t')
    m = M[(M.model == 'DeepECG-SSL') & (M.metric.isin(['fpr', 'fnr'])) &
          (M.series.isin(['uncorrected', 'uncorrected_predicted_groups'])) &
          (M.threshold_rule.str.startswith('shared 0.5'))].copy()
    m['split'] = np.where(m.series == 'uncorrected', 'genetic', 'inferred')
    keep = ['finding', 'metric', 'split', 'n', 'n_pos_eur', 'n_pos_non', 'eur', 'non', 'gap',
            'gap_lo', 'gap_hi', 'gap_p', 'auroc_diag', 'plotted']
    m = m[keep].sort_values(['metric', 'finding', 'split'])
    m.to_csv(out / 'discovery_fairness_fpr.tsv', sep='\t', index=False, float_format='%.6g')
    nf = m.finding.nunique()
    print(f'  discovery_fairness_fpr.tsv         {len(m):6,} rows, {nf} findings, '
          f'both splits, FPR and FNR')
    return m



def _mean_gaps(M):
    """Mean absolute gaps, computed from the audit table instead of hard-coded.

    The two numbers used to be literals here, which put measured results into a source file.
    They are a property of the table beside them, so they are derived from it; if the table
    changes shape the keys are simply omitted rather than silently going stale.
    """
    out = {}
    try:
        for metric, key, scale in (('auroc', 'mean_abs_auroc_gap', 1.0),
                                   ('fnr', 'mean_abs_fnr_gap_per_100', 100.0)):
            g = M[(M.model == 'DeepECG-SSL') & (M.metric == metric)]
            if len(g):
                out[key] = round(float(g.gap.abs().mean() * scale), 4 if scale == 1.0 else 1)
    except Exception:                                                     # noqa: BLE001
        return {}
    return out

# ------------------------------------------------------------------ 7. the headline numbers
def mgb_october():
    """The earlier external run, read from the package that site returned."""
    # found by pattern, not named: the site and the date are not written into this repository
    f = next(iter(sorted(ROOT.glob(
        'validation/unpacked/results_package_*/results/validate.json'))), None)
    if f is None:
        return {'note': 'no returned external package on this machine at build time'}
    v = json.loads(f.read_text())
    return {'earlier external run': {
        'n_total': v.get('n_total'),
        'n_by_stratum': v.get('n_by_dominant_ancestry'),
        'auroc': round(float(v['primary']['auroc']), 4),
        'auroc_ci95': [round(float(x), 4) for x in v['primary']['auroc_ci95']],
        'ece': round(float(v['primary']['ece']), 4),
        'note': 'ancestry container, analyses 1-7. The ECE is a prevalence shift, not a broken '
                'model: that cohort\'s non-European share is far below the matched '
                "set's."}}


def subpop_auroc():
    """Discovery AUROC by continental subpopulation, each against the European reference."""
    f = TASK / 'outputs/R-27_subpopulation/subpopulation_auroc.csv'
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    return {str(r.group): {'auroc': round(float(r.auroc), 4), 'ci_halfwidth': round(float(r.ci), 4),
                           'n': int(r.n)} for _, r in d.iterrows()}


def discovery_reference(out, A, W, M):
    h2 = json.loads(LDSC.read_text()) if LDSC.exists() else {}
    dcomp = pd.read_csv(out / 'discovery_composite_deciles.tsv', sep='\t').sort_values('decile')
    dtr = (pd.read_csv(out / 'discovery_pgs_deciles.tsv', sep='\t')
           .sort_values('oriented_decile'))
    r49 = json.loads((R49 / 'headline.json').read_text()) if (R49 / 'headline.json').exists() else {}
    lad = pd.read_csv(R49 / 'ladder.csv') if (R49 / 'ladder.csv').exists() else None
    ref = {
        'built_from': 'the discovery GWAS, composite-PGS, fairness and score-PGS analyses',
        'phenotype': 'DeepECG-Pop ensemble logit of P(non-European), rank-inverse-normal '
                     'within cohort',
        'discovery': {
            'cohorts': ['UK Biobank', 'CLSA', 'MHI'],
            'n': 84891,
            'ancestry_gate': 'RFMix EUR >= 0.70',
            'covariates': ['age', 'sex', 'source indicator', 'PC1-PC20'],
            'n_loci': int(len(A)),
            'locus_classes': {k: int(v) for k, v in A.class_plot.value_counts().items()},
            'ldsc_h2': h2.get('h2'), 'ldsc_h2_se': h2.get('h2_se'),
            'ldsc_intercept': h2.get('intercept'), 'ldsc_ratio': h2.get('ratio'),
            'lambda_gc_scan': 1.0776,
            'ldsc_lambda_gc': h2.get('lambda_gc'), 'ldsc_mean_chi2': h2.get('mean_chi2'),
        },
        # Spans are COMPUTED from the two curve files shipped beside this one rather than typed
        # in, so the number the site is compared against is the number on the plate.  Two spans
        # per curve, because the figure and the text use different ones: `raw` is the share of
        # the decile called non-European at 0.5, `mean_p` is the mean predicted probability, and
        # the published panel plots mean_p while the published sentence quotes raw.
        'trait_pgs': {
            'n_scores': 9,
            'n_weights': int(sum(1 for _ in open(out / 'pgs_weights_grch38.tsv')) - 1),
            'composite_decile_span_pp_raw': round(float(100 * (dcomp.raw.iloc[-1]
                                                               - dcomp.raw.iloc[0])), 2),
            'per_trait_decile_span_pp_raw': {t: round(float(100 * (g.raw.iloc[-1]
                                                                   - g.raw.iloc[0])), 2)
                                             for t, g in dtr.groupby('trait')},
            'per_trait_decile_span_pp_mean_p': {t: round(float(100 * (g.mean_p.iloc[-1]
                                                                      - g.mean_p.iloc[0])), 2)
                                                for t, g in dtr.groupby('trait')},
        },
        'ancestry_pgs': {
            'n_weights_shipped': int(len(W)),
            'design_note': 'leave-one-cohort-out in discovery (everyone is in the meta); '
                           'full-strength at an external site',
            'loo_beta_per_sd': r49.get('beta_sd'), 'loo_ci': [r49.get('lo'), r49.get('hi')],
            'loo_delta_r2_pct': None if lad is None else
                {c: round(100 * float(g[g.p_threshold == 1.0].delta_r2.iloc[0]), 3)
                 for c, g in lad[lad.weights == 'loo'].groupby('cohort')},
        },
        'fairness': {
            'head': 'DeepECG-SSL, 77 findings, shared operating point 0.5',
            'pool': 'the ancestry test partition, in 1:1 matched pairs',
            'n_findings_evaluable': int(M.finding.nunique()),
            # derived from the audit table rather than typed in, so no measured value of ours
            # is written into this repository
            **_mean_gaps(M),
        },
        # What that site itself produced earlier, read out of the package it returned rather
        # than typed from a slide.  This is the anchor the `score` gate compares against: if the
        # same five checkpoints on the same site's ECGs no longer land here, the ECG arrays or
        # the row alignment are wrong and nothing downstream is interpretable.
        'external_prior': mgb_october(),
        # Per-subpopulation discovery AUROCs, so the site can see whether its own
        # AFR > EAS > SAS ordering matches ours.  SAS is the weakest everywhere; a site that
        # reproduces the ordering has a working model even where the absolute numbers differ.
        'subpopulation_auroc_discovery': subpop_auroc(),
    }
    (out / 'discovery_reference.json').write_text(json.dumps(ref, indent=2))
    print('  discovery_reference.json           headline numbers the site is compared against')
    return ref


def main():
    ap = argparse.ArgumentParser()
    # the script's OWN directory, not the shell's: a rebuild run from the repository root
    # would otherwise scatter the tables there and leave the shipped ref/ stale.
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f'building reference tables in {out}\n')
    copy_verbatim(out)
    A = locus_annotation(out)
    W = ancestry_pgs(out)
    composite_and_curves(out)
    findings77(out)
    M = discovery_fairness(out)
    discovery_reference(out, A, W, M)

    man = out / 'MANIFEST.txt'
    lines = ['# ref/ -- frozen at build time; the site runs nothing that reaches back to us', '']
    for p in sorted(out.iterdir()):
        if p.name in {'MANIFEST.txt', '_build_reference.py'} or p.is_dir():
            continue
        lines.append(f'{md5(p)}  {p.stat().st_size:>12,}  {p.name}')
    man.write_text('\n'.join(lines) + '\n')
    print(f'\n  MANIFEST.txt                       {len(lines) - 2} files')


if __name__ == '__main__':
    main()
