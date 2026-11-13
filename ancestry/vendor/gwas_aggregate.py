"""Reduce the per-chromosome plink2 scan to the handful of aggregate tables a Manhattan needs.

WHAT THIS IS FOR.  03_gwas.sh leaves 22 full summary-statistic files on site -- tens of millions
of rows, one row per variant.  A full GWAS summary statistic is an export in its own right and
needs its own Lifelines approval; this script does not assume one.  It writes instead the small
set of tables a supplementary Manhattan and its cross-checks are built from:

    ll_manhattan.tsv        the thinned point cloud (~400k rows), CHR POS NEGLOG10P
    ll_qq.tsv               log-spaced quantiles of the observed against the null, + lambda_GC
    ll_leads.tsv            genome-wide-significant leads, distance-clumped
    ll_at_discovery_leads.tsv   the discovery leads looked up in the FULL scan
    ll_gwas_summary.json    n variants, lambda_GC, how many leads, the QC that was applied
    MANIFEST.txt            every file written, its rows and its md5

    python 04_gwas_aggregate.py --glm gwas/chr --out results_gwas/ --ref ref/ --build grch37

THE THINNING RULE IS THE DISCOVERY FIGURE'S, not a new one.  Keeping every variant with
P < 1e-4 and 1 in 50 of the rest -- the obvious rule, and the one stage 07 used -- puts a 25-fold
density step at exactly -log10(P) = 4 and draws a white band across the panel just under it,
which reads as a GWAS artefact and is not one.  Retention is min(1, 10^(0.5 (x - 4))) instead:
it is 1 at x = 4, so density is continuous there, and falls smoothly below.  Same seed (2000),
so the picture is reproducible.

NOTHING PER-PARTICIPANT IS WRITTEN HERE.  The smallest cell any output reports is a variant's
allele frequency and effect in ~35,000 people.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

SIG = 5e-8
SEED = 2000


def read_glm(pattern, chroms):
    """plink2 --glm output for each chromosome, reduced to the columns that are used.

    plink2 writes <prefix><chr>.Y.glm.linear; a few builds write .glm.linear without the
    phenotype name, and both are accepted rather than making the caller care.
    """
    frames = []
    # plink2 names the file after --out plus the phenotype column, and a few builds and
    # modifiers change that (no phenotype name, PHENO1, a .zst or .gz when output is
    # compressed).  All of them are accepted rather than making the caller care which.
    sfx = ('.pheno.glm.linear', '.Y.glm.linear', '.glm.linear', '.PHENO1.glm.linear',
           '.pheno.glm.linear.zst', '.pheno.glm.linear.gz',
           '.Y.glm.linear.zst', '.Y.glm.linear.gz',
           '.pheno.glm.logistic', '.Y.glm.logistic')
    for c in chroms:
        hits = [q for q in (Path(f'{pattern}{c}{s}') for s in sfx) if q.exists()]
        if not hits:
            print(f'! chr{c}: nothing at {pattern}{c}{{{",".join(sfx)}}}')
            continue
        if hits[0].name.endswith('.logistic'):
            sys.exit(f'{hits[0]} is a LOGISTIC fit: plink2 read the phenotype as a case/control '
                     'column.\n  The ancestry score is quantitative; check that pheno.tsv column '
                     'Y is numeric and not 0/1.')
        d = pd.read_csv(hits[0], sep='\t', low_memory=False)
        d.columns = [c_.lstrip('#') for c_ in d.columns]
        # plink2 emits one row per tested term unless --glm hide-covar; keep the variant's own.
        if 'TEST' in d.columns:
            d = d[d.TEST == 'ADD']
        d = d[d.P.notna() & (d.P > 0)]
        keep = [k for k in ('CHROM', 'POS', 'ID', 'REF', 'ALT', 'A1', 'A1_FREQ', 'MACH_R2',
                            'OBS_CT', 'BETA', 'SE', 'P') if k in d.columns]
        d = d[keep].copy()
        # int, NOT the string the --chroms list is split into: every lookup downstream keys on
        # (int(CHR), int(POS)) from the lead table, and against a string CHROM every one of the
        # lead set misses silently and the run reports "0 of N present".
        d['CHROM'] = int(c)
        frames.append(d)
        print(f'  chr{c}: {len(d):,} variants')
    if not frames:
        d = Path(pattern).parent
        there = sorted(q.name for q in d.glob('*')) if d.is_dir() else []
        logs = [q for q in (d / f'chr{c}.scan.log' for c in chroms) if q.exists()]
        msg = [f'no association results found under {pattern}*',
               f'  {d} holds {len(there)} files'
               + (': ' + ', '.join(there[:14]) + (' ...' if len(there) > 14 else '')
                  if there else ' (it is empty)')]
        if logs:
            msg.append(f'  the last lines of {logs[0]}, which is what plink2 itself said:')
            msg += ['    | ' + ln for ln in
                    logs[0].read_text(errors='replace').splitlines()[-12:]]
        else:
            msg.append('  and there is no chr*.scan.log either, so plink2 was never reached --'
                       '\n  the scan step was skipped. Re-run WITHOUT --aggregate-only, and '
                       'check that\n  --geno resolves for these chromosomes.')
        sys.exit('\n'.join(msg))
    return pd.concat(frames, ignore_index=True)


def thin(d, rng):
    """The discovery figure's density-continuous thinning."""
    x = d.NEGLOG10P.values
    keep = rng.random(len(x)) < np.minimum(1.0, 10.0 ** (0.5 * (x - 4.0)))
    return d[keep]


def clump(d, window=500_000):
    """Distance clumping: the strongest variant, then everything within `window` of it, repeat.

    No LD reference is needed and none is available on site, so this is deliberately a DISTANCE
    rule, not an r2 rule, and the count it produces is an upper bound on independent loci. The
    discovery loci were called the same way (07_loci.py), so the two are comparable.
    """
    out = []
    for c, g in d.groupby('CHROM'):
        g = g.sort_values('P').reset_index(drop=True)
        taken = []
        for r in g.itertuples():
            if all(abs(r.POS - t) > window for t in taken):
                taken.append(r.POS)
                out.append(r)
    return pd.DataFrame(out).drop(columns=['Index'], errors='ignore')


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def matching_allele_rows(rows, a1, a0):
    """Return only variants whose complete REF/ALT pair matches the discovery pair."""
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T
    if not {'REF', 'ALT'} <= set(rows.columns):
        return rows.iloc[0:0]
    expected = {str(a1).upper(), str(a0).upper()}
    ref = rows.REF.astype(str).str.upper()
    alt = rows.ALT.astype(str).str.upper()
    return rows[(ref.isin(expected)) & (alt.isin(expected)) & (ref != alt)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glm', required=True,
                    help="prefix the per-chromosome files share, e.g. gwas/chr -> gwas/chr1.Y...")
    ap.add_argument('--out', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--build', default='grch37', choices=['grch37', 'grch38'])
    ap.add_argument('--chroms', default=' '.join(str(i) for i in range(1, 23)))
    ap.add_argument('--maf', type=float, default=0.01)
    ap.add_argument('--info-min', type=float, default=0.4,
                    help='minimum MACH_R2, when plink2 wrote that column (imputed dosages)')
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = read_glm(a.glm, a.chroms.split())
    n_raw = len(d)

    # Kept whole, so a lead can be told apart from a lead that failed QC: "absent from the
    # release" and "imputed badly here" are different facts and the figure reports them apart.
    d_all = d[[c for c in ('CHROM', 'POS', 'REF', 'ALT') if c in d.columns]].copy()

    qc = {}
    if 'A1_FREQ' in d.columns and a.maf > 0:
        f = d.A1_FREQ.astype(float)
        n0 = len(d); d = d[(f >= a.maf) & (f <= 1 - a.maf)]
        qc['dropped_maf'] = n0 - len(d)
    if 'MACH_R2' in d.columns and a.info_min > 0:
        n0 = len(d); d = d[d.MACH_R2.astype(float) >= a.info_min]
        qc['dropped_info'] = n0 - len(d)
    # This stands in for the per-variant missingness filter (--geno) that an array GWAS would
    # run and an imputed one cannot: a dosage release has a value for every sample at every
    # variant, so OBS_CT is the keep-list size at every line and the filter would drop nothing.
    # Saying so out of the data beats asserting it: if OBS_CT is NOT flat, the release is not
    # fully imputed (hardcalls with missingness, or a chromosome that lost people) and the
    # missingness question comes back.
    if 'OBS_CT' in d.columns:
        o = d.OBS_CT.astype(int)
        qc['obs_ct_min'], qc['obs_ct_max'] = int(o.min()), int(o.max())
        if o.min() < 0.98 * o.median():
            print(f'! OBS_CT is not flat: {o.min():,} to {o.max():,} (median {int(o.median()):,}).'
                  '\n  A fully imputed dosage release has every sample at every variant. Some of'
                  '\n  these variants are hardcalls with missing genotypes, so the usual --geno'
                  '\n  missingness filter DOES apply to them; check the release before using'
                  '\n  the low-OBS_CT tail.')
    print(f'{len(d):,} variants after QC (from {n_raw:,})' +
          ''.join(f'; {k} {v:,}' for k, v in qc.items()))

    d['NEGLOG10P'] = -np.log10(d.P.astype(float))

    # lambda_GC over EVERY tested variant, from the median chi2 -- not re-derived from the
    # thinned cloud or from the log-spaced QQ file, both of which land thousands of ranks off
    # n/2 and give a value a few thousandths wrong.
    chi = chi2.isf(d.P.astype(float), 1)
    lam = float(np.median(chi) / chi2.ppf(0.5, 1))
    print(f'lambda_GC = {lam:.4f} over {len(d):,} variants')

    rng = np.random.default_rng(SEED)
    mh = thin(d[['CHROM', 'POS', 'NEGLOG10P']], rng).rename(columns={'CHROM': 'CHR'})
    mh = mh.sort_values(['CHR', 'POS'])
    mh.to_csv(out / 'll_manhattan.tsv', sep='\t', index=False,
              float_format='%.4f')

    # QQ, log-spaced in rank: 200 points carry a 20-million-variant plot without a 20-million-row
    # file.  Expected is -log10((i - 0.5) / n) at the ORIGINAL rank i, so the axis is the real
    # one even though the rows are sparse.
    p = np.sort(d.P.astype(float).values)
    n = len(p)
    idx = np.unique(np.r_[np.arange(1, 101),
                          np.geomspace(101, n, 400).astype(int)]) - 1
    qq = pd.DataFrame({'expected': -np.log10((idx + 0.5) / n),
                       'observed': -np.log10(p[idx])})
    qq.to_csv(out / 'll_qq.tsv', sep='\t', index=False, float_format='%.5f')

    sig = d[d.P.astype(float) < SIG]
    leads = clump(sig) if len(sig) else sig.copy()
    if len(leads):
        leads = leads.sort_values(['CHROM', 'POS'])
    leads.to_csv(out / 'll_leads.tsv', sep='\t', index=False)
    print(f'{len(sig):,} variants at P < 5e-8 in {len(leads)} distance-clumped loci')

    # The discovery leads, looked up in the full scan.  This is NOT the same table
    # 02_validate.py wrote: that one was regressed on the extracted dosages for 62 positions,
    # this one falls out of the genome-wide fit.  Agreement between them is a check that the
    # scan and the targeted replication are the same analysis; disagreement means one of the
    # two has a different analysis set or a different model, and the Manhattan should not be
    # shown beside the lead scatter until it is understood.
    L = pd.read_csv(Path(a.ref) / f'leads_{a.build}.tsv', sep='\t')
    key = d.set_index(['CHROM', 'POS'])
    release_key = d_all.set_index(['CHROM', 'POS'])
    rows = []
    for r in L.itertuples():
        cp = (int(r.CHR), int(r.POS))
        try:
            in_release = len(matching_allele_rows(release_key.loc[cp], r.A1, r.A0)) > 0
        except KeyError:
            in_release = False
        try:
            hit = key.loc[cp]
        except KeyError:
            rows.append({'KEY': r.KEY, 'CHR': r.CHR, 'POS': r.POS, 'found': False,
                         'in_release': in_release, 'nearest_gene': r.NEAREST_GENE})
            continue
        hit = matching_allele_rows(hit, r.A1, r.A0)
        if not len(hit):
            rows.append({'KEY': r.KEY, 'CHR': r.CHR, 'POS': r.POS, 'found': False,
                         'in_release': in_release, 'nearest_gene': r.NEAREST_GENE})
            continue
        hit = hit.iloc[hit.P.astype(float).values.argmin()]
        rows.append({'KEY': r.KEY, 'CHR': r.CHR, 'POS': r.POS, 'found': True,
                     'in_release': in_release,
                     'A1': hit.get('A1'), 'discovery_A1': r.A1, 'discovery_beta': r.BETA,
                     'beta': hit.BETA, 'se': hit.SE, 'p': hit.P,
                     'a1_freq': hit.get('A1_FREQ'), 'nearest_gene': r.NEAREST_GENE})
    A = pd.DataFrame(rows)
    # ALIGN TO THE DISCOVERY EFFECT ALLELE. plink2 counts whichever allele it counts, and at
    # about half of these leads that is the other one, so the raw betas are sign-concordant with
    # discovery exactly at chance (28/55 here) and an IVW slope comes out near 0.16 instead of
    # near 1. Everything else in the scan is P-based and immune, which is what makes this one
    # quiet: the Manhattan, the QQ and the locus count all look perfect while the single
    # comparison the figure exists to make is noise. The raw column is kept beside the aligned
    # one so the flip is auditable rather than assumed.
    if len(A) and {'A1', 'discovery_A1'} <= set(A.columns):
        same = A.A1.astype(str).str.upper() == A.discovery_A1.astype(str).str.upper()
        A['beta_aligned'] = np.where(same, A.beta, -A.beta)
        if 'a1_freq' in A.columns:
            A['a1_freq_aligned'] = np.where(same, A.a1_freq, 1 - A.a1_freq)
        A['allele_flipped'] = ~same
        tt = A[A.beta_aligned.notna()]
        if len(tt):
            conc = int((np.sign(tt.beta_aligned) == np.sign(tt.discovery_beta)).sum())
            print(f'effect alleles: {int((~same).sum())} of {len(A)} leads counted the other '
                  f'allele and were flipped;\n  {conc} of {len(tt)} are sign-concordant with '
                  'discovery after alignment')
            if conc < 0.7 * len(tt):
                print('! fewer than 70% concordant even after aligning. That is not a coding\n'
                      '  issue; check that the scan and the discovery leads are the same build\n'
                      '  and that strand-ambiguous (A/T, C/G) leads are not driving it.')
    A.to_csv(out / 'll_at_discovery_leads.tsv', sep='\t', index=False)
    nf = int(A.found.sum())
    nqc = int((A.in_release & ~A.found).sum())
    print(f'{nf} of {len(L)} discovery leads present in the scan'
          + (f'; {nqc} more are in the release but failed QC' if nqc else ''))
    if nf < 40:
        print('! most of the discovery leads were not found. Check --build: GRCh37 lead positions\n'
              '  against a GRCh38 scan (or the reverse) match almost nothing, and a handful of\n'
              '  coincidental hits looks like a weak replication rather than a build error.')

    # the analysis set size, so the figure does not have to hard-code it: OBS_CT is per variant
    # and dips where a dosage is missing, so the median is the number of people in the scan.
    n_part = int(d.OBS_CT.median()) if 'OBS_CT' in d.columns else None

    summ = {'n_participants': n_part,
            'n_variants_tested': int(n_raw), 'n_variants_after_qc': int(len(d)),
            'qc': qc, 'maf_min': a.maf, 'info_min': a.info_min,
            'lambda_gc': round(lam, 4),
            'n_sig_variants': int(len(sig)), 'n_loci': int(len(leads)),
            'n_discovery_leads_found': nf, 'n_discovery_leads': int(len(L)),
            'n_discovery_leads_failed_qc': nqc,
            'manhattan_rows': int(len(mh)), 'build': a.build, 'seed': SEED}
    (out / 'll_gwas_summary.json').write_text(json.dumps(summ, indent=2))

    man = ['# every file in this directory, with its row count and md5.',
           '# aggregate only: no row describes fewer than the whole analysis set.']
    for p_ in sorted(out.iterdir()):
        if p_.name == 'MANIFEST.txt':
            continue
        nrow = sum(1 for _ in open(p_, 'rb')) - 1 if p_.suffix in ('.tsv', '.csv') else ''
        man.append(f'{p_.name}\t{nrow}\t{md5(p_)}')
    (out / 'MANIFEST.txt').write_text('\n'.join(man) + '\n')

    print(f'\nwrote {out}  -- return this directory; it is aggregate only')
    print('NOTE: the per-chromosome .glm.linear files are NOT in it.  Exporting a full summary '
          'statistic\n      is a separate request to the Lifelines data manager.')


if __name__ == '__main__':
    main()
