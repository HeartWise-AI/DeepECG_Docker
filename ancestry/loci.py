"""Mode `loci` -- do the discovery leads associate with the ECG ancestry score at this site?

THE QUESTION, AND WHY IT IS NOT ONE QUESTION PER LEAD.  The leads reached genome-wide
significance in a within-European meta-analysis many times the size of a single hospital's
European stratum, so asking each lead to clear 5e-8 again -- or even a Bonferroni cut over the
whole lead set -- is asking for something the
sample cannot deliver and calling its absence a failure.  The headline is therefore the AGGREGATE:
how many of the leads point the same way (sign concordance, binomial), and what the slope of their
effects on ours is (inverse-variance weighted, through the origin).  A slope of 1 means the
discovery effects reproduce at full size; a slope below 1 with concordant signs means they
reproduce attenuated, which is what winner's curse predicts and what a reader should expect.
Per-lead p-values are reported beside it and are the supporting detail, not the claim.

THE NON-EUROPEAN STRATA ARE A DIFFERENT QUESTION AND ARE READ DIFFERENTLY.  In AFR, EAS and SAS the
leads are being asked something they have never been asked.  Two things work against them there
and neither is evidence about the biology: the strata are small, and a European lead SNP tags its
causal variant through European LD, which does not hold in African-ancestry genomes in particular.
An attenuated or absent slope in AFR is therefore consistent with the signal being perfectly real
and the instrument being wrong for that population.  The container reports the expected power for
each stratum next to the result, so an underpowered null is not read as a negative.

ORIENTATION.  Every effect is turned to OUR effect allele before anything is compared.  A site
that counts the other allele reports the opposite sign, and a sign-concordance statistic computed
without harmonising would be a coin flip dressed as a result.  A variant whose alleles match ours
in neither orientation is dropped and counted, never guessed at.

    python -m ancestry loci --work /out/work --results /out/results [--strata EUR AFR EAS SAS ADMIXED]

Writes results/loci/.  Everything there is per-variant or per-stratum; no person appears.
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

ALPHA = 0.05


def lead_table(ref: Path, build: str):
    sfx = 'grch37' if build == 'grch37' else 'grch38'
    D = pd.read_csv(ref / f'leads_{sfx}.tsv', sep='\t', dtype={'CHR': str})
    D['CHR'] = D.CHR.astype(str).str.replace('chr', '', regex=False)
    ann = ref / 'locus_annotation.tsv'
    if ann.exists():
        A = pd.read_csv(ann, sep='\t', dtype={'chr': str})
        D = D.merge(A[['key', 'rsid', 'class_plot', 'nearest_gene']]
                    .rename(columns={'key': 'KEY', 'nearest_gene': 'GENE_ANN'}),
                    on='KEY', how='left')
    return D


def write_permuted_pheno(source, destination, seed=2000):
    """Shuffle phenotype values without allowing pandas to reinterpret sample identifiers."""
    P = pd.read_csv(source, sep='\t',
                    dtype={'#FID': str, 'FID': str, '#IID': str, 'IID': str})
    # the ids keep their order and the phenotype values move, so every join downstream is
    # unchanged and only the pairing of person to phenotype is destroyed.
    # Addressed BY NAME, never by position: the file carries #FID and IID before the
    # phenotype, and `columns[1]` would permute the sample ids themselves -- which breaks
    # the FID/IID pairing and reads downstream as "no entries correspond to loaded sample
    # IDs" rather than as a shuffled phenotype.
    idc = [c for c in P.columns if c.lstrip('#').upper() in ('FID', 'IID')]
    val = [c for c in P.columns if c not in idc]
    if not val:
        raise SystemExit(f'{source}: no phenotype column beside {idc}')
    rng = np.random.default_rng(seed)
    for c in val:
        P[c] = rng.permutation(P[c].to_numpy())
    P.to_csv(destination, sep='\t', index=False)
    return Path(destination)


def run_stratum(st, work, ref, leads, build, threads, unrelated_only=True, permute=False):
    """The discovery leads in one stratum.  With `permute`, the SAME genotypes and covariates against a
    shuffled phenotype -- the negative control that makes the real number readable.

    WHY THE CONTROL IS NOT OPTIONAL.  The headline here is a count: how many leads point
    the same way as in discovery.  A count has no scale of its own.  Fifty-five of sixty-two
    sounds decisive and is decisive; forty of sixty-two sounds like weak support and is in fact
    nothing at all, because shuffling the phenotype gives thirty-one.  Rather than ask a reader
    to hold the binomial null in their head, the shuffled run is done for real, through the same
    plink call on the same people, and printed on the same table.  It also catches a whole class
    of pipeline error -- a phenotype joined to the wrong rows, a covariate file misaligned --
    that a null result alone would never reveal, because such a bug makes BOTH runs look alike.
    """
    d = Path(work) / 'geno' / st
    if not (d / 'pheno.tsv').exists():
        return None, f'{st}: no analysis set (run `geno`)'
    rng = d / 'lead_ranges.txt'
    G.write_extract(leads, rng)
    keep = d / 'keep_unrelated.txt'
    use_unrel = unrelated_only and keep.exists()
    pheno = d / 'pheno.tsv'
    if permute:
        pheno = write_permuted_pheno(pheno, d / 'pheno_permuted.tsv')
    A = G.glm(Path(work) / 'geno' / 'panel', pheno, d / 'covar.tsv',
              d / ('leads_perm' if permute else
                   'leads_unrel' if use_unrel else 'leads_all'),
              keep=str(keep) if use_unrel else None, extract=str(rng), threads=threads,
              maf=0.001, mac=10)
    A['CHR'] = A.CHR.astype(str).str.replace('chr', '', regex=False)
    A = A.rename(columns={'POS': 'POS', 'A1_FREQ': 'FREQ_SITE', 'OBS_CT': 'N_SITE'})
    if 'POS' not in A.columns:
        A['POS'] = A['BP'] if 'BP' in A.columns else np.nan

    M = leads.merge(A[['CHR', 'POS', 'A1', 'REF', 'ALT', 'BETA', 'SE', 'P', 'FREQ_SITE',
                       'N_SITE']].rename(columns={'A1': 'SITE_A1'}),
                    on=['CHR', 'POS'], how='left', suffixes=('', '_site'))
    other = np.where(M.SITE_A1 == M.ALT, M.REF, M.ALT)
    M['sign'] = L.allele_sign(M.SITE_A1, other, M.A1, M.A0)
    M['site_beta'] = M.BETA * M.sign
    M['site_freq'] = np.where(M.sign > 0, M.FREQ_SITE, 1 - M.FREQ_SITE)
    M['status'] = np.where(M.BETA.isna(), 'absent',
                           np.where(~np.isfinite(M.sign), 'allele mismatch', 'tested'))
    return M, ('permuted phenotype (negative control)' if permute
               else 'unrelated' if use_unrel else 'everyone')


def summarise(M, st, kind):
    """Sign concordance, the weighted slope, the aggregate z, and the power this n bought."""
    T = M[M.status == 'tested'].copy()
    out = dict(stratum=st, sample=kind, n_leads=int(len(M)), n_tested=int(len(T)),
               n_absent=int((M.status == 'absent').sum()),
               n_allele_mismatch=int((M.status == 'allele mismatch').sum()))
    if len(T) < 5:
        out['note'] = 'too few leads tested to summarise'
        return out, None
    disc, site, se = T.disc_beta.to_numpy(), T.site_beta.to_numpy(), T.SE.to_numpy()
    n_conc = int((np.sign(disc) == np.sign(site)).sum())
    out['n_concordant'] = n_conc
    out['pct_concordant'] = round(100 * n_conc / len(T), 1)
    out['sign_binomial_p'] = float(stats.binomtest(n_conc, len(T), 0.5,
                                                   alternative='greater').pvalue)
    w = 1 / se ** 2
    slope = float((w * disc * site).sum() / (w * disc * disc).sum())
    slope_se = float(np.sqrt(1 / (w * disc * disc).sum()))
    out['slope'] = round(slope, 4)
    out['slope_se'] = round(slope_se, 4)
    out['slope_lo'] = round(slope - 1.96 * slope_se, 4)
    out['slope_hi'] = round(slope + 1.96 * slope_se, 4)
    out['slope_p_vs_0'] = float(2 * stats.norm.sf(abs(slope / slope_se)))
    out['slope_p_vs_1'] = float(2 * stats.norm.sf(abs((slope - 1) / slope_se)))
    out['pearson_r'] = round(float(np.corrcoef(disc, site)[0, 1]), 4)
    # The site effect is harmonised to the discovery effect allele, but discovery effects can be
    # positive OR negative.  Orient each z toward replication before combining, otherwise equal
    # positive and negative discovery effects cancel even when every lead reproduces perfectly.
    z = np.sign(disc) * site / se
    out['aggregate_z'] = round(float(z.sum() / np.sqrt(len(z))), 3)
    out['aggregate_p'] = float(2 * stats.norm.sf(abs(out['aggregate_z'])))
    out['n_p05_same_sign'] = int(((T.P < ALPHA) & (np.sign(disc) == np.sign(site))).sum())
    out['n_bonferroni'] = int((T.P < ALPHA / len(T)).sum())
    out['n_genomewide'] = int((T.P < 5e-8).sum())
    out['median_n'] = int(np.nanmedian(T.N_SITE))
    # power: what this many people SHOULD have delivered if every discovery effect were exact
    ncp = (disc / se) ** 2
    crit_b = stats.chi2.isf(ALPHA / len(T), 1)
    crit_5 = stats.chi2.isf(ALPHA, 1)
    out['expected_n_bonferroni'] = round(float(stats.ncx2.sf(crit_b, 1, ncp).sum()), 1)
    out['expected_n_p05'] = round(float(stats.ncx2.sf(crit_5, 1, ncp).sum()), 1)
    out['mean_power_bonferroni'] = round(float(stats.ncx2.sf(crit_b, 1, ncp).mean()), 3)
    return out, T


def pooled_non_european(per_lead, leads):
    """One inverse-variance meta across every non-European stratum, per lead, then summarised.

    WHY THIS ARM EXISTS.  At a single hospital the non-European strata are small -- typically a
    few hundred people each with a linked ECG.  A lead-level replication in a few hundred people
    has no power at all, and four separate underpowered nulls invite exactly the wrong reading.
    Meta-analysed they are a few thousand, which is the only non-European statement this design
    can actually support, so it is computed and reported as one arm rather than left for a reader
    to assemble by eye.

    The strata are independent samples of different people, so a fixed-effect inverse-variance
    meta is the right combination, and Cochran's Q over the four is reported beside it because
    heterogeneity here is interesting in itself: these are different continental LD patterns
    tagging the same European lead SNPs.
    """
    D = pd.concat(per_lead, ignore_index=True)
    D = D[(D['sample'] == 'unrelated') & (D.stratum != L.PRIMARY) & (D.status == 'tested')]
    if D.stratum.nunique() < 2:
        return None, None
    rows = []
    for key, g in D.groupby('KEY'):
        m = L.meta_fe(g.site_beta.to_numpy(), g.site_se.to_numpy())
        if not np.isfinite(m['beta']):
            continue
        r = g.iloc[0]
        rows.append(dict(KEY=key, CHR=r.CHR, POS=r.POS, A1=r.A1, A0=r.A0,
                         rsid=r.get('rsid'), class_plot=r.get('class_plot'),
                         disc_beta=r.disc_beta, disc_se=r.disc_se, disc_freq=r.disc_freq,
                         site_beta=m['beta'], SE=m['se'], P=m['p'],
                         site_freq=float(g.site_freq.mean()),
                         N_SITE=int(g.site_n.sum()), k_strata=m['k'], het_q=m['q'], het_i2=m['i2'],
                         status='tested'))
    if not rows:
        return None, None
    M = pd.DataFrame(rows)
    M['concordant'] = np.sign(M.site_beta) == np.sign(M.disc_beta)
    # leads that no non-European stratum could test are still part of the denominator
    miss = leads[~leads.KEY.isin(M.KEY)]
    if len(miss):
        M = pd.concat([M, miss.assign(status='absent', site_beta=np.nan, SE=np.nan, P=np.nan,
                                      N_SITE=0, concordant=np.nan)], ignore_index=True)
    sm, _ = summarise(M, 'NON_EUR_META', 'unrelated')
    sm['note'] = ('fixed-effect meta of every non-European stratum; strata are independent '
                  'people, LD differs between them')
    out = M[[c for c in ['KEY', 'CHR', 'POS', 'A1', 'A0', 'rsid', 'class_plot', 'disc_beta',
                         'disc_se', 'disc_freq', 'site_beta', 'SE', 'P', 'site_freq', 'N_SITE',
                         'k_strata', 'het_i2', 'status', 'concordant'] if c in M.columns]].copy()
    out = out.rename(columns={'SE': 'site_se', 'P': 'site_p', 'N_SITE': 'site_n'})
    out.insert(0, 'sample', 'unrelated')
    out.insert(0, 'stratum', 'NON_EUR_META')
    return sm, out


def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry loci', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True,
                    help="the site's working directory: intermediates, per-person tables and "
                         'arrays. It stays at the site and is never packaged')
    ap.add_argument('--results', required=True,
                    help='the aggregate-only output directory; `package` zips exactly this')
    ap.add_argument('--ref', default=str(L.ref_dir()),
                    help='the frozen discovery tables (default: $REFERENCE_DIR, else the '
                         'reference/ folder of the pinned weights snapshot)')
    ap.add_argument('--build', default=None,
                    help='genome build; normally inherited from `geno`')
    ap.add_argument('--strata', nargs='+', default=None,
                    help='which ancestry strata to run (default: every stratum `geno` built)')
    ap.add_argument('--skip-permutation', action='store_true',
                    help='skip the shuffled-phenotype negative control (it is one extra '
                         'plink call over a few dozen variants; there is rarely a reason)')
    ap.add_argument('--threads', type=int, default=L.default_threads(),
                    help='threads for plink (default: this machine\'s cores, less two, capped at 16)')
    ap.add_argument('--include-related', action='store_true',
                    help='also report the analysis without the kinship filter')
    a = ap.parse_args()

    work = Path(a.work); res = Path(a.results) / 'loci'
    L.require('loci', [(Path(a.results) / 'geno' / 'geno_provenance.json',
                        'the per-stratum phenotype, covariates and variant panel', 'geno')])
    res.mkdir(parents=True, exist_ok=True)
    ref = Path(a.ref)
    prov = json.loads((Path(a.results) / 'geno' / 'geno_provenance.json').read_text())
    build = L.build_suffix(a.build or prov.get('build', 'grch38'))
    strata = a.strata or prov.get('strata', [])
    leads = lead_table(ref, build)
    leads = leads.rename(columns={'BETA': 'disc_beta', 'SE': 'disc_se', 'FREQ1': 'disc_freq',
                                  'NEGLOG10P': 'disc_neglog10p', 'N_TOTAL': 'disc_n'})
    L.banner(f'loci: the discovery leads in {", ".join(strata)}')

    rows, per_lead = [], []
    for st in strata:
        arms = [dict(unrelated_only=True)] + \
               ([dict(unrelated_only=False)] if a.include_related else []) + \
               ([] if a.skip_permutation else [dict(unrelated_only=True, permute=True)])
        for kw in arms:
            M, kind = run_stratum(st, work, ref, leads, build, a.threads, **kw)
            if M is None:
                print(f'  {kind}')
                continue
            M['concordant'] = np.where(M.status == 'tested',
                                       np.sign(M.site_beta) == np.sign(M.disc_beta), np.nan)
            s, T = summarise(M, st, kind)
            rows.append(s)
            print(f'\n  {st} ({kind}): {s.get("n_tested", 0)}/{len(M)} tested, '
                  f'{s.get("pct_concordant", float("nan"))}% concordant, '
                  f'slope {s.get("slope", float("nan"))} '
                  f'[{s.get("slope_lo", float("nan"))}, {s.get("slope_hi", float("nan"))}], '
                  f'aggregate P {s.get("aggregate_p", float("nan")):.3g}')
            keep = ['KEY', 'CHR', 'POS', 'A1', 'A0', 'rsid', 'GENE_ANN', 'NEAREST_GENE',
                    'class_plot', 'disc_beta', 'disc_se', 'disc_freq', 'disc_neglog10p',
                    'site_beta', 'SE', 'P', 'site_freq', 'N_SITE', 'status', 'concordant']
            out = M[[c for c in keep if c in M.columns]].copy()
            out.insert(0, 'sample', kind)
            out.insert(0, 'stratum', st)
            out = out.rename(columns={'SE': 'site_se', 'P': 'site_p', 'N_SITE': 'site_n',
                                      'NEAREST_GENE': 'nearest_gene_lead'})
            per_lead.append(out)

    if not rows:
        raise SystemExit('no stratum produced a lead analysis; run `geno` first')

    sm, om = pooled_non_european(per_lead, leads)
    if sm is not None:
        rows.append(sm); per_lead.append(om)
        print(f'\n  NON_EUR_META: {sm.get("n_tested", 0)}/{len(leads)} tested in '
              f'{int(om.k_strata.max())} strata, {sm.get("pct_concordant")}% concordant, '
              f'slope {sm.get("slope")} [{sm.get("slope_lo")}, {sm.get("slope_hi")}]')

    S = pd.DataFrame(rows)
    PV.write_table(S, res / 'replication_summary.csv', where='replication_summary.csv')
    PL = pd.concat(per_lead, ignore_index=True)
    # a per-variant effect and allele frequency is an aggregate only in a stratum big enough that
    # an allele count cannot be read back to a person; genoprep already refuses to build an
    # analysis set below the floor, and this is the assertion that keeps that true if it changes
    small = PL[(PL.site_n.fillna(0) > 0) & (PL.site_n < PV.MIN_N_FOR_VARIANT_STATS)]
    if len(small):
        raise SystemExit(f'refusing to export variant statistics computed in fewer than '
                         f'{PV.MIN_N_FOR_VARIANT_STATS} people '
                         f'({small.stratum.unique().tolist()})')
    PV.write_table(PL, res / 'leads_per_stratum.csv', where='leads_per_stratum.csv')

    # S['sample'], not S.sample: `sample` is a DataFrame METHOD, so attribute access returns
    # the bound method and the comparison silently evaluates to the scalar False -- which then
    # raised AttributeError on .any() every single run, after all the plink work was done.
    unrel = S['sample'].astype(str) == 'unrelated'
    head = S[unrel] if bool(unrel.any()) else S
    best = head[head.stratum == L.PRIMARY]
    (res / 'loci_headline.json').write_text(json.dumps(dict(
        version=L.VERSION, build=build,
        primary=None if not len(best) else best.iloc[0].to_dict(),
        all_strata=S.to_dict('records')), indent=2, default=float))
    print('\n' + S[['stratum', 'sample', 'n_tested', 'pct_concordant', 'sign_binomial_p', 'slope',
                    'slope_lo', 'slope_hi', 'aggregate_p', 'n_bonferroni',
                    'expected_n_bonferroni']].to_string(index=False))
    print(f'\n  wrote {res}/')


if __name__ == '__main__':
    main()
