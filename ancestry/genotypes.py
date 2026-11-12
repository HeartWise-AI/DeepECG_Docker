"""Mode `geno` -- everything the genetic analyses share: one variant subset, PCs, kinship, files.

Run once; `loci`, `pgs` and `scan` all read what it leaves in work/geno/.

WHAT IT BUILDS, AND THE REASONING BEHIND EACH CHOICE

  the subset       One bed/bim/fam holding only the variants this image ever uses -- the discovery
                   discovery leads, the 615 trait-PGS variants and the 21,055 ancestry-PGS panel
                   variants.  Extracted BY POSITION, never by variant ID: a hospital release
                   spells IDs however its imputation server did, and matching a spelling we do not
                   control is the commonest way an extraction silently returns an empty file.
                   Alleles are checked against ours afterwards and a variant that matches neither
                   orientation is dropped and counted, never guessed at.

  per-stratum PCs  Twenty principal components computed INSIDE each stratum, on LD-pruned common
                   variants.  Computed across all ancestries they would spend PC1-PC3 on the
                   ancestry split itself and then, used as covariates inside a stratum, adjust
                   away a different thing in each one.  Discovery computed its PCs inside the
                   European gate; every stratum here gets the same treatment inside its own.

  kinship          A maximal unrelated subset at KING 0.0884, the conventional third-degree
                   threshold (KING kinship 2^-3.5) -- the same cutoff as the discovery GWAS and
                   the Lifelines package.  A hospital biobank
                   holds families, and relatedness inflates the test statistic of every analysis
                   below.  Both the full and the unrelated analysis are reported; the unrelated
                   one is the headline.

  pheno / covar    plink-format files per stratum: the rank-inverse-normal ancestry score, and
                   age, sex and PC1-PC20.  The same covariate set as discovery, with the source
                   indicator dropped because MGB has only one source.

    python -m ancestry geno --work /out/work --results /out/results \\
        --geno '/path/to/release/chr@' [--build grch38] [--pca-geno '/path/array'] [--threads 8]

NOTHING LEAVES.  work/geno/ holds genotype subsets and id lists; results/geno/ holds the counts
and the PC variance, which are aggregates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from ancestry import privacy as PV                                       # noqa: E402
from ancestry import lib as L                                            # noqa: E402
from ancestry import geno_ops as G                                       # noqa: E402


def lift_panel(ref, work, build, chain):
    """Lift the ancestry-PGS panel into the site's build, once, using the vendored liftover.

    The discovery leads and the trait-PGS weights ship in both builds. The ancestry-PGS
    panel ships in GRCh38 only, because it is too large to have been lifted by hand and no chain
    file can be bundled -- UCSC's licence aside, this image carries no data it did not generate.
    A GRCh37 site that wants that one arm supplies the chain file itself; everything else runs on
    either build untouched. The lift is a point walk over the chain's aligned blocks: a variant
    in a gap is dropped and counted, never nudged to the nearest base.
    """
    sfx = L.build_suffix(build)
    out = Path(work) / 'ref' / f'ancestry_pgs_weights_{sfx}.tsv'
    if out.exists():
        print(f'  ancestry-PGS panel already lifted: {out}')
        return out
    src = Path(ref) / 'ancestry_pgs_weights_grch38.tsv'
    if not src.exists():
        print(f'  cannot lift: {src} is not in this image')
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'  lifting the ancestry-PGS panel GRCh38 -> {sfx} with {chain} ...')
    import runpy
    argv = sys.argv
    try:
        sys.argv = ['liftover_min.py', str(chain), str(src), str(out),
                    '--chr-col', 'CHR', '--pos-col', 'POS']
        runpy.run_path(str(HERE / 'vendor' / 'liftover_min.py'), run_name='__main__')
    except SystemExit as e:
        if e.code:
            print(f'  liftover failed: {e}')
            return None
    finally:
        sys.argv = argv
    if not out.exists():
        print('  liftover wrote nothing')
        return None
    n = sum(1 for _ in open(out)) - 1
    print(f'  wrote {out}  ({n:,} variants in {sfx}) -- `pgs` will use this automatically')
    return out


def wanted_variants(ref: Path, build: str, work=None):
    """The union of every variant this image scores, with our effect allele, one row each."""
    sfx = L.build_suffix(build)
    leads = pd.read_csv(ref / f'leads_{sfx}.tsv', sep='\t', dtype={'CHR': str})
    leads = leads.assign(SET='lead', OUR_A1=leads.A1, OUR_A0=leads.A0)[
        ['CHR', 'POS', 'OUR_A1', 'OUR_A0', 'SET']]
    tr = pd.read_csv(ref / f'pgs_weights_{sfx}.tsv', sep='\t', dtype={'CHR': str})
    tr = tr.assign(SET='trait_pgs', OUR_A1=tr.RISK_ALLELE,
                   OUR_A0=np.where(tr.PANEL_A1 == tr.RISK_ALLELE, tr.PANEL_A2, tr.PANEL_A1))[
        ['CHR', 'POS', 'OUR_A1', 'OUR_A0', 'SET']]
    parts = [leads, tr]
    apw = L.ref_table(ref, work, 'ancestry_pgs_weights', sfx, required=False)
    if apw is not None:
        ap = pd.read_csv(apw, sep='\t', dtype={'CHR': str})
        parts.append(ap.assign(SET='ancestry_pgs', OUR_A1=ap.A1, OUR_A0=ap.A0)[
            ['CHR', 'POS', 'OUR_A1', 'OUR_A0', 'SET']])
    else:
        print(f'  note: the ancestry-score PGS panel ships in GRCh38 only and this run is {sfx}, '
              f'so that ONE arm is skipped.\n'
              f'        Everything else -- the discovery leads, the nine trait PGSs, the scan, the '
              f'fairness audit -- runs on {sfx} unchanged.\n'
              f'        To get it too, re-run `geno` with --chain hg38ToHg19.over.chain.gz '
              f'(a 1.2 MB UCSC file; nothing is downloaded by this image).')
    V = pd.concat(parts, ignore_index=True)
    V['CHR'] = V.CHR.astype(str).str.replace('chr', '', regex=False)
    V['POS'] = pd.to_numeric(V.POS, errors='coerce').astype('Int64')
    agg = (V.groupby(['CHR', 'POS'], as_index=False)
           .agg(OUR_A1=('OUR_A1', 'first'), OUR_A0=('OUR_A0', 'first'),
                SET=('SET', lambda s: ','.join(sorted(set(s))))))
    print(f'  variants wanted: {len(agg):,} distinct positions '
          + ', '.join(f'{k} {v:,}' for k, v in V.SET.value_counts().items()))
    return agg


def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry geno', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True,
                    help="the site's working directory: intermediates, per-person tables and "
                         'arrays. It stays at the site and is never packaged')
    ap.add_argument('--results', required=True,
                    help='the aggregate-only output directory; `package` zips exactly this')
    ap.add_argument('--geno', required=True, help="genotypes, @ for the chromosome number")
    ap.add_argument('--pca-geno', default=None,
                    help='a different set for the PCs and kinship (e.g. the genotyping array); '
                         'pruning 22 imputed chromosomes is slow and buys nothing')
    ap.add_argument('--pca-single', action='store_true',
                    help='--pca-geno is one whole-genome set, not a per-chromosome pattern')
    ap.add_argument('--build', default='grch38',
                    help='grch38 (aliases hg38, b38) or grch37 (aliases hg19, b37). `survey` '
                         'looks the discovery leads up in both and tells you which your release is')
    ap.add_argument('--chain', default=None,
                    help='a UCSC chain file (e.g. hg38ToHg19.over.chain.gz) used ONCE to lift '
                         'the ancestry-PGS panel into your build. Only needed on GRCh37, and '
                         'only for that one arm; nothing is downloaded by this image')
    ap.add_argument('--ref', default=str(L.ref_dir()),
                    help='the frozen discovery tables (default: $REFERENCE_DIR, else the '
                         'reference/ folder of the pinned weights snapshot)')
    ap.add_argument('--strata', nargs='+', default=None,
                    help='which ancestry strata to run (default: every stratum `geno` built)')
    ap.add_argument('--threads', type=int, default=L.default_threads(),
                    help='threads for plink (default: this machine\'s cores, less two, capped at 16)')
    ap.add_argument('--npc', type=int, default=20,
                    help='within-stratum genotype PCs to compute and adjust for (default 20, '
                         'the discovery value)')
    ap.add_argument('--rfmix-covar', choices=['auto', 'always', 'never'],
                    default='auto',
                    help='add the measured RFMix proportions to the covariates. '
                         'auto (default) = every stratum except EUR, where they were shown to change nothing.')
    ap.add_argument('--skip-kinship', action='store_true',
                    help='do not prune relatives; only for a cohort you know is unrelated')
    a = ap.parse_args()
    a.geno = L.resolve_geno(a.work, a.geno)
    if a.pca_geno:
        a.pca_geno = L.resolve_geno(a.work, a.pca_geno, quiet=True)

    work = Path(a.work); res = Path(a.results)
    L.require('geno', [
        (work / 'roster.csv', 'the roster: who is in which ancestry stratum', 'prepare'),
        (work / 'scores.csv', 'the GWAS phenotype, built from the ancestry score', 'score')])
    gw = work / 'geno'; gw.mkdir(parents=True, exist_ok=True)
    (res / 'geno').mkdir(parents=True, exist_ok=True)
    ref = Path(a.ref)

    S = pd.read_csv(work / 'scores.csv', dtype={'participant_id': str, 'geno_id': str})
    S = S[S.pheno_rint.notna()]
    strata = a.strata or [s for s in L.STRATA
                          if (S.stratum == s).sum() >= PV.MIN_STRATUM]
    L.banner(f'geno: {", ".join(strata)}  (build {a.build})')

    # ------------------------------------------------------------------ the shared subset
    V = wanted_variants(ref, a.build)
    allkeep = G.keep_file(S.geno_id.unique(), gw / 'keep_all.txt', geno=a.geno)
    stem, bim = G.subset(a.geno, V, gw / 'panel', threads=a.threads, keep=allkeep)

    # which of ours the site actually has, and in which orientation
    bim['CHR'] = bim.chrom.astype(str).str.replace('chr', '', regex=False)
    M = V.merge(bim.rename(columns={'pos': 'POS'}), on=['CHR', 'POS'], how='left')
    M['sign'] = L.allele_sign(M.a1.str.upper(), M.a2.str.upper(), M.OUR_A1.str.upper())
    cov = pd.DataFrame([
        dict(set=s,
             wanted=int((V.SET.str.contains(s)).sum()),
             present=int(((M.SET.str.contains(s)) & M.vid.notna()).sum()),
             allele_match=int(((M.SET.str.contains(s)) & np.isfinite(M.sign)).sum()))
        for s in ['lead', 'trait_pgs', 'ancestry_pgs']])
    cov['pct_usable'] = (100 * cov.allele_match / cov.wanted.replace(0, np.nan)).round(1)
    PV.write_table(cov, res / 'geno' / 'variant_coverage.csv', where='variant_coverage.csv')
    print('\n' + cov.to_string(index=False))
    M.to_csv(gw / 'variant_map.csv', index=False)
    if cov.loc[cov.set == 'lead', 'allele_match'].iloc[0] < 20:
        print('\n  WARNING: fewer than 20 of the discovery leads are usable here. Check --build: a '
              'GRCh38 position looked up in a GRCh37 release matches almost nothing.')

    # ------------------------------------------------------------------ per stratum
    summary = []
    for st in strata:
        g = S[S.stratum == st]
        if len(g) < PV.MIN_STRATUM:
            print(f'  {st}: {len(g):,} -- below the minimum, skipped')
            continue
        d = gw / st; d.mkdir(exist_ok=True)
        print(f'\n  --- {st}: {len(g):,} people')
        keep = G.keep_file(g.geno_id.unique(), d / 'keep.txt', geno=a.geno)
        pca_src = a.pca_geno or a.geno
        P, pruned = G.pcs(pca_src, keep, d / 'pca', npc=a.npc, threads=a.threads,
                          single=bool(a.pca_geno and a.pca_single))
        unrel = None if a.skip_kinship else G.unrelated(pruned, d / 'king', threads=a.threads)
        if unrel is not None:
            G.keep_file(unrel, d / 'keep_unrelated.txt', geno=a.geno)

        D = g.merge(P.rename(columns={'IID': 'geno_id'}), on='geno_id', how='inner')
        if len(D) < 0.5 * len(g):
            D2, how = L.join_or_canon(g, P.rename(columns={'IID': 'geno_id'}), 'geno_id',
                                      what=f'{st} PCs')
            print(f'    id matching: {how}')
            D = D2
        # the PCs plink2 actually returned, not the ones that were asked for: a small stratum
        # can come back with fewer, and indexing by the requested count would raise here or,
        # worse, carry a column of NaNs into every covariate design below
        pcs = [c for c in P.columns if c.upper().startswith('PC')]
        sexnum = D.sex.astype(str).str.upper().map(
            {'M': 1, 'MALE': 1, '1': 1, 'F': 0, 'FEMALE': 0, '0': 0})
        ph = pd.DataFrame({'pheno': D.pheno_rint})
        cv = pd.DataFrame({'age': D.age, 'sex': sexnum})
        for c in pcs:
            cv[c] = D[c]
        # MEASURED ANCESTRY PROPORTIONS AS COVARIATES, in every stratum that is not European.
        #
        # Inside EUR >= 0.70 the principal components absorb what little structure is left, and
        # adding the RFMix proportions there was shown to change nothing (every locus
        # survive, slope 1.000).  Outside it they are not interchangeable: an AFR >= 0.70 stratum
        # still spans 0.70 to 1.00 African ancestry, and ADMIXED is nothing BUT that gradient, so
        # the leading PCs there ARE the admixture axis.  A scan run without these covariates in
        # ADMIXED is a scan of ancestry-informative markers wearing a Manhattan plot, and the
        # first reviewer to look at it will say so.  Adjusting for the proportions directly is
        # the same control the discovery analysis applied, and it is the default wherever it can matter.
        rfc = [c for c in D.columns if c.startswith('rf_') and c != 'rf_nonEUR']
        use_rf = (a.rfmix_covar == 'always' or
                  (a.rfmix_covar == 'auto' and st != L.PRIMARY)) and rfc
        if use_rf:
            # one proportion is dropped: they sum to 1 and the full set is collinear with the
            # intercept, which plink2 would silently drop for us at a column of its choosing
            for c in sorted(rfc)[:-1]:
                cv[c] = D[c].to_numpy(float)
            print(f'    RFMix proportions as covariates: {sorted(rfc)[:-1]} '
                  f'({sorted(rfc)[-1]} dropped as the reference)')
        G.add_fid(ph, D.geno_id, geno=a.geno).to_csv(d / 'pheno.tsv', sep='\t', index=False)
        G.add_fid(cv, D.geno_id, geno=a.geno).to_csv(d / 'covar.tsv', sep='\t', index=False)
        D.to_csv(d / 'analysis_set.csv', index=False)

        ev = pd.read_csv(f'{d}/pca.eigenval', sep=r'\s+', header=None).iloc[:, 0].to_numpy()
        summary.append(dict(stratum=st, n=len(D),
                            n_unrelated=len(unrel) if unrel is not None else len(D),
                            n_pc=len(pcs), pc1_var_pct=round(100 * ev[0] / ev.sum(), 2),
                            pc_var_pct_total=round(100 * ev[:len(pcs)].sum() / ev.sum(), 2),
                            kinship='KING 0.0884' if unrel is not None else 'not applied'))
        print(f'    {len(D):,} with PCs; unrelated '
              f'{len(unrel) if unrel is not None else len(D):,}')

    T = pd.DataFrame(summary)
    PV.write_table(T, res / 'geno' / 'analysis_sets.csv', where='analysis_sets.csv')
    (res / 'geno' / 'geno_provenance.json').write_text(json.dumps(
        dict(version=L.VERSION, build=a.build, geno=a.geno, pca_geno=a.pca_geno,
             npc=a.npc, kinship_cutoff=None if a.skip_kinship else G.KING,
             strata=list(T.stratum) if len(T) else []), indent=2))
    print('\n' + (T.to_string(index=False) if len(T) else '  no stratum was large enough'))


if __name__ == '__main__':
    main()
