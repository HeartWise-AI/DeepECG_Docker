"""Mode `scan` -- the OPTIONAL genome-wide association, one Manhattan per ancestry stratum.

THIS IS OPT-IN AND SEPARATE ON PURPOSE.  `loci` reads a few dozen positions and takes minutes; this reads
the whole imputed release and takes hours per chromosome per stratum.  Run it only after `loci`
has come back clean, because it is built on that run's analysis set, PCs and relatedness filter --
the same people, the same covariates -- so that the scan CONTAINS the lead replication rather than
being a second, differently-specified analysis that happens to be about the same thing.

WHAT IT IS FOR, AND WHAT IT IS NOT FOR.  MGB-MGH's European stratum is roughly a quarter of the
discovery meta-analysis, so this scan is not expected to rediscover the discovery loci and finding fewer is
not a failure.  It is here for three things a targeted lookup cannot give:

  * the genomic inflation factor, which says whether the phenotype behaves like a polygenic trait
    at this site or like a batch artefact.  Discovery's was 1.078 at the scan and LDSC put the
    stratification share at 8%;
  * whether anything NEW clears 5e-8 here -- a hospital cohort is not a volunteer cohort, and a
    locus that only appears in one is worth knowing about;
  * the non-European strata, where no genome-wide scan of this phenotype has ever been run.  With
    a few hundred to a few thousand people those Manhattans will be empty of significance, and
    the package says so in the same breath as it draws them: they are a description of where the
    phenotype's signal sits, not a test it passed or failed.

WHAT LEAVES THE SITE.  Not the summary statistics.  The per-chromosome .glm.linear files are a
full GWAS summary statistic and are an export in its own right, needing its own approval; they
stay in work/.  What leaves is the thinned point cloud the Manhattan is drawn from, the QQ
quantiles with lambda_GC, the distance-clumped significant leads, and the discovery leads
looked up in the full scan.

    python -m ancestry scan --work /out/work --results /out/results \\
        --geno '/path/chr@' [--strata EUR] [--chroms 1 2 3] [--threads 8] [--aggregate-only]

The reduction is done by vendor/gwas_aggregate.py, byte-identical to the script that produced the
Lifelines supplementary Manhattan, including its density-continuous thinning rule.
"""
from __future__ import annotations

import argparse
import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from ancestry import privacy as PV                                       # noqa: E402
from ancestry import lib as L                                            # noqa: E402
from ancestry import geno_ops as G                                       # noqa: E402


def chromosome_outputs(folder, chrom):
    """Fresh PLINK linear results for exactly one chromosome prefix."""
    folder = Path(folder)
    return sorted(set(folder.glob(f'chr{chrom}.glm.linear*')) |
                  set(folder.glob(f'chr{chrom}.*.glm.linear*')))


def scan_one(st, chrom, work, geno, threads, maf, mac, geno_miss):
    d = Path(work) / 'geno' / st
    out = Path(work) / 'scan' / st
    out.mkdir(parents=True, exist_ok=True)
    keep = d / 'keep_unrelated.txt'
    keep = keep if keep.exists() else d / 'keep.txt'
    args = (L.geno_args(geno, chrom) +
            ['--pheno', str(d / 'pheno.tsv'), '--pheno-name', 'pheno',
             '--covar', str(d / 'covar.tsv'), '--covar-variance-standardize',
             '--keep', str(keep),
             '--glm', 'hide-covar', 'cols=+a1freq,+machr2,+nobs',
             '--maf', str(maf), '--mac', str(mac), '--geno', str(geno_miss),
             '--out', str(out / f'chr{chrom}'), '--threads', str(threads)])
    # Never let a failed rerun reuse an old chromosome result.  PLINK writes the requested
    # phenotype name into the suffix on some versions, hence the two exact-prefix patterns.
    for stale in chromosome_outputs(out, chrom):
        stale.unlink()
    L.sh([L.plink2()] + [str(x) for x in args])
    hits = chromosome_outputs(out, chrom)
    if not hits:
        raise SystemExit(f'scan {st} chr{chrom}: PLINK exited successfully but wrote no fresh '
                         '.glm.linear output')
    return hits[0]


def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry scan', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True,
                    help="the site's working directory: intermediates, per-person tables and "
                         'arrays. It stays at the site and is never packaged')
    ap.add_argument('--results', required=True,
                    help='the aggregate-only output directory; `package` zips exactly this')
    ap.add_argument('--geno', default=None,
                    help='the genotypes: plink2 pgen/bed prefix, bgen, or VCF/BCF, with @ for '
                         'the chromosome number')
    ap.add_argument('--ref', default=str(L.ref_dir()),
                    help='the frozen discovery tables (default: $REFERENCE_DIR, else the '
                         'reference/ folder of the pinned weights snapshot)')
    ap.add_argument('--build', default=None,
                    help='genome build; normally inherited from `geno`')
    ap.add_argument('--strata', nargs='+', default=None,
                    help='which ancestry strata to run (default: every stratum `geno` built)')
    ap.add_argument('--chroms', nargs='+', default=[str(i) for i in range(1, 23)],
                    help='chromosomes to process (default 1-22)')
    ap.add_argument('--threads', type=int, default=L.default_threads(),
                    help='threads for plink (default: this machine\'s cores, less two, capped at 16)')
    ap.add_argument('--maf', type=float, default=0.01,
                    help='minor-allele frequency floor (default 0.01)')
    ap.add_argument('--mac', type=int, default=20,
                    help='minor-allele COUNT floor (default 20); the binding filter in a small '
                         'stratum, where a frequency floor alone lets through singletons')
    ap.add_argument('--geno-missing', type=float, default=0.10,
                    help='drop a variant missing in more than this fraction of people')
    ap.add_argument('--info-min', type=float, default=0.4,
                    help='imputation-quality floor, where the genotypes carry an INFO/R2 field')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='the per-chromosome scans already ran; just reduce them')
    ap.add_argument('--scan-only', action='store_true',
                    help='run the requested chromosome scans without aggregation; used by the '
                         'Slurm array before its single dependent aggregate job')
    a = ap.parse_args()

    work = Path(a.work); res = Path(a.results) / 'scan'; res.mkdir(parents=True, exist_ok=True)
    L.require('scan', [(Path(a.results) / 'geno' / 'geno_provenance.json',
                        'the per-stratum phenotype, covariates and unrelated keep-set', 'geno')])
    gp = json.loads((Path(a.results) / 'geno' / 'geno_provenance.json').read_text())
    build = a.build or gp.get('build', 'grch38')
    geno = L.resolve_geno(a.work, a.geno or gp.get('geno'))
    strata = a.strata or gp.get('strata', [])
    L.banner(f'scan: genome-wide, {", ".join(strata)}  ({len(a.chroms)} chromosomes)')
    if not a.aggregate_only and not geno:
        raise SystemExit('--geno is required (or run `geno` first so it can be read from there)')

    summary = []
    for st in strata:
        if not (Path(work) / 'geno' / st / 'pheno.tsv').exists():
            print(f'  {st}: no analysis set, skipped')
            continue
        n = sum(1 for _ in open(Path(work) / 'geno' / st / 'pheno.tsv')) - 1
        print(f'\n  --- {st}  ({n:,} people)')
        if not a.aggregate_only:
            for c in a.chroms:
                f = scan_one(st, c, work, geno, a.threads, a.maf, a.mac, a.geno_missing)
                print(f'    chr{c}: ok ({f.name})')
        missing = [c for c in a.chroms
                   if not chromosome_outputs(Path(work) / 'scan' / st, c)]
        if missing:
            raise SystemExit(f'{st}: scan is incomplete; missing fresh PLINK output for '
                             f'chromosome(s) {", ".join(missing)}')
        if a.scan_only:
            continue
        od = res / st; od.mkdir(parents=True, exist_ok=True)
        sys.argv = ['gwas_aggregate.py', '--glm', str(Path(work) / 'scan' / st / 'chr'),
                    '--out', str(od), '--ref', str(a.ref), '--build', build,
                    '--chroms', ' '.join(a.chroms), '--maf', str(a.maf),
                    '--info-min', str(a.info_min)]
        # Aggregation failures are stage failures.  Swallowing SystemExit lets Slurm release
        # packaging even though gwas_aggregate deliberately refused an incomplete input.
        runpy.run_path(str(HERE / 'vendor' / 'gwas_aggregate.py'), run_name='__main__')
        # the vendored script names its outputs ll_*; rename so the package reads as one thing
        for p in list(od.glob('ll_*')):
            p.rename(od / p.name.replace('ll_', 'scan_', 1))
        js = od / 'scan_gwas_summary.json'
        if js.exists():
            s = json.loads(js.read_text())
            s['stratum'] = st
            s['n_people'] = n
            s['note'] = ('powered for discovery' if n >= 20000 else
                         'DESCRIPTIVE ONLY: too small for genome-wide discovery; read the '
                         'targeted lead-level result instead')
            js.write_text(json.dumps(s, indent=2))
            summary.append(s)
        # bound the exported point cloud, whatever the thinning produced
        man = od / 'scan_manhattan.tsv'
        if man.exists():
            M = pd.read_csv(man, sep='\t')
            if len(M) > PV.MAX_MANHATTAN_ROWS:
                M = M.sort_values('NEGLOG10P', ascending=False).head(PV.MAX_MANHATTAN_ROWS)
                M.to_csv(man, sep='\t', index=False)
                print(f'    thinned further to {len(M):,} points for export')

    if a.scan_only:
        print(f'\n  scan-only complete; chromosome statistics stay in {work / "scan"}')
        return
    if summary:
        T = pd.DataFrame([{k: v for k, v in s.items()
                           if not isinstance(v, (dict, list))} for s in summary])
        PV.write_table(T, res / 'scan_summary.csv', where='scan_summary.csv')
        print('\n' + T.to_string(index=False))
    print(f'\n  wrote {res}/   (summary statistics stay in {work / "scan"})')


if __name__ == '__main__':
    main()
