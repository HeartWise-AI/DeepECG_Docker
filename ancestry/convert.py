"""Mode `convert` -- turn VCF/BCF genotypes into plink2 pgen, once, before anything reads them.

WHY THIS IS A STAGE AND NOT A CONVENIENCE.  plink2 reads a VCF directly, so nothing here is
strictly required -- but it re-parses the text on every single call, and this container makes a
lot of calls: an LD-pruning pass, a PCA, a kinship run and a regression per ancestry stratum, plus
one regression per chromosome per stratum if the genome-wide scan is run.  On a 20,000-person
imputed chromosome that is minutes of gzip and text parsing each time, repeated a hundred times.
Converting once turns the whole run from days into hours, and it is the difference between the
scan being practical at a hospital and not.

WHAT IT FIXES WHILE IT IS THERE.  A VCF from an imputation server needs four things done to it
before it behaves, and all four are silent failures if they are skipped:

  sample ids      plink's VCF reader puts the sample name in IID and `0` in FID unless told
                  otherwise.  `--double-id` puts it in both, which is what every join in this
                  image expects and what `canon_id` already strips back off.
  variant ids     imputed VCFs frequently carry `.` in the ID column.  plink2 refuses duplicate
                  ids, and `.` repeated ten million times is ten million duplicates, so ids are
                  rebuilt as chr:pos:ref:alt.  Nothing downstream matches on an id -- extraction
                  is by POSITION throughout -- but plink needs them to be unique.
  multi-allelics  split records and `*` alleles break the two-allele assumption that allele
                  harmonisation rests on.  Kept biallelic, and the count that was dropped is
                  reported rather than silently absorbed.
  half calls      `0/.` appears in some merged releases; plink2 errors out on it by default.
                  Treated as missing, which is the only defensible reading.

DOSAGES.  If the VCF carries a DS field the dosages are imported, because imputed data loses real
information when it is hard-called.  If it does not, the hard calls are imported instead.  Which
one happened is recorded in the provenance, because an analysis of dosages and an analysis of hard
calls are not the same analysis and the difference should not have to be guessed at later.

    python -m ancestry convert --work /out/work --vcf '/data/chr@.vcf.gz'      # one file per chromosome
    python -m ancestry convert --work /out/work --vcf /data/all.vcf.gz         # one file, split by chrom
    python -m ancestry convert --work /out/work --vcf '/data/chr@.bcf' --chroms 1 2 3   # or a subset

Writes work/geno/pgen/chr<N>.{pgen,pvar,psam} and work/geno/converted.json.  `geno` and `scan`
pick the converted set up automatically -- keep passing the original --vcf path and they will use
the pgen.  Nothing here leaves the site.  Prints no participant identifier.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from ancestry import lib as L                                            # noqa: E402

VCF_SUFFIXES = ('.vcf', '.vcf.gz', '.bcf', '.bcf.gz')

# chr:pos:ref:alt, with a length cap so an indel does not produce a kilobyte-long id
SET_IDS = ['--set-all-var-ids', '@:#:$r:$a', '--new-id-max-allele-len', '100', 'missing']


def looks_like_vcf(pattern):
    p = str(pattern or '').replace('@', '1')
    return p.endswith(VCF_SUFFIXES)


def has_dosage(path):
    """Is there a DS field in the header? Read the header only -- never the genotypes."""
    import gzip
    opener = gzip.open if str(path).endswith('.gz') else open
    try:
        with opener(path, 'rt', errors='replace') as fh:
            for line in fh:
                if not line.startswith('#'):
                    return False
                if line.startswith('##FORMAT=<ID=DS,'):
                    return True
    except OSError:
        return False
    return False


def convert_one(src, chrom, outdir, threads, dosage, keep_multiallelic, extra):
    out = Path(outdir) / f'chr{chrom}'
    args = ['--vcf', str(src)]
    if dosage:
        args += ['dosage=DS']
    args += ['--double-id', '--vcf-half-call', 'm'] + SET_IDS
    if not keep_multiallelic:
        args += ['--max-alleles', '2']
    # one chromosome per output even when the source holds the whole genome
    args += ['--chr', str(chrom)]
    args += ['--rm-dup', 'exclude-all', '--make-pgen', '--out', str(out),
             '--threads', str(threads)]
    r = L.sh([L.plink2()] + args + list(extra), check=False)
    ok = (out.with_suffix('.pgen')).exists()
    n_var = n_sam = 0
    if ok:
        pv = out.with_suffix('.pvar')
        n_var = sum(1 for ln in open(pv) if not ln.startswith('#'))
        ps = out.with_suffix('.psam')
        n_sam = sum(1 for ln in open(ps) if not ln.startswith('#')) if ps.exists() else 0
    return ok, n_var, n_sam, (r.stderr or '')[-400:] if hasattr(r, 'stderr') else ''


def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry convert', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True,
                    help="the site's working directory: intermediates, per-person tables and "
                         'arrays. It stays at the site and is never packaged')
    ap.add_argument('--vcf', required=True,
                    help="VCF/BCF, @ for the chromosome number, or one whole-genome file")
    ap.add_argument('--out', default=None, help='default work/geno/pgen')
    ap.add_argument('--chroms', nargs='+', default=[str(i) for i in range(1, 23)],
                    help='chromosomes to process (default 1-22)')
    ap.add_argument('--threads', type=int, default=L.default_threads(),
                    help="threads for plink (default: this machine's cores, less two, capped 16)")
    ap.add_argument('--dosage', choices=['auto', 'yes', 'no'], default='auto',
                    help='import DS dosages. auto reads the VCF header and decides (default)')
    ap.add_argument('--keep-multiallelic', action='store_true',
                    help='do not drop variants with more than two alleles (not advised: allele '
                         'harmonisation assumes two)')
    ap.add_argument('--force', action='store_true', help='re-convert chromosomes already done')
    a, extra = ap.parse_known_args()

    work = Path(a.work)
    outdir = Path(a.out) if a.out else work / 'geno' / 'pgen'
    outdir.mkdir(parents=True, exist_ok=True)
    one_file = '@' not in a.vcf

    L.banner(f'convert: {"one whole-genome file" if one_file else "one file per chromosome"} '
             f'-> plink2 pgen')
    print(f'  source : {a.vcf}')
    print(f'  output : {outdir}/chr<N>.pgen')

    rows, done, failed = [], [], []
    for c in a.chroms:
        src = Path(a.vcf if one_file else a.vcf.replace('@', str(c)))
        if not src.exists():
            print(f'  chr{c}: {src} does not exist, skipped')
            continue
        tgt = outdir / f'chr{c}.pgen'
        if tgt.exists() and not a.force:
            n = sum(1 for ln in open(outdir / f'chr{c}.pvar') if not ln.startswith('#'))
            print(f'  chr{c}: already converted ({n:,} variants) -- --force to redo')
            done.append(c)
            rows.append(dict(chrom=c, variants=n, reused=True))
            continue
        ds = (a.dosage == 'yes') or (a.dosage == 'auto' and has_dosage(src))
        ok, nv, ns, err = convert_one(src, c, outdir, a.threads, ds, a.keep_multiallelic, extra)
        if ok:
            print(f'  chr{c}: {nv:,} variants, {ns:,} samples'
                  f'  ({"DS dosages" if ds else "hard calls"})')
            done.append(c)
            rows.append(dict(chrom=c, variants=nv, samples=ns, dosage=bool(ds), reused=False))
        else:
            print(f'  chr{c}: FAILED\n     {err.strip()[:300]}')
            failed.append(c)

    if not done:
        raise SystemExit('nothing was converted. Check --vcf and the chromosome spelling: this '
                         'image uses bare numbers (1, 2, ... 22), so a release named chr1.vcf.gz '
                         "needs --vcf '/path/chr@.vcf.gz'.")

    pattern = str(outdir / 'chr@')
    prov = dict(version=L.VERSION, source=a.vcf, pattern=pattern, chroms=done,
                failed=failed, dosage=a.dosage, keep_multiallelic=bool(a.keep_multiallelic),
                per_chromosome=rows,
                note='`geno` and `scan` use this automatically when --geno points at the VCFs')
    (work / 'geno').mkdir(parents=True, exist_ok=True)
    (work / 'geno' / 'converted.json').write_text(json.dumps(prov, indent=2))

    total = sum(r.get('variants', 0) for r in rows)
    print(f'\n  {len(done)} chromosome(s), {total:,} variants in all')
    if failed:
        print(f'  !! {len(failed)} failed: {", ".join(failed)} -- the run can continue without '
              'them, but the scan will have holes and `loci` will report those leads absent')
    print(f'  wrote {work / "geno" / "converted.json"}')
    print(f'\n  from here on, pass the ORIGINAL --geno path: `geno` and `scan` read this file '
          'and use the pgen.')


if __name__ == '__main__':
    main()
