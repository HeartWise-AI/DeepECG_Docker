"""Everything that touches genotypes: subsetting, PCs, relatedness, association, scoring.

ALL OF IT GOES THROUGH plink.  Not because plink is fast -- though it is -- but because the
alternative is a dosage matrix in Python, and a 21,000-variant export over 20,000 people is a
two-gigabyte text file that has to be parsed, held and thrown away for every stratum.  plink2 does
the association and the scoring without ever materialising it, so the only thing this module reads
into memory is a per-variant results table.

WHAT EACH ROUTINE IS FOR

  subset()      one small bed/bim/fam holding ONLY the ~21,700 variants this image ever uses: the
                the discovery leads, the trait-PGS variants and the ancestry-PGS panel variants.
                Everything downstream of the genome-wide scan reads this instead of the release,
                so a stratum's analysis costs seconds rather than an hour.
  pcs()         within-stratum principal components.  WITHIN stratum: PCs computed across all
                ancestries would spend their first components on the ancestry split itself and
                then, used as covariates inside a stratum, would adjust away a different thing in
                each one.  Discovery computed its PCs inside the European gate; every stratum here
                gets the same treatment inside its own.
  unrelated()   a maximal unrelated subset at KING 0.0884 (second-degree).  Relatedness inflates
                a GWAS's test statistic, and a hospital biobank holds families.
  glm()         plink2 --glm linear, the association itself.
  score()       plink2 --score, the polygenic score, with allele harmonisation left to plink.
  clump()       LD clumping in the TARGET cohort's own genotypes, which is where LD has to be
                measured.  plink2's --clump when the build has it, plink1.9's when it does not,
                and a distance-only fallback that is reported as such rather than passed off as
                clumping.

NOTHING HERE WRITES ANYTHING EXPORTABLE.  Every output lands in work/ and is per-variant or
per-person; the modes above turn that into aggregates.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from ancestry import lib as L                                            # noqa: E402

KING = 0.0884            # second-degree; plink2's own documented cutoff for "unrelated"
PRUNE = ('1000', '50', '0.1')
MAF_PCA = 0.01
NPC = 20


def plink1():
    p = os.environ.get('PLINK1', shutil.which('plink') or '/opt/bin/plink')
    return p if (Path(p).exists() or shutil.which(p)) else None


def sample_table(pattern, chroms=range(1, 23)):
    """The dataset's own sample list (.psam or .fam), as {IID: FID}, or None.

    Reading it is what lets --keep carry a real FID. It also makes the id overlap checkable
    before plink runs, which is the check that actually matters: a site whose genotype ids are
    spelled differently from its ECG ids otherwise discovers it as an empty result.
    """
    pat = str(pattern)
    cands = []
    for c in list(chroms) + [None]:
        base = pat.replace('@', str(c)) if (c is not None and '@' in pat) else pat
        for ext in ('.psam', '.fam'):
            cands.append(Path(base + ext))
            if base.endswith(('.vcf.gz', '.vcf', '.bgen', '.pgen', '.bed')):
                cands.append(Path(base).with_suffix('').with_suffix(ext)
                             if base.endswith('.vcf.gz') else
                             Path(base).with_suffix(ext))
    for q in cands:
        if q.exists():
            try:
                d = pd.read_csv(q, sep=r'\s+', dtype=str,
                                header=0 if q.suffix == '.psam' else None)
            except Exception:                                            # noqa: BLE001
                continue
            if q.suffix == '.psam':
                cols = {c.lstrip('#').upper(): c for c in d.columns}
                if 'IID' not in cols:
                    continue
                iid = d[cols['IID']].astype(str)
                fid = d[cols['FID']].astype(str) if 'FID' in cols else iid
            else:
                if d.shape[1] < 2:
                    continue
                fid, iid = d[0].astype(str), d[1].astype(str)
            return dict(zip(iid, fid)), str(q)
    return None, None


def add_fid(df, iid, geno=None):
    """Put #FID and IID at the front of a plink2 sample table.

    Same reason as keep_file: an `#IID`-only --pheno/--covar file matches nothing on current
    plink2 builds, and the error ("No entries correspond to loaded sample IDs") reads as an id
    mismatch rather than a column-layout problem.
    """
    iid = pd.Series(iid, index=df.index if hasattr(df, 'index') else None).astype(str)
    fid_of, _ = (sample_table(geno) if geno else (None, None))
    fid = iid.map(fid_of).fillna(iid) if fid_of else iid
    out = df.copy()
    for c in ('#FID', 'IID', '#IID'):
        if c in out.columns:
            out = out.drop(columns=[c])
    out.insert(0, 'IID', iid.to_numpy())
    out.insert(0, '#FID', fid.to_numpy())
    return out


def keep_file(ids, path, geno=None):
    """plink2 --keep, written as FID + IID.

    An `#IID`-only list is documented to work and DOES NOT on current plink2 builds: it matches
    zero samples and the run then reports "no samples remaining", which looks exactly like an
    id mismatch. Verified against plink2 v2.0.0-b.1.4LM -- `#IID` 0 of 5,000, two-column
    FID+IID 5,000 of 5,000. So the FID is taken from the dataset's own .psam/.fam when we can
    read it, and otherwise mirrored from the IID, which is what --double-id produces and is
    what `convert` writes.
    """
    path = Path(path)
    ids = [str(x) for x in ids]
    fid_of, src = (sample_table(geno) if geno else (None, None))
    if fid_of:
        rows = [(fid_of[i], i) for i in ids if i in fid_of]
        n_hit = len(rows)
        if n_hit == 0:
            raise SystemExit(
                f'none of the {len(ids):,} requested ids appear in the genotype sample list '
                f'({src}).\n  The genotype ids are spelled differently from the ids in the '
                'ECG/RFMix tables.\n  Check --geno-id-col, and whether one side has a prefix '
                'or lost leading zeros.')
        if n_hit < len(ids):
            print(f'    --keep: {n_hit:,} of {len(ids):,} ids found in {Path(src).name}')
    else:
        rows = [(i, i) for i in ids]          # FID == IID, as --double-id writes it
    path.write_text('#FID\tIID\n' + '\n'.join(f'{f}\t{i}' for f, i in rows) + '\n')
    return str(path)


EXTRACT_MODE = 'bed1'      # plink2's spelling; plink1.9 and old plink2 say 'range'


def extract_args(path):
    """`--extract` for a 1-based, fully-closed interval file, across plink spellings.

    plink2 calls it `bed1`; older builds and plink1.9 call the same thing `range`. The mode that
    works is discovered once, on the first call that fails, rather than being assumed.
    """
    return ['--extract', EXTRACT_MODE, str(path)]


def _run(args, out_prefix=None, threads=4, memory=None, check=True):
    cmd = [L.plink2()] + [str(x) for x in args] + ['--threads', str(threads)]
    if memory:
        cmd += ['--memory', str(memory)]
    if out_prefix:
        cmd += ['--out', str(out_prefix)]
    return L.sh(cmd, check=check)


def _plink_reason(r):
    """plink2's own complaint, shortened, with the remedy where plink2 names one."""
    txt = ((getattr(r, 'stderr', '') or '') + '\n' + (getattr(r, 'stdout', '') or ''))
    low = txt.lower()
    if 'split chromosome' in low:
        return ('the genotype file has a split chromosome (variants are not in position '
                'order). Fix it once with:  plink2 --bfile <in> --make-pgen --sort-vars '
                '--out <sorted>   and pass the sorted files to --geno')
    if 'no samples remaining' in low or 'no people remaining' in low:
        return ('no samples remained after --keep: the genotype sample ids do not match the '
                'ids in the ECG/RFMix tables. Check --geno-id-col and the id spelling')
    if 'no variants remaining' in low or 'all variants excluded' in low:
        return ('no panel variant is present on this chromosome -- usually the wrong genome '
                'build (--build grch37/grch38) or chromosome naming (chr1 vs 1)')
    err = [ln.strip() for ln in txt.split('\n')
           if ln.strip().lower().startswith(('error', 'warning: no'))]
    if err:
        return err[0][:200]
    return 'plink2 gave no error text; see the log'


def _flip_extract_mode(r):
    """Switch bed1 <-> range once, if plink said it did not understand the one we used."""
    global EXTRACT_MODE
    txt = ((getattr(r, 'stderr', '') or '') + (getattr(r, 'stdout', '') or '')).lower()
    about_extract = ('--extract' in txt or 'bed1' in txt or 'bed0' in txt
                     or 'range' in txt or 'unrecognized' in txt or 'invalid' in txt)
    if about_extract:
        EXTRACT_MODE = 'range' if EXTRACT_MODE == 'bed1' else 'bed1'
        print(f'    (retrying with --extract {EXTRACT_MODE})')
        return True
    return False


# ------------------------------------------------------------------------------- subsetting
def write_extract(variants: pd.DataFrame, path):
    """plink2 --extract range: CHR START END, one line per variant (1 bp windows).

    Positions, not variant IDs. A hospital release spells IDs however its imputation server did --
    rsID, chr:pos:ref:alt, chr:pos_ref_alt -- and matching on a spelling we do not control is the
    commonest way an extraction silently returns nothing. A position cannot be spelled wrong.
    """
    d = variants[['CHR', 'POS']].copy()
    d['CHR'] = d.CHR.astype(str).str.replace('chr', '', regex=False)
    d['START'] = d.POS.astype(int)
    d['END'] = d.POS.astype(int)
    d['ID'] = d.CHR + ':' + d.POS.astype(str)
    d[['CHR', 'START', 'END', 'ID']].drop_duplicates().to_csv(path, sep='\t', header=False,
                                                              index=False)
    return str(path)


def clear_bed_files(prefix):
    """Remove a BED trio so a failed rerun cannot inherit successful-looking outputs."""
    for ext in ('.bed', '.bim', '.fam'):
        Path(f'{prefix}{ext}').unlink(missing_ok=True)


def complete_bed_files(prefix, result):
    return (getattr(result, 'returncode', 1) == 0
            and all(Path(f'{prefix}{ext}').exists() for ext in ('.bed', '.bim', '.fam')))


def subset(geno_pattern, variants, out_prefix, chroms=range(1, 23), threads=4, keep=None):
    """One bed/bim/fam holding just `variants`, merged over the chromosomes that have any."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rng = write_extract(variants, out_prefix.parent / 'extract_ranges.txt')
    parts, reasons = [], []
    for c in chroms:
        sub = variants[variants.CHR.astype(str).str.replace('chr', '', regex=False) == str(c)]
        if not len(sub):
            continue
        p = f'{out_prefix}_chr{c}'
        args = L.geno_args(geno_pattern, c) + extract_args(rng) + \
            ['--make-bed', '--max-alleles', '2']
        if keep:
            args += ['--keep', keep]
        clear_bed_files(p)
        r = _run(args, p, threads=threads, check=False)
        if not complete_bed_files(p, r) and _flip_extract_mode(r):
            clear_bed_files(p)
            args = L.geno_args(geno_pattern, c) + extract_args(rng) + \
                ['--make-bed', '--max-alleles', '2'] + (['--keep', keep] if keep else [])
            r = _run(args, p, threads=threads, check=False)
        if complete_bed_files(p, r):
            parts.append(p)
        else:
            clear_bed_files(p)
            why = _plink_reason(r)
            reasons.append(f'chr{c}: {why}')
            print(f'    chr{c}: nothing extracted (plink2 rc {r.returncode}) -- {why}')
    if not parts:
        # plink2 usually says exactly what is wrong and how to fix it. Reporting only
        # "nothing extracted" threw that away and pointed at --geno and --build, which is the
        # wrong place whenever the real cause was the file's own ordering or sample ids.
        uniq = []
        for q in reasons:
            tail = q.split(': ', 1)[-1]
            if tail not in [u.split(': ', 1)[-1] for u in uniq]:
                uniq.append(q)
        raise SystemExit(
            'nothing extracted from the genotypes.\n  what plink2 said:\n'
            + '\n'.join(f'    {q}' for q in uniq[:6])
            + '\n  if the message above does not name the cause, check that --geno points at '
              'the right files\n  and that --build matches the release '
              '(--build grch37 / grch38).')
    clear_bed_files(out_prefix)
    if len(parts) == 1:
        for e in ('bed', 'bim', 'fam'):
            shutil.copy(f'{parts[0]}.{e}', f'{out_prefix}.{e}')
    else:
        lst = out_prefix.parent / 'merge_list.txt'
        lst.write_text('\n'.join(parts[1:]) + '\n')
        r = _run(['--bfile', parts[0], '--pmerge-list', str(lst), 'bfile', '--make-bed'],
                 out_prefix, threads=threads, check=False)
        if not complete_bed_files(out_prefix, r):      # older plink2: --merge-list
            clear_bed_files(out_prefix)
            r = _run(['--bfile', parts[0], '--merge-list', str(lst), '--make-bed'],
                     out_prefix, threads=threads, check=False)
        if not complete_bed_files(out_prefix, r):
            clear_bed_files(out_prefix)
            raise SystemExit(f'failed to merge extracted chromosome panels: {_plink_reason(r)}')
    b = pd.read_csv(f'{out_prefix}.bim', sep=r'\s+', header=None,
                    names=['chrom', 'vid', 'cm', 'pos', 'a1', 'a2'], dtype={0: str})
    print(f'    subset: {len(b):,} of {len(variants):,} wanted variants present')
    for p in parts:
        for e in ('bed', 'bim', 'fam', 'log'):
            Path(f'{p}.{e}').unlink(missing_ok=True)
    return str(out_prefix), b


# ------------------------------------------------------------------------------- PCs, kinship
def pcs(geno_pattern, keep, out_prefix, chroms=range(1, 23), npc=NPC, threads=4,
        maf=MAF_PCA, single=False):
    """LD-pruned common variants -> `npc` within-stratum PCs. Returns a frame IID, PC1..PCn."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if single:
        base = [L.geno_args(geno_pattern)]
    else:
        base = [L.geno_args(geno_pattern, c) for c in chroms]
    pr = []
    for i, ga in enumerate(base):
        p = f'{out_prefix}_prune{i}'
        _run(ga + ['--keep', keep, '--maf', str(maf), '--geno', '0.05', '--hwe', '1e-6',
                   '--indep-pairwise', *PRUNE], p, threads=threads)
        if Path(f'{p}.prune.in').exists():
            pr.append(f'{p}.prune.in')
    if not pr:
        raise SystemExit('LD pruning produced nothing; check the genotypes')
    allin = out_prefix.parent / f'{out_prefix.name}.prune.in'
    with open(allin, 'w') as fh:
        for f in pr:
            fh.write(Path(f).read_text())
    # the pruned set, merged, then the decomposition
    parts = []
    for i, ga in enumerate(base):
        p = f'{out_prefix}_pr{i}'
        _run(ga + ['--keep', keep, '--extract', str(allin), '--make-bed'], p, threads=threads)
        if Path(f'{p}.bed').exists():
            parts.append(p)
    if len(parts) > 1:
        lst = out_prefix.parent / f'{out_prefix.name}_merge.txt'
        lst.write_text('\n'.join(parts[1:]) + '\n')
        _run(['--bfile', parts[0], '--pmerge-list', str(lst), 'bfile', '--make-bed'],
             f'{out_prefix}_all', threads=threads)
        stem = f'{out_prefix}_all'
    else:
        stem = parts[0]
    _run(['--bfile', stem, '--pca', str(npc), 'approx'], out_prefix, threads=threads)
    # plink2 writes '#FID IID PC1 ...' when the fileset has FIDs and '#IID PC1 ...' when it does
    # not.  Both are read by NAME: taking columns by position silently shifts every PC by one in
    # whichever case was not anticipated, and nothing downstream would notice.
    ev = read_plink_output(f'{out_prefix}.eigenvec', sep=r'\s+')
    ev.columns = [c.lstrip('#') for c in ev.columns]
    if 'IID' not in ev.columns:
        raise SystemExit(f'{out_prefix}.eigenvec has no IID column: {list(ev.columns)[:6]}')
    pcs = [c for c in ev.columns if c.upper().startswith('PC')]
    if len(pcs) < npc:
        print(f'    note: plink2 returned {len(pcs)} PCs, not {npc}')
        npc = len(pcs)
    ev['IID'] = ev.IID.astype(str)
    return ev[['IID'] + pcs[:npc]], stem


def unrelated(pruned_stem, out_prefix, cutoff=KING, threads=4):
    """A maximal unrelated subset at KING `cutoff`. Returns the list of IIDs to keep."""
    _run(['--bfile', pruned_stem, '--king-cutoff', str(cutoff), '--make-just-fam'],
         out_prefix, threads=threads)
    f = Path(f'{out_prefix}.king.cutoff.in.id')
    if not f.exists():
        print('    KING cutoff produced no list; keeping everyone and saying so')
        return None
    d = pd.read_csv(f, sep=r'\s+', dtype=str)
    col = 'IID' if 'IID' in d.columns else d.columns[-1]
    return d[col].astype(str).tolist()


# ------------------------------------------------------------------------------- association
def glm(bfile, pheno, covar, out_prefix, keep=None, extract=None, threads=4,
        maf=0.005, mac=20, geno_miss=0.10, covar_names=None):
    """plink2 --glm linear.  Returns the parsed .glm.linear frame (ADD rows only).

    `--covar-variance-standardize` is on because the PCs and age are on wildly different scales and
    plink's default reports the coefficient in the covariate's own units; nothing downstream reads
    the covariate coefficients, but a conditioned design is better numerically.
    """
    args = ['--bfile', str(bfile), '--pheno', str(pheno), '--pheno-name', 'pheno',
            '--covar', str(covar), '--covar-variance-standardize',
            '--glm', 'hide-covar', 'cols=+a1freq,+machr2,+nobs',
            '--maf', str(maf), '--mac', str(mac), '--geno', str(geno_miss)]
    if covar_names:
        args += ['--covar-name', ','.join(covar_names)]
    if keep:
        args += ['--keep', str(keep)]
    if extract:
        args += extract_args(extract)
    _run(args, out_prefix, threads=threads)
    f = Path(f'{out_prefix}.pheno.glm.linear')
    if not f.exists():
        f = Path(f'{out_prefix}.glm.linear')
    if not f.exists():
        cand = sorted(Path(out_prefix).parent.glob(Path(out_prefix).name + '*.glm.linear'))
        if not cand:
            raise SystemExit(f'plink2 wrote no .glm.linear for {out_prefix}; see the .log')
        f = cand[0]
    d = pd.read_csv(f, sep='\t', dtype={'#CHROM': str})
    d = d.rename(columns={'#CHROM': 'CHR', 'ID': 'SNP'})
    if 'TEST' in d.columns:
        d = d[d.TEST == 'ADD']
    return d


def score(bfile, weights_file, out_prefix, keep=None, threads=4):
    """plink2 --score: columns of `weights_file` are ID, effect allele, weight (1 2 3)."""
    args = ['--bfile', str(bfile), '--score', str(weights_file), '1', '2', '3', 'header',
            'no-mean-imputation', 'cols=+scoresums']
    if keep:
        args += ['--keep', str(keep)]
    _run(args, out_prefix, threads=threads)
    f = Path(f'{out_prefix}.sscore')
    if not f.exists():
        raise SystemExit(f'plink2 wrote no .sscore for {out_prefix}')
    d = read_plink_output(f, sep='\t')
    d.columns = [c.lstrip('#') for c in d.columns]
    d['IID'] = d.IID.astype(str)
    return d


def read_plink_output(path, sep):
    """Read PLINK tables without destroying leading zeroes in FID/IID identifiers."""
    return pd.read_csv(path, sep=sep,
                       dtype={'#FID': str, 'FID': str, '#IID': str, 'IID': str})


def clump(bfile, assoc_file, out_prefix, p1=1.0, r2=0.1, kb=500, threads=4):
    """LD clumping in the target cohort.  Returns (kept variant ids, how it was done).

    Three routes, tried in order, and the one that was used is reported rather than hidden: a
    distance-only fallback is a weaker instrument than LD clumping and a reader has to know which
    one produced the score they are looking at.
    """
    args = ['--bfile', str(bfile), '--clump', str(assoc_file),
            '--clump-p1', str(p1), '--clump-r2', str(r2), '--clump-kb', str(kb)]
    f = Path(f'{out_prefix}.clumps')
    f.unlink(missing_ok=True)
    try:
        cmd = [L.plink2(), *args, '--clump-field', 'P', '--clump-id-field', 'SNP',
               '--out', str(out_prefix), '--threads', str(threads)]
        r = L.sh(cmd, check=False)
        if getattr(r, 'returncode', 1) == 0 and f.exists():
            d = pd.read_csv(f, sep=r'\s+')
            return d['ID'].astype(str).tolist(), 'plink2 --clump'
    except SystemExit:
        pass
    f.unlink(missing_ok=True)
    p1bin = plink1()
    if p1bin:
        f = Path(f'{out_prefix}.clumped')
        f.unlink(missing_ok=True)
        cmd = [p1bin, '--bfile', str(bfile), '--clump', str(assoc_file), '--clump-p1', str(p1),
               '--clump-r2', str(r2), '--clump-kb', str(kb), '--clump-field', 'P',
               '--clump-snp-field', 'SNP', '--out', str(out_prefix), '--threads', str(threads)]
        r = L.sh(cmd, check=False)
        if getattr(r, 'returncode', 1) == 0 and f.exists():
            d = pd.read_csv(f, sep=r'\s+')
            return d['SNP'].astype(str).tolist(), 'plink1.9 --clump'
        f.unlink(missing_ok=True)
    # distance only: keep the most significant variant per `kb` window, no LD at all
    A = pd.read_csv(assoc_file, sep=r'\s+')
    A = A.sort_values('P')
    kept, used = [], {}
    for _, row in A.iterrows():
        c, p = str(row['CHR']), int(row['BP']) if 'BP' in A.columns else int(row['POS'])
        w = used.setdefault(c, [])
        if any(abs(p - q) <= kb * 1000 for q in w):
            continue
        w.append(p); kept.append(str(row['SNP']))
    return kept, f'distance only ({kb} kb), NO LD -- plink --clump was unavailable'
