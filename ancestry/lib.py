"""Shared machinery for the MGB-MGH validation image: strata, regression, clumping, plink2, IO.

Every mode imports from here, so a definition that has to be the same everywhere -- what an
ancestry stratum IS, how a partial effect is computed, how a results directory is replaced -- is
written once and only once.

THREE CONVENTIONS THAT THE WHOLE IMAGE DEPENDS ON, stated here because they are the ones a reader
of any single module would otherwise have to infer:

  1. STRATUM.  A participant belongs to stratum X when their RFMix X fraction is >= `gate`
     (0.70 by default, the discovery gate), and to ADMIXED when no single ancestry reaches it.
     The strata are therefore disjoint and their union is everyone genotyped.  EUR is the
     discovery setup; AFR, EAS and SAS are new here and are where the discovery leads are being asked a
     question they have never been asked.

  2. PHENOTYPE.  The GWAS phenotype is the ensemble LOGIT of P(non-European), rank-inverse-normal
     transformed WITHIN STRATUM.  Within stratum, because the strata are analysed separately and
     a score that is compressed against 1.0 in an African-ancestry stratum and spread out in a
     European one must not be put on one scale by accident.  RINT makes every stratum's phenotype
     mean 0, variance 1, so a beta is in SD of the score per copy of the allele in every stratum
     and the strata can be laid side by side.

  3. NOTHING PER-PERSON IS WRITTEN OUTSIDE `work/`.  `work/` stays at the site.  Everything under
     `results/` goes through privacy.write_table, which refuses an identifier column and refuses a
     table long enough to be per-person.
"""
from __future__ import annotations

import datetime
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ancestry import VERSION                                              # noqa: E402
BUILD_NOTE = 'external validation: genetic replication + fairness audit'

# --------------------------------------------------------------------------------- strata
ANCESTRIES = ['EUR', 'AFR', 'EAS', 'SAS']
STRATA = ANCESTRIES + ['ADMIXED']
GATE = 0.70
# Which strata carry which analysis by default.  Everything runs everywhere it can; this is only
# the ORDER the reports are written in and the stratum the discovery comparison is anchored on.
PRIMARY = 'EUR'
PC = [f'PC{i}' for i in range(1, 21)]


def assign_stratum(props: pd.DataFrame, ancestries=ANCESTRIES, gate=GATE) -> np.ndarray:
    """-> one of EUR/AFR/EAS/SAS/ADMIXED per row of `props` (columns = ancestry fractions)."""
    P = props[list(ancestries)].to_numpy(dtype=float)
    dom = np.argmax(P, axis=1)
    val = P[np.arange(len(P)), dom]
    return np.where(val >= gate, np.asarray(ancestries)[dom], 'ADMIXED').astype(object)


# --------------------------------------------------------------------------------- ids
def canon_id(x):
    """The one id spelling every table is compared on when exact matching fails.

    Copied from the Lifelines package, where it was the fix for a live failure: the same barcode
    is '0082438624' in one file and '82438624' in another, and plink exports it as '1_0082438624'.
    Exact matching is tried FIRST and is the normal path; this is the fallback, and any run that
    uses it says so.
    """
    x = str(x)
    x = re.sub(r'^\d+_', '', x)              # plink's FID_IID
    if x.isdigit():
        x = x.lstrip('0') or '0'             # a numeric parse upstream dropped leading zeros
    return x


def join_or_canon(left, right, on, min_frac=0.20, what=''):
    """Merge on `on`; if under `min_frac` of `left` matches, retry on the canonical spelling.

    Returns (merged, how) where how is 'exact' or 'canonical'. Raises if the canonical retry is
    also below the floor -- a 5% join is not a join, and silently analysing 5% of a cohort is the
    failure mode this exists to prevent.
    """
    m = left.merge(right, on=on, how='inner')
    if len(m) >= min_frac * len(left):
        return m, 'exact'
    l2 = left.copy(); r2 = right.copy()
    for c in ([on] if isinstance(on, str) else on):
        l2[c] = l2[c].map(canon_id); r2[c] = r2[c].map(canon_id)
    m2 = l2.merge(r2, on=on, how='inner')
    if len(m2) < min_frac * len(left):
        raise SystemExit(
            f'{what}: only {len(m):,} exact and {len(m2):,} canonical matches out of '
            f'{len(left):,} -- the two tables do not spell ids the same way. Fix the crosswalk '
            f'rather than analysing a fraction of the cohort.')
    return m2, 'canonical'


# --------------------------------------------------------------------------------- statistics
def rint(x):
    """Blom rank-inverse-normal, ties averaged. NaNs are carried through as NaN."""
    s = pd.Series(np.asarray(x, dtype=float))
    ok = s.notna()
    out = np.full(len(s), np.nan)
    r = s[ok].rank(method='average').to_numpy()
    out[ok.to_numpy()] = stats.norm.ppf((r - 0.375) / (ok.sum() + 0.25))
    return out


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def partial(y, x, C, add_const=True):
    """Frisch-Waugh-Lovell: the effect of x on y adjusted for C, by residualising both.

    Returns beta (per SD of x, because x is standardised after residualising), its SE, the
    two-sided P, the partial correlation, the incremental R^2 over C alone, and n. Doing it this
    way rather than by reading one coefficient out of a 25-column design is not a shortcut: it
    makes the incremental R^2 fall out of the same arithmetic as the beta, so the two cannot
    disagree.
    """
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(C).all(axis=1)
    y, x, C = y[ok], x[ok], C[ok]
    n = len(y)
    if n < 20 or np.std(x) == 0:
        return dict(n=int(n), beta_sd=np.nan, se=np.nan, p=np.nan, partial_r=np.nan,
                    delta_r2=np.nan, r2_base=np.nan)
    D = np.column_stack([np.ones(n), C]) if add_const else C
    # SVD-backed least squares projects onto the actual column space.  An unpivoted QR returns
    # one Q column per input column even when sex is constant or two PCs/covariates are redundant,
    # which removes arbitrary directions that are not in the covariate space.
    coef, _, rank, _ = np.linalg.lstsq(D, np.column_stack([y, x]), rcond=None)
    adjusted = np.column_stack([y, x]) - D @ coef
    ry, rx = adjusted[:, 0], adjusted[:, 1]
    df = n - int(rank) - 1
    sx = np.std(rx)
    if sx == 0 or df <= 0:
        return dict(n=int(n), beta_sd=np.nan, se=np.nan, p=np.nan, partial_r=np.nan,
                    delta_r2=np.nan, r2_base=np.nan)
    rxs = rx / sx
    b = float(rxs @ ry / (rxs @ rxs))
    resid = ry - b * rxs
    se = float(np.sqrt((resid @ resid) / df / (rxs @ rxs)))
    t = b / se if se > 0 else np.nan
    r = float(np.corrcoef(rx, ry)[0, 1])
    r2_base = 1 - np.var(ry) / np.var(y) if np.var(y) > 0 else np.nan
    return dict(n=int(n), beta_sd=b, se=se, p=float(2 * stats.t.sf(abs(t), df)),
                partial_r=r, delta_r2=float(r * r * (1 - r2_base)), r2_base=float(r2_base))


# --------------------------------------------------------------------------------- tables
# Sites export tables from Excel, R, SAS and their EHR, and three things vary without anyone
# thinking to mention them: the delimiter (a .csv is often tab- or semicolon-separated), the
# compression, and how a yes/no field is spelled. None of these is interesting, all of them
# used to produce either a cryptic error or -- worse -- a silent column of NaNs, so they are
# handled once here rather than in each reader.
SEPS = (',', '\t', ';', '|')
SEP_NAME = {',': 'comma', '\t': 'tab', ';': 'semicolon', '|': 'pipe'}

TRUE_WORDS = {'1', 'y', 'yes', 'true', 't', 'pos', 'positive', 'present', 'abnormal', 'oui'}
FALSE_WORDS = {'0', 'n', 'no', 'false', 'f', 'neg', 'negative', 'absent', 'normal', 'non'}


def sniff_sep(path):
    """The delimiter actually used, from the header line. Extension is not evidence."""
    import gzip
    op = gzip.open if str(path).endswith('.gz') else open
    try:
        with op(path, 'rt', errors='replace') as fh:
            head = fh.readline()
    except OSError as e:
        raise SystemExit(f'{path}: cannot be read ({e})')
    best, n_best = ',', 1
    for sep in SEPS:
        n = len(head.split(sep))
        if n > n_best:
            best, n_best = sep, n
    return best


def read_table(path, what='', dtype=None, sep=None, **kw):
    """Read a delimited table without requiring the site to describe it.

    Sniffs the delimiter, follows .gz, and refuses an empty file by saying it is empty rather
    than letting a zero-row frame travel downstream and fail later as a join that matched
    nothing -- which is the same symptom as a genuine id mismatch and sends the site looking in
    the wrong place.
    """
    p = Path(path)
    # `what` is the flag the site typed ('--ses'); the path is appended here so a caller that
    # already spelled the path into its label does not print it twice.
    label = f'{what}: {p}' if what and str(p) not in what else (what or str(p))
    if not p.exists():
        raise SystemExit(f'{label} does not exist')
    if p.stat().st_size == 0:
        raise SystemExit(f'{label} is an empty file (0 bytes)')
    sep = sep or sniff_sep(p)
    try:
        D = pd.read_csv(p, sep=sep, dtype=dtype, **kw)
    except Exception as e:                                               # noqa: BLE001
        raise SystemExit(f'{label} could not be parsed as a {SEP_NAME.get(sep, sep)}-'
                         f'separated table ({type(e).__name__}: {e})')
    if len(D) == 0:
        raise SystemExit(f'{label} has a header but no rows.')
    # A one-column frame is legitimate (an id list), so it is only an error when the single
    # column NAME still has a delimiter in it: that is a header which was never split, and the
    # symptom the site would otherwise see is "no id column found" listing one absurd name.
    if D.shape[1] == 1:
        only = str(list(D.columns)[0])
        stuck = [c for c in list(SEPS) + [':', ' '] if c in only]
        if stuck:
            shown = ', '.join(SEP_NAME.get(c, repr(c)) for c in stuck)
            raise SystemExit(
                f'{label} parsed to a single column, named {only!r}.'
                f'\n  its header still contains: {shown}'
                f'\n  the delimiter was read as {SEP_NAME.get(sep, sep)}, which did not split '
                f'this header.'
                f'\n  save the file as comma- or tab-separated, or pass the delimiter '
                f'explicitly where the flag allows it.')
    return D


def pick_column(df, wanted, candidates=(), what='', kind='column', flag=''):
    """Find a column by any of `candidates`, case- and space-insensitively, or fail usefully.

    The failure names what WAS there and gives the exact flag to pass, because "no id column
    found" plus a list of twenty column names is a puzzle, not an error message.
    """
    lower = {str(c).strip().lower(): c for c in df.columns}
    if wanted and wanted in df.columns:
        return wanted
    if wanted and str(wanted).strip().lower() in lower:
        return lower[str(wanted).strip().lower()]
    for c in candidates:
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    cols = list(df.columns)
    hint = f'\n  pass {flag} <name>' if flag else ''
    guess = ''
    if flag:
        likely = [c for c in cols if any(k in str(c).lower()
                                         for k in ('id', 'mrn', 'empi', 'subject', 'patient'))]
        if likely:
            guess = f'\n  {flag} {likely[0]}   <- looks like the one'
    raise SystemExit(f'{what or "table"}: no {kind} found.'
                     f'\n  columns present: {cols[:15]}{" ..." if len(cols) > 15 else ""}'
                     f'{hint}{guess}')


def to_binary(sr):
    """0/1 from whatever the site wrote. Returns (values, how, n_unmapped).

    0/1 is what the documentation asks for, but clinical exports say Yes/No, Y/N, true/false,
    positive/negative. Coercing with to_numeric turns every one of those into NaN silently, and
    a findings table that is entirely NaN looks exactly like a cohort with no positives.
    """
    v = pd.to_numeric(sr, errors='coerce')
    if v.notna().mean() > 0.5:
        return (v >= 1).astype(float).where(v.notna()), 'numeric', int(v.isna().sum())
    t = sr.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=sr.index, dtype=float)
    out[t.isin(TRUE_WORDS)] = 1.0
    out[t.isin(FALSE_WORDS)] = 0.0
    return out, 'yes/no words', int(out.isna().sum())


# --------------------------------------------------------------------------------- order
CHAIN_ORDER = ('prepare -> score -> traits -> geno -> loci / pgs / scan / fairness / ses '
               '-> figures -> package')


def require(mode, items):
    """Refuse to start when an upstream stage has not run, and say which one.

    Without this the failure is a FileNotFoundError on a path the site has never heard of,
    several frames into pandas. Naming the stage that produces the file turns "what is
    geno_provenance.json" into "run geno first", which is the whole of the fix.
    """
    missing = [(q, what, stage) for q, what, stage in items if not Path(q).exists()]
    if not missing:
        return
    out = [f'{mode}: cannot start -- {len(missing)} input(s) from an earlier stage '
           f'{"is" if len(missing) == 1 else "are"} missing.', '']
    for q, what, stage in missing:
        out += [f'  {what}',
                f'    expected at : {q}',
                f'    produced by : `{stage}`  (run that first)', '']
    out += [f'The stages run in this order:', f'  {CHAIN_ORDER}', '',
            'Or let the chain do it:',
            '  python -m ancestry all --work <work> --results <results> --rfmix ... --geno ...']
    raise SystemExit('\n'.join(out))


# --------------------------------------------------------------------------------- builds
# Every reference table that carries a POSITION ships in two builds, and which one a site needs
# is the single commonest way this container silently produces nothing: a GRCh38 position looked
# up in a GRCh37 release matches almost no variants, and the few it does match are coincidences.
# So the build is named once, normalised here, and carried everywhere through provenance rather
# than being re-stated per mode.
BUILD_ALIASES = {
    'grch38': 'grch38', 'hg38': 'grch38', 'b38': 'grch38', '38': 'grch38', 'grch38.p13': 'grch38',
    'grch37': 'grch37', 'hg19': 'grch37', 'b37': 'grch37', '37': 'grch37', 'hs37d5': 'grch37',
}


def build_suffix(build):
    """'hg19' -> 'grch37'. Any of the usual spellings, in any case, for either build."""
    b = str(build or 'grch38').strip().lower().replace('_', '').replace('-', '')
    if b not in BUILD_ALIASES:
        raise SystemExit(
            f'--build {build!r} is not a build this image knows. Use grch38 (aliases: hg38, b38) '
            f'or grch37 (aliases: hg19, b37). If you do not know which your release is, run '
            f'`survey` -- it looks the discovery leads up in both and tells you which one matches.')
    return BUILD_ALIASES[b]


def ref_table(ref, work, stem, build, required=True):
    """The build-specific reference table, preferring one the site lifted itself.

    Looks for work/ref/<stem>_<build>.tsv first, then ref/<stem>_<build>.tsv. The work/ copy is
    how a GRCh37 site gets the ancestry-PGS panel, which ships in GRCh38 only: `geno --chain`
    lifts it once and leaves it there. Returns None when it is absent and not required, so the
    arm that needs it can say so and be skipped rather than silently reading the wrong build.
    """
    sfx = build_suffix(build)
    name = f'{stem}_{sfx}.tsv'
    for base in ([Path(work) / 'ref'] if work else []) + [Path(ref)]:
        f = Path(base) / name
        if f.exists():
            return f
    if required:
        raise SystemExit(f'{name} is not in {ref} and was not lifted into {work}/ref. '
                         f'This image cannot run that arm on {sfx}.')
    return None


def default_threads():
    """How many threads plink gets unless the site says otherwise.

    Eight was a safe guess for one node; it is a poor one for a whole-genome scan on a 64-core
    hospital server, where it leaves most of the machine idle for hours, and a bad one on a
    shared 4-core VM, where it oversubscribes.  plink2 scales well to about sixteen threads and
    poorly past it, so take what the machine reports, keep two cores for everything else, and cap
    at sixteen.  `--threads` overrides this everywhere.
    """
    n = os.cpu_count() or 4
    return max(1, min(16, n - 2 if n > 4 else n))


def meta_fe(beta, se):
    """Inverse-variance fixed-effect meta, with Cochran's Q and I^2."""
    beta = np.asarray(beta, float); se = np.asarray(se, float)
    ok = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    if ok.sum() == 0:
        return dict(beta=np.nan, se=np.nan, p=np.nan, q=np.nan, i2=np.nan, k=0)
    w = 1 / se[ok] ** 2
    b = float((w * beta[ok]).sum() / w.sum())
    s = float(np.sqrt(1 / w.sum()))
    q = float((w * (beta[ok] - b) ** 2).sum())
    k = int(ok.sum())
    i2 = float(max(0.0, (q - (k - 1)) / q) * 100) if q > 0 and k > 1 else 0.0
    return dict(beta=b, se=s, p=float(2 * stats.norm.sf(abs(b / s))), q=q, i2=i2, k=k)


def auroc(y, p):
    """AUROC by the rank identity, so sklearn is not a hard dependency of anything here."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = stats.rankdata(p)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ece(y, p, bins=10):
    """Expected calibration error on equal-width probability bins.

    Reported, never acted on: a site whose non-European prevalence differs from the matched set
    the model was calibrated in HAS a large ECE and a perfectly good AUROC, which is exactly what
    MGB-MGH saw in October (0.30).  It is here so that a large value is recognised as the
    prevalence shift it is rather than read as a broken model.
    """
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if not len(y):
        return np.nan
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def boot_ci(stat_fn, n, n_boot=1000, seed=2000, clusters=None):
    """Percentile bootstrap of stat_fn(idx). Resamples `clusters` when given, rows otherwise."""
    rng = np.random.default_rng(seed)
    if clusters is None:
        draws = [stat_fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    else:
        cl = pd.Series(np.arange(n)).groupby(np.asarray(clusters)).apply(list)
        keys = np.asarray(cl.index)
        members = list(cl.values)
        draws = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(keys), len(keys))
            idx = np.concatenate([members[i] for i in pick])
            draws.append(stat_fn(idx))
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 50:
        return np.nan, np.nan
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# --------------------------------------------------------------------------------- alleles
def allele_sign(site_a1, site_a2, want_a1, want_a2):
    """+1 for the expected allele pair, -1 for its reverse, NaN for a different variant.

    NaN rather than a guess. A site whose alleles match ours as a pair in neither orientation is a
    strand ambiguity or a multi-allelic collapse, and there is no safe inference from a position
    and two letters; such a variant is dropped and counted in the report.
    """
    a1 = np.char.upper(np.asarray(site_a1, dtype=str))
    a2 = np.char.upper(np.asarray(site_a2, dtype=str))
    w1 = np.char.upper(np.asarray(want_a1, dtype=str))
    w2 = np.char.upper(np.asarray(want_a2, dtype=str))
    same = (a1 == w1) & (a2 == w2)
    flipped = (a1 == w2) & (a2 == w1)
    return np.where(same, 1.0, np.where(flipped, -1.0, np.nan))


def key_sorted(chrom, pos, a1, a2):
    """chr:pos:<alleles in alphabetical order> -- the allele-order-free variant key."""
    c = pd.Series(chrom).astype(str).str.replace('chr', '', regex=False)
    p = pd.Series(pos).astype(str)
    s = [':'.join(sorted([str(x).upper(), str(y).upper()])) for x, y in zip(a1, a2)]
    return (c + ':' + p + ':' + pd.Series(s, index=c.index)).to_numpy()


def clump(W, D, pos, r2=0.1, win_kb=500):
    """Greedy p-value clumping in the TARGET cohort's own genotypes.

    W: frame with columns P and ri (row index into D). D: (variants x people), already centred.
    Keeps the most significant variant, drops everything within `win_kb` correlated above `r2`,
    repeats. Uses no phenotype, so computing LD in the target cohort leaks nothing -- and LD is a
    property of the target cohort, so computing it anywhere else would be wrong.
    """
    order = np.argsort(W.P.to_numpy())
    taken, dead = [], np.zeros(len(W), dtype=bool)
    ri = W.ri.to_numpy()
    sd = D.std(axis=1)
    for i in order:
        if dead[i]:
            continue
        taken.append(i)
        near = np.where((np.abs(pos - pos[i]) <= win_kb * 1000) & ~dead)[0]
        if len(near) == 0:
            continue
        a = D[ri[i]]
        sa = sd[ri[i]]
        if sa == 0:
            continue
        B = D[ri[near]]
        sb = sd[ri[near]]
        with np.errstate(invalid='ignore', divide='ignore'):
            r = (B @ a) / (len(a) * sa * np.where(sb > 0, sb, np.nan))
        dead[near[np.nan_to_num(r ** 2) > r2]] = True
        dead[i] = False
    return np.asarray(sorted(taken))


# --------------------------------------------------------------------------------- shell
def sh(cmd, check=True, quiet=False, **kw):
    """Run a command, echoing it. Raises on failure unless check=False."""
    if isinstance(cmd, (str, bytes)):
        raise TypeError('sh() requires an argument sequence; shell command strings are refused')
    if not quiet:
        print('    $ ' + ' '.join(map(str, cmd)), flush=True)
    r = subprocess.run([str(value) for value in cmd], shell=False,
                       capture_output=True, text=True, **kw)
    if r.returncode and check:
        sys.stderr.write(r.stdout[-4000:] + '\n' + r.stderr[-4000:] + '\n')
        raise SystemExit(f'command failed ({r.returncode})')
    return r


def plink2():
    p = os.environ.get('PLINK2', shutil.which('plink2') or '/opt/bin/plink2')
    if not Path(p).exists() and not shutil.which(p):
        raise SystemExit('plink2 not found; set PLINK2=/path/to/plink2')
    return p


def resolve_geno(work, pattern, quiet=False):
    """Swap a VCF pattern for the pgen `convert` produced, if it did.

    The site keeps passing the same --geno it always passed; this is what makes `convert`
    optional rather than a second thing to remember. If there is no converted set the original
    pattern is returned untouched and plink reads the VCF directly, which works and is slow.
    """
    import json
    if not pattern:
        return pattern
    f = Path(work) / 'geno' / 'converted.json'
    if not f.exists():
        return pattern
    try:
        prov = json.loads(f.read_text())
    except (OSError, ValueError):
        return pattern
    new = prov.get('pattern')
    if not new or new == pattern:
        return pattern
    if not quiet:
        print(f'  using the converted genotypes from `convert`: {new}')
        print(f'    (built from {prov.get("source")}, {len(prov.get("chroms", []))} chromosomes)')
    return new


def geno_args(pattern, chrom=None):
    """--pfile/--bfile/--vcf/--bgen arguments for one genotype set.

    `pattern` takes @ where the chromosome number goes, as every script in this image does.
    """
    p = pattern.replace('@', str(chrom)) if chrom is not None else pattern
    if p.endswith('.vcf.gz') or p.endswith('.vcf'):
        return ['--vcf', p, 'dosage=DS']
    if p.endswith('.bgen'):
        sample = p[:-5] + '.sample'
        a = ['--bgen', p, 'ref-first']
        if Path(sample).exists():
            a += ['--sample', sample]
        return a
    if Path(p + '.pgen').exists():
        return ['--pfile', p]
    if Path(p + '.bed').exists():
        return ['--bfile', p]
    if Path(p + '.pgen').exists() or Path(p).suffix == '.pgen':
        return ['--pfile', str(Path(p).with_suffix(''))]
    raise SystemExit(f'no pgen/bed/vcf/bgen at {p}')


# --------------------------------------------------------------------------------- directories
def fresh_results(path: Path):
    """Never leave a half-new results directory behind.

    The Lifelines run that taught this lesson re-ran a step that crashed, wrote nothing, and left
    the previous run's files in place -- so the directory looked finished and the numbers were
    three days old. Anything already here is moved aside with a timestamp before a run starts.
    """
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        prev = path.parent / f'{path.name}_prev_{datetime.datetime.now():%Y%m%d_%H%M%S}'
        shutil.move(str(path), str(prev))
        print(f'  previous results moved to {prev}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def failed(path: Path, exc):
    """Write FAILED.txt into results/ so a site that sends the directory sends the reason too."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / 'FAILED.txt').write_text(
            f'{datetime.datetime.now().isoformat()}\nversion {VERSION}\n\n'
            + ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass


def banner(what):
    print(f'\n{"=" * 92}\n  {what}\n{"=" * 92}', flush=True)


HERE_REF = Path(__file__).resolve().parent / 'reference'


def reference_is_populated(d):
    """True when `d` holds the tables MANIFEST.txt names, not just the manifest itself."""
    d = Path(d)
    man = d / 'MANIFEST.txt'
    if not man.exists():
        return False
    names = [line.split()[-1] for line in man.read_text().splitlines()
             if line.strip() and not line.startswith('#') and len(line.split()) >= 2]
    return bool(names) and all((d / n).exists() for n in names)


def ref_dir():
    """Where the frozen discovery tables are, WITHOUT reaching the network.

    This is an argparse default -- it is evaluated by `--help` too -- so it must never download.
    Order: REFERENCE_DIR, then a populated copy beside this package (a mounted or baked-in set),
    then a snapshot already on disk from an earlier fetch. When none of those holds the tables it
    returns the in-package path anyway, so `--help` prints something, and `fetch_reference()`
    is what turns that into an error a person can act on.
    """
    try:
        from ancestry import weights as W
        src = W.source()
    except Exception:
        src = 'auto'
    env = os.environ.get('REFERENCE_DIR')
    if env and src != 'huggingface':
        return Path(env)
    if src != 'huggingface' and reference_is_populated(HERE_REF):
        return HERE_REF
    try:
        from ancestry import weights as W
        cached = W.resolve('reference_tables', download=False)
        if reference_is_populated(cached):
            return Path(cached)
    except Exception:
        pass
    return HERE_REF


def fetch_reference(ref=None, api_key='', download=True):
    """The reference directory, downloading the pinned snapshot if that is the only copy.

    Raises SystemExit naming what to do, because the alternative is a pandas error from inside
    whichever table a mode happened to read first.
    """
    d = Path(ref) if ref else ref_dir()
    if reference_is_populated(d):
        return d
    from ancestry import weights as W
    try:
        got = Path(W.resolve('reference_tables', api_key=api_key, download=download))
    except W.WeightsUnavailable as e:
        raise SystemExit(
            f'the frozen discovery tables are not here: {e}\n'
            '  They are not in this repository on purpose -- it is public, and they are '
            'aggregates of\n  the discovery cohorts. They ship under reference/ in the pinned '
            'ancestry model\n  snapshot, so either mount them and set REFERENCE_DIR, or '
            'build the image with\n  --build-arg RUN_MODELS_SETUP=true and a Hugging Face token '
            'that can read it.')
    if not reference_is_populated(got):
        raise SystemExit(f'{got} does not hold the tables MANIFEST.txt names')
    return got
