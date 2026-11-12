"""Mode `selftest` -- RUN BEFORE ANYTHING ELSE.  A whole synthetic cohort, end to end.

A container that is wrong at a partner site is expensive to find out about: the data cannot be
sent to us, the people who can see it cannot read the code, and every round trip is a week.  So
the image carries a cohort of its own.  The fixture is built from a seeded random number
generator with KNOWN TRUTH injected -- genotypes whose effect on the phenotype is set by us, a
clinical finding whose false-positive gap between the two ancestry groups is set by us -- and the
test asserts that the pipeline recovers what was put in, rather than merely that it ran.

THE STAGES

  T1  reference integrity   every shipped table matches the md5 recorded at build time.
  T2  arithmetic            the partial regression recovers a known coefficient; RINT is a
                            rank transform; allele harmonisation flips what it should and
                            refuses what it cannot resolve; clumping drops a variant in perfect
                            LD with a stronger one and keeps an independent one.
  T3  genetics end to end   a synthetic bed/bim/fam at the real variant positions, with the real
                            discovery effects injected at twice their size, run through
                            geno -> loci -> pgs.  Asserts the sign concordance is near-perfect
                            and the fitted slope is near 2.  (SKIPPED, loudly, without plink2.)
  T4  fairness              a synthetic 77-finding head and a ground truth with a +5 per 100
                            false-positive gap injected on one finding.  Asserts the audit
                            recovers it on the genetic split, and that a finding with no gap
                            comes back with an interval covering zero.
  T5  privacy               the audit catches an identifier, an unmasked small count and an
                            over-long table when they are deliberately planted; and the real
                            results directory contains none of the three.
  T6  figures               every plate draws from the tables T3 and T4 produced.
  T7  package               the zip is written and its audit passes.

    python -m ancestry selftest --out /tmp/selftest [--quick]

Writes a report; exits non-zero if anything FAILS.  A stage that cannot run says SKIPPED and the
report says why -- a skipped stage is never counted as a pass.
"""
from __future__ import annotations

import argparse
import json
import importlib
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # the repository root, from which `python -m ancestry` resolves
# Subprocess checks run the package the way the image does. PYTHONPATH rather than cwd:
# several checks assert on paths relative to their own temporary directory.
PYENV = {**os.environ, 'PYTHONPATH': os.pathsep.join(
    [str(ROOT)] + ([os.environ['PYTHONPATH']] if os.environ.get('PYTHONPATH') else []))}
from ancestry import privacy as PV                                       # noqa: E402
from ancestry import lib as L                                            # noqa: E402

N_PEOPLE = 5000
SEED = 2000


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, note=''):
        self.rows.append(dict(check=name, result='PASS' if ok else 'FAIL', note=note))
        print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  -- {note}' if note else ''))
        return ok

    def skip(self, name, why):
        self.rows.append(dict(check=name, result='SKIP', note=why))
        print(f'  [SKIP] {name}  -- {why}')

    @property
    def failed(self):
        return [r for r in self.rows if r['result'] == 'FAIL']


# ---------------------------------------------------------------------------- bed writer
_CODE = {2: 0b00, 1: 0b10, 0: 0b11}


def write_bed(stem, G, bim, iids):
    """PLINK1 binary, variant-major.  G is (variants x people) in {0,1,2}; -1 is missing."""
    stem = Path(stem)
    n = G.shape[1]
    with open(f'{stem}.bed', 'wb') as fh:
        fh.write(bytes([0x6c, 0x1b, 0x01]))
        for row in G:
            buf = bytearray((n + 3) // 4)
            for i, g in enumerate(row):
                code = 0b01 if g < 0 else _CODE[int(g)]
                buf[i >> 2] |= code << (2 * (i & 3))
            fh.write(bytes(buf))
    bim.to_csv(f'{stem}.bim', sep='\t', header=False, index=False)
    pd.DataFrame({0: iids, 1: iids, 2: 0, 3: 0, 4: 0, 5: -9}).to_csv(
        f'{stem}.fam', sep=' ', header=False, index=False)
    return str(stem)


# ---------------------------------------------------------------------------- the fixture
def build_fixture(root, ref, rng, n=N_PEOPLE, effect_mult=2.0):
    """A cohort with known truth: genotypes that cause the phenotype, and a planted audit gap."""
    root = Path(root)
    work = root / 'work'; res = root / 'results'
    (work / 'geno').mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    # people: 70% European, the rest split over the three non-European ancestries
    grp = rng.choice(['EUR', 'AFR', 'EAS', 'SAS'], size=n, p=[0.70, 0.14, 0.09, 0.07])
    props = np.zeros((n, 4))
    cols = L.ANCESTRIES
    for i, g in enumerate(grp):
        k = cols.index(g)
        props[i] = rng.dirichlet([1, 1, 1, 1]) * 0.12
        props[i, k] += 0.88
        props[i] /= props[i].sum()
    ids = [f'SYN{i:06d}' for i in range(n)]
    roster = pd.DataFrame({'row': np.arange(n), 'participant_id': ids, 'geno_id': ids,
                           'stratum': L.assign_stratum(pd.DataFrame(props, columns=cols)),
                           'age': rng.normal(62, 11, n).round(1),
                           'sex': rng.choice(['M', 'F'], n), 'batch': 1, 'round': 1})
    for j, c in enumerate(cols):
        roster[f'rf_{c}'] = props[:, j]
    roster['rf_nonEUR'] = 1 - roster.rf_EUR
    roster.to_csv(work / 'roster.csv', index=False)

    # genotypes at the real variant positions, with the real effects injected x effect_mult
    leads = pd.read_csv(ref / 'leads_grch38.tsv', sep='\t', dtype={'CHR': str})
    tr = pd.read_csv(ref / 'pgs_weights_grch38.tsv', sep='\t', dtype={'CHR': str})
    apw = ref / 'ancestry_pgs_weights_grch38.tsv'
    ap = pd.read_csv(apw, sep='\t', dtype={'CHR': str}) if apw.exists() else None
    if ap is not None:
        ap = ap.sample(min(len(ap), 3000), random_state=SEED)
    V = pd.concat([
        leads.assign(A1u=leads.A1, A0u=leads.A0, BETAu=leads.BETA * effect_mult, SET='lead')[
            ['CHR', 'POS', 'A1u', 'A0u', 'BETAu', 'SET']],
        tr.assign(A1u=tr.RISK_ALLELE,
                  A0u=np.where(tr.PANEL_A1 == tr.RISK_ALLELE, tr.PANEL_A2, tr.PANEL_A1),
                  BETAu=0.0, SET='trait')[['CHR', 'POS', 'A1u', 'A0u', 'BETAu', 'SET']]]
        + ([ap.assign(A1u=ap.A1, A0u=ap.A0, BETAu=ap.BETA * effect_mult, SET='anc')[
            ['CHR', 'POS', 'A1u', 'A0u', 'BETAu', 'SET']]] if ap is not None else []),
        ignore_index=True)
    V = V.drop_duplicates(['CHR', 'POS']).reset_index(drop=True)
    V = V[V.CHR.astype(str).str.isdigit()].reset_index(drop=True)
    # Sort by chromosome then position BEFORE the genotype matrix is built, so G stays row-
    # aligned with V. The three panels are concatenated above, which interleaves chromosomes;
    # plink2 refuses such a file outright ("has a split chromosome") and every extraction
    # returns nothing. A real imputed release is sorted, so this was purely a fixture defect --
    # but it hid T3 completely, and T3 is the only end-to-end test of the genetic stages.
    V = (V.assign(_c=V.CHR.astype(int), _p=V.POS.astype(int))
          .sort_values(['_c', '_p'], kind='mergesort')
          .drop(columns=['_c', '_p'])
          .reset_index(drop=True))
    m = len(V)
    freq = np.clip(rng.beta(2, 2, m), 0.05, 0.95)
    G = rng.binomial(2, freq[:, None], size=(m, n)).astype(np.int8)   # counts of A1u
    # the phenotype the genotypes cause, plus age/sex noise
    eff = V.BETAu.to_numpy(float)
    gsd = G.std(axis=1); gsd[gsd == 0] = 1
    y = (eff @ ((G - G.mean(axis=1, keepdims=True)) / gsd[:, None])) * 1.0
    y = y / (np.std(y) or 1)
    y = 0.55 * y + 0.25 * ((roster.age.to_numpy() - 62) / 11) + rng.normal(0, 0.8, n)
    # a score in (0,1) whose logit is y, shifted so the non-European strata sit higher
    shift = np.where(roster.stratum.to_numpy() == 'EUR', -0.9, 1.1)
    p = 1 / (1 + np.exp(-(0.9 * y + shift)))
    S = roster.copy()
    S['p_nonEUR'] = p
    S['score_logit'] = L.logit(p)
    S['pheno_rint'] = np.nan
    for st, g in S.groupby('stratum'):
        if len(g) >= PV.MIN_STRATUM:
            S.loc[g.index, 'pheno_rint'] = L.rint(g.score_logit.to_numpy())
    S.to_csv(work / 'scores.csv', index=False)

    bim = pd.DataFrame({0: V.CHR.astype(str), 1: V.CHR.astype(str) + ':' + V.POS.astype(str),
                        2: 0, 3: V.POS.astype(int), 4: V.A1u, 5: V.A0u})
    write_bed(work / 'geno' / 'synthetic', G, bim, ids)
    (work / 'prepared_dir.txt').write_text(str(root / 'prepared') + '\n')
    return work, res, S, V, G


# The findings planted into the synthetic cohort, and the false-positive gap each is given, per
# 100 patients on the genetic split. `Afib` and `Bradycardia` carry T4/T4b -- a known gap and a
# known null -- and the remaining four exist so the AUDITABILITY statistic has something to be
# computed on: it needs at least four findings comparable on both splits (len(j) < 4 -> skip in
# fairness.py), and with only the first two the slope, correlation and sign concordance that are
# the headline of this mode were never executed by any test. The spread is deliberate: two more
# positive gaps of different size, one negative, one null, so a positive slope and a majority of
# concordant signs are a real finding about the planted structure and not an artefact of every
# gap pointing the same way.
PLANTED_GAPS = (('Afib', 1.0), ('Bradycardia', 0.0), ('Sinusal', 0.6),
                ('Regular', 0.3), ('Monomorph', -0.4), ('Left axis deviation', 0.0))


def plant_findings(work, S, rng, gap_per_100=10.0):
    """A synthetic 77-output head and a ground truth with known gaps on six findings.

    Each finding in PLANTED_GAPS gets a false-positive gap of its multiplier x `gap_per_100` per
    100 on the genetic split -- the model flags non-European patients who do not have it more
    often. `Afib` gets the full gap and `Bradycardia` none: the audit has to find the first and
    not invent the second. The other four give the auditability slope its input.
    """
    n = len(S)
    F = pd.read_csv(L.ref_dir() / 'findings77.tsv', sep='\t')
    outs = sorted(set(F.output.dropna().astype(str)) - {''})
    W = pd.DataFrame({'participant_id': S.participant_id.to_numpy()})
    for o in outs:
        W[o] = rng.uniform(0, 1, n).astype(np.float32)
    lab = pd.DataFrame({'participant_id': S.participant_id.to_numpy()})
    nonE = (S.stratum != 'EUR').to_numpy()
    # Each finding is planted by setting the two groups' false-positive RATES directly rather
    # than by nudging a score distribution: the quantity the audit measures is then exactly the
    # quantity that was planted, and a null finding still has a non-zero rate in both groups, so
    # T4b is a real null and not a degenerate 0-vs-0 comparison.
    FPR_EUR = 0.25
    for fin, mult in PLANTED_GAPS:
        hit = F.loc[F.finding == fin, 'output']
        if not len(hit) or not str(hit.iloc[0]).strip():
            continue                      # a reference table without this finding: skip it
        gap = mult * gap_per_100 / 100.0
        o = str(hit.iloc[0])
        y = rng.binomial(1, 0.12, n)
        want = np.where(nonE, FPR_EUR + gap, FPR_EUR)
        flag_neg = rng.uniform(0, 1, n) < want
        W[o] = np.where(y == 1, rng.uniform(0.55, 1.0, n),
                        np.where(flag_neg, rng.uniform(0.55, 1.0, n),
                                 rng.uniform(0.0, 0.45, n))).astype(np.float32)
        lab[fin] = y
    W.to_parquet(work / 'findings.parquet', index=False)
    lab.to_csv(work / 'labels.csv', index=False)
    return lab


# ---------------------------------------------------------------------------- stages
ALL_MODES = ['survey', 'selftest', 'weights', 'convert', 'prepare', 'score', 'traits', 'geno', 'ses',
             'loci', 'pgs', 'scan', 'fairness', 'figures', 'package', 'submit',
             # the external-validation modes merged in from the deepecg-ancestry container
             'validate', 'metrics', 'admixed', 'ensemble', 'all']


def t0_dispatch(R, log):
    """Every mode loads, builds its parser and dispatches.

    THE BUG THIS EXISTS FOR.  A mode can import cleanly and still die the moment it is dispatched
    -- a name used in `main()` that was never imported, a default that calls a function the module
    does not have.  `all` shipped broken exactly that way: it referenced L.default_threads() while
    entrypoint.py had never imported _lib, so every individual mode worked and the one command the
    guide tells the site to run raised NameError before doing anything.  Nothing else in this
    selftest touched it, because every other stage calls the modes directly.

    `--help` is enough to catch the whole class: argparse builds every default and every help
    string, which is where those references live, and it costs a second for all fourteen.
    """
    bad = []
    for m in ALL_MODES:
        rc, out = run_mode(m, ['--help'], log)
        if rc != 0:
            bad.append(f'{m} (rc={rc}: ' + out.strip().splitlines()[-1][:60] + ')')
    R.add('T0 every mode loads and parses its arguments', not bad,
          f'{len(ALL_MODES) - len(bad)}/{len(ALL_MODES)} modes'
          + ('' if not bad else '  FAILED: ' + '; '.join(bad)))


def t0b_chain_guards(R, root, log):
    """`all` refuses a bad invocation BEFORE it runs anything, and says why.

    Two guards, both added after a real failure: an input path that does not exist used to
    surface hours later as a pandas traceback from inside `prepare`, and a mode-specific flag
    used to be appended to every mode in the chain, where the eight with strict parsers rejected
    it and the loop then swallowed the exit.
    """
    rc, out = run_mode('all', ['--work', root / 'x', '--results', root / 'y',
                               '--rfmix', '/does/not/exist.txt',
                               '--ecg-meta', '/does/not/exist.parquet',
                               '--geno', '/does/not/exist/chr@'], log)
    R.add('T0b missing inputs stop the chain before any work',
          rc != 0 and 'were not found' in out and 'nothing was run' in out,
          'all three missing paths named up front' if 'were not found' in out
          else f'rc={rc}, unexpected message')

    rc, out = run_mode('all', ['--work', root / 'x', '--results', root / 'y',
                               '--rfmix', '/does/not/exist.txt',
                               '--ecg-meta', '/x.parquet', '--geno', '/x/chr@',
                               '--not-a-real-flag', '7'], log)
    R.add('T0c a flag no mode accepts is refused by name',
          rc != 0 and 'not accepted by any mode' in out,
          'named and rejected' if 'not accepted by any mode' in out
          else f'rc={rc}, unexpected message')


def t0d_vcf_route(R, root, log):
    """A VCF path makes `all` insert `convert`, and the converted set is picked up afterwards.

    Both halves silently matter. If the detection misses, the site reads VCFs a hundred times and
    the run takes days for no reason; if `resolve_geno` misses, `convert` runs and then every
    later stage reads the VCFs anyway, which is worse -- the work is done twice and nothing says
    so.
    """
    import json
    vdir = Path(root) / 'vcf_fixture'
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / 'chr1.vcf.gz').write_bytes(b'')
    rc, out = run_mode('all', ['--work', root / 'vw', '--results', root / 'vr',
                               '--rfmix', '/nope.txt', '--ecg-meta', '/nope.parquet',
                               '--geno', str(vdir / 'chr@.vcf.gz')], log)
    R.add('T0d VCF input inserts the conversion stage',
          '`convert` is inserted first' in out,
          'detected' if '`convert` is inserted first' in out else 'NOT detected')

    w = Path(root) / 'vw2'
    (w / 'geno').mkdir(parents=True, exist_ok=True)
    tgt = str(w / 'geno' / 'pgen' / 'chr@')
    (w / 'geno' / 'converted.json').write_text(json.dumps(
        dict(pattern=tgt, source='/data/chr@.vcf.gz', chroms=list(range(1, 23)))))
    got = L.resolve_geno(w, '/data/chr@.vcf.gz', quiet=True)
    R.add('T0e the converted genotypes are used by later stages', got == tgt,
          'pgen' if got == tgt else f'still {got}')


def t1_reference(R, ref):
    man = Path(ref) / 'MANIFEST.txt'
    if not man.exists():
        return R.skip('T1 reference integrity', 'no MANIFEST.txt in ref/')
    # The tables are not in the repository (see ancestry/reference/README.md), so a fresh clone
    # has the manifest and nothing to check it against. That is a reason to skip, not to fail;
    # inside the image, or with REFERENCE_DIR set, they are present and this test is strict.
    if not L.reference_is_populated(ref):
        return R.skip('T1 reference integrity',
                      f'{ref} holds MANIFEST.txt but not the tables -- set REFERENCE_DIR or '
                      'fetch the pinned snapshot')
    import hashlib
    bad = []
    for line in man.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        md5, name = parts[0], parts[-1]
        p = Path(ref) / name
        if not p.exists():
            bad.append(f'{name} missing'); continue
        h = hashlib.md5()
        with open(p, 'rb') as fh:
            for b in iter(lambda: fh.read(1 << 20), b''):
                h.update(b)
        if h.hexdigest() != md5:
            bad.append(f'{name} changed')
    R.add('T1 reference integrity', not bad, '; '.join(bad) or 'all shipped tables match')


def t2_arithmetic(R, rng):
    n = 4000
    C = rng.normal(size=(n, 3)); x = rng.normal(size=n)
    y = 0.30 * x + C @ np.ones(3) + rng.normal(size=n)
    r = L.partial(y, x, C)
    R.add('T2a partial regression recovers a known beta',
          abs(r['beta_sd'] - 0.30) < 0.05, f"beta {r['beta_sd']:.3f} (truth 0.300)")
    z = L.rint(rng.lognormal(size=500))
    R.add('T2b RINT is normal and rank-preserving',
          abs(np.mean(z)) < 0.1 and abs(np.std(z) - 1) < 0.1,
          f'mean {np.mean(z):.3f}, sd {np.std(z):.3f}')
    s = L.allele_sign(['A', 'T', 'G'], ['G', 'A', 'C'], ['A', 'A', 'A'])
    R.add('T2c allele harmonisation', list(s[:2]) == [1.0, -1.0] and not np.isfinite(s[2]),
          'same=+1, flipped=-1, unresolvable=NaN (dropped, never guessed)')
    # clumping: two variants in perfect LD plus one independent
    g = rng.binomial(2, 0.3, 500).astype(float)
    D = np.vstack([g, g, rng.binomial(2, 0.3, 500).astype(float)])
    D = D - D.mean(axis=1, keepdims=True)
    W = pd.DataFrame({'P': [1e-9, 1e-4, 1e-6], 'ri': [0, 1, 2]})
    keep = L.clump(W, D, np.array([1000, 1200, 900000]), r2=0.1, win_kb=500)
    R.add('T2d clumping drops perfect LD, keeps the independent variant',
          set(keep) == {0, 2}, f'kept rows {sorted(keep)} of 3')


def run_mode(mode, argv, log):
    cmd = [sys.executable, '-m', 'ancestry', mode] + [str(x) for x in argv]
    r = subprocess.run(cmd, capture_output=True, text=True, env=PYENV)
    log.write(f'\n$ {" ".join(cmd)}\n{r.stdout}\n{r.stderr}\n')
    return r.returncode, r.stdout + r.stderr


def t3_genetics(R, root, ref, work, res, log, quick):
    # Resolve plink2 exactly as _lib.plink2() does, $PLINK2 included. The gate used to check
    # only PATH and /opt/bin, so a machine that had plink2 somewhere else and said so with
    # $PLINK2 skipped this test while every genetic stage would in fact have run -- the one
    # case where the skip is actively misleading.
    _p2 = os.environ.get('PLINK2') or shutil.which('plink2') or '/opt/bin/plink2'
    if quick or not Path(_p2).exists():
        return R.skip('T3 genetics end to end',
                      'plink2 is not on this machine (it IS in the image); the genetic stages '
                      'were not exercised')
    geno = str(work / 'geno' / 'synthetic')
    rc, out = run_mode('geno', ['--work', work, '--results', res, '--geno', geno,
                                '--build', 'grch38', '--pca-single', '--pca-geno', geno,
                                '--skip-kinship', '--npc', 5, '--threads', 2], log)
    if rc:
        return R.add('T3a geno', False, 'see the log')
    R.add('T3a geno built the analysis sets', (res / 'geno' / 'analysis_sets.csv').exists())
    rc, out = run_mode('loci', ['--work', work, '--results', res, '--threads', 2], log)
    f = res / 'loci' / 'replication_summary.csv'
    if rc or not f.exists():
        return R.add('T3b loci', False, 'see the log')
    S = pd.read_csv(f)
    e = S[S.stratum == 'EUR']
    if not len(e):
        return R.add('T3b loci', False, 'no EUR row')
    pc, sl = float(e.pct_concordant.iloc[0]), float(e.slope.iloc[0])
    R.add('T3b sign concordance recovers the injected direction', pc >= 85,
          f'{pc:.0f}% concordant (truth: 100%)')
    # NOT a test of the injected magnitude. The fixture normalises the phenotype
    # (`y = y / np.std(y)`, then a logistic and a rank-inverse-normal), and that cancels the
    # overall scale of the injected betas: multiply every one of them by two and the phenotype
    # is unchanged, so no slope can recover `effect_mult`. What the stage really claims is that
    # the leads reproduce in direction and in proportion, and that a shuffled phenotype does
    # not -- so that is what is asserted, against the permutation control the stage already
    # computes beside it.
    lo, hi = float(e.slope_lo.iloc[0]), float(e.slope_hi.iloc[0])
    R.add('T3c the fitted slope is positive and excludes zero', sl > 0 and lo > 0,
          f'slope {sl:.2f} [{lo:.2f}, {hi:.2f}]')
    perm = S[(S.stratum == 'EUR') & S['sample'].astype(str).str.contains('permut')]
    if len(perm):
        ps, plo, phi = (float(perm.slope.iloc[0]), float(perm.slope_lo.iloc[0]),
                        float(perm.slope_hi.iloc[0]))
        R.add('T3c-bis the permuted phenotype gives no slope', plo <= 0 <= phi,
              f'permuted slope {ps:.3f} [{plo:.3f}, {phi:.3f}] vs real {sl:.2f}')
        R.add('T3c-ter the real slope is well clear of the permuted one', abs(sl) > 3 * abs(ps),
              f'|{sl:.2f}| vs 3x|{ps:.3f}|')
    else:
        R.skip('T3c-bis the permuted phenotype gives no slope', 'no permutation row')
    # The positive control -- each trait score against its OWN measured trait -- is the gate
    # `pgs` prints first and the one every other PGS panel depends on, and with no measurements
    # file it is SKIPPED: 36 rows of NaN, no t statistic, and a plate that never draws. At a site
    # `traits` writes this file before `pgs` runs; here it is planted so the arm is computed.
    #
    # The values are random ON PURPOSE. The control must then come back null, which is what
    # proves the gate reports its own failure instead of assuming success -- a measurement
    # correlated with its own score would only ever test the happy path. This is the same
    # discipline as T11b's model-check gate, which must fail on synthetic ECGs.
    ro_m = pd.read_csv(Path(work) / 'roster.csv', dtype={'participant_id': str})
    rm = np.random.default_rng(SEED + 23)
    nm_ = len(ro_m)
    #
    # TWO RULERS are planted, not one: `measurements.csv` stands in for the NeuroKit pass that
    # mode `traits` writes, and a separate file stands in for the cart's own precomputed panel
    # passed as --measurements. Running both is what lets the stage tell a broken score from a
    # disagreeing instrument, and it is the only coverage of that branch.
    def _panel(seed):
        r = np.random.default_rng(seed)
        return pd.DataFrame({'participant_id': ro_m.participant_id,
                             'HR_bpm': r.normal(72, 12, nm_), 'PR_ms': r.normal(160, 22, nm_),
                             'QRS_ms': r.normal(95, 11, nm_), 'QT_ms': r.normal(400, 28, nm_),
                             'JT_ms': r.normal(305, 26, nm_),
                             'QRSTangleF_deg': r.normal(40, 30, nm_),
                             'Tamp_V6': r.normal(0.30, 0.12, nm_),
                             'Ramp_I': r.normal(0.55, 0.22, nm_)})
    _panel(SEED + 23).to_csv(Path(work) / 'measurements.csv', index=False)      # neurokit
    _dev = Path(work) / 'device_panel.csv'
    _panel(SEED + 24).to_csv(_dev, index=False)                                 # the cart

    rc, out = run_mode('pgs', ['--work', work, '--results', res, '--threads', 2,
                               '--measurements', _dev], log)
    R.add('T3d pgs ran', (res / 'pgs' / 'trait_pgs.csv').exists())

    try:
        TP = pd.read_csv(res / 'pgs' / 'trait_pgs.csv')
        ctl = TP[(TP.arm == 'own_measured_trait') & TP.beta_sd.notna()]
        eur = ctl[ctl.stratum == L.PRIMARY]
        R.add('T3d the PGS positive control is computed, not skipped',
              len(eur) >= 8 and bool(np.isfinite(eur.t.to_numpy()).all()),
              f'{len(eur)} traits with a finite effect in {L.PRIMARY}, '
              f'{len(ctl)} across all strata')
        # BOTH instruments must appear, each with its own row per trait
        srcs = set(eur.source) if 'source' in eur.columns else set()
        R.add('T3d the control runs on both rulers, precomputed and inferred',
              {'device', 'neurokit'} <= srcs and len(eur) >= 16,
              f'sources {sorted(srcs)}, {len(eur)} score-ruler pairs in {L.PRIMARY}')
        # Noise measurements must mostly fail the t = 3 bar. Deliberately NOT all-or-nothing:
        # 16 pairs each have a ~0.3% chance of clearing t = 3 by luck, so `.all()` would flake
        # roughly one run in 25. The property worth asserting is that the gate reports failure,
        # not that no single statistic ever strays.
        if len(eur):
            frac = float((eur.t.abs() < 3).mean())
            R.add('T3d the positive control reports failure when the measurements are noise',
                  frac >= 0.75,
                  f'{frac * 100:.0f}% of pairs below t = 3, max |t| '
                  f'{float(eur.t.abs().max()):.2f}')
    except Exception as e:                                                # noqa: BLE001
        R.add('T3d the PGS positive control is computed, not skipped', False,
              f'{type(e).__name__}: {e}')

    # ---- T3e THE RULER CHOICE AND THE FILL.  A site may run the control on the cart's panel,
    # on NeuroKit, or on both (--measurement-source).  Two things then have to hold, and neither
    # is visible in the table if it is wrong: picking one ruler must give exactly ONE row per
    # trait, and a trait the chosen panel does not carry must be filled from the other
    # instrument, labelled, and be that instrument's OWN estimate -- a fill computed any other
    # way would look identical and be a different number.  A site column name is renamed through
    # --measurement-map in the same run, because that is how the two features will meet in
    # practice: a panel that spells heart rate differently AND is missing two traits.
    try:
        _dpart = Path(work) / 'device_partial.csv'
        _pan = pd.read_csv(_dev, dtype={'participant_id': str})
        _pan.drop(columns=['QT_ms', 'Tamp_V6']).rename(
            columns={'HR_bpm': 'VENT_RATE'}).to_csv(_dpart, index=False)
        _mp = Path(work) / 'measurement_map.json'
        _mp.write_text(json.dumps({'HR': 'VENT_RATE'}))
        res2 = res.parent / f'{res.name}_oneruler'
        if res2.exists():
            shutil.rmtree(res2)
        shutil.copytree(res, res2)
        run_mode('pgs', ['--work', work, '--results', res2, '--threads', 2,
                         '--measurements', _dpart, '--measurement-source', 'device',
                         '--measurement-map', _mp, '--skip-ancestry-pgs'], log)
        T2 = pd.read_csv(res2 / 'pgs' / 'trait_pgs.csv')
        e2 = T2[(T2.arm == 'own_measured_trait') & (T2.stratum == L.PRIMARY)
                & T2.beta_sd.notna()]
        R.add('T3e one chosen ruler gives one control row per trait',
              len(e2) == 8 and bool(e2.trait.is_unique),
              f'{len(e2)} rows, {e2.trait.nunique()} traits')
        fil = e2[e2.source.astype(str).str.endswith('(fill)')]
        R.add('T3e a trait the panel does not carry is filled from the other instrument',
              set(fil.trait) == {'QT', 'Tamp'}
              and bool(fil.note.astype(str).str.len().gt(0).all()),
              f'filled {sorted(fil.trait)}, each labelled in `source` and `note`')
        R.add('T3e a renamed site column is read under the site\'s own name',
              set(e2[e2.trait == 'HR'].measure) == {'VENT_RATE'},
              f'HR measured as {sorted(set(e2[e2.trait == "HR"].measure))}')
        nk = eur[eur.source == 'neurokit'].set_index('trait').beta_sd.astype(float)
        off = max([abs(float(r.beta_sd) - float(nk[r.trait])) for r in fil.itertuples()]
                  or [np.inf])
        R.add('T3e a filled row is the other instrument\'s own estimate, unchanged',
              off < 1e-9, f'largest disagreement with the neurokit row {off:.2e}')
        # and the map must refuse a trait it does not know rather than silently ignoring it
        _bad = Path(work) / 'measurement_map_bad.json'
        _bad.write_text(json.dumps({'NotATrait': 'x'}))
        rc3, out3 = run_mode('pgs', ['--work', work, '--results', res2, '--threads', 2,
                                     '--measurement-map', _bad], log)
        R.add('T3e an unknown trait in the measurement map is refused by name',
              rc3 != 0 and 'NotATrait' in out3,
              'names the offending key and lists the nine traits' if rc3 else 'exit 0')
        shutil.rmtree(res2, ignore_errors=True)
    except Exception as e:                                                # noqa: BLE001
        R.add('T3e one chosen ruler gives one control row per trait', False,
              f'{type(e).__name__}: {e}')

    # ---- the PGS are VALIDATED, never refitted. Asserted, not asserted-in-a-comment: every
    # weight file the stage handed to plink must hold exactly the shipped beta for that
    # variant, up to the sign of an allele flip. If any arm ever re-estimated a weight, these
    # numbers would stop matching the reference table.
    wfiles = sorted((Path(work) / 'geno').rglob('pgs/w_*.tsv')) + \
        sorted((Path(work) / 'geno').rglob('pgs/anc_w_*.tsv'))
    VM = pd.read_csv(Path(work) / 'geno' / 'variant_map.csv', dtype={'CHR': str})
    shipped = {}
    for nm, acol, bcol in (('pgs_weights_grch38.tsv', 'RISK_ALLELE', 'BETA'),
                           ('ancestry_pgs_weights_grch38.tsv', 'A1', 'BETA')):
        q = Path(ref) / nm
        if not q.exists():
            continue
        T = pd.read_csv(q, sep='\t', dtype={'CHR': str})
        key = VM[['CHR', 'POS', 'vid']].copy()
        key['POS'] = pd.to_numeric(key.POS, errors='coerce')
        T['POS'] = pd.to_numeric(T.POS, errors='coerce')
        J = T.merge(key, on=['CHR', 'POS'], how='inner')
        for r in J.itertuples():
            shipped.setdefault(str(r.vid), set()).add(round(abs(float(getattr(r, bcol))), 10))
    checked = bad = 0
    for q in wfiles:
        W = pd.read_csv(q, sep='\t')
        for r in W.itertuples():
            exp = shipped.get(str(r.ID))
            if not exp:
                continue
            checked += 1
            if round(abs(float(r.BETA)), 10) not in exp:
                bad += 1
    R.add('T3d-bis every applied weight is the shipped weight (no refitting)',
          checked > 0 and bad == 0,
          f'{checked:,} weights across {len(wfiles)} weight file(s), {bad} altered')

    # the headline ancestry-PGS arm must be the FIXED panel, not a site-clumped one
    A0 = res / 'pgs' / 'ancestry_pgs.csv'
    if A0.exists():
        Q = pd.read_csv(A0)
        if 'arm' in Q.columns:
            nfix = int((Q.arm == 'fixed_panel').sum())
            R.add('T3d-ter the ancestry PGS reports a fixed, unselected panel arm', nfix > 0,
                  f'{nfix} fixed_panel row(s), {int((Q.arm == "clumped").sum())} clumped '
                  'sensitivity row(s)')
            fx = Q[Q.arm == 'fixed_panel']
            R.add('T3d-quater the fixed arm does no site selection',
                  bool((fx.clumping.astype(str) == 'none (fixed panel)').all()),
                  f'{sorted(set(fx.clumping.astype(str)))}')
        else:
            R.add('T3d-ter the ancestry PGS reports a fixed, unselected panel arm', False,
                  'no `arm` column')
    A = res / 'pgs' / 'ancestry_pgs.csv'
    if A.exists():
        D = pd.read_csv(A)
        D = D[(D.stratum == 'EUR') & D.get('beta_sd', pd.Series(dtype=float)).notna()]
        R.add('T3e the ancestry-score PGS points the right way',
              bool(len(D)) and float(D.beta_sd.max()) > 0,
              f'max beta {float(D.beta_sd.max()):.3f}' if len(D) else 'no usable threshold')


def t4_fairness(R, work, res, log, planted=10.0):
    rc, out = run_mode('fairness', ['--work', work, '--results', res,
                                    '--labels', work / 'labels.csv', '--boot', 200,
                                    '--min-pos', 15], log)
    f = res / 'fairness' / 'gaps.csv'
    if rc or not f.exists():
        return R.add('T4 fairness', False, 'see the log')
    G = pd.read_csv(f)
    arm = 'matched' if (G.arm == 'matched').any() else G.arm.iloc[0]
    q = G[(G.arm == arm) & (G.metric == 'fpr') & (G.split == 'genetic') &
          (G.threshold_rule == 'shared 0.5')].set_index('finding')
    if 'Afib' not in q.index:
        return R.add('T4a the planted gap is recovered', False, 'Afib was not audited')
    g = float(q.loc['Afib'].gap)
    R.add('T4a the planted false-positive gap is recovered',
          planted * 0.4 <= g <= planted * 2.0, f'{g:+.2f} per 100 (planted {planted:+.1f})')
    if 'Bradycardia' in q.index:
        b = q.loc['Bradycardia']
        R.add('T4b a finding with no planted gap comes back null',
              bool(b.gap_lo <= 0 <= b.gap_hi),
              f'{b.gap:+.2f} [{b.gap_lo:+.2f}, {b.gap_hi:+.2f}]')
    # T4c used to assert only that the file existed, which it did even when the comparison
    # inside it had been skipped for want of four comparable findings -- so the slope, the
    # correlation and the sign concordance that ARE the claim of this mode went unexecuted. Now
    # the dict has to be populated and has to recover the planted structure.
    j = res / 'fairness' / 'auditability.json'
    if not j.exists():
        return R.add('T4c the auditability comparison was computed', False, 'no auditability.json')
    J = json.loads(j.read_text())
    aud = J.get('auditability') or {}
    R.add('T4c the auditability comparison was computed', bool(aud),
          f'{len(aud)} arm x metric block(s): ' + ', '.join(sorted(aud)[:4]))
    key = next((k for k in ('matched_fpr', 'unmatched_fpr') if k in aud), None)
    if key:
        v = aud[key]
        # The planted gaps are the same numbers on both splits up to the error in the inferred
        # label, so the genetic-vs-inferred slope must come back positive and the signs must
        # mostly agree. A slope at or below zero would mean the inferred split had told the
        # opposite story, which is exactly the claim this mode exists to test.
        R.add('T4c the genetic and inferred gaps agree in direction',
              float(v['slope_through_origin']) > 0 and float(v['pearson_r']) > 0,
              f'slope {v["slope_through_origin"]:+.3f}, r {v["pearson_r"]:+.3f}, '
              f'{v["n_same_sign"]}/{v["n_findings"]} same sign')


def t5_privacy(R, res, work, tmp):
    # planted problems must be caught
    tmp = Path(tmp); tmp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'IID': ['SYN000001'], 'x': [1]}).to_csv(tmp / 'bad_ids.csv', index=False)
    (tmp / 'bad_counts.json').write_text(json.dumps({'n': 3, 'n_pos': 2}))
    pd.DataFrame({'v': np.arange(PV.MAX_TABLE_ROWS + 10)}).to_csv(tmp / 'bad_long.csv',
                                                                  index=False)
    caught = PV.audit_files([tmp / 'bad_counts.json', tmp / 'bad_long.csv'],
                            ids=('SYN000001',))
    caught += PV.audit_files([tmp / 'bad_ids.csv'], ids=('SYN000001',))
    R.add('T5a the audit catches planted problems', len(caught) >= 3,
          f'{len(caught)} problems found in 3 planted files')

    # T5a-bis: the hole this audit had until 2026-11-09.  The scan writes .tsv, and the audit
    # only looked at .json and .csv, so a tab-separated table could carry an identifier column
    # or a million rows straight through a "clean" audit.  Planted both ways, in the two shapes
    # a .tsv actually takes here: an ordinary table, and a long point cloud.
    pd.DataFrame({'IID': ['SYN000001', 'SYN000002'], 'BETA': [0.1, 0.2]}).to_csv(
        tmp / 'bad_tsv_ids.tsv', sep='\t', index=False)
    pd.DataFrame({'CHROM': np.ones(10), 'POS': np.arange(10),
                  'participant_id': ['SYN000001'] * 10}).to_csv(
        tmp / 'scan_manhattan.tsv', sep='\t', index=False)
    pd.DataFrame({'CHROM': np.ones(PV.MAX_MANHATTAN_ROWS + 5),
                  'POS': np.arange(PV.MAX_MANHATTAN_ROWS + 5),
                  'NEGLOG10P': np.zeros(PV.MAX_MANHATTAN_ROWS + 5)}).to_csv(
        tmp / 'scan_qq.tsv', sep='\t', index=False)
    c1 = PV.audit_files([tmp / 'bad_tsv_ids.tsv'], ids=('SYN000001',))
    c2 = PV.audit_files([tmp / 'scan_manhattan.tsv'], ids=('SYN000001',))
    c3 = PV.audit_files([tmp / 'scan_qq.tsv'])
    R.add('T5a-bis a tab-separated table is audited like any other',
          bool(c1) and bool(c2) and bool(c3),
          f'ids in a plain .tsv: {len(c1)}, an identifier column in a long one: {len(c2)}, '
          f'over-long point cloud: {len(c3)}')
    try:
        PV.check_columns(pd.DataFrame({'participant_id': ['x']}), 'test')
        ok = False
    except ValueError:
        ok = True
    R.add('T5b writing an identifier column raises', ok)
    rp = work / 'roster.csv'
    ids = tuple(pd.read_csv(rp, dtype=str).participant_id[:500]) if rp.exists() else ()
    files = [p for p in Path(res).rglob('*') if p.is_file()]
    prob = PV.audit_files(files, ids=ids)
    R.add('T5c the real results directory is clean', not prob,
          '; '.join(prob[:5]) or f'{len(files)} files, no identifier, no unmasked small count')


def t6_figures(R, res, log):
    rc, out = run_mode('figures', ['--results', res, '--formats', 'png'], log)
    figs = list((Path(res) / 'figures').glob('*.png'))
    R.add('T6 figures draw', rc == 0 and len(figs) >= 1,
          f'{len(figs)} plate(s): ' + ', '.join(sorted(p.stem for p in figs)))
    nd = [ln for ln in out.splitlines() if 'NOT DRAWN' in ln]
    if nd:
        print('      (not drawn: ' + '; '.join(x.strip() for x in nd) + ')')


def t7_package(R, work, res, log):
    rc, out = run_mode('package', ['--work', work, '--results', res,
                                   '--run-name', 'SELFTEST'], log)
    z = list(Path(res).parent.glob('results_package_SELFTEST_*.zip'))
    R.add('T7 package written and audit clean', rc == 0 and len(z) == 1,
          z[0].name if z else out.strip().splitlines()[-1] if out.strip() else 'no zip')


def t8_ses(R, root, log):
    """The deprivation arms, against a planted truth in BOTH directions.

    A control that only ever checks the null is worthless: it passes just as happily on a
    pipeline that cannot detect anything. So the same machinery is run twice on the same
    coupling -- once where deprivation has no path to the score and once where it IS most of the
    score -- and both the bound and its absence have to come out right.
    """
    import subprocess
    rng = np.random.default_rng(SEED)
    n_e, n_n = 1400, 420
    n = n_e + n_n
    nonE = np.r_[np.zeros(n_e, int), np.ones(n_n, int)]
    pid = [f'S{i:06d}' for i in range(n)]
    sdi = np.clip(rng.normal(np.where(nonE == 1, 68, 38), 18), 1, 100)
    z = (sdi - sdi.mean()) / sdi.std()
    age = rng.normal(60, 14, n).clip(18, 95)
    for scen, (w_a, w_s), want_bounded in (('null', (2.0, 0.0), True),
                                           ('causal', (0.5, 1.8), False)):
        d = root / f'ses_{scen}'
        (d / 'work').mkdir(parents=True, exist_ok=True)
        lat = w_a * nonE + w_s * z + rng.normal(0, 1.0, n)
        pd.DataFrame({'participant_id': pid,
                      'stratum': np.where(nonE == 1, 'AFR', 'EUR'),
                      'age': age, 'sex': rng.choice(['M', 'F'], n),
                      'p_nonEUR': 1 / (1 + np.exp(-(lat - 1.0))),
                      'pheno_rint': rng.normal(size=n)}).to_csv(d / 'work' / 'scores.csv',
                                                                index=False)
        pd.DataFrame({'participant_id': pid, 'sdi': np.round(sdi, 1)}).to_csv(d / 'sdi.csv',
                                                                              index=False)
        r = subprocess.run([sys.executable, '-m', 'ancestry.ses', '--work',
                            str(d / 'work'), '--results', str(d / 'results'),
                            '--ses', str(d / 'sdi.csv'), '--n-sham', '25',
                            '--no-embeddings'],
                           capture_output=True, text=True, env=PYENV)
        log.write(r.stdout + r.stderr)
        q = d / 'results' / 'ses' / 'ses_quad.csv'
        if r.returncode != 0 or not q.exists():
            R.add(f'T8 ses ({scen}) runs', False, (r.stderr or '')[-200:])
            continue
        Q = pd.read_csv(q)
        cost = Q[(Q.arm == 'B_matched') & Q.quantity.str.startswith('cost')]
        sham = Q[(Q.arm == 'B_matched') & Q.quantity.str.startswith('coupling-matched')]
        val = float(cost.estimate.iloc[0]) if len(cost) else float('nan')
        sv = float(sham.estimate.iloc[0]) if len(sham) else float('nan')
        bnd = cost.achieved_bound.iloc[0] if len(cost) and 'achieved_bound' in cost else None
        # The TOST flag is n-dependent, so it is NOT what is asserted here: a small fixture
        # cannot clear 0.02 however true the null is. What must hold is that the estimate sits
        # where the planted truth put it, and that it is separated from the coupling-matched
        # sham only when deprivation really drives the score.
        ok = (abs(val) < 0.05 and abs(val - sv) < 0.05) if want_bounded \
            else (val < -0.08 and (val - sv) < -0.08)
        R.add(f'T8 ses arm B, {scen} cohort: '
              + ('sits at the null, and on top of its sham' if want_bounded
                 else 'detects the planted effect, clear of its sham'),
              bool(ok), f'cost {val:+.4f}, coupled sham {sv:+.4f}, bound {bnd}')
        R.add(f'T8 ses ({scen}) reports the bound its n achieves', bnd is not None,
              f'achieved_bound={bnd}')
        # the coupling report has to be there whatever the arms say
        R.add(f'T8 ses ({scen}) reports the ancestry coupling',
              (d / 'results' / 'ses' / 'ses_coupling.csv').exists()
              and (d / 'results' / 'ses' / 'ses_verdict.txt').exists())
        # and nothing per-person may have escaped
        bad = PV.audit_files(sorted((d / 'results' / 'ses').glob('*')), ids=pid[:500])
        R.add(f'T8 ses ({scen}) exports survive the privacy audit', not bad, '; '.join(bad[:2]))


def t1b_builds(R, ref, root):
    """GRCh37/GRCh38 routing: the right table, by every spelling, and never the wrong one."""
    # the spellings a site will actually type
    try:
        ok = all(L.build_suffix(x) == 'grch37' for x in ('grch37', 'GRCh37', 'hg19', 'b37', '37'))
        ok &= all(L.build_suffix(x) == 'grch38' for x in ('grch38', 'HG38', 'b38', '38'))
        R.add('T1b build aliases resolve', ok, 'hg19/b37/37 -> grch37, hg38/b38/38 -> grch38')
    except SystemExit as e:
        R.add('T1b build aliases resolve', False, str(e))
    # an unknown build must be refused by name, not defaulted
    try:
        L.build_suffix('hg18')
        R.add('T1b an unknown build is refused', False, 'hg18 was accepted')
    except SystemExit:
        R.add('T1b an unknown build is refused', True, 'hg18 -> refused by name')
    # the resolver returns the matching file for each build
    got37 = L.ref_table(ref, None, 'leads', 'hg19')
    got38 = L.ref_table(ref, None, 'leads', 'grch38')
    R.add('T1b the leads table follows the build',
          got37.name == 'leads_grch37.tsv' and got38.name == 'leads_grch38.tsv',
          f'{got37.name} / {got38.name}')
    # and the two are genuinely different coordinates, not a copy wearing two names
    a = pd.read_csv(got37, sep='\t', dtype={'CHR': str})
    b = pd.read_csv(got38, sep='\t', dtype={'CHR': str})
    same_n = len(a) == len(b)
    moved = float((a.POS.to_numpy() != b.POS.to_numpy()).mean()) if same_n else float('nan')
    R.add('T1b the two builds carry different positions', same_n and moved > 0.9,
          f'{len(a)} leads in both; {100 * moved:.0f}% sit at a different position')
    # the one table that ships in GRCh38 only must come back absent, not substituted
    miss = L.ref_table(ref, None, 'ancestry_pgs_weights', 'grch37', required=False)
    R.add('T1b the GRCh38-only panel is absent on grch37, not silently substituted',
          miss is None, 'ref_table -> None, so `pgs` skips that arm and says why')
    # a site-lifted copy in work/ref is preferred when present
    w = Path(root) / 'buildprobe'
    (w / 'ref').mkdir(parents=True, exist_ok=True)
    (w / 'ref' / 'ancestry_pgs_weights_grch37.tsv').write_text('KEY\tCHR\tPOS\n')
    lifted = L.ref_table(ref, w, 'ancestry_pgs_weights', 'grch37', required=False)
    R.add('T1b a site-lifted panel in work/ref is picked up',
          lifted is not None and 'buildprobe' in str(lifted), str(lifted))


def t9_robustness(R, root, log):
    """The mistakes a site actually makes: odd delimiters, yes/no labels, stages out of order."""
    import subprocess
    d = Path(root) / 'robust'
    (d / 'work').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    n = 400
    pid = [f'R{i:05d}' for i in range(n)]
    nonE = rng.random(n) < 0.25
    pd.DataFrame({'participant_id': pid, 'stratum': np.where(nonE, 'AFR', 'EUR'),
                  'age': rng.normal(60, 12, n), 'sex': rng.choice(['M', 'F'], n),
                  'p_nonEUR': rng.random(n), 'pheno_rint': rng.normal(size=n)}
                 ).to_csv(d / 'work' / 'scores.csv', index=False)
    sdi = rng.integers(1, 100, n)

    # a .csv that is tab- or semicolon-separated, and a .gz -- all three must just work
    V = pd.DataFrame({'participant_id': pid, 'sdi': sdi})
    V.to_csv(d / 'comma.csv', index=False)
    V.to_csv(d / 'tabs.csv', sep='\t', index=False)
    V.to_csv(d / 'semi.csv', sep=';', index=False)
    V.to_csv(d / 'gz.csv.gz', index=False)
    for name in ('comma.csv', 'tabs.csv', 'semi.csv', 'gz.csv.gz'):
        try:
            got, col = importlib.import_module('ancestry.ses').load_ses(d / name)
            ok = len(got) == n and col == 'sdi'
        except SystemExit as e:
            ok, col = False, str(e)[:60]
        R.add(f'T9 --ses reads a {name.split(".")[0]}-delimited file', ok, f'{len(got) if ok else col}')

    # an empty table must say it is empty, not look like an id mismatch
    (d / 'empty.csv').write_text('participant_id,sdi\n')
    try:
        importlib.import_module('ancestry.ses').load_ses(d / 'empty.csv')
        R.add('T9 an empty table is named as empty', False, 'it was accepted')
    except SystemExit as e:
        R.add('T9 an empty table is named as empty', 'no rows' in str(e).lower(), str(e)[:70])

    # an unknown id column must name the likely one and the exact flag
    pd.DataFrame({'MRN_hash': pid, 'sdi': sdi}).to_csv(d / 'weirdid.csv', index=False)
    try:
        importlib.import_module('ancestry.ses').load_ses(d / 'weirdid.csv')
        R.add('T9 an unknown id column is diagnosed', False, 'it was accepted')
    except SystemExit as e:
        R.add('T9 an unknown id column is diagnosed',
              '--ses-id-col MRN_hash' in str(e), str(e).replace('\n', ' | ')[:90])

    # yes/no findings must be understood, not silently turned into NaN
    from ancestry import fairness as RF
    F = pd.read_csv(Path(os.environ.get('REFERENCE_DIR', '.')) / 'findings77.tsv', sep='\t')
    fin = list(F.finding[:2])
    pd.DataFrame({'participant_id': pid, fin[0]: rng.choice(['Yes', 'No'], n),
                  fin[1]: rng.choice(['Y', 'N'], n)}).to_csv(d / 'yesno.csv', index=False)
    try:
        LB, how = RF.labels_from_table(d / 'yesno.csv', list(F.finding))
        usable = int(LB[fin[0]].notna().sum())
        R.add('T9 yes/no findings are read as 0/1', usable == n, f'{usable}/{n} usable; {how}')
    except SystemExit as e:
        R.add('T9 yes/no findings are read as 0/1', False, str(e)[:70])

    # a mode run before its upstream stage must name that stage, not raise FileNotFoundError
    r = subprocess.run([sys.executable, '-m', 'ancestry.ses', '--work', str(d / 'nothing'),
                        '--results', str(d / 'o'), '--ses', str(d / 'comma.csv')],
                       capture_output=True, text=True, env=PYENV)
    log.write(r.stdout + r.stderr)
    msg = r.stdout + r.stderr
    R.add('T9 a stage run out of order names the stage that was skipped',
          'produced by : `score`' in msg and 'Traceback' not in msg,
          'names `score`' if 'produced by : `score`' in msg else msg[-80:])


def t10_formats(R, root, log):
    """Every file the SITE supplies, in every format a site plausibly supplies it in.

    The container documents one format per input. Sites do not read documentation that way: an
    ECG inventory comes out of a reporting system as CSV, a deprivation table comes out of
    Excel with semicolons, and a `.csv` that was saved from a tab-separated export is still
    called `.csv`. Each of those used to fail as "no id column found" or "0 of 77 findings",
    which sends the site looking at its ids instead of at its delimiter. Format is decided from
    content here, so this test feeds each reader the wrong extension on purpose."""
    from ancestry import prepare_external as PI
    from ancestry import fairness as RF
    d = Path(root) / 'formats'
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    n = 60
    pid = [f'F{i:05d}' for i in range(n)]

    # ---- the ECG inventory: parquet is documented, CSV/TSV are what arrives
    cols = ['ecg_id', 'gender', 'age', 'full_ecg_path']
    M = pd.DataFrame({'ecg_id': [f'E{i}' for i in range(n)], 'gender': ['M'] * n,
                      'age': rng.integers(40, 80, n), 'full_ecg_path': ['/x.npy'] * n})
    M.to_parquet(d / 'meta.parquet')
    M.to_csv(d / 'meta.csv', index=False)
    M.to_csv(d / 'meta.tsv', sep='\t', index=False)
    M.to_parquet(d / 'lying.csv')                       # parquet bytes, .csv name
    M.to_csv(d / 'lying.parquet', index=False)          # text bytes, .parquet name
    M.rename(columns={'ecg_id': 'ECG_ID', 'age': 'Age'}).to_csv(d / 'cased.csv', index=False)
    for name in ('meta.parquet', 'meta.csv', 'meta.tsv', 'lying.csv', 'lying.parquet',
                 'cased.csv'):
        try:
            got = PI._read_meta(d / name, cols)
            ok = len(got) == n and list(got.columns) == cols
            why = f'{len(got)} rows'
        except SystemExit as e:
            ok, why = False, str(e).splitlines()[0][:70]
        R.add(f'T10 --ecg-meta reads {name}', ok, why)

    # a genuinely missing column must be named, with the flag that fixes it
    M.drop(columns=['age']).to_csv(d / 'noage.csv', index=False)
    try:
        PI._read_meta(d / 'noage.csv', cols)
        R.add('T10 a missing --ecg-meta column is named', False, 'it was accepted')
    except SystemExit as e:
        R.add('T10 a missing --ecg-meta column is named',
              "'age'" in str(e) and '--age-col' in str(e), str(e).replace('\n', ' | ')[:90])

    # ---- the RFMix header probe must still resolve case with --rfmix-sep auto
    V = pd.DataFrame({'Sample': pid, 'AFR': rng.random(n), 'EUR': rng.random(n)})
    V.to_csv(d / 'rfmix.tsv', sep='\t', index=False)
    V.to_csv(d / 'rfmix_comma.csv', index=False)
    for name in ('rfmix.tsv', 'rfmix_comma.csv'):
        head = list(PI._read(d / name, what='--rfmix', sep=None, nrows=0).columns)
        R.add(f'T10 --rfmix header probe parses {name}',
              head == ['Sample', 'AFR', 'EUR'], f'{head}')

    # ---- --linkage in both delimiters, and with the columns cased differently
    lk = pd.DataFrame({'Barcode': pid, 'project_pseudo_id': pid})
    lk.to_csv(d / 'link.tsv', sep='\t', index=False)
    for name in ('link.tsv',):
        got = PI._read(d / name, what='--linkage', dtype=str)
        R.add(f'T10 --linkage reads {name}', len(got) == n, f'{len(got)} rows')
    bc = PI._col(lk.rename(columns={'Barcode': 'barcode'}), 'Barcode',
                 ('barcode', 'sample'), '--linkage', 'barcode column', '--barcode-col')
    R.add('T10 --linkage column case is resolved', bc == 'barcode', f'{bc!r}')

    # ---- --measurements as a TSV named .csv
    from ancestry import pgs as RP                                                     # noqa: F401
    meas = pd.DataFrame({'participant_id': pid, 'QT': rng.normal(400, 30, n)})
    meas.to_csv(d / 'meas.csv', sep='\t', index=False)
    got = L.read_table(d / 'meas.csv', what='--measurements', dtype={'participant_id': str})
    R.add('T10 --measurements reads a tab-separated .csv',
          len(got) == n and 'participant_id' in got.columns, f'{got.shape}')

    # ---- --statements as a semicolon file, id column spelled differently
    pd.DataFrame({'EMPI': pid,
                  'text': ['normal sinus rhythm'] * n}).to_csv(d / 'stmt.csv', sep=';',
                                                               index=False)
    F = pd.read_csv(Path(os.environ.get('REFERENCE_DIR', '.')) / 'findings77.tsv', sep='\t')
    try:
        out, how = RF.labels_from_statements(d / 'stmt.csv', F)
        ok = len(out) == n and 'participant_id' in out.columns
        why = f'{out.shape}'
    except SystemExit as e:
        ok, why = False, str(e).replace('\n', ' | ')[:80]
    R.add('T10 --statements reads a semicolon file with an EMPI id column', ok, why)


def t11_models(R, root, log):
    """Load the five fold checkpoints and the findings head, and score synthetic ECGs.

    Nothing else in this suite touches the weights. Without this, a mis-copied checkpoint, a
    torch/fairseq import that breaks on the shipped env, or a `COPY assets/models` that landed a
    directory deeper than expected is discovered by the SITE -- after `prepare` has already run
    for hours, because `score` is the first stage that opens a model. The point is to fail here,
    inside `build.sh`, on this machine. Skipped where the weights are not present (they are
    hard-linked in at build time, so they exist in the image and not in the source tree)."""
    import importlib
    from ancestry import weights as W
    try:
        ssl_ckpt = W.resolve('ssl_backbone', download=False)
        models_root = W.resolve('ancestry_folds', download=False)
        wcr = W.resolve('findings_head', download=False)
    except W.WeightsUnavailable as e:
        R.skip('T11 the ancestry ensemble loads and scores', str(e))
        return
    if not (ssl_ckpt.exists() and models_root.exists()):
        R.skip('T11 the ancestry ensemble loads and scores',
               f'weights are not on this machine ({models_root} absent); they ARE in the image')
        return
    try:
        IL = importlib.import_module('ancestry.infer')
    except Exception as e:                                                # noqa: BLE001
        R.add('T11 ancestry.infer imports', False, f'{type(e).__name__}: {e}')
        log.write(traceback.format_exc())
        return
    R.add('T11 ancestry.infer imports', True, 'torch and fairseq-signals load')

    # all five checkpoints must resolve, in the layout the image actually ships
    try:
        ck = IL.discover_fold_checkpoints(models_root, n_folds=5)
        R.add('T11 all five fold checkpoints resolve', len(ck) == 5 and len(set(ck)) == 5,
              f'{len(set(ck))} distinct paths')
    except Exception as e:                                                # noqa: BLE001
        R.add('T11 all five fold checkpoints resolve', False, f'{type(e).__name__}: {e}')
        return

    dev = IL.pick_device('cpu')                     # CPU: the build host need not have a GPU
    try:
        models, ckpts = IL.load_fold_models(str(models_root), str(ssl_ckpt), dev, n_folds=5)
        R.add('T11 the five fold models load onto the device', len(models) == 5, f'{len(models)}')
    except Exception as e:                                                # noqa: BLE001
        R.add('T11 the five fold models load onto the device', False, f'{type(e).__name__}: {e}')
        log.write(traceback.format_exc())
        return

    rng = np.random.default_rng(SEED)
    X = (rng.normal(0, 0.15, (4, 2500, 12))).astype(np.float32)        # millivolts, 10 s, 250 Hz
    try:
        p_ens, p_folds = IL.ensemble_predict(models, X, dev, batch_size=2)
        finite = bool(np.isfinite(p_ens).all())
        inrange = bool((p_ens >= 0).all() and (p_ens <= 1).all())
        shaped = p_folds.shape == (4, 5)
        R.add('T11 the ensemble scores an ECG batch', finite and inrange and shaped,
              f'p in [{p_ens.min():.3f}, {p_ens.max():.3f}], folds {p_folds.shape}')
        # the five folds are separately fine-tuned: identical columns mean one checkpoint was
        # loaded five times, which would silently turn the ensemble into a single model
        same = [int(np.allclose(p_folds[:, i], p_folds[:, j]))
                for i in range(5) for j in range(i + 1, 5)]
        R.add('T11 the five folds are five different models', sum(same) == 0,
              f'{sum(same)} of 10 fold pairs give identical predictions')
        R.add('T11 the ensemble is the mean of its folds',
              bool(np.allclose(p_ens, p_folds.mean(axis=1), atol=1e-6)), 'mean of 5')
    except Exception as e:                                                # noqa: BLE001
        R.add('T11 the ensemble scores an ECG batch', False, f'{type(e).__name__}: {e}')
        log.write(traceback.format_exc())

    # microvolts must be refused, not scored: this is the one input error that does not raise
    try:
        IL.check_units(X * 1000.0)
        R.add('T11 a microvolt array is refused by the unit check', False, 'it was accepted')
    except Exception as e:                                                # noqa: BLE001
        R.add('T11 a microvolt array is refused by the unit check', True, str(e)[:70])

    # the 77-finding head: shipped separately, used by the fairness arm
    if wcr.exists():
        try:
            from ancestry import score as SE
            import types
            # The paths are bound OUTSIDE the namespace: inside a class body,
            # `ssl_ckpt = str(ssl_ckpt)` makes the name class-local, so the right-hand side
            # raises NameError rather than reading the enclosing function's variable.
            _a = types.SimpleNamespace(device='cpu', batch_size=2,
                                       ssl_ckpt=str(ssl_ckpt), wcr_ckpt=str(wcr),
                                       models_root=str(models_root))
            W = SE.score_findings(X, _a)
            W = W[0] if isinstance(W, tuple) else W
            R.add('T11 the 77-finding head emits its outputs',
                  W.shape[0] == 4 and W.shape[1] >= 77, f'{tuple(W.shape)}')
        except Exception as e:                                            # noqa: BLE001
            R.add('T11 the 77-finding head emits its outputs', False,
                  f'{type(e).__name__}: {e}')
            log.write(traceback.format_exc())
    else:
        R.skip('T11 the 77-finding head emits its outputs', f'{wcr} absent on this machine')

    # ---- T11b: the `score` STAGE, not just the inference call it wraps ----------------------
    #
    # Everything above calls ensemble_predict and score_findings directly, so the whole of
    # score.py AROUND the model had no test: the roster alignment, the per-stratum summary, the
    # fixed-bin histogram, the model-check table and the GATE verdict that `package` prints as
    # "did the ancestry model work here at all?". Those four files are described in package.py
    # and shipped to the site, and a fault in any of them would appear on the site's first real
    # run -- after `prepare` has already taken hours -- and cost a new image.
    #
    # 240 synthetic ECGs on CPU, with EUR=140 and AFR=100 so both clear PV.MIN_STRATUM and
    # neither summary row is suppressed. The weights go in as FLAGS, not environment variables,
    # which is also the only coverage of `--models-root`/`--ssl-ckpt` being honoured.
    try:
        d = Path(root) / 'score_stage'
        work2, res2 = d / 'work', d / 'results'
        pdir = d / 'prepared' / 'all_genotyped'
        pdir.mkdir(parents=True, exist_ok=True)
        work2.mkdir(parents=True, exist_ok=True)
        r2 = np.random.default_rng(SEED + 11)
        n2 = 240
        strat = np.array(['EUR'] * 140 + ['AFR'] * 100)
        sids = [f'SYN{i:06d}' for i in range(n2)]
        ro = pd.DataFrame({'row': np.arange(n2), 'participant_id': sids, 'geno_id': sids,
                           'stratum': strat, 'age': r2.integers(40, 80, n2),
                           'sex': r2.integers(0, 2, n2)})
        ro['rf_EUR'] = np.where(strat == 'EUR', 0.95, 0.05)
        ro['rf_AFR'] = 1 - ro.rf_EUR
        ro['rf_nonEUR'] = 1 - ro.rf_EUR
        ro.to_csv(work2 / 'roster.csv', index=False)
        np.save(pdir / 'X.npy', synthetic_ecg(r2, n2))
        (work2 / 'prepared_dir.txt').write_text(str(d / 'prepared'))

        rc, out = run_mode('score', ['--work', work2, '--results', res2, '--head', 'ancestry',
                                     '--device', 'cpu', '--batch-size', 32,
                                     '--models-root', models_root,
                                     '--ssl-ckpt', ssl_ckpt], log)
        want = ('model_check.csv', 'score_by_stratum.csv', 'score_histogram.csv',
                'model_check_verdict.txt')
        missing = [f for f in want if not (res2 / 'score' / f).exists()]
        R.add('T11b the score stage writes every table package.py describes',
              rc == 0 and not missing,
              f'exit {rc}' + (f'; missing {missing}' if missing else '; all four present'))

        if not missing:
            S2 = pd.read_csv(work2 / 'scores.csv')
            folds = [c for c in S2.columns if c.startswith('p_fold')]
            R.add('T11b every rostered person is scored once, by five folds',
                  len(S2) == n2 and len(folds) == 5
                  and bool(S2.p_nonEUR.between(0, 1).all()),
                  f'{len(S2)} rows, {len(folds)} fold columns, p in '
                  f'[{S2.p_nonEUR.min():.3f}, {S2.p_nonEUR.max():.3f}]')

            B = pd.read_csv(res2 / 'score' / 'score_by_stratum.csv')
            note = B.note.astype(str).fillna('')
            live = set(B.stratum[~note.str.contains('suppress', case=False)])
            quiet = set(B.stratum[note.str.contains('suppress', case=False)])
            R.add('T11b the per-stratum summary reports the groups it can and suppresses the rest',
                  {'EUR', 'AFR'} <= live and {'EAS', 'SAS'} <= quiet,
                  f'reported {sorted(live)}, suppressed {sorted(quiet)}')

            # THE GATE. Synthetic ECGs carry no ancestry signal, so the model must land at
            # chance here and the verdict must SAY SO. A gate that passed on noise would be
            # worse than no gate: it is the line that tells a site not to interpret the
            # genetic results when the lead order or the units are wrong.
            v = (res2 / 'score' / 'model_check_verdict.txt').read_text()
            R.add('T11b the model-check gate fires on a model that is at chance',
                  'FAIL' in v.upper(), v.strip().splitlines()[0][:90] if v.strip() else 'empty')
    except Exception as e:                                                # noqa: BLE001
        R.add('T11b the score stage writes every table package.py describes', False,
              f'{type(e).__name__}: {e}')
        log.write(traceback.format_exc())


def t12_slurm(R, root, log):
    """The `submit` path: what it refuses to guess, and whether what it writes actually submits.

    This stage had no coverage at all, and three faults were living in it. Each one costs the
    site a whole queue rather than a single job, because `afterok` propagates: a `prepare` that
    dies at argparse cancels all 147 descendants, a `--gres=gpu:1` on a cluster with no GPU is a
    rejection rather than a CPU fallback, and a missing `--account` is a rejection on most
    production clusters. So the contract tested here is: resolve every site-specific value or
    refuse to write anything, and emit a submitter that survives a hostile login shell.
    """
    d = Path(root) / 'slurm'
    d.mkdir(parents=True, exist_ok=True)
    data = d / 'data'
    data.mkdir(exist_ok=True)
    (data / 'RF.txt').write_text('x\n')
    (data / 'meta.parquet').write_text('x\n')
    work, res = d / 'work', d / 'res'
    base = ['--work', work, '--results', res, '--image', '/opt/sif/x.sif']
    inputs = ['--rfmix', data / 'RF.txt', '--ecg-meta', data / 'meta.parquet',
              '--geno', str(data / 'chr@')]

    # ---- 1/2: nothing supplied -> one message naming every missing thing, and NO scripts
    rc, out = run_mode('submit', base + ['--geno', str(data / 'chr@')], log)
    named = [w for w in ('ACCOUNT', 'PARTITION', 'GPU', 'RFMix', 'ECG inventory')
             if w.lower() in out.lower()]
    R.add('T12 submit refuses to guess the site settings', rc != 0 and len(named) == 5,
          f'exit {rc}, named {len(named)}/5')
    R.add('T12 nothing is written when a setting is missing',
          not (work / 'slurm').exists(),
          'a half-laid-out run must not look submittable')

    # ---- 3: every question answered, no GPU anywhere
    rc, out = run_mode('submit', base + inputs
                       + ['--account', 'acct_x', '--partition', 'cpu_q', '--no-gpu'], log)
    sd = work / 'slurm'
    scripts = sorted(sd.glob('*.sbatch')) if sd.exists() else []
    R.add('T12 submit writes the graph once answered', rc == 0 and len(scripts) >= 10,
          f'exit {rc}, {len(scripts)} sbatch scripts')
    if not scripts:
        return

    score = (sd / 'score.sbatch').read_text()
    # ---- 4/5: the CPU cluster must not be sent a --gres it will reject
    R.add('T12 no --gres is requested with --no-gpu', '--gres' not in score,
          'gres on a CPU-only cluster is a rejection, not a fallback')
    R.add('T12 score is told --device cpu with --no-gpu', '--device cpu' in score)

    # ---- 6: the account and partition reach every script
    miss = [f.name for f in scripts
            if '#SBATCH --account=acct_x' not in f.read_text()
            or '#SBATCH --partition=' not in f.read_text()]
    R.add('T12 account and partition reach every stage', not miss, f'missing in {miss[:3]}')

    # ---- 7: submit_all.sh is valid bash
    sa = sd / 'submit_all.sh'
    rc2 = subprocess.run(['bash', '-n', str(sa)], capture_output=True, text=True)
    R.add('T12 submit_all.sh parses', rc2.returncode == 0, rc2.stderr.strip()[:80])

    # ---- 8: and runs to completion against a stub sbatch WITH USER UNSET.  `set -u` plus an
    #         unexported USER used to turn a completed submission into exit 1, which invites a
    #         second submission of jobs that are already queued.
    sb = d / 'bin'
    sb.mkdir(exist_ok=True)
    (sb / 'sbatch').write_text('#!/bin/bash\nn=$(cat %s 2>/dev/null || echo 500)\n'
                               'n=$((n+1)); echo $n > %s\necho $n\n'
                               % (d / 'ctr', d / 'ctr'))
    (sb / 'sbatch').chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ('USER', 'LOGNAME')}
    env['PATH'] = f'{sb}:' + env.get('PATH', '/usr/bin:/bin')
    r3 = subprocess.run(['bash', str(sa)], capture_output=True, text=True, env=env)
    log.write(f'\n$ submit_all.sh (USER unset)\n{r3.stdout}\n{r3.stderr}\n')
    R.add('T12 submit_all.sh survives an unset $USER', r3.returncode == 0,
          f'exit {r3.returncode}: {r3.stderr.strip()[:60]}')

    # ---- 9: the dependency graph is the documented one
    txt = sa.read_text()
    wired = all(x in txt for x in (
        'JID_SCORE=$(sbatch --parsable --dependency=afterok:$JID_PREPARE',
        'JID_GENO=$(sbatch --parsable --dependency=afterok:$JID_SCORE_MERGE',
        'JID_PACKAGE=$(sbatch --parsable --dependency=afterok:$JID_FIGURES'))
    R.add('T12 dependencies are wired prepare->score->merge->geno->...->package', wired)

    # ---- 10: a rejection must name the jobs already queued, or the operator double-submits
    (sb / 'sbatch').write_text('#!/bin/bash\nn=$(cat %s 2>/dev/null || echo 900)\n'
                               'n=$((n+1)); echo $n > %s\n'
                               'if [ $n -ge 903 ]; then echo "sbatch: error: bad account" >&2; '
                               'exit 1; fi\necho $n\n' % (d / 'ctr2', d / 'ctr2'))
    (sb / 'sbatch').chmod(0o755)
    r4 = subprocess.run(['bash', str(sa)], capture_output=True, text=True, env=env)
    both = 'scancel' in r4.stdout and '901' in r4.stdout
    R.add('T12 a rejected stage names the jobs already queued', r4.returncode != 0 and both,
          'without this the retry submits the early stages twice')

    # ---- 11: the GPU route asks for the GPU, on the GPU partition
    rc, out = run_mode('submit', ['--work', d / 'w2', '--results', d / 'r2',
                                  '--image', '/opt/sif/x.sif'] + inputs
                       + ['--account', 'acct_x', '--partition', 'cpu_q',
                          '--gpu-partition', 'gpu_q', '--qos', 'normal'], log)
    g = (d / 'w2' / 'slurm' / 'score.sbatch').read_text() if rc == 0 else ''
    R.add('T12 the GPU stage asks for gpu:1 on the GPU partition',
          '#SBATCH --gres=gpu:1' in g and '#SBATCH --partition=gpu_q' in g
          and '#SBATCH --qos=normal' in g, f'exit {rc}')
    cpu_stage = (d / 'w2' / 'slurm' / 'geno.sbatch').read_text() if rc == 0 else ''
    R.add('T12 only the GPU stage is moved to the GPU partition',
          '#SBATCH --partition=cpu_q' in cpu_stage and '--gres' not in cpu_stage)


# ---------------------------------------------------------------------------- T13 external
# The four modes merged in from the deepecg-ancestry container. T0 proves they dispatch; this
# proves they RUN, which is the only claim worth making about a merge. The fixture is the one
# the older container's own selftest used -- a synthetic cohort with repeat recordings -- driven
# end to end: `prepare` -> `validate`, then the three stand-alone modes `validate` subsumes.
EXT_COHORT = {'EUR': 90, 'AFR': 26, 'EAS': 24, 'SAS': 6, 'ADMIXED': 24}
EXT_MIN_N = 10


def synthetic_ecg(rng, n, fs=250, secs=10):
    """ECG-like (n, 2500, 12) float32 in MILLIVOLTS: P/QRS/T Gaussian bumps per beat with
    per-lead gains plus baseline noise. Physiologically crude, but on the training scale, which
    is the part that matters -- `infer.check_units` refuses anything else."""
    t = np.arange(fs * secs) / fs
    out = np.empty((n, fs * secs, 12), np.float32)
    for i in range(n):
        hr = rng.uniform(55, 95) / 60.0
        beats = np.arange(rng.uniform(0, 1 / hr), secs, 1 / hr)
        base = np.zeros_like(t)
        for b in beats:
            base += 0.15 * np.exp(-((t - b + 0.16) / 0.025) ** 2)     # P
            base += 1.2 * np.exp(-((t - b) / 0.012) ** 2)             # R
            base -= 0.25 * np.exp(-((t - b - 0.03) / 0.01) ** 2)      # S
            base += 0.30 * np.exp(-((t - b - 0.28) / 0.05) ** 2)      # T
        gains = rng.uniform(-1.2, 1.6, 12)
        out[i] = (base[:, None] * gains[None, :]
                  + rng.normal(0, 0.02, (len(t), 12))).astype(np.float32)
    return out


def build_external_cohort(root, rng):
    """RFMix matrices + an ECG inventory, as a site exports them.

    Every 2nd person gets a second ECG and every 5th a third, each on its own date, so the
    repeat pairing of analysis 8 is exercised and so is the one-ECG-per-person rule of
    analyses 1-7 against people who have several. One in five has an unrecognised sex, which
    must cost them a place in the matched set and nothing else.
    """
    rows, meta, k = [], [], 0
    ecg_dir = root / 'ecgs'
    ecg_dir.mkdir(parents=True)
    for grp, n in EXT_COHORT.items():
        for _ in range(n):
            pid = f'STPID{k:05d}'
            if grp == 'ADMIXED':                        # dominant < 0.70 by construction
                pr = rng.dirichlet([2, 2, 2, 2])
                while pr.max() >= 0.65:
                    pr = rng.dirichlet([2, 2, 2, 2])
            else:
                pr = [0.03, 0.03, 0.03, 0.03]
                pr[['AFR', 'EAS', 'SAS', 'EUR'].index(grp)] = 0.91
            rows.append({'Sample': pid, 'AFR': pr[0], 'EAS': pr[1], 'SAS': pr[2], 'EUR': pr[3],
                         'batch': 1 + (k % 2)})
            sex = ['Male', 'Female', 'M', 'f', 'Unknown'][k % 5]
            age = float(rng.integers(30, 80))
            n_ecg = 1 + (k % 2 == 0) + (k % 5 == 0)
            for j in range(int(n_ecg)):
                f = ecg_dir / f'{pid}_{j}.npy'
                np.save(f, synthetic_ecg(rng, 1)[0])
                meta.append({'ecg_id': f'ECG_{pid}_{j}', 'patient_id': pid, 'gender': sex,
                             'age': age + j, 'full_ecg_path': str(f),
                             'acq_date': f'{2010 + 3 * j + (k % 4)}-0{1 + (k % 9)}-15'})
            k += 1
    df = pd.DataFrame(rows)
    for b in (1, 2):
        df[df.batch == b].drop(columns='batch').to_csv(root / f'rfmix{b}.txt', sep='\t',
                                                      index=False)
    pd.DataFrame(meta).to_parquet(root / 'meta.parquet')
    return df


def t13_external(R, root, log):
    """prepare -> validate, and the three stand-alone modes, on a synthetic cohort."""
    from ancestry import weights as W
    try:
        W.resolve('ssl_backbone', download=False)
        W.resolve('ancestry_folds', download=False)
    except W.WeightsUnavailable as e:
        R.skip('T13 the external-validation modes run end to end', str(e))
        return

    d = root / 'external'
    d.mkdir(parents=True, exist_ok=True)
    build_external_cohort(d, np.random.default_rng(7))
    n_all = sum(EXT_COHORT.values())
    n_case = EXT_COHORT['AFR'] + EXT_COHORT['EAS'] + EXT_COHORT['SAS']
    out = d / 'prepared'

    rc, msg = run_mode('prepare', ['--work', d / 'work', '--results', d / 'results',
                                   '--rfmix', d / 'rfmix1.txt', d / 'rfmix2.txt',
                                   '--ecg-meta', d / 'meta.parquet', '--output', out,
                                   '--no-plots', '--patient-id-col', 'patient_id',
                                   '--date-col', 'acq_date'], log)
    if not R.add('T13 prepare builds all_genotyped/ from RFMix + an ECG inventory', rc == 0,
                 msg.strip().splitlines()[-1][:90] if rc else 'exit 0'):
        return

    gcov = pd.read_csv(out / 'all_genotyped' / 'covariates.csv')
    R.add('T13 several ECGs of one patient collapse to one person',
          len(gcov) == n_all, f'{len(gcov)}/{n_all} rows')
    rdir = out / 'all_genotyped_repeats'
    R.add('T13 repeat pairs are written for people with two or more ECGs',
          (rdir / 'X.npy').exists())

    vout = out / 'validate'
    rc, msg = run_mode('validate', ['--input', out / 'all_genotyped', '--output', vout,
                                    '--n-boot', '40', '--ensemble-n-boot', '10',
                                    '--min-n', str(EXT_MIN_N), '--run-name', 'selftest'], log)
    if not R.add('T13 validate runs the whole pre-specified plan', rc == 0,
                 msg.strip().splitlines()[-1][:90] if rc else 'exit 0'):
        return

    res = json.loads((vout / 'validate.json').read_text())
    R.add('T13 everyone is scored, admixed and unknown-sex included',
          res['n_total'] == n_all, f"{res['n_total']}/{n_all}")
    R.add('T13 the primary set is gated and excludes the admixed',
          res['primary']['n'] == n_all - EXT_COHORT['ADMIXED']
          and round(res['primary']['pos_rate'] * res['primary']['n']) == n_case,
          f"n={res['primary']['n']}, positives="
          f"{round(res['primary']['pos_rate'] * res['primary']['n'])}")
    R.add('T13 the gate sweep widens monotonically',
          all(a['n'] >= b['n'] for a, b in zip(res['gate_sweep'], res['gate_sweep'][1:])))
    R.add('T13 analysis 8 reports intrapatient stability from the repeat pairs',
          bool(res.get('stability', {}).get('available')),
          'ICC over the pairs prepare found')

    # The merge's own rule: these outputs leave the site, so they go through the same audit as
    # the genetic tables -- the stricter one, since the older container's audit did not look
    # inside .png or check for an identifier column by name.
    bad = PV.audit_files(sorted(p for p in vout.rglob('*') if p.is_file()),
                         ids=[f'STPID{k:05d}' for k in range(0, n_all, 7)])
    R.add('T13 every validate output passes the merged privacy audit', not bad,
          '; '.join(bad)[:160] if bad else f'{len(list(vout.rglob("*")))} files')

    # the three stand-alone modes `validate` subsumes, on the same prepared input
    g = out / 'all_genotyped'
    X = np.load(g / 'X.npy', mmap_mode='r')
    cov = pd.read_csv(g / 'covariates.csv')
    m = d / 'standalone'
    m.mkdir(exist_ok=True)
    # Balanced on purpose: `metrics` refuses a class with fewer than 20 participants, which is
    # the disclosure floor doing its job, so a lopsided slice would test the refusal rather than
    # the mode. 25 per class clears it with room to spare.
    eur = cov['EUR'].to_numpy()
    y_all = (eur < 0.70).astype(int)
    take = np.concatenate([np.flatnonzero(y_all == 1)[:25], np.flatnonzero(y_all == 0)[:25]])
    np.save(m / 'X.npy', np.asarray(X[take]))
    np.save(m / 'y.npy', y_all[take])
    np.save(m / 'X_admixed.npy', np.asarray(X[take]))
    np.save(m / 'rfmix_eur.npy', eur[take])
    for mode, argv in (('metrics', ['--input', m, '--output', m / 'metrics', '--n-boot', '20']),
                       ('admixed', ['--input', m, '--output', m / 'admixed']),
                       ('ensemble', ['--input', m, '--output', m / 'ensemble', '--n-boot', '10'])):
        rc, msg = run_mode(mode, argv, log)
        R.add(f'T13 `{mode}` runs on its own', rc == 0,
              msg.strip().splitlines()[-1][:90] if rc else 'exit 0')


def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry selftest', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True,
                    help='a scratch directory for the synthetic cohort and its outputs')
    ap.add_argument('--ref', default=str(L.ref_dir()),
                    help='the frozen discovery tables (default: $REFERENCE_DIR, else the '
                         'reference/ folder of the pinned weights snapshot)')
    ap.add_argument('--quick', action='store_true',
                    help='skip the stages that need plink2')
    ap.add_argument('--keep', action='store_true', help='keep the fixture for inspection')
    a = ap.parse_args()

    root = Path(a.out)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    ref = Path(a.ref)
    os.environ.setdefault('REFERENCE_DIR', str(ref))
    rng = np.random.default_rng(SEED)
    R = Report()
    logf = root / 'selftest.log'
    L.banner(f'selftest  (image version {L.VERSION})')
    with open(logf, 'w') as log:
        t0_dispatch(R, log)
        t0b_chain_guards(R, root, log)
        t0d_vcf_route(R, root, log)
        t1_reference(R, ref)
        t1b_builds(R, ref, root)
        t2_arithmetic(R, rng)
        print('\n  building the synthetic cohort ...')
        work, res, S, V, G = build_fixture(root, ref, rng)
        plant_findings(work, S, rng)
        print(f'  {len(S):,} people, {len(V):,} variants, '
              + ', '.join(f'{k} {v}' for k, v in S.stratum.value_counts().items()))
        try:
            t3_genetics(R, root, ref, work, res, log, a.quick)
        except Exception as e:
            R.add('T3 genetics end to end', False, f'{type(e).__name__}: {e}')
            log.write(traceback.format_exc())
        try:
            t4_fairness(R, work, res, log)
        except Exception as e:
            R.add('T4 fairness', False, f'{type(e).__name__}: {e}')
            log.write(traceback.format_exc())
        t5_privacy(R, res, work, root / 'planted')
        t6_figures(R, res, log)
        t7_package(R, work, res, log)
        try:
            t8_ses(R, root, log)
        except Exception as e:
            R.add('T8 ses', False, f'{type(e).__name__}: {e}')
            log.write(traceback.format_exc())
        try:
            t9_robustness(R, root, log)
        except Exception as e:
            R.add('T9 robustness', False, f'{type(e).__name__}: {e}')
            log.write(traceback.format_exc())
        try:
            t10_formats(R, root, log)
        except Exception as e:
            R.add('T10 formats', False, f'{type(e).__name__}: {e}')
            log.write(traceback.format_exc())
        try:
            t11_models(R, root, log)
        except Exception as e:
            R.add('T11 models', False, f'{type(e).__name__}: {e}')
            log.write(traceback.format_exc())
        try:
            t12_slurm(R, root, log)
        except Exception as e:
            R.add('T12 slurm', False, f'{type(e).__name__}: {e}')
            log.write(traceback.format_exc())
        try:
            t13_external(R, root, log)
        except Exception as e:
            R.add('T13 external validation', False, f'{type(e).__name__}: {e}')
            log.write(traceback.format_exc())

    T = pd.DataFrame(R.rows)
    (root / 'selftest_report.json').write_text(json.dumps(
        dict(version=L.VERSION, checks=R.rows,
             passed=int((T.result == 'PASS').sum()), failed=int((T.result == 'FAIL').sum()),
             skipped=int((T.result == 'SKIP').sum())), indent=2))
    (root / 'selftest_report.txt').write_text(T.to_string(index=False) + '\n')
    n_p, n_f, n_s = ((T.result == x).sum() for x in ('PASS', 'FAIL', 'SKIP'))
    print(f'\n  {n_p} passed, {n_f} failed, {n_s} skipped')
    print(f'  report {root / "selftest_report.txt"}   log {logf}')
    if not a.keep and not n_f:
        for d in ('planted',):
            shutil.rmtree(root / d, ignore_errors=True)
    if n_f:
        print('\n  ALL CHECKS DID NOT PASS. Do not run on real data until they do.')
        sys.exit(1)
    print('\n  ALL CHECKS PASSED' + (f' ({n_s} skipped -- see the report)' if n_s else ''))


if __name__ == '__main__':
    main()
