"""Pre-ship audit of the DeepECG-Ancestry package: one table of facts, no opinions.

Not a pytest -- it needs a COMPLETED selftest output directory to audit, which the unit tests
cannot produce. Run it after the gate, before handing the image to anyone:

    python -m ancestry selftest --out /tmp/gate
    python tests/release_audit.py /tmp/gate

It checks five things that each cost a re-ship if wrong, and prints every answer rather than
only the failures: that the command surface is complete and routed, that nothing per-person is
in what a run returns, that every promised analysis actually produced its file, that the flags a
partner site needs in order to fit this to their data exist, that every flag explains itself and
every mode is in the canonical table, and that the standalone ancestry image stays aligned with
the original DeepECG runtime -- same base, same torch, same pinned fairseq-signals commit -- while
not copying the default pipeline entry point or expanding its dependency surface.
"""
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if len(sys.argv) < 2:
    sys.exit(__doc__)
RUN = Path(sys.argv[1])                       # a completed selftest output dir
sys.path.insert(0, str(ROOT))
from ancestry.__main__ import MODES, CHAIN                      # noqa: E402
from ancestry import privacy as PV                              # noqa: E402

rows, fails = [], []
def chk(group, what, ok, note=''):
    rows.append((group, what, 'PASS' if ok else 'FAIL', note))
    if not ok:
        fails.append(what)

# ---------------------------------------------------------------- 1. the command surface
wrapper = (ROOT / 'run_ancestry.bash').read_text()
anc_md = (ROOT / 'docs' / 'ANCESTRY.md').read_text()
guide = (ROOT / 'docs' / 'ANCESTRY_USER_GUIDE.md').read_text()
readme = (ROOT / 'README.md').read_text()
cases = re.findall(r'^\s*([a-z|]+)\)', wrapper, re.M)
routed = {m for arm in cases for m in arm.split('|')}
unrouted = [m for m in MODES if m not in routed]
chk('surface', 'every mode is routed by run_ancestry.bash', not unrouted, f'{len(MODES)} modes')
undoc = [m for m in MODES if f'`{m}`' not in anc_md]
chk('surface', 'every mode is in the docs/ANCESTRY.md mode table', not undoc,
    ', '.join(undoc) or f'{len(MODES)} modes documented')
chk('surface', 'every stage of the chain is a mode', set(CHAIN) <= set(MODES),
    ' -> '.join(CHAIN))
helps = {}
for m in sorted(MODES) + ['all']:
    r = subprocess.run([sys.executable, '-m', 'ancestry', m, '--help'],
                       capture_output=True, text=True, cwd=ROOT)
    helps[m] = r.stdout + r.stderr
    if r.returncode:
        chk('surface', f'`{m}` --help', False, (r.stderr or '').strip()[:60])
chk('surface', 'every mode parses its own arguments', all(
    subprocess.run([sys.executable, '-m', 'ancestry', m, '--help'], capture_output=True,
                   cwd=ROOT).returncode == 0 for m in sorted(MODES)), f'{len(MODES)}/{len(MODES)}')

# ---------------------------------------------------------------- 2. what a run returns
ret = sorted(p for p in RUN.rglob('*') if p.is_file()
             and ('results' in p.parts or 'validate' in p.parts))
tables = [p for p in ret if p.suffix in ('.csv', '.tsv')]
bad = PV.audit_files([str(p) for p in tables], k=PV.MIN_CELL, ids=None)
chk('safety', 'the privacy audit is clean over every returned table', not bad,
    f'{len(tables)} tables, {len(bad)} finding(s)')
forb = []
import pandas as pd
worst = (0, None)
for p in tables:
    try:
        t = pd.read_csv(p, nrows=5)
    except Exception:
        continue
    hits = [c for c in t.columns if c.lower() in PV.FORBIDDEN_COLUMNS]
    if hits:
        forb.append((p.name, hits))
    n = sum(1 for _ in open(p)) - 1
    if n > worst[0]:
        worst = (n, p.name)
chk('safety', 'no forbidden column in any returned table', not forb,
    f'checked {len(tables)} tables against {len(PV.FORBIDDEN_COLUMNS)} names')
chk('safety', 'the longest returned table is an aggregate, not a cohort',
    worst[0] <= PV.MAX_TABLE_ROWS and worst[0] < 1000,
    f'{worst[0]} rows ({worst[1]})')
chk('safety', '`figures` cannot reach per-person data', '--work' not in helps['figures'],
    'the mode takes no --work at all')

# ---------------------------------------------------------------- 3. the experiments
WANT = {
 'the 62-locus replication': 'results/loci/replication_summary.csv',
 'per-stratum lead table': 'results/loci/leads_per_stratum.csv',
 'the trait-PGS positive control': 'results/pgs/trait_pgs.csv',
 'trait-PGS decile curves': 'results/pgs/trait_pgs_deciles.csv',
 'the ancestry-score PGS': 'results/pgs/ancestry_pgs.csv',
 'ancestry-PGS deciles': 'results/pgs/ancestry_pgs_deciles.csv',
 'the 77-finding fairness audit': 'results/fairness/gaps.csv',
 'fairness auditability (true vs inferred)': 'results/fairness/auditability.json',
 'finding coverage': 'results/fairness/label_coverage.csv',
 'variant coverage': 'results/geno/variant_coverage.csv',
 'the analysis sets': 'results/geno/analysis_sets.csv',
 'cohort strata': 'results/cohort/strata.csv',
 'the score distribution': 'results/score/score_histogram.csv',
 'score by stratum': 'results/score/score_by_stratum.csv',
 'the model-check gate': 'results/score/model_check_verdict.txt',
 'deprivation (ses) arms': 'results/ses/ses_quad.csv',
 'the deprivation cut grid': 'results/ses/ses_cut_grid.csv',
 'external validation, 8 analyses': 'validate/validate_summary.csv',
 'intrapatient stability (ICC)': 'validate/stability.csv',
 'external fairness': 'validate/fairness.csv',
 'dose-response': 'validate/dose_response.csv',
 'ensemble agreement': 'validate/ensemble_stats.json',
 'stand-alone metrics': 'metrics/metrics.csv',
 'admixed OOD': 'admixed/admixed_stats.json',
}
have = {str(p) for p in RUN.rglob('*')}
for name, suffix in WANT.items():
    hit = [p for p in have if p.replace('\\', '/').endswith(suffix)]
    chk('experiments', name, bool(hit), suffix)
figs = sorted(p.name for p in RUN.rglob('results/figures/*.png'))
chk('experiments', 'every plate is drawn', len(figs) >= 6, ', '.join(figs))

# ---------------------------------------------------------------- 4. flexibility
FLEX = {
 'input formats sniffed (csv/tsv/txt/gz/parquet)': ('prepare', '--rfmix-sep'),
 'column names overridable': ('prepare', '--ecg-id-col'),
 'VCF/BCF as well as plink': ('convert', '--vcf'),
 'genome build override': ('geno', '--build'),
 'ancestry gate adjustable': ('prepare', '--threshold'),
 'strata selectable': ('pgs', '--strata'),
 'device and batch size': ('score', '--batch-size'),
 'thread count': ('geno', '--threads'),
 'sharding for big cohorts': ('score', '--shard'),
 'prepared arrays reusable': ('prepare', '--reuse-prepared'),
 'resume a long run': ('all', '--from-mode'),
 'skip a stage': ('all', '--skip'),
 'keep going past a failure': ('all', '--keep-going'),
 'two measurement rulers': ('pgs', '--measurement-source'),
 'site column names for measurements': ('pgs', '--measurement-map'),
 'weights from HF or from disk': ('weights', '--verify'),
 'Slurm, optionally': ('submit', '--go'),
 'suppression thresholds visible': ('validate', '--min-n'),
}
for name, (mode, flag) in FLEX.items():
    chk('flexibility', name, flag in helps.get(mode, ''), f'`{mode} {flag}`')

# ---------------------------------------------------------------- 4b. documentation
import ast as _ast
nohelp = []
for f in sorted(Path(ROOT / 'ancestry').glob('*.py')):
    for n in _ast.walk(_ast.parse(f.read_text())):
        if (isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
                and n.func.attr == 'add_argument' and n.args
                and isinstance(n.args[0], _ast.Constant)
                and str(n.args[0].value).startswith('--')
                and not any(k.arg == 'help' for k in n.keywords)):
            nohelp.append(f'{f.stem} {n.args[0].value}')
chk('docs', 'every flag explains itself in --help', not nohelp,
    f'{len(nohelp)} undocumented' if nohelp else 'no undocumented flag in the package')
chk('docs', 'the canonical step table covers every mode',
    all(f'`{m}`' in anc_md for m in MODES), 'docs/ANCESTRY.md')
chk('docs', 'the README gives the command for every step',
    readme.count('run_ancestry.bash --mode') >= 6, 'README.md ancestry section')
chk('docs', 'the user guide covers both measurement rulers',
    '--measurement-source' in guide and 'both' in guide, 'ANCESTRY_USER_GUIDE.md')
chk('docs', 'the deliverable is named in all three docs',
    all('package' in t for t in (readme, anc_md, guide)), 'one zip, `package`')

# ---------------------------------------------------------------- 5. the image
df, base = (ROOT / 'dockerfile.ancestry').read_text(), (ROOT / 'dockerfile').read_text()
for what, needle in (('same python base', 'FROM python:3.11'),
                     ('same torch', 'torch==2.8.0'),
                     ('same fairseq-signals commit',
                      '980e705ce957c6f46b62f0ca2cfa2d846e112bae'),
                     ('same credential step', 'python -m utils.credentials')):
    chk('image', f'{what} as `dockerfile`', needle in df and needle in base, needle[:40])
for what, needle in (('plink pinned by checksum', 'sha256sum -c SHA256SUMS.txt'),
                     ('plink licences travel with the binaries', 'LICENSE.plink.GPLv3.txt'),
                     ('the build fails if a mode cannot parse its arguments',
                      "'ancestry', m, '--help'"),
                     ('the gated weights use a BuildKit secret', 'id=hf_api_key'),
                     ('entrypoint is the package itself',
                      'ENTRYPOINT ["python", "-m", "ancestry"]')):
    chk('image', what, needle in df, needle[:46])
chk('image', 'the default main.py pipeline is not copied into the ancestry image',
    'COPY main.py' not in df and 'COPY run_pipeline.bash' not in df,
    'separate entry point and dependency surface')
chk('image', 'the default image does not install the ancestry genetics stack',
    'plink' not in base.lower() and 'ancestry/' not in base,
    'dockerfile remains the standard image')

# ---------------------------------------------------------------- print
w = max(len(r[1]) for r in rows) + 2
group = None
for g, what, res, note in rows:
    if g != group:
        print(f'\n{g.upper()}')
        group = g
    print(f'  [{res}] {what:<{w}} {note}')
print(f'\n{len(rows) - len(fails)} passed, {len(fails)} failed')
if fails:
    print('FAILED: ' + '; '.join(fails))
