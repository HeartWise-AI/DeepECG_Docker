"""DeepECG-Ancestry: external validation of the ECG ancestry score at one site.

Genetic replication of the score, and a fairness audit of the released 77-finding model.

Two claims are being validated here, in one image, because they need the same three inputs --
the ECGs, the genotypes and the ancestry calls -- and splitting them into two containers would
mean preparing all three twice.

  THE GENETIC SIGNATURE   The ECG-predicted ancestry score is heritable and its genome-wide
                          loci are cardiac-conduction loci.  Does that hold in a hospital cohort
                          that no part of the discovery meta-analysis touched?  Asked four ways:
                          the discovery leads looked up directly (`loci`), a whole-genome scan per
                          ancestry stratum (`scan`), the nine ECG-trait polygenic scores against
                          the site's own measured ECGs and against the ancestry call (`pgs`), and
                          the ancestry-score PGS itself, which at a site in no arm of the
                          discovery meta-analysis can use the full-strength three-cohort weights.

  THE FAIRNESS AUDIT      The released 77-finding model fails unequally across ancestry groups.
                          Can a hospital SEE that without genotyping anyone -- by splitting its
                          patients on ancestry inferred from the ECG instead?  (`fairness`)

MODES, in the order they are run

  survey     READ-ONLY. What is on this machine, is it millivolts, which build, how many people
             per stratum, is there ground truth. Run this first and send the output back.
  selftest   A full synthetic end-to-end run with privacy assertions. Run this second.
  convert    OPTIONAL but recommended for VCF sites: VCF/BCF -> plink2 pgen, once.
             Everything reads a VCF directly without it, perhaps a hundred times over.
  prepare    RFMix + ECG metadata -> the roster, the ancestry strata, model-ready arrays.
  score      DeepECG-Pop (ancestry) and DeepECG-SSL (77 findings) over every ECG, once each.
  traits     NeuroKit2 ECG measurements, if the site has no cart measurement panel.
  geno       One variant subset, within-stratum PCs, kinship, plink-format phenotype files.
  loci       The discovery leads, per stratum.  THE HEADLINE GENETIC RESULT.
  pgs        Trait PGS positive control, the decile curves, and the ancestry-score PGS.
  scan       OPTIONAL and expensive: the genome-wide scan and its Manhattan, per stratum.
  fairness   The 77-finding audit, on genetic ancestry and on ancestry inferred from the ECG.
  ses        OPTIONAL. Is the ancestry signal explained by area-level socioeconomic deprivation?
             Four arms with sham and potency controls. Runs only if --ses is given.
  figures    Every plate, drawn from the tables above and computing nothing.
  package    RUN LAST. One privacy-audited zip. The only thing to send back.
  submit     OPTIONAL. If the site has Slurm, write (and with --go, submit) the whole graph as
             job arrays: the scan fans out over stratum x chromosome, scoring over shards.
             Slurm is never required -- `all` does the same work on one machine.
  all        prepare -> score -> traits -> geno -> loci -> pgs -> fairness -> figures -> package
             (add --with-scan for the genome-wide scan, --ses FILE for the deprivation arms)

NOTHING PER-PATIENT EVER LEAVES.  work/ carries the ids, the scores, the genotype subsets and the
summary statistics, and stays at the site.  results/ is aggregate by construction: every table is
written through a check that refuses an identifier column and refuses a table long enough to be
per-person, counts of 1-9 are masked, and no metric is computed on fewer than 20 people per class
or a finding with fewer than 15 positives per group.  `package` re-audits every file again before
it writes the zip.

  `<mode> --help` lists that mode's full options.
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
PKG = __package__ or 'ancestry'
from ancestry import lib as L                                            # noqa: E402

# mode -> the module of this package that implements it. Each one exposes `main()` and is
# imported and called, rather than executed by path: the package is the unit, so a mode that
# does not import is a broken build and says so here instead of at the first runpy.
MODES = {
    'survey': 'survey', 'selftest': 'selftest', 'weights': 'weights', 'prepare': 'prepare',
    'score': 'score', 'traits': 'traits', 'geno': 'genotypes',
    'loci': 'loci', 'pgs': 'pgs', 'scan': 'scan',
    'fairness': 'fairness', 'figures': 'figures', 'package': 'package',
    'ses': 'ses',
    'convert': 'convert',
    'submit': 'submit',
    # the external-validation modes: one inference pass, metrics against the frozen discovery
    # reference, no genotypes required. `validate` is the whole pre-specified plan; the four
    # below it are the stand-alone pieces it already covers.
    'validate': 'validate', 'metrics': 'metrics', 'admixed': 'admixed', 'ensemble': 'ensemble',
}
CHAIN = ['prepare', 'score', 'traits', 'geno', 'loci', 'pgs', 'fairness', 'figures', 'package']


# What the image is expected to carry, and which modes break without each. The Dockerfile sets
# every one of these; the check exists because the failure otherwise arrives LATE -- `score` is
# the first thing that touches the backbone, and by then `prepare` has run for hours. It is also
# the one honest way to answer "is this self-contained": assert it at startup rather than claim
# it in a README.
NEEDS = {
    # `score` is on this list because the model-check gate compares against the frozen
    # discovery and October anchors, not only because the heads need weights
    'REFERENCE_DIR': ('the frozen discovery tables',
                      ('score', 'loci', 'pgs', 'fairness', 'figures', 'package', 'selftest',
                       'survey')),
    'FAIRSEQ_ROOT': ('the fairseq-signals source the backbone loads through',
                     ('score', 'validate', 'metrics', 'admixed', 'ensemble')),
}

# The weights are not a bare path check any more: they may come from the environment variable the
# image sets OR from the pinned revisions in the model manifests, which the optional
# `ancestry weights --download` build step puts under weights/. ancestry.weights knows the order;
# this only decides when to ask it, and asks
# with download=False so a startup check never reaches the network.
NEEDS_WEIGHTS = {
    'ssl_backbone': ('the shared SSL backbone',
                     ('score', 'validate', 'metrics', 'admixed', 'ensemble')),
    'findings_head': ('the released 77-finding head', ('score',)),
    'ancestry_folds': ('the five ancestry fold checkpoints',
                       ('score', 'validate', 'metrics', 'admixed', 'ensemble')),
}


def check_environment(mode, argv=()):
    """Fail at startup, by name, when something the image should carry is not there.

    `--help` is exempt: a user reading the flags of a mode they have not set up yet should get
    the flags, not a lecture about the environment. An override on the command line (`--ref`)
    is honoured too -- the check is for what the image must provide, not for what the operator
    has already answered.
    """
    if any(x in ('-h', '--help') for x in argv):
        return
    from ancestry import weights as W
    try:                     # a typo in weights_source should not be a traceback
        W.source()
    except W.WeightsUnavailable as e:
        raise SystemExit(f'{e}\n  set it in ancestry.config (weights_source) or in the '
                         'environment (ANCESTRY_WEIGHTS_SOURCE).')
    bad = []
    for var, (what, modes) in NEEDS.items():
        if mode not in modes and mode != 'all':
            continue
        if var == 'REFERENCE_DIR':
            # The tables are not in this repository -- it is public and they are aggregates of
            # the discovery cohorts -- so "the directory exists" is not the question. The
            # question is whether it holds the tables MANIFEST.txt names, wherever it came from:
            # an operator's --ref, REFERENCE_DIR, a baked-in copy, or the pinned snapshot.
            argv = list(argv)
            given = (argv[argv.index('--ref') + 1]
                     if '--ref' in argv and argv.index('--ref') + 1 < len(argv) else None)
            d = Path(given) if given else L.ref_dir()
            if not L.reference_is_populated(d):
                bad.append(f'the frozen discovery tables are not in {d} ({what}). They ship '
                           'under reference/ in the pinned ancestry model snapshot: mount '
                           'them and set REFERENCE_DIR, or build with RUN_MODELS_SETUP=true')
            continue
        q = os.environ.get(var)
        if not q:
            bad.append(f'{var} is not set ({what})')
        elif not Path(q).exists():
            bad.append(f'{var}={q} does not exist ({what})')
    # The command line answers the same question the environment variable does, so a path
    # passed to the mode must satisfy the check for that artefact. Without this, `score
    # --models-root /mnt/weights` was refused for "MODELS_ROOT is not set" -- the flag exists
    # precisely to replace that variable, and `score --help` advertises it. `--ref` is handled
    # above; these are the other three.
    FLAG_FOR = {'ssl_backbone': '--ssl-ckpt', 'findings_head': '--wcr-ckpt',
                'ancestry_folds': '--models-root'}
    av = list(argv)

    def _given(flag):
        return av[av.index(flag) + 1] if flag in av and av.index(flag) + 1 < len(av) else None

    head = (_given('--head') or 'both') if mode in ('score', 'all') else 'both'
    for artefact, (what, modes) in NEEDS_WEIGHTS.items():
        if mode not in modes and mode != 'all':
            continue
        # `score --head ancestry` never opens the findings head, and `--head findings` never
        # opens the fold checkpoints: demanding the other one blocks a run that would work.
        if head == 'ancestry' and artefact == 'findings_head':
            continue
        if head == 'findings' and artefact == 'ancestry_folds':
            continue
        try:
            W.resolve(artefact, _given(FLAG_FOR.get(artefact, '')), download=False)
        except W.WeightsUnavailable as e:
            bad.append(f'{what}: {e}')
    if bad:
        raise SystemExit(
            'this image is missing something it should carry:\n  ' + '\n  '.join(bad)
            + '\n\nInside the container the paths are set by the dockerfile and the weights are '
              'placed by `ancestry weights --download`, so seeing this means either the image was built '
              'without them (build with --build-arg RUN_MODELS_SETUP=true, or mount them) or the '
              'package is being run outside it. Outside the image, point SSL_CKPT / WCR_CKPT / '
              'MODELS_ROOT / FAIRSEQ_ROOT / REFERENCE_DIR at the files on disk.')


def run(mode, argv):
    check_environment(mode, argv)
    sys.argv = [f'{PKG} {mode}'] + list(argv)
    importlib.import_module(f'{PKG}.{MODES[mode]}').main()


def split_for_chain(mode, argv, common, extra_by_mode):
    out = list(common)
    out += extra_by_mode.get(mode, [])
    return out


def mode_flags(mode):
    """Which --flags a mode's parser declares, read from its source.

    `all` used to append every unrecognised argument to EVERY mode, and eight of the ten modes
    use a strict parser, so one mode-specific flag -- `--rfmix-covar never`, `--caliper 3`,
    `--skip-permutation` -- made the modes that do not know it exit 2. Combined with the loop
    below swallowing those exits, a single tuning flag could skip most of the run and still end
    with a cheerful summary. Routing each flag only to the modes that declare it fixes the cause;
    failing loudly on one no mode declares fixes the rest.
    """
    src = (APP / f'{MODES[mode]}.py').read_text()
    return set(re.findall(r"add_argument\(\s*'(--[a-z0-9][a-z0-9-]*)'", src))


def common_args(mode, work, results):
    """Only the shared paths a mode actually declares.

    In particular, convert and traits take --work but not --results, while figures takes
    --results but not --work.  Passing the union to every stage leaks unknown flags through
    traits into its strict vendored worker.
    """
    flags = mode_flags(mode)
    out = []
    if '--work' in flags:
        out += ['--work', work]
    if '--results' in flags:
        out += ['--results', results]
    return out


def ses_mode_args(a):
    """Arguments shared by the single-machine SES stage and its standalone parser."""
    return (['--ses', a.ses]
            + (['--ses-col', a.ses_col] if a.ses_col else [])
            + (['--ses-name', a.ses_name] if a.ses_name else [])
            + (['--ses-geography', a.ses_geography] if a.ses_geography else [])
            + (['--measurements', a.measurements] if a.measurements else []))


def route_extras(extra, modes):
    """Split `extra` into {mode: [args]}. Raises on a flag no mode declares."""
    known = {m: mode_flags(m) for m in modes}
    out = {m: [] for m in modes}
    i, unknown = 0, []
    while i < len(extra):
        tok = extra[i]
        if not tok.startswith('--'):
            unknown.append(tok); i += 1; continue
        flag = tok.split('=', 1)[0]
        vals = [tok]
        if '=' not in tok:
            j = i + 1
            while j < len(extra) and not extra[j].startswith('--'):
                vals.append(extra[j]); j += 1
            i = j
        else:
            i += 1
        takers = [m for m in modes if flag in known[m]]
        if not takers:
            unknown.append(flag)
            continue
        for m in takers:
            out[m] += vals
    if unknown:
        # A near-miss is named outright. The site is typing a long command at a shell prompt
        # inside a hospital network; "check the spelling" against sixty possible flags is a
        # worse answer than one guess that is usually right.
        import difflib
        # Search EVERY mode, not only the chain's: the most useful answer to `--margn` is
        # `--margin`, which belongs to `ses` -- a mode that joins the chain only when --ses is
        # passed. Guessing inside the chain alone would answer `--merge`, which is worse than
        # no guess at all. The owning mode is named so the site can see whether it is even
        # running that stage.
        allf = {m: mode_flags(m) for m in MODES}
        every = sorted(set().union(*allf.values()))
        lines = [f'these arguments are not accepted by any mode in the chain: {unknown}']
        for u in unknown:
            near = difflib.get_close_matches(u, every, n=1, cutoff=0.6)
            if near:
                owners = [m for m in MODES if near[0] in allf[m]]
                inchain = [m for m in owners if m in modes]
                where = (f'taken by: {", ".join(inchain)}' if inchain else
                         f'taken by `{owners[0]}`, which this chain is not running')
                lines.append(f'  {u}  ->  did you mean {near[0]}?  ({where})')
            else:
                lines.append(f'  {u}  ->  no close match in any mode')
        lines.append('  run that mode on its own with --help to see every flag it takes.')
        raise SystemExit('\n'.join(lines))
    return out


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        print('  modes: ' + ', '.join(list(MODES) + ['all']))
        return
    mode = sys.argv[1]
    rest = sys.argv[2:]
    if mode in MODES:
        run(mode, rest)
        return
    if mode != 'all':
        import difflib
        all_modes = list(MODES) + ['all']
        near = difflib.get_close_matches(mode, all_modes, n=1, cutoff=0.5)
        hint = f'\n  did you mean {near[0]!r}?' if near else ''
        sys.exit(f'unknown mode {mode!r}{hint}\n  modes: ' + ', '.join(all_modes))

    ap = argparse.ArgumentParser(prog='python -m ancestry all', add_help=True,
                                 description='Run every mode in order with one set of paths.')
    ap.add_argument('--work', required=True,
                    help="the site's working directory: intermediates, per-person tables and "
                         'arrays. It stays at the site and is never packaged')
    ap.add_argument('--results', required=True,
                    help='the aggregate-only output directory; `package` zips exactly this')
    ap.add_argument('--rfmix', nargs='+', required=True,
                    help='RFMix weighted-sum matrix/matrices: ancestry proportions per sample')
    ap.add_argument('--ecg-meta', nargs='+', required=True,
                    help='the ECG inventory: one row per recording (id, path, age, sex, and a '
                         'patient id and date where you have them)')
    ap.add_argument('--geno', required=True,
                    help='the genotypes: plink2 pgen/bed prefix, bgen, or VCF/BCF with @ for '
                         'the chromosome number')
    ap.add_argument('--labels', default=None,
                    help='adjudicated 0/1 labels for the 77 findings, one row per person '
                         '(yes/no, Y/N, true/false are all read)')
    ap.add_argument('--statements', default=None,
                    help='free-text ECG statements instead of labels; mapped to the 77 '
                         'findings by the shipped rules')
    ap.add_argument('--measurements', default=None,
                    help="the cart's precomputed ECG measurement panel; `pgs` reads it beside "
                         'the NeuroKit measurements and compares the two rulers')
    ap.add_argument('--reuse-prepared', default=None,
                    help='a prepared/ directory from an earlier run, used instead of '
                         'rebuilding the arrays -- also the way to guarantee the same cohort')
    ap.add_argument('--run-name', default='MGB-MGH',
                    help='your cohort name, printed in the returned tables and used in the '
                         'package filename')
    ap.add_argument('--threads', type=int, default=L.default_threads(),
                    help='threads for plink (default: this machine\'s cores, less two, capped at 16)')
    ap.add_argument('--jobs', type=int, default=8,
                    help='worker processes for the CPU measurement pass')
    ap.add_argument('--device', default='auto',
                    help="cpu, cuda, cuda:1 ...; 'auto' takes a GPU when there is one")
    ap.add_argument('--batch-size', type=int, default=64,
                    help='ECGs per forward pass; lower it if the GPU runs out of memory')
    ap.add_argument('--build', default='grch38',
                    help='grch38 (aliases hg38, b38) or grch37 (aliases hg19, b37); `survey` '
                         'tells you which your release is')
    ap.add_argument('--chain', default=None,
                    help='UCSC chain file, GRCh37 sites only: lifts the ancestry-PGS panel, '
                         'which ships in GRCh38 alone. Every other arm runs on either build')
    ap.add_argument('--with-scan', action='store_true',
                    help='also run the genome-wide scan: hours per stratum, and not needed for '
                         'the headline locus-level result')
    ap.add_argument('--ses', default=None,
                    help='area-level deprivation per patient (csv with participant_id + a score '
                         'column, e.g. the Social Deprivation Index). Adds the `ses` stage')
    ap.add_argument('--ses-col', default=None,
                    help='the deprivation column of --ses (an index or a household income)')
    ap.add_argument('--ses-name', default=None,
                    help='what that variable is called in your cohort, for the tables')
    ap.add_argument('--ses-geography', default=None,
                    help='the unit it is measured on (person, postcode, census tract ...); an '
                         'area-level index and an individual income are different claims')
    ap.add_argument('--save-embeddings', action='store_true',
                    help="keep each fold's representation (work/embeddings.npy) so the `ses` "
                         'arms A and D can run on the embedding rather than on their weaker '
                         'fallbacks. Costs a second forward pass in `score`')
    ap.add_argument('--skip', nargs='*', default=[],
                    help='stages to leave out, by mode name')
    ap.add_argument('--from-mode', default=None,
                    help='resume at this stage, trusting the work/ left by the ones before it')
    ap.add_argument('--no-convert', action='store_true',
                    help='do not convert VCF input to pgen first; read the VCFs '
                         'directly every time (much slower, same answers)')
    ap.add_argument('--keep-going', action='store_true',
                    help='carry on after a stage fails instead of stopping (the '
                         'default is to stop, because every stage feeds the next)')
    a, extra = ap.parse_known_args(rest)

    per = {
        'convert': ['--vcf', a.geno],
        'prepare': ['--rfmix', *a.rfmix, '--ecg-meta', *a.ecg_meta]
                   + (['--reuse-prepared', a.reuse_prepared] if a.reuse_prepared else []),
        'score': ['--device', a.device, '--batch-size', str(a.batch_size)]
                 + (['--save-embeddings'] if a.save_embeddings else []),
        'traits': ['--jobs', str(a.jobs)] if not a.measurements else None,
        'geno': ['--geno', a.geno, '--build', a.build, '--threads', str(a.threads)]
                + (['--chain', a.chain] if a.chain else []),
        'loci': ['--threads', str(a.threads)],
        'pgs': ['--threads', str(a.threads)]
               + (['--measurements', a.measurements] if a.measurements else []),
        'scan': ['--geno', a.geno, '--threads', str(a.threads)],
        'fairness': (['--labels', a.labels] if a.labels else
                     ['--statements', a.statements] if a.statements else None),
        'ses': ses_mode_args(a) if a.ses else None,
        # --run-name reaches `figures` as well as `package`: the plates carry the site's name
        # on their axes, and a package labelled one way with figures labelled another is
        # worse than either.
        'figures': ['--results', a.results, '--run-name', a.run_name],
        'package': ['--run-name', a.run_name],
    }
    chain = CHAIN if not a.with_scan else CHAIN[:CHAIN.index('fairness')] + ['scan'] + \
        CHAIN[CHAIN.index('fairness'):]
    # `ses` needs an input most sites will not have, so it is added only when the file is given
    # rather than being a stage that is skipped nine times out of ten.
    if a.ses:
        chain = chain[:chain.index('fairness')] + ['ses'] + chain[chain.index('fairness'):]
        if not a.save_embeddings:
            print('  --ses given without --save-embeddings: arms A and D will use their '
                  'fallbacks (the measured trait panel and a score-level adjustment). Add '
                  '--save-embeddings for the embedding versions, at the cost of a second '
                  'forward pass in `score`.')
    # A site that hands us VCFs gets the conversion inserted for them rather than being told
    # about it in a document they will read after the run. plink would read the VCFs directly,
    # correctly, and about a hundred times over; converting once is strictly better and there is
    # no reason to make it opt-in.
    from ancestry import convert as CV
    if CV.looks_like_vcf(a.geno) and not a.no_convert:
        chain = ['convert'] + chain
        print('  --geno points at VCF/BCF, so `convert` is inserted first (--no-convert to skip; '
              'everything works without it, far more slowly)')
    routed = route_extras(extra, chain)
    # PRE-FLIGHT.  Every one of these is read by a stage that may not run for hours, and the
    # failure then arrives as a pandas traceback from inside `prepare`. Checking first costs
    # nothing and turns "FileNotFoundError on line 1661" into a list of what is missing.
    missing = []
    for label, paths in (('--rfmix', a.rfmix), ('--ecg-meta', a.ecg_meta),
                         ('--labels', [a.labels] if a.labels else []),
                         ('--statements', [a.statements] if a.statements else []),
                         ('--measurements', [a.measurements] if a.measurements else []),
                         ('--ses', [a.ses] if a.ses else []),
                         ('--reuse-prepared', [a.reuse_prepared] if a.reuse_prepared else [])):
        for q in paths:
            if q and not Path(q).exists():
                missing.append(f'{label} {q}')
    if a.geno:
        # the pattern carries @ for the chromosome; check that chromosome 1 resolves
        try:
            L.geno_args(a.geno, 1)
        except SystemExit as e:
            missing.append(f'--geno {a.geno}  ({e})')
    if missing:
        raise SystemExit('these inputs were not found, so nothing was run:\n  '
                         + '\n  '.join(missing))

    for m, v in routed.items():
        if v:
            print(f'  {m}: extra arguments {" ".join(v)}')
    if a.from_mode:
        if a.from_mode not in chain:
            sys.exit(f'--from-mode {a.from_mode!r} is not one of {chain}')
        chain = chain[chain.index(a.from_mode):]
    for m in chain:
        if m in a.skip:
            print(f'\n==> {m}: skipped by --skip')
            continue
        args = per.get(m)
        if args is None:
            print(f'\n==> {m}: skipped (nothing to run it on -- see --help)')
            continue
        argv = common_args(m, a.work, a.results) + args + routed.get(m, [])
        print(f'\n{"#" * 92}\n#  {m}\n{"#" * 92}', flush=True)
        try:
            run(m, argv)
        except Exception:
            print(f'\n!! {m} failed. The traceback follows; the rest of the chain was not run.')
            print(f'   Once it is fixed, resume with:  --from-mode {m}')
            raise
        except SystemExit as e:
            if e.code:
                print(f'\n!! {m} stopped: {e}')
                # Every stage feeds something downstream, so by default a failure stops the
                # chain. The old rule -- stop only for prepare/score/geno -- meant a broken
                # `loci` or a `package` that refused its own privacy audit scrolled past and the
                # run still looked like it had worked.
                if not a.keep_going:
                    raise SystemExit(
                        f'{m} failed, so the rest of the chain was not run. Fix it and resume '
                        f'with --from-mode {m}, or pass --keep-going to push past failures.')
        finally:
            sys.argv = [PKG]


if __name__ == '__main__':
    main()
