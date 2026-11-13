"""Mode `submit` -- find Slurm, and if it is there, fan the run out across it.

NOTHING HERE IS REQUIRED.  `python -m ancestry all ...` runs the whole study on one machine and always
will; that is the contract the image was built on and this mode does not change it.  What it adds
is the option, for a site that has a scheduler, of not spending a week of wall-clock on work that
is embarrassingly parallel.  The genome-wide scan is 22 chromosomes x up to 5 ancestry strata of
independent plink jobs, scoring is a GPU pass that shards cleanly by recording, and the NeuroKit
measurements are one recording at a time.  On one node those are days; across a queue they are
hours.

WHAT IT DOES, AND WHY IT IS SPLIT IN TWO

  Writing the scripts never needs Slurm.  Submitting them does.  Those are separated on purpose,
  because `sbatch` is usually NOT reachable from inside a container -- the scheduler's binaries,
  its munge socket and its config live on the host -- so a mode that could only work by calling
  sbatch itself would fail on exactly the sites that have a cluster.  So:

    submit            writes work/slurm/*.sbatch and work/slurm/submit_all.sh, and stops.
                      Run submit_all.sh from a login node.  Works from anywhere.
    submit --go       additionally submits them, with the dependencies already wired, if and only
                      if sbatch is reachable from where this is running.

THE DEPENDENCY GRAPH, which is the part worth reading

    convert    (VCF sites only: array over chromosomes)
    prepare
      |-- score      (array over shards, GPU)  --> score_merge
      |-- traits     (array over shards, CPU)  --> traits_merge      [skipped with --measurements]
      '-> geno       (one job, CPU: shared panel once, then PCA + kinship per stratum)
            |-- loci
            |-- pgs
            '-- scan (array over stratum x chromosome) --> scan_aggregate
                  '-> fairness --> figures --> package

  Everything after `prepare` that can start, starts.  `loci`, `pgs` and `scan` do not wait for
  each other.  `package` waits for everything, because it audits everything.

  Dependencies are `afterok`, so a failed stage stops its descendants instead of packaging a
  half-finished result.  A failed ARRAY TASK is different: `afterok` on an array waits for every
  task, so one dead chromosome blocks the aggregate.  That is deliberate -- a genome-wide scan
  missing chromosome 9 without saying so is worse than one that stops.

PRIVACY.  The scripts contain paths and parameters, never data.  Slurm's own stdout/stderr go to
work/slurm/logs/, which stays at the site like the rest of work/; nothing in this mode writes to
results/ and nothing it produces is exported.

    python -m ancestry submit --work /out/work --results /out/results --rfmix RF.txt \\
                       --ecg-meta meta.parquet --geno '/data/chr@' \\
                       --image /opt/sif/deepecg-ancestry_1.0.0.sif \\
                       --account <slurm account> --partition <queue> \\
                       --gpu-partition <gpu queue>|--no-gpu [--go]

WHAT THE SITE HAS TO TELL IT.  The account, the partition and whether there is a GPU are asked
for, not guessed: each is looked up when the cluster is reachable (sacctmgr, sinfo) and demanded
when it is not, and all of the missing ones are reported in one message before anything is
written.  `--no-account`, `--no-partition` and `--no-gpu` are the ways to say "this cluster does
not use that".  The reason is narrow and practical: `--gres=gpu:1` on a cluster with no GPU is a
REJECTION rather than a fallback to CPU, and a missing --account is a rejection on most sites, so
a guess costs a queue of 147 dead jobs instead of one question.

Prints no participant identifier.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from ancestry import lib as L                                            # noqa: E402

OK, NO, HM = '  [ok] ', '  [--] ', '  [??] '


# ============================================================================== detection
def detect():
    """What scheduler is visible from here, and can we actually submit to it?"""
    d = {'sbatch': shutil.which('sbatch'), 'squeue': shutil.which('squeue'),
         'sinfo': shutil.which('sinfo'), 'scontrol': shutil.which('scontrol')}
    d['inside_job'] = bool(os.environ.get('SLURM_JOB_ID'))
    d['job_id'] = os.environ.get('SLURM_JOB_ID')
    # Are we inside a container?  It matters because it is the usual reason sbatch is missing
    # while a perfectly good cluster is running on the other side of the mount namespace.
    d['in_container'] = bool(os.environ.get('APPTAINER_CONTAINER')
                             or os.environ.get('SINGULARITY_CONTAINER')
                             or Path('/.dockerenv').exists()
                             or Path('/app/ancestry').exists())
    d['can_submit'] = bool(d['sbatch'])
    d['partitions'] = []
    if d['sinfo']:
        try:
            r = subprocess.run([d['sinfo'], '-h', '-o', '%P|%a|%l|%D|%G'],
                               capture_output=True, text=True, timeout=20)
            for line in r.stdout.strip().splitlines():
                p = line.split('|')
                if len(p) >= 4:
                    d['partitions'].append(dict(partition=p[0].rstrip('*'), state=p[1],
                                                timelimit=p[2], nodes=p[3],
                                                gres=p[4] if len(p) > 4 else ''))
        except (OSError, subprocess.SubprocessError):
            pass
    d['gpu_partitions'] = [p['partition'] for p in d['partitions']
                           if 'gpu' in (p.get('gres') or '').lower()
                           or 'gpu' in p['partition'].lower()]
    d['assoc'] = associations()
    return d


def associations():
    """The accounts, partitions and QoS this user is actually allowed to charge.

    Worth asking the cluster rather than the human: on a site where the user has exactly one
    association there is nothing to ask, and on a site where they have several, listing the real
    ones beats a guess.  Returns {} when sacctmgr is not reachable -- which is the normal case
    from inside a container, and the reason the flags exist at all.
    """
    sa = shutil.which('sacctmgr')
    user = os.environ.get('USER') or os.environ.get('LOGNAME') or ''
    if not (sa and user):
        return {}
    try:
        r = subprocess.run([sa, '-nP', 'show', 'assoc', f'user={user}',
                            'format=Account,Partition,QOS'],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return {}
    acc, part, qos = [], [], []
    for line in r.stdout.strip().splitlines():
        f = line.split('|')
        if len(f) > 0 and f[0].strip():
            acc.append(f[0].strip())
        if len(f) > 1 and f[1].strip():
            part.append(f[1].strip())
        if len(f) > 2 and f[2].strip():
            qos += [q.strip() for q in f[2].split(',') if q.strip()]
    return {'accounts': sorted(set(acc)), 'partitions': sorted(set(part)),
            'qos': sorted(set(qos))}


def report(d, prefix='scheduler'):
    """One readable block, used by `survey` as well as by this mode."""
    print(f'\n{prefix}')
    if d['sbatch']:
        print(OK + f'sbatch: {d["sbatch"]}  -- this run can submit directly')
    elif d['in_container']:
        print(HM + 'sbatch is not visible from inside the container. That is normal: Slurm '
                   'lives on the host.')
        print(HM + '  `submit` will still WRITE the job scripts; run work/slurm/submit_all.sh '
                   'on a login node.')
    else:
        print(NO + 'no sbatch: this machine has no Slurm. Use `all`, which needs no scheduler.')
    if d['inside_job']:
        print(OK + f'running inside Slurm job {d["job_id"]}')
    if d['partitions']:
        up = [p for p in d['partitions'] if p['state'].lower().startswith('up')]
        print(OK + f'{len(d["partitions"])} partition(s), {len(up)} up: '
                   + ', '.join(p['partition'] for p in d['partitions'][:8]))
        if d['gpu_partitions']:
            print(OK + f'GPU partition(s): {", ".join(d["gpu_partitions"][:4])}  -- `score` '
                       'belongs here')
        else:
            print(HM + 'no partition advertises a GPU. Pass --no-gpu (score runs on CPU, '
                       'same numbers) or --gpu-partition if the name is not obvious.')
    A = d.get('assoc') or {}
    if A.get('accounts'):
        print(OK + f'account(s) for this user: {", ".join(A["accounts"][:6])}'
                   + ('  -- pass one with --account' if len(A['accounts']) > 1 else
                      '  -- adopted automatically'))
    if A.get('qos'):
        print(OK + f'QoS available: {", ".join(A["qos"][:6])}')
    if not A and d['sbatch']:
        print(HM + 'sacctmgr did not answer, so the account cannot be looked up. `submit` will '
                   'ask for --account.')
    return d


# ====================================================== what the site has to tell us
# A plan of 147 jobs, every one rejected because `--account` was absent, is the worst possible
# first contact with this image -- and the fix is a single word the site's own admin already
# knows. So the decisions a production scheduler will not take a job without are RESOLVED BEFORE
# ANYTHING IS WRITTEN, from the cluster where it can be asked and from the operator where it
# cannot, and every one still missing is reported in ONE message rather than three runs.

def _bullet(what, flag, why, how):
    return (f'  {what}\n'
            f'      pass   {flag}\n'
            f'      why    {why}\n'
            f'      find   {how}\n')


def resolve_site(a, d):
    """Pin down account / partition / GPU / QoS, or explain exactly what is missing.

    Returns a dict of the resolved values plus, for each, HOW it was resolved -- printed, because
    "we adopted the only account sacctmgr lists" and "you told us" should not look the same in a
    log someone reads six months later.
    """
    assoc = d.get('assoc') or {}
    accounts = assoc.get('accounts') or []
    parts = [p['partition'] for p in d.get('partitions') or []]
    qoss = assoc.get('qos') or []
    site, ask = {}, []

    # ---- account
    if a.no_account:
        site['account'], site['account_how'] = None, 'not used (--no-account)'
    elif a.account:
        site['account'], site['account_how'] = a.account, 'given with --account'
    elif len(accounts) == 1:
        site['account'], site['account_how'] = accounts[0], 'the only account sacctmgr lists'
    else:
        ask.append(_bullet(
            'Slurm ACCOUNT' + (f' -- sacctmgr lists {len(accounts)}: {", ".join(accounts[:8])}'
                               if accounts else ''),
            '--account <name>     (or --no-account if this cluster does not use them)',
            'most production clusters reject every job that does not charge an account',
            'sacctmgr -nP show assoc user=$USER format=Account,Partition,QOS'))

    # ---- partition
    if a.no_partition:
        site['partition'], site['partition_how'] = None, 'not used (--no-partition)'
    elif a.partition:
        site['partition'], site['partition_how'] = a.partition, 'given with --partition'
    elif len(parts) == 1:
        site['partition'], site['partition_how'] = parts[0], 'the only partition sinfo reports'
    else:
        ask.append(_bullet(
            'Slurm PARTITION' + (f' -- sinfo reports {len(parts)}: {", ".join(parts[:8])}'
                                 if parts else ''),
            '--partition <name>   (or --no-partition to take the cluster default)',
            'a cluster with no usable default partition rejects the job outright',
            'sinfo -s      (the one marked * is the default, if there is one)'))

    # ---- the GPU question.  `score` is the only stage that wants one, and getting this wrong is
    # silent in the worst direction: an unconditional --gres=gpu:1 is REJECTED on a CPU-only
    # cluster, so the stage never runs at all rather than running slowly.
    if a.no_gpu:
        site['gpu'], site['gpu_partition'] = False, None
        site['gpu_how'] = 'off (--no-gpu): score runs on CPU, same numbers, slower'
    elif a.gpu_partition:
        site['gpu'], site['gpu_partition'] = True, a.gpu_partition
        site['gpu_how'] = 'given with --gpu-partition'
    elif d.get('gpu_partitions'):
        site['gpu'], site['gpu_partition'] = True, d['gpu_partitions'][0]
        site['gpu_how'] = 'sinfo advertises a GPU on this partition'
    else:
        ask.append(_bullet(
            'whether the cluster has a GPU for `score`'
            + ('' if d.get('sinfo') else ' -- sinfo is not reachable from here, so we cannot look'),
            '--gpu-partition <name>   or   --no-gpu',
            'score asks for --gres=gpu:1, and a cluster with no GPU REJECTS that job rather '
            'than running it on CPU',
            'sinfo -o "%P %G"      (a partition with gpu:N in the last column)'))

    # ---- QoS: genuinely optional everywhere, so never asked for
    if a.qos:
        site['qos'], site['qos_how'] = a.qos, 'given with --qos'
    else:
        site['qos'] = None
        site['qos_how'] = ('not set' + (f'; available: {", ".join(qoss[:6])}' if qoss else ''))

    # ---- the inputs the stages themselves require.  argparse inside the job would catch these,
    # but twelve hours later and after `afterok` has cancelled everything downstream.
    if not a.reuse_prepared:
        if not a.rfmix:
            ask.append(_bullet('the RFMix ancestry file', '--rfmix <file> [file ...]',
                               '`prepare` requires it; without it every one of the jobs below '
                               'fails at argparse and afterok cancels the rest',
                               'the per-patient RFMix Q/global-ancestry table'))
        if not a.ecg_meta:
            ask.append(_bullet('the ECG inventory', '--ecg-meta <file> [file ...]',
                               '`prepare` requires it',
                               'the parquet or CSV listing one row per ECG recording'))
    if not a.geno:
        ask.append(_bullet('the genotypes', '--geno \'/data/chr@\'',
                           '`geno`, `loci`, `pgs` and `scan` all need them; @ stands for the '
                           'chromosome number',
                           'the pgen/bed/VCF release, one file set per chromosome'))

    if ask:
        raise SystemExit(
            '\nthis run needs ' + (f'{len(ask)} things' if len(ask) > 1 else 'one thing')
            + ' the site has to tell it, and nothing has been written:\n\n'
            + '\n'.join(ask)
            + '\n  Nothing was guessed on purpose: every one of these is a value a wrong guess '
              'turns into\n  a queue full of rejected jobs. `survey` reports what it can see of '
              'the cluster,\n  and `python -m ancestry all ...` needs no scheduler at all.')
    return site


def report_site(site):
    print('\n  site settings')
    for k, lbl in (('account', 'account'), ('partition', 'partition'), ('qos', 'QoS')):
        v = site.get(k)
        print(f'    {lbl:12s} {str(v) if v else "--":24s} {site.get(k + "_how", "")}')
    g = f'gpu:1 on {site["gpu_partition"]}' if site.get('gpu') else 'none'
    print(f'    {"GPU":12s} {g:24s} {site.get("gpu_how", "")}')


# ============================================================================== the plan
class Stage:
    """One node of the graph: a mode, its resources, and what it waits for."""

    def __init__(self, name, mode, args, after=(), array=None, cpus=8, mem='32G',
                 time='08:00:00', gpus=0, partition=None):
        self.name, self.mode, self.args = name, mode, list(args)
        self.after = list(after)
        self.array = array          # list of per-task argument strings, or None
        self.cpus, self.mem, self.time, self.gpus = cpus, mem, time, gpus
        self.partition = partition


def build_plan(a, strata, chroms, site):
    """The graph above, with MGB-sized resource requests.

    The numbers are deliberately generous rather than tuned: a job that dies at its memory limit
    eleven hours in costs far more than an over-request that the scheduler backfills around.
    Every one of them can be overridden per stage from the command line.
    """
    base = ['--work', a.work, '--results', a.results]
    P = []

    # VCF sites convert first, one chromosome per task -- the single most parallel stage here
    # and the one that most changes the wall-clock of everything after it.
    from ancestry import convert as CV
    if CV.looks_like_vcf(a.geno) and not a.no_convert:
        P.append(Stage('convert', 'convert',
                       ['--work', a.work, '--vcf', a.geno, '--threads', str(a.scan_threads)],
                       array=[f'--chroms {c}' for c in chroms],
                       cpus=a.scan_threads, mem='32G', time='08:00:00'))

    P.append(Stage('prepare', 'prepare',
                   base + ['--rfmix', *a.rfmix, '--ecg-meta', *a.ecg_meta]
                   + (['--reuse-prepared', a.reuse_prepared] if a.reuse_prepared else []),
                   cpus=8, mem='64G', time='12:00:00'))

    # ---- score: the GPU pass, sharded by recording
    sh = [f'--shard {i}/{a.score_shards}' for i in range(1, a.score_shards + 1)]
    _emb = ['--save-embeddings'] if getattr(a, 'save_embeddings', False) else []
    # The GPU is requested only where one is known to exist.  `--gres=gpu:1` on a CPU-only
    # cluster is a REJECTION, not a fallback, so the stage would never run at all -- and the
    # numbers are identical either way, so CPU is a slow answer rather than a wrong one.
    _gpu = bool(site.get('gpu'))
    P.append(Stage('score', 'score', base + _emb
                   + ['--device', a.device if _gpu else 'cpu',
                      '--batch-size', str(a.batch_size)],
                   after=['prepare'], array=sh, cpus=8, mem='48G',
                   time='12:00:00' if _gpu else '24:00:00',
                   gpus=1 if _gpu else 0, partition=site.get('gpu_partition')))
    P.append(Stage('score_merge', 'score',
                   base + ['--merge', str(Path(a.work) / 'scores_*.csv')],
                   after=['score'], cpus=4, mem='32G', time='02:00:00'))

    # ---- traits: only when the site has no cart measurements of its own
    if not a.measurements:
        sh = [f'--shard {i}/{a.trait_shards}' for i in range(1, a.trait_shards + 1)]
        P.append(Stage('traits', 'traits', base[:2] + ['--jobs', str(a.jobs)],
                       after=['prepare'], array=sh, cpus=a.jobs, mem='32G', time='24:00:00'))
        P.append(Stage('traits_merge', 'traits',
                       base[:2] + ['--merge', str(Path(a.work) / 'traits_*.csv')],
                       after=['traits'], cpus=2, mem='16G', time='01:00:00'))

    # ---- geno: one job builds the shared position-matched panel once, then processes each
    #      stratum in isolated work/geno/<stratum>/ directories and writes one combined
    #      provenance file.  Making this an array races on work/geno/panel and lets the last
    #      task overwrite provenance for every other stratum.
    P.append(Stage('geno', 'geno',
                   base + ['--geno', a.geno, '--build', a.build]
                   + (['--chain', a.chain] if a.chain else []) + [
                           '--threads', str(a.threads), '--strata', *strata],
                   after=['score_merge'] + (['convert'] if any(x.name == 'convert' for x in P)
                                            else []),
                   cpus=a.threads, mem='96G', time='24:00:00'))

    P.append(Stage('loci', 'loci', base + ['--threads', str(a.threads)],
                   after=['geno'], cpus=a.threads, mem='32G', time='04:00:00'))

    P.append(Stage('pgs', 'pgs', base + ['--threads', str(a.threads)]
                   + (['--measurements', a.measurements] if a.measurements else []),
                   after=['geno'] + ([] if a.measurements else ['traits_merge']),
                   cpus=a.threads, mem='64G', time='08:00:00'))

    # ---- scan: the reason this mode exists.  stratum x chromosome, all independent.
    if a.with_scan:
        tasks = [f'--strata {s} --chroms {c}' for s in strata for c in chroms]
        P.append(Stage('scan', 'scan',
                       base + ['--geno', a.geno, '--threads', str(a.scan_threads),
                               '--scan-only'],
                       after=['geno'], array=tasks, cpus=a.scan_threads, mem='48G',
                       time='12:00:00'))
        P.append(Stage('scan_aggregate', 'scan',
                       base + ['--aggregate-only', '--chroms', *map(str, chroms)],
                       after=['scan'], cpus=4, mem='64G', time='04:00:00'))

    fair = (['--labels', a.labels] if a.labels else
            ['--statements', a.statements] if a.statements else None)
    if fair is not None:
        P.append(Stage('fairness', 'fairness', base + fair,
                       after=['score_merge'], cpus=4, mem='48G', time='06:00:00'))

    if a.ses:
        ses_after = ['score_merge']
        if not a.save_embeddings and not a.measurements:
            # Without saved embeddings arm A falls back to the measured trait panel. When no
            # site panel was supplied, that panel is produced asynchronously by traits_merge.
            ses_after.append('traits_merge')
        P.append(Stage('ses', 'ses', base + ['--ses', a.ses]
                       + (['--ses-col', a.ses_col] if a.ses_col else [])
                       + (['--ses-name', a.ses_name] if a.ses_name else [])
                       + (['--ses-geography', a.ses_geography] if a.ses_geography else []),
                       after=ses_after, cpus=8, mem='32G', time='04:00:00'))

    last = [s.name for s in P if s.name in ('loci', 'pgs', 'scan_aggregate', 'fairness', 'ses')]
    P.append(Stage('figures', 'figures', ['--results', a.results],
                   after=last, cpus=2, mem='16G', time='01:00:00'))
    P.append(Stage('package', 'package', base + ['--run-name', a.run_name],
                   after=['figures'], cpus=2, mem='16G', time='01:00:00'))
    return P


# ============================================================================== script writing
def launcher_cmd(a):
    """How a job re-enters this image.  Returns (prologue_lines, command_prefix)."""
    binds = set(a.bind or [])
    for p in [a.work, a.results] + list(a.rfmix) + list(a.ecg_meta) + [a.geno, a.labels,
                                                                      a.statements,
                                                                      a.measurements,
                                                                      a.reuse_prepared]:
        if not p:
            continue
        # the genotype pattern carries an @ for the chromosome; bind its directory
        d = Path(str(p).replace('@', '1')).expanduser()
        d = d if d.is_dir() else d.parent
        binds.add(str(d.resolve()) if d.exists() else str(d))
    b = ' '.join(f'-B {_q(x)}' for x in sorted(binds))
    if a.launcher == 'none':
        return [], f'PYTHONPATH={_q(HERE.parent)} {_q(sys.executable)} -m ancestry'
    if a.launcher == 'docker':
        v = ' '.join(f'-v {_q(f"{x}:{x}")}' for x in sorted(binds))
        return [], f'docker run --rm $GPUFLAG {v} {_q(a.image)}'
    pro = ['# --nv is added only for the GPU stage; it fails on nodes with no driver',
           f'module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true']
    return pro, f'apptainer exec $GPUFLAG {b} {_q(a.image)} python -m ancestry'


SCRIPT = """#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --output={stdout}
#SBATCH --error={stderr}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
{extra}
set -euo pipefail

# Written by `python -m ancestry submit`. Contains paths and parameters only -- no patient data.
# Slurm's own logs land in {logs}, which stays at the site like the rest of work/.

GPUFLAG="{gpuflag}"
{prologue}
{taskline}
echo "[$(date -Is)] {name} starting on $(hostname)"
{cmd} {mode} {args} {taskargs}
echo "[$(date -Is)] {name} done"
"""


def write_scripts(plan, a, site):
    out = Path(a.work) / 'slurm'
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    pro, cmd = launcher_cmd(a)
    written = []
    for s in plan:
        extra = []
        if s.partition or site.get('partition'):
            extra.append('#SBATCH --partition=' + _slurm_token(
                s.partition or site['partition'], 'partition'))
        if site.get('account'):
            extra.append('#SBATCH --account=' + _slurm_token(site['account'], 'account'))
        if site.get('qos'):
            extra.append('#SBATCH --qos=' + _slurm_token(site['qos'], 'qos'))
        if s.gpus:
            extra.append(f'#SBATCH --gres=gpu:{s.gpus}')
        taskline, taskargs, tasksuffix = '', '', ''
        if s.array:
            tf = out / f'tasks_{s.name}.txt'
            tf.write_text('\n'.join(s.array) + '\n')
            extra.append(f'#SBATCH --array=1-{len(s.array)}'
                         + (f'%{a.max_concurrent}' if a.max_concurrent else ''))
            # the task's own arguments are read from a file rather than built from the index, so
            # the mapping from array index to (stratum, chromosome) is visible and checkable
            taskline = (f'TASK_ARGS=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {_q(tf)})\n'
                        f'echo "task ${{SLURM_ARRAY_TASK_ID}}: $TASK_ARGS"')
            taskargs = '$TASK_ARGS'
            tasksuffix = '_%a'
        f = out / f'{s.name}.sbatch'
        suffix = f'-%A{tasksuffix}'
        f.write_text(SCRIPT.format(
            jobname=_slurm_token(f'{a.run_name}-{s.name}', 'run-name'), name=s.name,
            stdout=_slurm_token(logs / f'{s.name}{suffix}.out', 'work path'),
            stderr=_slurm_token(logs / f'{s.name}{suffix}.err', 'work path'),
            logs=_q(logs), cpus=s.cpus, mem=s.mem,
            time=s.time, extra='\n'.join(extra), gpuflag=('--nv' if s.gpus and
                                                          a.launcher == 'apptainer'
                                                          else '--gpus all' if s.gpus and
                                                          a.launcher == 'docker' else ''),
            prologue='\n'.join(pro), taskline=taskline, cmd=cmd, mode=s.mode,
            tasksuffix=tasksuffix,
            args=' '.join(_q(x) for x in s.args), taskargs=taskargs))
        f.chmod(0o755)
        written.append(f)

    # the one-shot submitter, for a site whose container cannot see sbatch
    lines = ['#!/bin/bash', 'set -euo pipefail',
             '# Submit the whole graph with its dependencies. Run this from a login node.',
             '#',
             '# If sbatch rejects a stage, this stops there -- and the stages ALREADY submitted',
             '# are still queued. The trap prints their ids so they can be cancelled before a',
             '# second attempt; resubmitting without doing so runs the early stages twice.',
             'SUBMITTED=()',
             'trap \'if (( ${#SUBMITTED[@]} )); then echo; echo "a stage was rejected. These '
             'were already submitted:"; echo "  ${SUBMITTED[*]}"; echo "cancel them before '
             'retrying:  scancel ${SUBMITTED[*]}"; fi\' ERR',
             f'cd {_q(out)}', '']
    for s in plan:
        dep = ''
        if s.after:
            dep = '--dependency=afterok:' + ':'.join('$' + _var(x) for x in s.after) + ' '
        lines.append(f'{_var(s.name)}=$(sbatch --parsable {dep}{s.name}.sbatch)')
        lines.append(f'SUBMITTED+=("${_var(s.name)}")')
        lines.append(f'echo "  {s.name:16s} job ${_var(s.name)}"')
    # Single-quoted on purpose: $USER is advice to READ, and expanding it here under `set -u`
    # turned a completed submission into exit 1 on any shell where USER is not exported.
    lines += ['', 'echo', "echo 'submitted. watch with: squeue -u $USER'",
              f'echo logs: {_q(logs)}']
    sa = out / 'submit_all.sh'
    sa.write_text('\n'.join(lines) + '\n')
    sa.chmod(0o755)
    return out, written, sa


def _q(x):
    return shlex.quote(str(x))


_SLURM_TOKEN = re.compile(r'[A-Za-z0-9_./:@+,%=-]+')


def _slurm_token(value, label):
    """A single value in an #SBATCH directive; refuse whitespace and control characters."""
    value = str(value)
    if not _SLURM_TOKEN.fullmatch(value):
        raise SystemExit(
            f'{label} contains characters Slurm directives cannot safely represent: {value!r}')
    return value


def _var(name):
    return 'JID_' + name.upper().replace('-', '_')


# ============================================================================== submission
def submit(plan, a, outdir):
    """Submit with dependencies wired. Only called when sbatch is actually reachable."""
    ids = {}
    for s in plan:
        cmd = ['sbatch', '--parsable']
        if s.after:
            dep = ':'.join(ids[x] for x in s.after if x in ids)
            if dep:
                cmd.append(f'--dependency=afterok:{dep}')
        cmd.append(str(outdir / f'{s.name}.sbatch'))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(f'  !! {s.name} was rejected: {r.stderr.strip()}')
            print('     the remaining stages were not submitted; the scripts are all written, '
                  f'so fix the request and run {outdir / "submit_all.sh"}')
            return ids
        ids[s.name] = r.stdout.strip().split(';')[0]
        n = f' (array of {len(s.array)})' if s.array else ''
        print(f'  {s.name:16s} job {ids[s.name]}{n}')
    return ids


# ============================================================================== main
def main():
    ap = argparse.ArgumentParser(prog='python -m ancestry submit', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True,
                    help="the site's working directory: intermediates, per-person tables and "
                         'arrays. It stays at the site and is never packaged')
    ap.add_argument('--results', required=True,
                    help='the aggregate-only output directory; `package` zips exactly this')
    ap.add_argument('--rfmix', nargs='+', default=[], help='as for `all`, forwarded to the jobs')
    ap.add_argument('--ecg-meta', nargs='+', default=[], help='as for `all`, forwarded to the jobs')
    ap.add_argument('--geno', default=None,
                    help='the genotypes: plink2 pgen/bed prefix, bgen, or VCF/BCF, with @ for '
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
    ap.add_argument('--ses', default=None,
                    help='area-level deprivation per patient; adds the `ses` stage, which waits '
                         'on the scores and runs beside loci/pgs/fairness')
    ap.add_argument('--ses-col', default=None,
                    help='the deprivation column of --ses (an index or a household income)')
    ap.add_argument('--ses-name', default=None,
                    help='what that variable is called in your cohort, for the tables')
    ap.add_argument('--ses-geography', default=None,
                    help='the unit it is measured on (person, postcode, census tract ...); an '
                         'area-level index and an individual income are different claims')
    ap.add_argument('--save-embeddings', action='store_true',
                    help="keep each fold's representation so the `ses` arms A and D run on the "
                         'embedding rather than their fallbacks')
    ap.add_argument('--reuse-prepared', default=None,
                    help='a prepared/ directory from an earlier run, used instead of '
                         'rebuilding the arrays -- also the way to guarantee the same cohort')
    ap.add_argument('--run-name', default='MGB-MGH',
                    help='your cohort name, printed in the returned tables and used in the '
                         'package filename')
    ap.add_argument('--build', default='grch38',
                    help='grch38 (aliases hg38, b38) or grch37 (aliases hg19, b37); `survey` '
                         'tells you which your release is')
    ap.add_argument('--chain', default=None,
                    help='UCSC chain file, GRCh37 sites only: lifts the ancestry-PGS panel, '
                         'which ships in GRCh38 alone. Every other arm runs on either build')
    ap.add_argument('--with-scan', action='store_true',
                    help='include the genome-wide scan (the stage parallelism is really for)')
    ap.add_argument('--strata', nargs='+', default=None,
                    help='strata to fan out over; read from cohort/strata.csv when prepare has '
                         'already run')
    ap.add_argument('--chroms', nargs='+', default=[str(i) for i in range(1, 23)],
                    help='chromosomes to process (default 1-22)')

    ap.add_argument('--launcher', choices=['apptainer', 'docker', 'none'], default='apptainer',
                    help='how a job re-enters this image (default apptainer, which is what an '
                         'HPC site almost always has)')
    ap.add_argument('--image', default=None, help='the .sif or docker image the jobs run')
    ap.add_argument('--bind', nargs='*', default=[],
                    help='extra paths to bind into the container, beyond those derived from the '
                         'arguments')

    # The four site-specific values.  Each has an explicit "this cluster does not use it"
    # opt-out, so that a missing value is always a decision someone made and never a default
    # that quietly produced a queue of rejected jobs.
    ap.add_argument('--partition', default=None,
                    help='queue to submit to; adopted automatically when sinfo reports exactly '
                         'one. `sinfo -s` lists them')
    ap.add_argument('--no-partition', action='store_true',
                    help='this cluster has a usable default partition; do not name one')
    ap.add_argument('--gpu-partition', default=None,
                    help='partition holding the GPUs, for `score`. Detected when sinfo '
                         'advertises gpu: in a partition gres')
    ap.add_argument('--no-gpu', action='store_true',
                    help='no GPU anywhere: `score` runs on CPU and no --gres is requested. '
                         'Identical numbers, slower')
    ap.add_argument('--account', default=None,
                    help='account to charge; adopted automatically when sacctmgr lists exactly '
                         'one for this user')
    ap.add_argument('--no-account', action='store_true',
                    help='this cluster does not use accounts')
    ap.add_argument('--qos', default=None, help='optional; never required')
    ap.add_argument('--max-concurrent', type=int, default=0,
                    help='cap on simultaneously running array tasks (sbatch --array=1-N%%K). '
                         'Set it if the queue is shared; 0 means no cap.')
    ap.add_argument('--score-shards', type=int, default=8,
                    help='how many GPU array tasks the scoring pass is split into')
    ap.add_argument('--trait-shards', type=int, default=16,
                    help='how many CPU array tasks the measurement pass is split into')
    ap.add_argument('--threads', type=int, default=16, help='threads per genetic job')
    ap.add_argument('--scan-threads', type=int, default=8, help='threads per genome-wide scan job')
    ap.add_argument('--jobs', type=int, default=16, help='worker processes inside each measurement task')
    ap.add_argument('--device', default='auto',
                    help="cpu, cuda, cuda:1 ...; 'auto' takes a GPU when there is one")
    ap.add_argument('--batch-size', type=int, default=64,
                    help='ECGs per forward pass; lower it if the GPU runs out of memory')
    ap.add_argument('--no-convert', action='store_true',
                    help='skip the VCF -> pgen conversion stage')
    ap.add_argument('--go', action='store_true',
                    help='actually submit, instead of only writing the scripts')
    a = ap.parse_args()

    L.banner('submit: fan the run out across Slurm')
    d = report(detect())

    # Resolved FIRST: this raises with everything that is missing, before a single script is
    # written, so a half-laid-out run never sits in work/slurm/ looking submittable.
    site = resolve_site(a, d)
    report_site(site)

    if a.launcher != 'none' and not a.image:
        raise SystemExit('--image is required unless --launcher none (the jobs have to know '
                         'which image to run; `apptainer exec` cannot guess it from inside)')

    # strata: whatever prepare found, when it has run; otherwise all five and let `geno` skip
    strata = a.strata
    if not strata:
        sf = Path(a.results) / 'cohort' / 'strata.csv'
        if sf.exists():
            import pandas as pd
            S = pd.read_csv(sf)
            col = 'analysed' if 'analysed' in S.columns else None
            strata = [str(r.stratum) for _, r in S.iterrows()
                      if col is None or str(r[col]).startswith('yes')]
            print(f'\n  strata from {sf.name}: {", ".join(strata)}')
        else:
            strata = list(L.STRATA)
            print(f'\n  prepare has not run yet, so all {len(strata)} strata are laid out; the '
                  'ones below the minimum will skip themselves')

    plan = build_plan(a, strata, a.chroms, site)
    outdir, written, sa = write_scripts(plan, a, site)

    print('\n  the graph')
    for s in plan:
        n = f'  array of {len(s.array)}' if s.array else ''
        dep = ('  after ' + ', '.join(s.after)) if s.after else '  (first)'
        gres = f', {s.gpus} GPU' if s.gpus else ''
        print(f'    {s.name:16s} {s.cpus:3d} cpu, {s.mem:>5s}, {s.time}{gres}{n}{dep}')
    total = sum(len(s.array) if s.array else 1 for s in plan)
    print(f'\n  {len(plan)} stages, {total} jobs in all')
    print(f'  scripts: {outdir}')

    if a.go and d['can_submit']:
        print('\n  submitting')
        ids = submit(plan, a, outdir)
        (outdir / 'submitted.json').write_text(json.dumps(ids, indent=2))
        print(f'\n  watch:  squeue -u $USER\n  logs:   {outdir / "logs"}')
    elif a.go:
        print('\n  !! --go was given but sbatch is not reachable from here.')
        print(f'     Everything is written. Run this on a login node:\n       bash {sa}')
    else:
        print(f'\n  nothing was submitted (no --go). To submit:\n       bash {sa}')
        print('     or re-run with --go from somewhere sbatch is reachable.')
    print('\n  Slurm is never required: `python -m ancestry all ...` runs the same graph on one '
          'machine.')


if __name__ == '__main__':
    main()
