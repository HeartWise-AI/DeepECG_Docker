"""The ancestry pipeline's command surface: every mode dispatches, and nothing is guessed.

These run with no weights, no plink and no network, which is the point -- they are the part of
the pipeline CI can check. The parts that need the models are in `ancestry/selftest.py`, which
runs inside the image.
"""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ancestry.__main__ import CHAIN, MODES, mode_flags  # noqa: E402


def run(*argv, expect=0):
    r = subprocess.run([sys.executable, '-m', 'ancestry', *map(str, argv)],
                       capture_output=True, text=True, cwd=ROOT)
    out = r.stdout + r.stderr
    assert r.returncode == expect, f'expected exit {expect}, got {r.returncode}:\n{out}'
    return out


def test_every_mode_has_a_module_with_a_main():
    for mode, module in MODES.items():
        path = ROOT / 'ancestry' / f'{module}.py'
        assert path.exists(), f'{mode} -> {module}.py is missing'
        assert '\ndef main(' in path.read_text(), f'{module}.py has no main()'


def test_every_chain_stage_is_a_mode():
    assert set(CHAIN) <= set(MODES), set(CHAIN) - set(MODES)


@pytest.mark.parametrize('mode', sorted(MODES))
def test_mode_parses_its_own_arguments(mode):
    """--help builds every default and every help string, which is where the dispatch bugs live."""
    assert run(mode, '--help')


def test_bare_help_lists_every_mode():
    out = run('--help')
    for mode in MODES:
        assert mode in out, f'{mode} missing from the help'


def test_unknown_mode_suggests_the_near_miss():
    out = run('figure', expect=1)
    assert 'figures' in out


def test_unknown_flag_in_the_chain_is_named_not_ignored():
    """A flag no mode declares used to be forwarded to every mode, and the strict parsers then
    exited 2 one by one -- which skipped most of the run and still ended with a summary."""
    out = run('all', '--work', 'w', '--results', 'r', '--rfmix', 'a', '--ecg-meta', 'b',
              '--geno', 'c', '--not-a-real-flag', expect=1)
    assert '--not-a-real-flag' in out


def test_all_refuses_inputs_that_do_not_exist_before_running_anything():
    out = run('all', '--work', 'w', '--results', 'r', '--rfmix', '/nope/rfmix.txt',
              '--ecg-meta', '/nope/meta.parquet', '--geno', '/nope/chr@', expect=1)
    assert 'were not found' in out and 'nothing was run' in out


def test_submit_refuses_to_guess_the_site_settings(tmp_path):
    """One dead stage cancels every afterok descendant, so the account, the partition and the
    GPU are asked for rather than guessed -- and nothing is written while one is missing."""
    out = run('submit', '--work', tmp_path / 'w', '--results', tmp_path / 'r',
              '--rfmix', 'a', '--ecg-meta', 'b', '--geno', 'c', expect=1)
    assert 'nothing has been written' in out
    assert not list(tmp_path.rglob('*.sbatch'))


def test_submit_uses_shell_quoting_and_rejects_directive_injection(tmp_path):
    from ancestry import submit

    assert submit._q("a'b") == "'a'\"'\"'b'"
    assert submit._slurm_token('account-1', 'account') == 'account-1'
    with pytest.raises(SystemExit, match='cannot safely represent'):
        submit._slurm_token('account\n#SBATCH --exclusive', 'account')
    out = run(
        'submit', '--work', tmp_path / 'w', '--results', tmp_path / 'r',
        '--image', '/opt/ancestry.sif', '--rfmix', 'a', '--ecg-meta', 'b', '--geno', 'c',
        '--account', 'account\n#SBATCH --exclusive', '--partition', 'cpu', '--no-gpu', expect=1)
    assert 'cannot safely represent' in out
    assert not list(tmp_path.rglob('*.sbatch'))


def test_command_runner_refuses_shell_strings():
    from ancestry import lib

    with pytest.raises(TypeError, match='argument sequence'):
        lib.sh('echo unsafe')


def test_flags_are_read_from_each_mode_source():
    assert '--rfmix' in mode_flags('prepare')
    assert '--ses' in mode_flags('ses')
    assert '--threads' in mode_flags('loci')


# --- a path given to the mode answers the startup check -------------------------------------
#
# check_environment's own contract is "the check is for what the image must provide, not for
# what the operator has already answered", but it honoured only --ref: `score --models-root
# /mnt/weights` was refused for "MODELS_ROOT is not set", which is the variable that flag
# exists to replace. These two pin the fixed behaviour from both sides.

def _problems(msg):
    """The named problem list, without the trailing 'point SSL_CKPT / ... MODELS_ROOT' advice."""
    return msg.split('Inside the container')[0]


def _check(argv, monkeypatch):
    import ancestry.__main__ as M
    for v in ('MODELS_ROOT', 'SSL_CKPT', 'WCR_CKPT'):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv('ANCESTRY_WEIGHTS_SOURCE', 'local')
    try:
        M.check_environment('score', argv)
        return ''
    except SystemExit as e:
        return str(e)


def test_a_weight_path_given_on_the_command_line_satisfies_the_startup_check(tmp_path,
                                                                             monkeypatch):
    root = tmp_path / 'weights'
    (root / 'raw').mkdir(parents=True)
    ck = tmp_path / 'backbone.pt'
    ck.write_bytes(b'not a real checkpoint')
    msg = _problems(_check(['--head', 'ancestry', '--models-root', str(root),
                            '--ssl-ckpt', str(ck)], monkeypatch))
    assert 'ancestry_folds' not in msg, msg
    assert 'ssl_backbone' not in msg, msg
    # and --head ancestry must not demand the findings head it will never open
    assert 'findings_head' not in msg, msg


def test_a_weight_path_that_does_not_exist_is_still_refused_by_name(tmp_path, monkeypatch):
    ck = tmp_path / 'backbone.pt'
    ck.write_bytes(b'x')
    msg = _problems(_check(['--head', 'ancestry', '--models-root', '/nonexistent/weights',
                            '--ssl-ckpt', str(ck)], monkeypatch))
    assert 'ancestry_folds' in msg and '/nonexistent/weights' in msg, msg
    assert 'given explicitly' in msg, msg


# ---------------------------------------------------------------- the two measurement rulers
#
# `pgs` can read its ECG measurements from the site's cart panel, from NeuroKit2, or from both,
# and `both` has to stay the default: it is the only setting that distinguishes a broken trait
# score from two instruments disagreeing, and a site that silently got one ruler would read the
# positive control wrongly with no sign on the page. CI cannot run the stage (it needs plink and
# genotypes), but it can hold the contract: the choice exists, `both` is the default, the fill is
# advertised, and the documentation a site actually reads says all three.

def test_the_positive_control_lets_a_site_choose_its_measurement_ruler():
    out = run('pgs', '--help')
    assert '--measurement-source' in out
    assert '{both,device,neurokit}' in out.replace('\n', '').replace(' ', '')
    assert '--measurement-map' in out


def test_both_rulers_are_the_default_and_the_recommendation():
    # the real default, read off the parser rather than off the help text
    import argparse
    from ancestry import pgs as P
    seen = {}
    real = argparse.ArgumentParser.add_argument

    def spy(self, *a, **k):
        if a and a[0] == '--measurement-source':
            seen['default'] = k.get('default')
            seen['choices'] = list(k.get('choices') or [])
        return real(self, *a, **k)

    argparse.ArgumentParser.add_argument = spy
    try:
        with pytest.raises(SystemExit):
            P.main.__wrapped__() if hasattr(P.main, '__wrapped__') else P.main()
    except Exception:                                                    # noqa: BLE001
        pass
    finally:
        argparse.ArgumentParser.add_argument = real
    assert seen.get('default') == 'both', seen
    assert seen.get('choices') == ['both', 'device', 'neurokit'], seen
    out = run('pgs', '--help')
    flat = ' '.join(out.split())
    assert 'both (default, RECOMMENDED)' in flat, flat[flat.find('--measurement-source'):][:400]
    # and the fallback has to be described where the flag is, not only in the docs
    assert 'falling back to NeuroKit' in flat or 'falling back to' in flat


def test_a_pinned_ruler_must_name_one_of_the_two_instruments():
    """The map may pin a trait to an instrument; a typo there must not become a dropped trait."""
    src = (ROOT / 'ancestry' / 'pgs.py').read_text()
    assert "if v['source'] not in ('device', 'neurokit')" in src
    assert 'the only two' in src


@pytest.mark.parametrize('doc', ['README.md',
                                 'docs/ANCESTRY.md',
                                 'docs/ANCESTRY_USER_GUIDE.md'])
def test_the_documentation_tells_a_site_to_try_both_rulers(doc):
    text = (ROOT / doc).read_text()
    assert '--measurement-source' in text, f'{doc} does not mention the flag'
    assert 'both' in text


def test_every_flag_a_site_can_set_explains_itself():
    """A flag with no --help is a flag a partner site guesses at.

    `--help` is the only documentation that cannot go stale, and at a site with no network it
    may be the only documentation there is. This asserts the whole command surface carries one:
    190 flags did not, including the ancestry gate and every column-name override in `prepare`,
    which is the first mode anyone runs.
    """
    import ast
    missing = []
    for path in sorted((ROOT / 'ancestry').glob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument' and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and str(node.args[0].value).startswith('--')
                    and not any(k.arg == 'help' for k in node.keywords)):
                missing.append(f'{path.stem} {node.args[0].value}')
    assert not missing, f'{len(missing)} flag(s) with no help text: ' + ', '.join(missing[:12])


@pytest.mark.parametrize('mode', sorted(MODES))
def test_every_mode_is_in_the_reference_table(mode):
    """docs/ANCESTRY.md carries the canonical table: one row per mode, what it needs, what it
    writes, and whether it comes back. A mode absent from it is a mode a site cannot find."""
    assert f'`{mode}`' in (ROOT / 'docs' / 'ANCESTRY.md').read_text()
