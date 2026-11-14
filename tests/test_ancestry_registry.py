"""The standalone image resolves immutable model artefacts and frozen reference tables.

The regular DeepECG task registry is intentionally unchanged. The ancestry image reads its own
manifest entry and local overrides through ``ancestry.weights``.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REFERENCE = ROOT / 'ancestry' / 'reference'
PENDING = 'PENDING-UPLOAD'

def test_every_ancestry_manifest_revision_is_an_exact_commit_or_declared_pending():
    manifest = json.loads((ROOT / 'ancestry' / 'model_manifest.json').read_text())
    assert manifest['schema_version'] == 1
    for name, entry in manifest['models'].items():
        assert entry['repository'].startswith('heartwise/'), name
        assert re.fullmatch(r'[0-9a-f]{40}', entry['revision']) or entry['revision'] == PENDING, \
            f'{name}: {entry["revision"]} is neither a 40-character commit nor {PENDING}'


def test_the_ancestry_weights_are_in_the_manifest():
    ancestry_manifest = json.loads(
        (ROOT / 'ancestry' / 'model_manifest.json').read_text())['models']
    main_manifest = json.loads((ROOT / 'models' / 'manifest.json').read_text())['models']
    # The backbone is NOT a separate entry: it is base_ssl.pt inside the public, pinned
    # wcr_77_classes snapshot, verified byte-identical to the one the folds were tuned
    # from. Only the fold checkpoints need a repository of their own.
    assert 'deepecg_pop_ancestry' in ancestry_manifest
    assert 'wcr_77_classes' in main_manifest
    assert 'deepecg_pop_ancestry' not in main_manifest


def test_a_pending_revision_cannot_be_downloaded_by_accident():
    """The standalone resolver refuses a pending pin before any network call."""
    from ancestry import weights
    manifest = json.loads((ROOT / 'ancestry' / 'model_manifest.json').read_text())['models']
    pending = [name for name, entry in manifest.items() if entry['revision'] == PENDING]
    for name in pending:
        with pytest.raises(weights.WeightsUnavailable, match='40-character commit'):
            weights.entry_for(name)


def test_no_discovery_table_is_tracked_in_this_public_repository():
    """The one that matters: the tables are cohort aggregates and this repository is public.

    A developer copy of them normally sits in ancestry/reference/ while working, so the test is
    on what git TRACKS, not on what is on disk.
    """
    out = subprocess.run(['git', 'ls-files', 'ancestry/reference'], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip('not a git checkout')
    tracked = {Path(x).name for x in out.stdout.split()}
    allowed = {'MANIFEST.txt', 'README.md', '_build_reference.py',
               '_build_external_reference.py'}
    assert tracked <= allowed, f'cohort data tracked in a public repo: {sorted(tracked - allowed)}'


def test_the_frozen_reference_tables_match_their_manifest():
    lines = [l.split() for l in (REFERENCE / 'MANIFEST.txt').read_text().splitlines()
             if l.strip() and not l.startswith('#')]
    assert lines, 'MANIFEST.txt lists nothing'
    if not all((REFERENCE / parts[-1]).exists() for parts in lines if len(parts) >= 2):
        pytest.skip('the tables are not in the repository; set REFERENCE_DIR to check a copy')
    checked = 0
    for parts in lines:
        if len(parts) < 2:
            continue
        digest, name = parts[0], parts[-1]
        if not re.fullmatch(r'[0-9a-f]{32}', digest):
            continue
        path = REFERENCE / name
        assert path.exists(), f'{name} is in MANIFEST.txt but not shipped'
        assert hashlib.md5(path.read_bytes()).hexdigest() == digest, f'{name} has changed'
        checked += 1
    assert checked >= 10, f'only {checked} tables fingerprinted'


def test_the_weight_resolver_never_reaches_the_network_when_told_not_to(monkeypatch):
    from ancestry import weights
    monkeypatch.delenv('SSL_CKPT', raising=False)
    with pytest.raises(weights.WeightsUnavailable, match='SSL_CKPT'):
        weights.resolve('ssl_backbone', download=False)


def test_an_explicit_path_wins_over_the_environment(monkeypatch, tmp_path):
    """An air-gapped site that was handed the weights on a disk must never reach the network."""
    from ancestry import weights
    given = tmp_path / 'given.pt'
    given.write_bytes(b'')
    monkeypatch.setenv('SSL_CKPT', str(tmp_path / 'other.pt'))
    assert weights.resolve('ssl_backbone', given) == given


def test_a_snapshot_directory_becomes_the_path_the_loader_opens(tmp_path):
    """The fallback path used to hand WCREcgTransformer a DIRECTORY where it wants a checkpoint,
    and the failure then arrived from inside fairseq-signals. A HeartWise model repository ships
    the backbone as base_ssl.pt with one head beside it, exactly as
    models/wcr_ecg_transformer.py reads it."""
    from ancestry import weights
    (tmp_path / 'base_ssl.pt').write_bytes(b'')
    (tmp_path / 'wcr_77_classes_fold0.pt').write_bytes(b'')
    assert weights._member(tmp_path, 'base_ssl.pt', 'ssl_backbone').name == 'base_ssl.pt'
    assert weights._member(tmp_path, weights.HEAD, 'findings_head').name \
        == 'wcr_77_classes_fold0.pt'
    assert weights._member(tmp_path, None, 'ancestry_folds') == tmp_path


def test_an_ambiguous_snapshot_is_refused_rather_than_guessed(tmp_path):
    from ancestry import weights
    (tmp_path / 'base_ssl.pt').write_bytes(b'')
    (tmp_path / 'a.pt').write_bytes(b'')
    (tmp_path / 'b.pt').write_bytes(b'')
    with pytest.raises(weights.WeightsUnavailable, match='exactly one head checkpoint'):
        weights._member(tmp_path, weights.HEAD, 'findings_head')


def test_the_findings_head_is_pinned_to_a_repository_that_already_exists():
    """Unlike the two ancestry entries, the 77-finding head is an already-released model, so the
    fallback is usable today -- which is the point of auditing the released model rather than a
    re-implementation of it."""
    import json
    from ancestry import weights
    manifest = json.loads((ROOT / 'models' / 'manifest.json').read_text())['models']
    key = weights.ARTEFACTS['findings_head'][1]
    assert re.fullmatch(r'[0-9a-f]{40}', manifest[key]['revision'])


def test_the_ancestry_snapshot_can_be_pointed_somewhere_else(monkeypatch):
    """A site must be able to name a staged or project-specific gated snapshot."""
    from ancestry import weights
    monkeypatch.setenv('ANCESTRY_MODEL_REPO', 'someorg/their_copy')
    monkeypatch.setenv('ANCESTRY_MODEL_REVISION', 'a' * 40)
    assert weights.entry_for('deepecg_pop_ancestry') == {
        'repository': 'someorg/their_copy', 'revision': 'a' * 40}
    # the public heads are NOT overridable this way: repinning those is a reviewed change
    assert weights.entry_for('wcr_77_classes')['repository'] == 'heartwise/wcr_77_classes'


def test_an_override_without_an_exact_commit_is_refused(monkeypatch):
    """A branch name would make a validation run unreproducible."""
    from ancestry import weights
    monkeypatch.setenv('ANCESTRY_MODEL_REPO', 'someorg/their_copy')
    for revision in ('', 'main', 'v1.0', 'a' * 39):
        monkeypatch.setenv('ANCESTRY_MODEL_REVISION', revision)
        with pytest.raises(weights.WeightsUnavailable) as excinfo:
            weights.entry_for('deepecg_pop_ancestry')
        assert '40-character' in str(excinfo.value)


def test_a_snapshot_is_verified_by_layout_and_checksum(tmp_path):
    """`weights --verify` is what a colleague runs before uploading and after downloading."""
    from ancestry import weights
    for k in range(1, 6):
        d = tmp_path / 'raw' / f'fold_{k}' / 'bs_128'
        d.mkdir(parents=True)
        (d / 'checkpoint_best.pt').write_bytes(b'not a real checkpoint')
    ref = tmp_path / 'reference'
    ref.mkdir()
    (ref / 'table.tsv').write_text('a\tb\n1\t2\n')
    digest = hashlib.md5((ref / 'table.tsv').read_bytes()).hexdigest()
    (ref / 'MANIFEST.txt').write_text(f'# frozen\n{digest}  9  table.tsv\n')
    fold = tmp_path / 'raw' / 'fold_1' / 'bs_128' / 'checkpoint_best.pt'
    fold_digest = hashlib.md5(fold.read_bytes()).hexdigest()
    (tmp_path / 'CHECKSUMS.md5').write_text(
        f'{fold_digest}  raw/fold_1/bs_128/checkpoint_best.pt\n')
    rows = dict((what, ok) for what, ok, _ in weights.verify_snapshot(tmp_path))
    assert rows['the five fold checkpoints']
    assert rows['the reference tables match MANIFEST.txt']
    assert rows['CHECKSUMS.md5']
    # and a changed table is caught
    (ref / 'table.tsv').write_text('a\tb\n9\t9\n')
    rows = dict((what, ok) for what, ok, _ in weights.verify_snapshot(tmp_path))
    assert not rows['the reference tables match MANIFEST.txt']


def test_a_missing_fold_is_caught_before_anything_is_uploaded(tmp_path):
    from ancestry import weights
    (tmp_path / 'reference').mkdir()
    rows = dict((what, ok) for what, ok, _ in weights.verify_snapshot(tmp_path))
    assert not rows['the five fold checkpoints']


# --- the wrapper and the package must agree on what the modes are -------------------------
#
# `run_ancestry.bash` names its modes three times: in usage(), in the first case block (the
# modes that take no study paths) and in the second (the ones that do). A mode added to
# ancestry/__main__.py and to usage() but to neither case block is advertised and then refused
# with "Invalid mode", which is how `weights` shipped broken until this test existed.

def _runner_text():
    return (ROOT / 'run_ancestry.bash').read_text()


def _declared_modes():
    body = (ROOT / 'ancestry' / '__main__.py').read_text().split('MODES = {')[1].split('}')[0]
    return set(re.findall(r"'(\w+)'\s*:\s*'\w+'", body))


def _case_labels(text):
    """Every `label)` arm of every case statement, with `a|b)` split out."""
    return {m for group in re.findall(r'^\s{4}([\w|]+)\)', text, re.M)
            for m in group.split('|')}


@pytest.mark.parametrize('mode', sorted(_declared_modes()))
def test_every_mode_is_routed_by_the_wrapper(mode):
    assert mode in _case_labels(_runner_text()), (
        f'`{mode}` is in MODES but no case arm of run_ancestry.bash matches it, so '
        f'`--mode {mode}` falls through to "Invalid mode"')


@pytest.mark.parametrize('mode', sorted(_declared_modes()))
def test_every_mode_is_listed_in_the_wrapper_usage(mode):
    usage = _runner_text().split('usage() {')[1].split('}')[0]
    assert re.search(rf'\b{mode}\b', usage), f'`{mode}` is a mode but usage() does not list it'


def test_the_wrapper_advertises_nothing_it_cannot_run():
    """The pipe-separated list in usage() must name only real modes.

    Parsed from the `--mode` listing itself -- the span between the `--mode` column and the
    `(default:` note -- rather than from the whole block, which also contains prose.
    """
    usage = _runner_text().split('usage() {')[1].split('\n}')[0]
    listing = usage.split('--mode', 1)[1].split('(default:')[0]
    advertised = {w for w in (tok.strip(' "|\\\n') for tok in listing.split('|')) if w.isalpha()}
    assert advertised, 'the usage() mode listing could not be parsed'
    unknown = advertised - (_declared_modes() | {'all'})
    assert not unknown, f'usage() advertises modes that do not exist: {sorted(unknown)}'
