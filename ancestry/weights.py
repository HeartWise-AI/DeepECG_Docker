"""Where the model weights come from, in the three situations this pipeline actually runs in.

A partner site is usually air-gapped, so the weights are baked into the image and named by
environment variable. A developer on a laptop has neither, and should get the same artefact the
site has -- which is what the main and ancestry model manifests are for: pinned repositories and
exact 40-character commits, downloaded once through the repository's own HuggingFaceWrapper.

Resolution order, per artefact:

Note on the backbone: the ancestry fold heads and the released 77-finding head sit on ONE shared
SSL backbone, so there is nothing to upload -- it is already `base_ssl.pt` inside the pinned,
PUBLIC `wcr_77_classes` snapshot this repository has always used. Verified rather than assumed,
on 2026-11-10, against the backbone the released fold checkpoints were fine-tuned from:

  sha256 39985e904bfcc4e3fa58f3e631d84598c7228bfa821a3279109dc2e05f54383c, 1090680436 bytes

That matters because a DIFFERENT backbone would not fail -- it would silently change every
score. Anyone repinning `wcr_77_classes` to a new revision must re-check that hash.

  1. an explicit path, from a command-line flag                 (the site's own layout)
  2. the environment variable the image sets                    (the baked-in weights)
  3. the pinned Hugging Face revision in the model manifests    (a clone, with network)

Order matters: 1 and 2 must win, because a site that has been given the weights on a disk must
never be made to reach the network, and a run that silently downloaded a different artefact from
the one audited in MODEL_PROVENANCE.txt would invalidate the validation it is part of.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

# artefact -> (environment variable, manifest key, what to pick inside the snapshot)
#
# `member` is how a downloaded snapshot DIRECTORY becomes the path the loader wants, and the three
# differ on purpose:
#   'base_ssl.pt'  the backbone, which a HeartWise model repository ships under that exact name
#   HEAD           the one head checkpoint beside it -- the name varies by repository, so it is
#                  found the way models/wcr_ecg_transformer.py finds it: the single .pt that is
#                  not base_ssl.pt
#   None           a directory IS the answer: discover_fold_checkpoints() walks the five folds
#
# Without this, falling through to the manifest handed `WCREcgTransformer` a directory where it
# expects a checkpoint file, and the failure arrived from inside fairseq-signals.
HEAD = object()

ARTEFACTS = {
    'ssl_backbone': ('SSL_CKPT', 'wcr_77_classes', 'base_ssl.pt'),
    'findings_head': ('WCR_CKPT', 'wcr_77_classes', HEAD),
    'ancestry_folds': ('MODELS_ROOT', 'deepecg_pop_ancestry', None),
    # The frozen discovery tables travel WITH the weights rather than in this repository, which
    # is public: they are aggregates of UK Biobank, MHI and CLSA participants, and whether they
    # may be released is a data-access question, not a packaging one. They sit under reference/
    # in the same pinned snapshot as the fold checkpoints, so one access grant covers both and
    # the tables can never drift from the model they describe.
    'reference_tables': ('REFERENCE_DIR', 'deepecg_pop_ancestry', 'reference'),
}


class WeightsUnavailable(RuntimeError):
    """Named, actionable: which artefact, which variable to set, which repository it came from."""


# The ancestry snapshot may be staged in a project-specific gated repository before the shared
# manifest entry is pinned, so its name and revision must be settable without editing this
# repository. ANCESTRY_MODEL_REPO / ANCESTRY_MODEL_REVISION (or the matching ancestry.config
# keys, which run_ancestry.bash exports) override the manifest for it and for the reference
# tables that travel inside it. The public heads are NOT overridable this way -- repinning those
# is a change to this repository, reviewed like any other.
OVERRIDABLE = {'deepecg_pop_ancestry'}
SHA = re.compile(r'[0-9a-f]{40}')
ANCESTRY_MANIFEST = Path(__file__).with_name('model_manifest.json')

# Where artefacts may come from, as a deliberate choice rather than only an order of precedence.
# ANCESTRY_WEIGHTS_SOURCE, or `weights_source` in ancestry.config:
#
#   auto         (default) a path given on the command line, then the environment variable, then
#                a snapshot already fetched, then the pinned revision from Hugging Face
#   local        never reach the network. Only an explicit path, an environment variable or a
#                snapshot already on disk. This is what an air-gapped site wants, and it turns a
#                missing mount into an error instead of a download attempt
#   huggingface  ignore the environment variables and use the pinned revision -- the way to be
#                certain a run used the published artefact and not something left in a mount
#
# A path passed directly to a mode (--models-root, --ssl-ckpt) always wins: it is the most
# specific instruction there is, and silently ignoring it would be worse than any of this.
SOURCES = ('auto', 'local', 'huggingface')


def source() -> str:
    s = (os.environ.get('ANCESTRY_WEIGHTS_SOURCE') or 'auto').strip().lower()
    if s not in SOURCES:
        raise WeightsUnavailable(
            f'ANCESTRY_WEIGHTS_SOURCE={s!r} is not one of {", ".join(SOURCES)}')
    return s


def _ancestry_manifest() -> dict:
    data = json.loads(ANCESTRY_MANIFEST.read_text())
    if data.get('schema_version') != 1 or not isinstance(data.get('models'), dict):
        raise WeightsUnavailable(f'invalid ancestry model manifest: {ANCESTRY_MANIFEST}')
    return data['models']


def entry_for(manifest_key: str) -> dict:
    """{'repository', 'revision'} for `manifest_key`, honouring the override for the ancestry snapshot."""
    repo = (os.environ.get('ANCESTRY_MODEL_REPO') or '').strip()
    rev = (os.environ.get('ANCESTRY_MODEL_REVISION') or '').strip()
    if repo and manifest_key in OVERRIDABLE:
        if not SHA.fullmatch(rev):
            raise WeightsUnavailable(
                f'ANCESTRY_MODEL_REPO={repo} was given without an exact 40-character '
                'ANCESTRY_MODEL_REVISION. A branch name would make the run unreproducible, so '
                'it is refused: get the commit with\n'
                f'  huggingface-cli repo info {repo} --repo-type model\n'
                'and set ANCESTRY_MODEL_REVISION to it (or put both in ancestry.config).')
        return {'repository': repo, 'revision': rev}
    if manifest_key in OVERRIDABLE:
        found = _ancestry_manifest().get(manifest_key)
        manifest_path = 'ancestry/model_manifest.json'
    else:
        from utils.model_manifest import load_manifest
        found = load_manifest().get(manifest_key)
        manifest_path = 'models/manifest.json'
    if not found:
        raise WeightsUnavailable(
            f'{manifest_path} has no {manifest_key!r} entry. Either add the pinned '
            'repository and revision, or set ANCESTRY_MODEL_REPO / ANCESTRY_MODEL_REVISION.')
    return found


def resolve(artefact: str, explicit: str | Path | None = None,
            api_key: str = '', download: bool = True) -> Path:
    """The path to `artefact`, by the order in this module's docstring."""
    if artefact not in ARTEFACTS:
        raise KeyError(f'unknown artefact {artefact!r}; known: {sorted(ARTEFACTS)}')
    env_var, manifest_key, member = ARTEFACTS[artefact]
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise WeightsUnavailable(f'{artefact}: {p} does not exist (given explicitly)')
        return p
    src = source()
    if src != 'huggingface' and os.environ.get(env_var):
        p = Path(os.environ[env_var])
        if not p.exists():
            raise WeightsUnavailable(f'{artefact}: {env_var}={p} does not exist')
        return p
    cached = _cached(manifest_key)
    if cached is not None:
        return _member(cached, member, artefact)
    if src == 'local':
        raise WeightsUnavailable(
            f'{artefact}: weights_source is "local", so nothing is downloaded. Set {env_var} to '
            f'the copy on disk, pass the path to the mode, or use weights_source "huggingface" '
            f'(ANCESTRY_WEIGHTS_SOURCE) to fetch the pinned revision.')
    if not download:
        raise WeightsUnavailable(
            f'{artefact}: no path given, {env_var} is not set, and no completed snapshot of '
            f'{manifest_key} is under weights/. Inside the image the snapshots are placed by '
            f'`ancestry weights --download` (or build with --build-arg RUN_MODELS_SETUP=true) '
            'or the variable is '
            f'set to mounted weights; seeing this means neither happened.')
    # A clone with network access: the same pinned artefact the image was built from.
    entry = entry_for(manifest_key)
    token = api_key or _token()
    if os.environ.get('ANCESTRY_MODEL_REPO') and manifest_key in OVERRIDABLE:
        # An overridden repository is not in the manifest, so HuggingFaceWrapper -- which looks
        # the revision up there -- cannot fetch it. Pin it explicitly instead, same layout, so
        # _cached() finds it next time and the run is still tied to one commit.
        from huggingface_hub import snapshot_download
        folder = _snapshot_dir(manifest_key, entry)
        folder.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=entry['repository'], revision=entry['revision'],
                          local_dir=str(folder), repo_type='model', token=token or None)
    else:
        from utils.huggingface_wrapper import HuggingFaceWrapper
        folder = Path(HuggingFaceWrapper.get_model(entry['repository'],
                                                   Path('weights') / manifest_key, token))
    return _member(folder, member, artefact)


def _token(api_key: str = '') -> str:
    """The Hugging Face token, or '' -- a public repository needs none."""
    if api_key:
        return api_key
    if os.environ.get('HUGGING_FACE_API_KEY'):
        return os.environ['HUGGING_FACE_API_KEY']
    try:
        from utils.credentials import read_api_key
        k = read_api_key(os.environ.get('HF_API_KEY_PATH', 'api_key.json')
                         )['HUGGING_FACE_API_KEY']
        if k:
            return k
    except Exception:                                                     # noqa: BLE001
        pass
    # LAST, the token `huggingface-cli login` writes. It is the route the Hugging Face docs give
    # and the one our own UPLOAD.md tells an operator to use, so ignoring it meant a correctly
    # logged-in user was told there was no token at all. It comes last on purpose: an explicit
    # --api-key, the environment and api_key.json still win, so a container that mounts a
    # specific key cannot be overridden by whatever happens to be cached in the home directory.
    try:
        from huggingface_hub import get_token
        return get_token() or ''
    except Exception:                                                     # noqa: BLE001
        return ''


def _snapshot_dir(manifest_key: str, entry: dict) -> Path:
    """The local path HuggingFaceWrapper publishes a snapshot to, reproduced exactly."""
    return (Path('weights') / manifest_key
            / entry['repository'].replace('/', '--') / entry['revision'])


def _cached(manifest_key: str):
    """A completed snapshot already on disk, or None -- without any network access.

    This mirrors HuggingFaceWrapper's own layout and completion marker rather than calling it,
    because calling it would reach the Hub when the snapshot is absent, and the whole point of
    download=False is that a startup check never does that.
    """
    try:
        entry = entry_for(manifest_key)
        folder = _snapshot_dir(manifest_key, entry)
        marker = folder / '.deepecg-complete.json'
        if marker.exists():
            import json
            saved = json.loads(marker.read_text())
            if any(saved.get(k) != v for k, v in entry.items()):
                return None
            files = saved.get('files') or {}
            if files and all((folder / n).is_file() and (folder / n).stat().st_size == s
                             for n, s in files.items()):
                return folder
            return None
        # A snapshot fetched by an explicit override has no marker (snapshot_download writes
        # none). It is still pinned by revision, which is in the path, so presence is enough.
        if folder.is_dir() and any(folder.rglob('*.pt')):
            return folder
        return None
    except Exception:
        return None


def _member(folder: Path, member, artefact: str) -> Path:
    """The path inside a downloaded snapshot that the loader actually opens."""
    if member is None:
        return folder
    if member is HEAD:
        heads = sorted(p for p in folder.glob('*.pt') if p.name != 'base_ssl.pt')
        if len(heads) != 1:
            raise WeightsUnavailable(
                f'{artefact}: expected exactly one head checkpoint beside base_ssl.pt in '
                f'{folder}, found {[p.name for p in heads]}')
        return heads[0]
    path = folder / member
    if not path.exists():
        raise WeightsUnavailable(f'{artefact}: {member} is not in {folder}')
    return path


def report(api_key: str = '', download: bool = False) -> dict[str, str]:
    """For `survey`: every artefact, and where it would be read from. Never downloads by default."""
    out = {}
    for name in ARTEFACTS:
        try:
            out[name] = str(resolve(name, download=download, api_key=api_key))
        except WeightsUnavailable as e:
            out[name] = f'NOT AVAILABLE: {e}'
    return out


# --------------------------------------------------------------------------------- mode `weights`
USAGE = """Mode `weights` -- where the model artefacts come from, and how to put them there.

    python -m ancestry weights                      # report, reaching the network for nothing
    python -m ancestry weights --download           # fetch the pinned snapshot, then verify it
    python -m ancestry weights --upload DIR         # what WOULD be pushed (a dry run)
    python -m ancestry weights --upload DIR --go    # actually push it

The ancestry snapshot carries the five fold heads AND the frozen reference tables, so one fetch
and one access grant serve a whole validation run. Until the shared manifest entry is pinned,
name a staged snapshot with --repo/--revision, ANCESTRY_MODEL_REPO/ANCESTRY_MODEL_REVISION, or
the matching keys in ancestry.config. A revision is always required: a branch name would make a
run unreproducible.
"""


def verify_snapshot(folder):
    """Check a snapshot the way the site will need it: the layout, and every checksum shipped."""
    import hashlib
    folder = Path(folder)
    out = []
    folds = [folder / 'raw' / f'fold_{k}' / 'bs_128' / 'checkpoint_best.pt' for k in range(1, 6)]
    missing = [str(f.relative_to(folder)) for f in folds if not f.is_file()]
    out.append(('the five fold checkpoints', not missing, ', '.join(missing) or 'all present'))
    ref = folder / 'reference'
    man = ref / 'MANIFEST.txt'
    if man.is_file():
        bad = []
        for line in man.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            f = ref / parts[-1]
            if not f.is_file():
                bad.append(f'{parts[-1]} missing')
            else:
                try:
                    digest = hashlib.md5(f.read_bytes()).hexdigest()
                except OSError as error:
                    bad.append(f'{parts[-1]} unreadable ({error.__class__.__name__})')
                else:
                    if digest != parts[0]:
                        bad.append(f'{parts[-1]} changed')
        out.append(('the reference tables match MANIFEST.txt', not bad,
                    '; '.join(bad) or 'all match'))
    else:
        out.append(('reference/MANIFEST.txt', False, 'missing -- no tables in this snapshot'))
    sums = next((folder / name for name in ('CHECKSUMS.md5', 'CHECKSUMS.txt')
                 if (folder / name).is_file()), folder / 'CHECKSUMS.md5')
    if sums.is_file():
        bad = []
        for line in sums.read_text().splitlines():
            parts = line.split()
            if len(parts) < 2 or parts[-1] in {'CHECKSUMS.md5', 'CHECKSUMS.txt'}:
                continue
            f = folder / parts[-1]
            if not f.is_file():
                bad.append(f'{parts[-1]} missing')
            else:
                try:
                    h = hashlib.md5()
                    with f.open('rb') as fh:
                        for b in iter(lambda: fh.read(1 << 22), b''):
                            h.update(b)
                except OSError as error:
                    bad.append(f'{parts[-1]} unreadable ({error.__class__.__name__})')
                else:
                    if h.hexdigest() != parts[0]:
                        bad.append(f'{parts[-1]} changed')
        out.append((sums.name, not bad, '; '.join(bad[:4]) or 'every file matches'))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(prog='python -m ancestry weights', description=USAGE,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--download', action='store_true', help='fetch the pinned ancestry snapshot')
    ap.add_argument('--upload', metavar='DIR', default=None,
                    help='a staged directory to publish (dry run unless --go)')
    ap.add_argument('--go', action='store_true', help='with --upload: actually push')
    ap.add_argument('--repo', default=None,
                    help='repository id, e.g. heartwise/DeepECG-SSL_Ancestry')
    ap.add_argument('--revision', default=None, help='the exact 40-character commit to pin')
    ap.add_argument('--public', action='store_true',
                    help='with --upload --go: create the repository PUBLIC. The default is '
                         'private, because these weights are derived from participant recordings')
    ap.add_argument('--api-key', default='', help='Hugging Face token; otherwise api_key.json')
    ap.add_argument('--weights-source', choices=SOURCES, default=None,
                    help='auto (default), local (never reach the network), or huggingface '
                         '(ignore the environment variables and use the pinned revision)')
    ap.add_argument('--verify', metavar='DIR', default=None,
                    help='check a snapshot already on disk and exit')
    a = ap.parse_args()
    if a.repo:
        os.environ['ANCESTRY_MODEL_REPO'] = a.repo
    if a.revision:
        os.environ['ANCESTRY_MODEL_REVISION'] = a.revision
    if a.weights_source:
        os.environ['ANCESTRY_WEIGHTS_SOURCE'] = a.weights_source

    if a.verify:
        print(f'\n  verifying {a.verify}\n')
        rows = verify_snapshot(a.verify)
        for what, ok, note in rows:
            print('    %-42s %s  %s' % (what, 'ok  ' if ok else 'FAIL', note))
        raise SystemExit(0 if all(ok for _, ok, _ in rows) else 1)

    if a.upload:
        src = Path(a.upload)
        if not src.is_dir():
            raise SystemExit(f'{src} is not a directory')
        # The upload path needs the repository NAME, never a revision: the commit it would pin
        # does not exist until this push creates it. entry_for() demands one as soon as
        # ANCESTRY_MODEL_REPO is set, which made `--upload --repo X --go` refuse itself and sent
        # the operator looking for a placeholder sha to satisfy a check that does not apply here.
        if a.repo:
            repo = a.repo
        else:
            found = _ancestry_manifest().get('deepecg_pop_ancestry') or {}
            repo = found.get('repository')
            if not repo:
                raise SystemExit(
                    'no repository to upload to: pass --repo ORG/NAME, or add a '
                    "'deepecg_pop_ancestry' entry to ancestry/model_manifest.json")
        files = sorted(q for q in src.rglob('*') if q.is_file())
        skip = {'UPLOAD.md', 'verify_before_upload.sh'}
        send = [q for q in files if q.name not in skip]
        print(f'\n  repository : {repo}   ({"PUBLIC" if a.public else "private"})')
        print(f'  from       : {src}')
        print(f'  {len(send)} files, {sum(q.stat().st_size for q in send) / 2**30:.2f} GiB'
              f'   (skipping {", ".join(sorted(skip))})')
        rows = verify_snapshot(src)
        print()
        for what, ok, note in rows:
            print('    %-42s %s  %s' % (what, 'ok  ' if ok else 'FAIL', note))
        if not all(ok for _, ok, _ in rows):
            raise SystemExit('\n  refusing to upload a directory that does not verify.')
        if not a.go:
            print('\n  dry run. Add --go to push.'
                  '\n  The default is a PRIVATE repository; --public is the only way to make it '
                  'open, and it cannot be undone for anyone who has already cloned it.')
            return
        from huggingface_hub import HfApi
        token = _token(a.api_key)
        if not token:
            raise SystemExit('no Hugging Face token: pass --api-key, set HUGGING_FACE_API_KEY, '
                             'or put one in api_key.json')
        api = HfApi()
        api.create_repo(repo_id=repo, repo_type='model', private=not a.public, exist_ok=True,
                        token=token)
        api.upload_folder(repo_id=repo, folder_path=str(src), repo_type='model', token=token,
                          ignore_patterns=sorted(skip),
                          commit_message='DeepECG-Pop ancestry ensemble + frozen reference')
        sha = api.model_info(repo, token=token).sha
        print(f'\n  pushed. revision {sha}')
        print('  pin it: ancestry/model_manifest.json -> "deepecg_pop_ancestry": '
              f'{{"repository": "{repo}", "revision": "{sha}"}}')
        print('  or, without editing the repository:')
        print(f'    export ANCESTRY_MODEL_REPO={repo}')
        print(f'    export ANCESTRY_MODEL_REVISION={sha}')
        return

    if a.download:
        folder = resolve('ancestry_folds', api_key=a.api_key, download=True)
        backbone = resolve('ssl_backbone', api_key=a.api_key, download=True)
        findings = resolve('findings_head', api_key=a.api_key, download=True)
        print(f'\n  snapshot: {folder}\n')
        rows = verify_snapshot(folder)
        for what, ok, note in rows:
            print('    %-42s %s  %s' % (what, 'ok  ' if ok else 'FAIL', note))
        if not all(ok for _, ok, _ in rows):
            raise SystemExit('\n  the snapshot is incomplete; delete it and fetch again.')
        print('\n  for the validation run:')
        print(f'    export MODELS_ROOT={folder}')
        print(f'    export REFERENCE_DIR={folder}/reference')
        print(f'    export SSL_CKPT={backbone}')
        print(f'    export WCR_CKPT={findings}')
        return

    print(f'\n  weights_source: {source()}'
          '   (auto | local = never download | huggingface = ignore the mounts)')
    print('\n  where each artefact resolves from, right now (no network):\n')
    for name in ARTEFACTS:
        env_var, key, _ = ARTEFACTS[name]
        try:
            where = str(resolve(name, download=False, api_key=a.api_key))
        except WeightsUnavailable as e:
            where = f'NOT AVAILABLE -- {str(e).splitlines()[0]}'
        try:
            ent = entry_for(key)
            # Shorten a commit sha, but print anything else whole: truncating the
            # PENDING-UPLOAD placeholder to PENDING-UPLO made a deliberate marker look
            # like a corrupted value.
            rv = ent['revision']
            rv = rv[:12] if re.fullmatch(r'[0-9a-f]{40}', rv) else rv
            pin = f"{ent['repository']} @ {rv}"
        except WeightsUnavailable as e:
            pin = f'unpinned ({str(e).splitlines()[0][:60]})'
        print(f'    {name:<17} {env_var:<14} {pin}\n    {"":<32} -> {where}\n')
    print('  --download fetches what is missing; --upload DIR publishes a staged snapshot.')
