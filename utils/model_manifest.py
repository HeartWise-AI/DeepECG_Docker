"""Immutable model revisions shared by setup, inference and reproducibility checks."""
import json
from pathlib import Path
import re

MANIFEST_PATH = Path(__file__).resolve().parents[1] / 'models' / 'manifest.json'


class ModelPreparationError(RuntimeError):
    """A safe, actionable model preparation error, without credential details."""


def load_manifest():
    with MANIFEST_PATH.open() as stream:
        manifest = json.load(stream)
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('models'), dict):
        raise ModelPreparationError('Invalid models/manifest.json schema.')
    return manifest['models']


def model_entry(repo_id):
    entries = [entry for entry in load_manifest().values() if entry.get('repository') == repo_id]
    if len(entries) != 1:
        raise ModelPreparationError(f'{repo_id}: expected one entry in models/manifest.json.')
    entry = entries[0]
    if not isinstance(entry.get('revision'), str) or not re.fullmatch(r'[0-9a-f]{40}', entry['revision']):
        raise ModelPreparationError(f'{repo_id}: manifest revision must be an exact 40-character commit SHA.')
    return entry
