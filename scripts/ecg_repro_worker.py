"""Container-side preparation and deterministic invocation of the real CLI."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(stream)


def _digest(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(chunk)
    return result.hexdigest()


def prepare():
    from utils.huggingface_wrapper import HuggingFaceWrapper
    from utils.credentials import read_api_key
    from utils.model_manifest import model_entry
    from utils.constants import MODEL_MAPPING

    token = read_api_key('/run/hf-key.json')['HUGGING_FACE_API_KEY']
    previous = json.loads(Path('/work/reference.json').read_text())['model_manifest']
    manifest = {}
    names = {name for config in MODEL_MAPPING.values() for name in config.values() if name is not None}
    if names != set(previous):
        raise ValueError('Configured models do not match the reference')
    for name in sorted(names):
        repo = f'heartwise/{name}'
        revision = previous[name]['revision']
        if model_entry(repo)['revision'] != revision:
            raise ValueError(f'Manifest revision differs from the reference for {name}')
        folder = Path(HuggingFaceWrapper.get_model(repo, Path('/app/weights') / name, token))
        manifest[name] = {
            'repository': repo, 'revision': revision,
            'files': {str(p.relative_to(folder)): digest(p)
                      for p in sorted(folder.rglob('*'))
                      if p.is_file() and '.cache' not in p.relative_to(folder).parts
                      and p.name != '.deepecg-complete.json'},
        }
        if manifest[name] != previous[name]:
            raise ValueError(f'Downloaded files do not match the reference for {name}')
        Path('/work/models.json').write_text(json.dumps(manifest, indent=2))
        print(f'Pinned model {name}', flush=True)
    packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    Path('/work/packages.txt').write_text(packages)
    revision = subprocess.check_output(['git', '-c', 'safe.directory=/app/fairseq-signals', '-C', '/app/fairseq-signals', 'rev-parse', 'HEAD'], text=True).strip()
    Path('/work/fairseq-revision.txt').write_text(revision + '\n')


def run():
    import numpy as np
    import torch

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(),
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'seed': 2026, 'deterministic_algorithms': True,
                'python': sys.version,
                'packages': {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()},
                'fairseq_commit': subprocess.check_output(['git', '-c', 'safe.directory=/app/fairseq-signals', '-C', '/app/fairseq-signals', 'rev-parse', 'HEAD'], text=True).strip()}
    Path('/run-output/environment.json').write_text(json.dumps(metadata, indent=2))
    sys.argv = ['main.py', *sys.argv[2:]]
    runpy.run_path('/app/main.py', run_name='__main__')


if __name__ == '__main__':
    sys.path.insert(0, '/app')
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
