"""Container-side preparation and deterministic invocation of the PNG pipeline."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys


MODEL_KEYS = (
    'Bert_diagnosis2classification_En_Fr',
    'siglip2_base_encoder',
    'siglip2_77_classes',
    'siglip2_afib_5y',
    'siglip2_lvef_equal_under_40',
    'siglip2_lvef_under_50',
    'siglip2_shd',
)


def digest(path):
    with path.open('rb') as stream:
        if hasattr(hashlib, 'file_digest'):
            return hashlib.file_digest(stream, 'sha256').hexdigest()
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
        return result.hexdigest()


def prepare():
    from utils.credentials import read_api_key
    from utils.huggingface_wrapper import HuggingFaceWrapper
    from utils.model_manifest import load_manifest

    token = read_api_key('/run/hf-key.json')['HUGGING_FACE_API_KEY']
    configured = load_manifest()
    missing = sorted(set(MODEL_KEYS) - set(configured))
    if missing:
        raise ValueError(f'PNG reproducibility models are absent from the manifest: {missing}')
    snapshots = {}
    for key in MODEL_KEYS:
        entry = configured[key]
        folder = Path(HuggingFaceWrapper.get_model(
            entry['repository'], Path('/app/weights') / key, token
        ))
        snapshots[key] = {
            'repository': entry['repository'],
            'revision': entry['revision'],
            'files': {
                str(path.relative_to(folder)): digest(path)
                for path in sorted(folder.rglob('*'))
                if path.is_file()
                and '.cache' not in path.relative_to(folder).parts
                and path.name != '.deepecg-complete.json'
            },
        }
        print(f'Pinned model {key}', flush=True)
    Path('/work/models.json').write_text(json.dumps(snapshots, indent=2))
    Path('/work/packages.txt').write_text(
        subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    )
    revision = subprocess.check_output(
        ['git', '-c', 'safe.directory=/app/fairseq-signals', '-C', '/app/fairseq-signals',
         'rev-parse', 'HEAD'], text=True
    ).strip()
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
    metadata = {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'seed': 2026,
        'deterministic_algorithms': True,
        'python': sys.version,
        'packages': {
            dist.metadata['Name']: dist.version
            for dist in importlib.metadata.distributions()
        },
        'fairseq_commit': subprocess.check_output(
            ['git', '-c', 'safe.directory=/app/fairseq-signals', '-C', '/app/fairseq-signals',
             'rev-parse', 'HEAD'], text=True
        ).strip(),
    }
    Path('/run-output/environment.json').write_text(json.dumps(metadata, indent=2))
    sys.argv = ['main.py', *sys.argv[2:]]
    runpy.run_path('/app/main.py', run_name='__main__')


if __name__ == '__main__':
    sys.path.insert(0, '/app')
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
