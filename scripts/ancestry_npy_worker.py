"""Score a fixed NPY cohort with the standalone DeepECG-Pop ensemble.

This worker is intended for container-to-container regression comparisons. It records input
hashes and raw fold probabilities, but it does not interpret ancestry or clinical performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--models-root', type=Path, required=True)
    parser.add_argument('--ssl-ckpt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=25)
    parser.add_argument('--input-scale', type=float, default=1.0,
                        help='multiply loaded samples by this documented unit conversion')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--legacy-module-root', type=Path,
                        help='load infer_lib.py from the original deployed Ancestry image')
    args = parser.parse_args()

    if args.count <= 0 or args.batch_size <= 0:
        parser.error('count and batch-size must be positive')
    if not math.isfinite(args.input_scale) or args.input_scale <= 0:
        parser.error('input-scale must be finite and positive')

    files = sorted(args.input_dir.glob('*.npy'))
    if len(files) != args.count:
        raise SystemExit(f'expected {args.count} NPY files in {args.input_dir}, found {len(files)}')
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise SystemExit('the comparison cohort must contain regular, non-symlink NPY files')

    signals = []
    for path in files:
        signal = np.load(path, allow_pickle=False).squeeze()
        if signal.ndim != 2 or 12 not in signal.shape:
            raise SystemExit(f'{path.name}: expected a 12-lead array, got {signal.shape}')
        if signal.shape[0] == 12:
            signal = signal.T
        if signal.shape[0] < 2500 or signal.shape[0] % 2500:
            raise SystemExit(f'{path.name}: sample count must be a positive multiple of 2500')
        signals.append(signal[::signal.shape[0] // 2500][:2500])
    signals = np.stack(signals).astype(np.float32, copy=False)
    signals *= np.float32(args.input_scale)

    if args.legacy_module_root:
        sys.path.insert(0, str(args.legacy_module_root))
        import infer_lib as infer  # noqa: PLC0415
    else:
        from ancestry import infer  # noqa: PLC0415
    device = infer.pick_device(args.device)
    models, checkpoints = infer.load_fold_models(
        args.models_root, args.ssl_ckpt, device, n_folds=5)
    ensemble, folds = infer.ensemble_predict(
        models, signals, device, batch_size=args.batch_size, check=True)

    predictions = {}
    for index, path in enumerate(files):
        fold_values = [float(value) for value in folds[index]]
        value = float(ensemble[index])
        if not math.isfinite(value) or not all(math.isfinite(x) for x in fold_values):
            raise SystemExit(f'non-finite prediction for {path.name}')
        predictions[path.name] = {'ensemble': value, 'folds': fold_values}

    result = {
        'schema_version': 1,
        'input_scale': args.input_scale,
        'inputs_sha256': {path.name: sha256(path) for path in files},
        'model_files': [str(Path(path).relative_to(args.models_root)) for path in checkpoints],
        'environment': {
            'python_device': str(device),
            'numpy': np.__version__,
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
        },
        'predictions': predictions,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
