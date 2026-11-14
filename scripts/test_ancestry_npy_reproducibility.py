"""Compare two Ancestry Docker images on the approved 25-NPY NAS cohort."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'scripts' / 'ancestry_npy_worker.py'


def image_id(image: str) -> str:
    """Resolve a mutable local tag to the immutable image ID recorded in the report."""
    return subprocess.check_output(
        ['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True).strip()


def run_image(image: str, label: str, args, work: Path, legacy: bool) -> dict:
    output = work / f'{label}.json'
    log = work / f'{label}.log'
    command = [
        'docker', 'run', '--rm', '--network', 'none',
        '--entrypoint', 'python',
        '-v', f'{WORKER}:/app/ancestry_npy_worker.py:ro',
        '-v', f'{args.input_dir}:/inputs:ro',
        '-v', f'{args.models_root}:/models:ro',
        '-v', f'{work}:/comparison',
        image, '/app/ancestry_npy_worker.py',
        '--input-dir', '/inputs',
        '--models-root', '/opt/ancestry/models' if legacy else '/models',
        '--ssl-ckpt', ('/opt/ancestry/checkpoint_last.pt' if legacy
                       else '/comparison/checkpoint_last.pt'),
        '--output', f'/comparison/{label}.json',
        '--count', str(args.count),
        '--input-scale', str(args.input_scale),
        '--device', args.device,
        '--batch-size', str(args.batch_size),
    ]
    # The deployed legacy image made its fairseq-signals tree root-only. Preserve its native
    # runtime user for the baseline; exercise the candidate under the caller's non-root UID.
    if not legacy:
        command[3:3] = ['--user', f'{os.getuid()}:{os.getgid()}']
    if legacy:
        command.extend(['--legacy-module-root', '/opt/ancestry/app'])
    if args.device.startswith('cuda'):
        command[3:3] = ['--gpus', 'all']
    with log.open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{label} image failed with exit {result.returncode}; see {log}')
    return json.loads(output.read_text())


def compare(baseline: dict, candidate: dict, tolerance: float) -> dict:
    for field in ('schema_version', 'input_scale', 'inputs_sha256', 'model_files'):
        if baseline[field] != candidate[field]:
            raise ValueError(f'comparison metadata differs for {field}')
    if baseline['predictions'].keys() != candidate['predictions'].keys():
        raise ValueError('the images did not score the same NPY files')

    differences = []
    for name in baseline['predictions']:
        old = baseline['predictions'][name]
        new = candidate['predictions'][name]
        differences.append(abs(old['ensemble'] - new['ensemble']))
        if len(old['folds']) != len(new['folds']):
            raise ValueError(f'fold count differs for {name}')
        differences.extend(abs(a - b) for a, b in zip(old['folds'], new['folds']))
    return {
        'pass': all(value <= tolerance for value in differences),
        'absolute_tolerance': tolerance,
        'ecgs': len(baseline['predictions']),
        'values_compared': len(differences),
        'max_absolute_difference': max(differences, default=0.0),
        'differing_values': sum(value > tolerance for value in differences),
        'baseline_environment': baseline['environment'],
        'candidate_environment': candidate['environment'],
    }


def extract_backbone(image: str, work: Path) -> None:
    """Copy the readable, provenance-matched backbone out of the deployed baseline image."""
    copy_command = [
        'docker', 'run', '--rm', '--network', 'none',
        '--entrypoint', 'cp',
        '-v', f'{work}:/comparison',
        image,
        '/opt/ancestry/checkpoint_last.pt', '/comparison/checkpoint_last.pt',
    ]
    subprocess.run(copy_command, check=True, stdout=subprocess.DEVNULL)
    chmod_command = [
        'docker', 'run', '--rm', '--network', 'none',
        '--entrypoint', 'chmod',
        '-v', f'{work}:/comparison',
        image, '0444', '/comparison/checkpoint_last.pt',
    ]
    subprocess.run(chmod_command, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-image', required=True)
    parser.add_argument('--candidate-image', required=True)
    parser.add_argument('--input-dir', type=Path,
                        default=Path('/mnt/data1/datasets/unit_tests/ECGs/npy'))
    parser.add_argument('--models-root', type=Path, required=True,
                        help='approved local/NAS ancestry model bundle')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=25)
    parser.add_argument('--input-scale', type=float, default=0.00488,
                        help='documented raw-ADC to millivolt conversion; recorded in the report')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument(
        '--atol', type=float, default=1e-6,
        help='absolute probability tolerance (default: 1e-6; use 0 for bitwise comparison)')
    args = parser.parse_args()

    if args.work_dir.exists():
        parser.error('work-dir must not already exist')
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error('atol must be finite and nonnegative')
    args.input_dir = args.input_dir.resolve(strict=True)
    args.models_root = args.models_root.resolve(strict=True)
    if not (args.models_root / 'shared' / 'checkpoint_last.pt').is_file():
        parser.error('models-root is missing shared/checkpoint_last.pt')
    args.work_dir.mkdir(parents=True)

    extract_backbone(args.baseline_image, args.work_dir)
    baseline = run_image(args.baseline_image, 'baseline', args, args.work_dir, legacy=True)
    candidate = run_image(args.candidate_image, 'candidate', args, args.work_dir, legacy=False)
    report = compare(baseline, candidate, args.atol)
    report.update({'baseline_image': args.baseline_image,
                   'baseline_image_id': image_id(args.baseline_image),
                   'candidate_image': args.candidate_image,
                   'candidate_image_id': image_id(args.candidate_image)})
    (args.work_dir / 'report.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    if not report['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
