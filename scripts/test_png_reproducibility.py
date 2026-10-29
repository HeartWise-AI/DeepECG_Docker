"""Build once, run the 25 PNG fixtures, and compare all SigLIP2 heads with a reference.

Only use synthetic or approved de-identified ECG pages. Artifacts stay in --work-dir.
The normal pipeline configuration and file extensions select the image models; the test does
not pass a PNG mode or image-model flag.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


TASKS = {
    'ecg_machine_diagnosis': ('77_classes_ecg_file_name', 'deepecg_siglip2_77_classes', 77),
    'afib_5y': ('afib_ecg_file_name', 'deepecg_siglip2_afib_5y', 1),
    'lvef_40': ('lvef_40_ecg_file_name', 'deepecg_siglip2_lvef_equal_under_40', 1),
    'lvef_50': ('lvef_50_ecg_file_name', 'deepecg_siglip2_lvef_under_50', 1),
    'shd': ('shd_ecg_file_name', 'deepecg_siglip2_shd', 1),
}
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg'}
BERT_MODEL = 'Bert_diagnosis2classification_En_Fr'


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def validate_input_csv(path, images, count):
    """Validate one fixed, basename-only image cohort used by every task."""
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {
            name
            for task, (file_column, _, _) in TASKS.items()
            for name in (task, file_column)
        }
        if (reader.fieldnames is None or not required.issubset(reader.fieldnames)
                or len(reader.fieldnames) != len(set(reader.fieldnames))):
            raise ValueError('Input CSV must contain every PNG task and filename column')
        rows = list(reader)
    if len(rows) != count:
        raise ValueError('Input CSV must contain the requested number of ECG pages')
    files = []
    for row in rows:
        if not (row.get('ecg_machine_diagnosis') or '').strip():
            raise ValueError('PNG reproducibility rows require diagnosis text for the 77-class test')
        filename = row['77_classes_ecg_file_name']
        if (not filename or Path(filename).name != filename
                or Path(filename).suffix.lower() not in IMAGE_SUFFIXES):
            raise ValueError('ECG page filenames must be PNG or JPEG basenames')
        for task, (column, _, _) in TASKS.items():
            if row[column] != filename:
                raise ValueError('All tasks must use the same ECG page in each row')
            label = (row[task] or '').strip()
            if task != 'ecg_machine_diagnosis' and label not in ('', '0', '1'):
                raise ValueError('Binary outcome columns must be empty, 0 or 1')
        image = images / filename
        if image.is_symlink() or not image.is_file():
            raise ValueError('Input CSV references a missing ECG page or a symlink')
        files.append(image)
    if len({image.name for image in files}) != count:
        raise ValueError('Input CSV must contain unique ECG pages')
    return files


def read_predictions(path, expected_names, width):
    with path.open(newline='') as stream:
        reader = csv.reader(stream)
        columns = next(reader, [])
        if width == 77:
            valid = (len(columns) == 155 and columns[0] == 'file_name'
                     and all(column.endswith('_bert_model') for column in columns[1:78])
                     and all(column.endswith('_sig_model') for column in columns[78:])
                     and [column[:-11] for column in columns[1:78]]
                     == [column[:-10] for column in columns[78:]])
            probability_start = 1
        else:
            valid = columns == ['file_name', 'ground_truth', 'predictions']
            probability_start = 2
        if not valid or len(set(columns)) != len(columns):
            raise ValueError('Wrong PNG prediction schema')
        rows = {}
        for row in reader:
            if len(row) != len(columns) or row[0] in rows:
                raise ValueError('Duplicate ECG page or malformed prediction row')
            values = tuple(float(value) for value in row[probability_start:])
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                raise ValueError('Non-finite or out-of-range PNG prediction')
            rows[row[0]] = values
    if set(rows) != set(expected_names):
        raise ValueError('Missing or unexpected ECG page predictions')
    return [columns[0], *columns[probability_start:]], rows


def capture_run(folder, names):
    """Read the five image-model outputs and assert that no signal preprocessing occurred."""
    preprocessing = folder / 'preprocessing'
    if any(preprocessing.iterdir()):
        raise ValueError('PNG rows unexpectedly created signal preprocessing artifacts')
    outputs = folder / 'outputs'
    if len(list(outputs.glob('*_probabilities.csv'))) != len(TASKS):
        raise ValueError('Expected predictions from all five SigLIP2 heads')
    if list(outputs.glob('missing_files_*.csv')) or list(outputs.glob('unreadable_images_*.csv')):
        raise ValueError('Pipeline reported missing or unreadable ECG pages')
    models = {}
    for task, (_, model, width) in TASKS.items():
        matches = list(outputs.glob(f'{model}_*_{task}_probabilities.csv'))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one predictions file for {model}')
        columns, rows = read_predictions(matches[0], names, width)
        if width == 77:
            models[BERT_MODEL] = {
                'columns': columns[1:78],
                'predictions': {name: list(values[:77]) for name, values in rows.items()},
            }
            columns = columns[78:]
            rows = {name: values[77:] for name, values in rows.items()}
        else:
            columns = columns[1:]
        models[model] = {
            'columns': columns,
            'predictions': {name: list(values) for name, values in rows.items()},
        }
    return {
        'environment': json.loads((folder / 'environment.json').read_text()),
        'models': models,
    }


def compare_reference(reference, folder, names, atol=0.0):
    if not math.isfinite(atol) or atol < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    actual = capture_run(folder, names)
    if set(reference['models']) != set(actual['models']):
        raise ValueError('Reference must contain BERT and all five SigLIP2 models')
    results = {}
    for model, current in actual['models'].items():
        saved = reference['models'][model]
        if saved['columns'] != current['columns']:
            raise ValueError(f'Reference prediction labels do not match {model}')
        if saved['predictions'].keys() != current['predictions'].keys():
            raise ValueError(f'Reference ECG pages do not match {model}')
        differences = []
        for name, values in current['predictions'].items():
            previous = saved['predictions'][name]
            if (len(previous) != len(values)
                    or not all(isinstance(value, (int, float)) and math.isfinite(value)
                               and 0 <= value <= 1 for value in previous)):
                raise ValueError(f'Invalid reference probabilities for {model}')
            differences.extend(abs(first - second) for first, second in zip(previous, values))
        results[model] = {
            'ecgs': len(names),
            'probabilities_per_ecg': len(current['columns']),
            'max_absolute_difference': max(differences),
            'differing_values': sum(value > atol for value in differences),
            'pass': all(value <= atol for value in differences),
        }
    return {
        'pass': all(result['pass'] for result in results.values()),
        'environment_identical': reference['environment'] == actual['environment'],
        'absolute_tolerance': atol,
        'models': results,
    }


def command(arguments, log):
    with log.open('w') as output:
        result = subprocess.run(arguments, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed (exit {result.returncode}); see {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png-dir', type=Path,
                        default=Path('/mnt/data1/datasets/unit_tests/ECGs/png'),
                        help='Test ECG page folder')
    parser.add_argument('--input-csv', type=Path, required=True,
                        help='CSV with one ECG page per row and every task column')
    parser.add_argument('--reference', type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / 'tests/fixtures/reproducibility/data_rows_png_reproducibility_25.reference.json')
    parser.add_argument('--record-reference', action='store_true',
                        help='Create --reference once; refuses to overwrite an existing file')
    parser.add_argument('--api-key-file', type=Path, default=Path('api_key.json'))
    parser.add_argument('--work-dir', type=Path, required=True,
                        help='New private artifact directory')
    parser.add_argument('--count', type=int, default=25)
    parser.add_argument('--gpu', help='Host GPU index; omit for CPU')
    parser.add_argument('--atol', type=float, default=0.0)
    args = parser.parse_args()
    if args.count <= 0 or not math.isfinite(args.atol) or args.atol < 0:
        parser.error('count must be positive and atol finite and nonnegative')

    source = Path(__file__).resolve().parents[1]
    images = args.png_dir.resolve(strict=True)
    sys.path.insert(0, str(source))
    from utils.credentials import read_api_key
    read_api_key(args.api_key_file)
    key = args.api_key_file.resolve(strict=True)
    input_csv = args.input_csv.resolve(strict=True)
    files = validate_input_csv(input_csv, images, args.count)
    reference_path = args.reference.resolve() if args.record_reference else args.reference.resolve(strict=True)
    if args.record_reference and reference_path.exists():
        raise FileExistsError(f'Refusing to overwrite reference: {reference_path}')
    reference = None if args.record_reference else json.loads(reference_path.read_text())
    if reference is not None:
        if reference.get('schema_version') != 1:
            raise ValueError('Unsupported PNG reference format')
        if sha256(input_csv) != reference['input_csv_sha256']:
            raise ValueError('Input CSV differs from the saved PNG reference')
        if {path.name: sha256(path) for path in files} != reference['pngs']:
            raise ValueError('Input ECG pages differ from the saved PNG reference')

    os.umask(0o077)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    context = work / 'build-context'
    context.mkdir()
    for name in ('dockerfile', 'requirements.txt', 'heartwise.config', 'main.py',
                 'models_setup.py', 'run_pipeline.bash'):
        shutil.copyfile(source / name, context / name)
    for folder in ('data', 'models', 'utils', 'thresholds'):
        (context / folder).mkdir()
        for path in (source / folder).iterdir():
            if path.suffix in ('.py', '.json'):
                shutil.copyfile(path, context / folder / path.name)
    for path in context.rglob('*'):
        path.chmod(0o755 if path.is_dir() else 0o644)
    checks = work / 'checks'
    checks.mkdir()
    for name in ('png_repro_worker.py', 'test_png_reproducibility.py'):
        shutil.copyfile(source / 'scripts' / name, checks / name)
    shutil.copyfile(input_csv, work / 'inputs.csv')
    if reference is not None:
        shutil.copyfile(reference_path, work / 'reference.json')

    input_hash = sha256(input_csv)
    png_hashes = {path.name: sha256(path) for path in files}
    provenance = {
        'source_commit': subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True
        ).strip(),
        'source_files': {
            str(path.relative_to(context)): sha256(path)
            for path in sorted(context.rglob('*')) if path.is_file()
        },
        'test_files': {path.name: sha256(path) for path in sorted(checks.iterdir())},
        'input_csv_sha256': input_hash,
        'pngs': png_hashes,
    }
    (work / 'provenance.json').write_text(json.dumps(provenance, indent=2))

    image = 'deepecg-png-repro:' + hashlib.sha256(str(work).encode()).hexdigest()[:12]
    print('Building image without cached layers...', flush=True)
    command(['docker', 'build', '--pull', '--no-cache', '-t', image, str(context)],
            work / 'build.log')
    (work / 'image.json').write_text(
        subprocess.check_output(['docker', 'image', 'inspect', image], text=True)
    )
    (work / 'weights').mkdir()
    base = [
        'docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}',
        '--env', 'HF_HOME=/tmp/huggingface', '--env', 'HF_HUB_DISABLE_TELEMETRY=1',
        '--env', 'PYTHONHASHSEED=2026', '--env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
        '--volume', f'{checks}:/checks:ro',
        '--volume', f'{key}:/run/hf-key.json:ro', '--entrypoint', 'python',
    ]
    print('Downloading the pinned BERT, encoder, and five classifier snapshots...', flush=True)
    command([
        *base, '--volume', f'{work}:/work', '--volume', f'{work / "weights"}:/app/weights',
        image, '/checks/png_repro_worker.py', 'prepare',
    ], work / 'models.log')
    model_manifest = json.loads((work / 'models.json').read_text())
    if reference is not None and model_manifest != reference['model_manifest']:
        raise ValueError('Downloaded PNG model snapshots differ from the saved reference')

    folder = work / 'run'
    folder.mkdir()
    for name in ('preprocessing', 'outputs'):
        (folder / name).mkdir()
    print(f'Running BERT and all five SigLIP2 heads on {len(files)} ECG pages...', flush=True)
    gpu = ['--gpus', f'device={args.gpu}'] if args.gpu is not None else []
    device = 'cuda:0' if args.gpu is not None else 'cpu'
    command([
        *base, *gpu, '--network', 'none', '--env', 'HF_HUB_OFFLINE=1',
        '--volume', f'{images}:/ecgs:ro',
        '--volume', f'{work / "weights"}:/app/weights:ro',
        '--volume', f'{work / "inputs.csv"}:/input.csv:ro',
        '--volume', f'{folder}:/run-output',
        image, '/checks/png_repro_worker.py', 'run',
        '--mode', 'full_run', '--data_path', '/input.csv',
        '--ecg_signals_path', '/ecgs', '--output_folder', '/run-output/outputs',
        '--preprocessing_folder', '/run-output/preprocessing',
        '--hugging_face_api_key_path', '/run/hf-key.json', '--batch_size', '8',
        '--preprocessing_n_workers', '1', '--signal_processing_device', device,
        '--diagnosis_classifier_device', device,
    ], folder / 'pipeline.log')

    if {path.name: sha256(path) for path in files} != png_hashes:
        raise ValueError('Input ECG pages changed during the test')
    if sha256(work / 'inputs.csv') != input_hash:
        raise ValueError('Input CSV changed during the test')
    actual = capture_run(folder, [path.name for path in files])
    if args.record_reference:
        document = {
            'schema_version': 1,
            'input_csv_sha256': input_hash,
            'pngs': png_hashes,
            'model_manifest': model_manifest,
            **actual,
            'source_run': f'{work.name}/run',
        }
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        with reference_path.open('x') as stream:
            json.dump(document, stream, indent=2)
            stream.write('\n')
        report = {'pass': True, 'recorded_reference': str(reference_path)}
    else:
        report = compare_reference(reference, folder, [path.name for path in files], args.atol)
    report['input_csv_sha256'] = input_hash
    (work / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f'Artifacts: {work}')
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
