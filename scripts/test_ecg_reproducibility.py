"""Build Docker once, run 50 ECGs once, and compare with a saved reference.

Only use synthetic or approved de-identified ECGs. Artifacts stay in --work-dir.
This verifies prediction repeatability on one environment, not clinical accuracy.
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
    'ecg_machine_diagnosis': ('77_classes_ecg_file_name', '77_classes', 77),
    'afib_5y': ('afib_ecg_file_name', 'afib_5y', 1),
    'lvef_40': ('lvef_40_ecg_file_name', 'lvef_equal_under_40', 1),
    'lvef_50': ('lvef_50_ecg_file_name', 'lvef_under_50', 1),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_predictions(path, expected_names, width):
    with path.open(newline='') as stream:
        reader = csv.reader(stream)
        columns = next(reader, [])
        if width == 77:
            valid = (len(columns) == 155 and columns[0] == 'file_name'
                     and all(c.endswith('_bert_model') for c in columns[1:78])
                     and all(c.endswith('_sig_model') for c in columns[78:])
                     and [c[:-11] for c in columns[1:78]] == [c[:-10] for c in columns[78:]])
            probability_start = 1
        else:
            valid = columns == ['file_name', 'ground_truth', 'predictions']
            probability_start = 2
        if not valid or len(set(columns)) != len(columns):
            raise ValueError('Wrong prediction schema')
        rows = {}
        for row in reader:
            if len(row) != len(columns) or row[0] in rows:
                raise ValueError('Duplicate ECG or malformed prediction row')
            if width == 1 and row[1] not in ('0', '1', '0.0', '1.0'):
                raise ValueError('Invalid binary input label in output')
            values = tuple(float(value) for value in row[probability_start:])
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ValueError('Non-finite or out-of-range prediction')
            rows[row[0]] = values
        if set(rows) != set(expected_names):
            raise ValueError('Missing or unexpected ECG predictions')
    return [columns[0], *columns[probability_start:]], rows


def validate_input_csv(path, ecgs, count):
    """Require the normal paired label/filename schema and one fixed ECG cohort."""
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {column for task, (filename, _, _) in TASKS.items() for column in (task, filename)}
        if (reader.fieldnames is None or not required.issubset(reader.fieldnames)
                or len(reader.fieldnames) != len(set(reader.fieldnames))):
            raise ValueError('Input CSV must contain all diagnosis/outcome and filename columns')
        rows = list(reader)
    if len(rows) != count:
        raise ValueError('Input CSV must contain the requested number of ECGs')
    files = []
    for row in rows:
        if not (row.get('ecg_machine_diagnosis') or '').strip():
            raise ValueError('Input CSV has an empty diagnosis')
        filename = row['77_classes_ecg_file_name']
        if not filename or Path(filename).name != filename or Path(filename).suffix.lower() != '.xml':
            raise ValueError('ECG filenames must be XML basenames')
        for task, (column, _, _) in TASKS.items():
            if row[column] != filename:
                raise ValueError('All tasks must use the same ECG in each row')
            if task != 'ecg_machine_diagnosis' and row[task] not in ('0', '1'):
                raise ValueError('Binary outcome columns must contain 0 or 1')
        file = ecgs / filename
        if file.is_symlink() or not file.is_file():
            raise ValueError('Input CSV references a missing ECG or a symlink')
        files.append(file)
    if len({file.stem for file in files}) != count:
        raise ValueError('Input CSV must contain unique ECGs')
    return files


def capture_run(folder, names):
    """Read validated outputs, keeping BERT once and each signal model separately."""
    expected = {Path(n).stem + '.base64' for n in names}
    if not expected or len(expected) != len(names):
        raise ValueError('ECG basenames must be unique and nonempty')
    if {p.name for p in (folder / 'preprocessing').glob('*.base64')} != expected:
        raise ValueError('Preprocessing did not produce every ECG')
    if len(list((folder / 'outputs').glob('*_probabilities.csv'))) != 8:
        raise ValueError('Expected predictions from all eight signal models')
    if list((folder / 'outputs').glob('missing_files_*.csv')):
        raise ValueError('Pipeline reported missing ECGs')
    reports = list((folder / 'outputs').glob('batch_*/ecg_processing_detailed_report.csv'))
    if not reports:
        raise ValueError('Missing preprocessing reports')
    statuses = []
    for report in reports:
        with report.open() as stream:
            statuses.extend(csv.DictReader(stream))
    if (len(statuses) != len(names)
            or {r['file_id'] for r in statuses} != {Path(n).stem for n in names}
            or any(r['status'] != 'Success' for r in statuses)):
        raise ValueError('Incomplete or failed preprocessing report')
    models = {}
    for task, (_, suffix, width) in TASKS.items():
        for family in ('wcr', 'efficientnetv2'):
            model = f'{family}_{suffix}'
            files = list((folder / 'outputs').glob(f'{model}_*_{task}_probabilities.csv'))
            if len(files) != 1:
                raise ValueError(f'Expected exactly one predictions file for {model}')
            columns, rows = read_predictions(files[0], expected, width)
            if width == 77:
                bert = {'columns': columns[1:78],
                        'predictions': {name: list(values[:77]) for name, values in rows.items()}}
                bert_name = 'Bert_diagnosis2classification_En_Fr'
                if bert_name in models and models[bert_name] != bert:
                    raise ValueError('BERT predictions disagree between the two signal-model outputs')
                models[bert_name] = bert
                columns = columns[78:]
                rows = {name: values[77:] for name, values in rows.items()}
            else:
                columns = columns[1:]
            models[model] = {'columns': columns,
                             'predictions': {name: list(values) for name, values in rows.items()}}
    return {'environment': json.loads((folder / 'environment.json').read_text()),
            'preprocessing_sha256': {name: sha256(folder / 'preprocessing' / name) for name in sorted(expected)},
            'models': models}


def compare_reference(reference, folder, names, atol=0.0):
    """Compare one new run to an existing reference; never update that reference."""
    if not math.isfinite(atol) or atol < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    actual = capture_run(folder, names)
    if set(reference['models']) != set(actual['models']):
        raise ValueError('Reference must contain BERT and all eight signal models')
    results = {}
    for model, current in actual['models'].items():
        saved = reference['models'][model]
        if saved['columns'] != current['columns']:
            raise ValueError(f'Reference prediction labels do not match {model}')
        if saved['predictions'].keys() != current['predictions'].keys():
            raise ValueError(f'Reference ECGs do not match {model}')
        differences = []
        for name, values in current['predictions'].items():
            previous = saved['predictions'][name]
            if (len(previous) != len(values)
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in previous)):
                raise ValueError(f'Invalid reference probabilities for {model}')
            differences.extend(abs(a - b) for a, b in zip(previous, values))
        results[model] = {'ecgs': len(names), 'probabilities_per_ecg': len(current['columns']),
                         'max_absolute_difference': max(differences),
                         'differing_values': sum(d > atol for d in differences),
                         'pass': all(d <= atol for d in differences)}
    signals_match = reference['preprocessing_sha256'] == actual['preprocessing_sha256']
    return {'pass': signals_match and all(r['pass'] for r in results.values()),
            'preprocessing_identical': signals_match,
            'environment_identical': reference['environment'] == actual['environment'],
            'absolute_tolerance': atol, 'models': results}


def command(args, log):
    with log.open('w') as output:
        result = subprocess.run(args, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed (exit {result.returncode}); see {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ecg-dir', type=Path, required=True)
    parser.add_argument('--input-csv', type=Path, required=True, help='CSV with diagnosis text, binary labels, and ECG filenames')
    parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[1] / 'inputs/data_rows_reproducibility_50.reference.json', help='Saved expected outputs; never overwritten by the test')
    parser.add_argument('--api-key-file', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True, help='New private artifact directory')
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--gpu', help='Host GPU index; omit for CPU')
    parser.add_argument('--atol', type=float, default=0.0)
    args = parser.parse_args()
    if args.count <= 0 or not math.isfinite(args.atol) or args.atol < 0:
        parser.error('count must be positive and atol finite and nonnegative')
    source = Path(__file__).resolve().parents[1]
    ecgs = args.ecg_dir.resolve(strict=True)
    key = args.api_key_file.resolve(strict=True)
    input_csv = args.input_csv.resolve(strict=True)
    files = validate_input_csv(input_csv, ecgs, args.count)
    reference_path = args.reference.resolve(strict=True)
    reference = json.loads(reference_path.read_text())
    reference_hash = sha256(reference_path)
    if reference.get('schema_version') != 1:
        raise ValueError('Unsupported reference format')
    if sha256(input_csv) != reference['input_csv_sha256']:
        raise ValueError('Input CSV differs from the saved reference')
    if {p.name: sha256(p) for p in files} != reference['ecgs']:
        raise ValueError('Input ECGs differ from the saved reference')
    os.umask(0o077)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    context = work / 'build-context'
    context.mkdir()
    for name in ('dockerfile', 'requirements.txt', 'heartwise.config', 'main.py', 'models_setup.py', 'run_pipeline.bash'):
        shutil.copyfile(source / name, context / name)
    for folder in ('data', 'models', 'utils', 'thresholds'):
        (context / folder).mkdir()
        for path in (source / folder).iterdir():
            if path.suffix in ('.py', '.json'):
                shutil.copyfile(path, context / folder / path.name)
    # The image runs as the invoking user, while COPY creates root-owned files.
    # Only application source goes into this context; credentials and ECGs do not.
    for path in context.rglob('*'):
        path.chmod(0o755 if path.is_dir() else 0o644)
    checks = work / 'checks'
    checks.mkdir()
    for name in ('ecg_repro_worker.py', 'test_ecg_reproducibility.py'):
        shutil.copyfile(source / 'scripts' / name, checks / name)
    provenance = {'source_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                  'source_files': {str(p.relative_to(context)): sha256(p) for p in sorted(context.rglob('*')) if p.is_file()},
                  'test_files': {p.name: sha256(p) for p in sorted(checks.iterdir())},
                  'input_csv_sha256': sha256(input_csv),
                  'reference_sha256': reference_hash,
                  'ecgs': {p.name: sha256(p) for p in files}}
    (work / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    shutil.copyfile(input_csv, work / 'inputs.csv')
    shutil.copyfile(reference_path, work / 'reference.json')
    image = 'deepecg-repro:' + hashlib.sha256(str(work).encode()).hexdigest()[:12]
    print('Building image without cached layers...', flush=True)
    command(['docker', 'build', '--pull', '--no-cache', '-t', image, str(context)], work / 'build.log')
    (work / 'image.json').write_text(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))
    (work / 'weights').mkdir()
    base = ['docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}',
            '--env', 'HF_HOME=/tmp/huggingface', '--env', 'HF_HUB_DISABLE_TELEMETRY=1',
            '--env', 'PYTHONHASHSEED=2026', '--env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
            '--volume', f'{checks}:/checks:ro',
            '--volume', f'{key}:/run/hf-key.json:ro', '--entrypoint', 'python']
    print('Downloading model snapshots at recorded commit revisions...', flush=True)
    command([*base, '--volume', f'{work}:/work', '--volume', f'{work / "weights"}:/app/weights',
             image, '/checks/ecg_repro_worker.py', 'prepare'], work / 'models.log')
    folder = work / 'run'
    folder.mkdir()
    for name in ('preprocessing', 'outputs'):
        (folder / name).mkdir()
    print(f'Running BERT and eight signal models on {len(files)} ECGs once...', flush=True)
    gpu = ['--gpus', f'device={args.gpu}'] if args.gpu is not None else []
    device = 'cuda:0' if args.gpu is not None else 'cpu'
    command([*base, *gpu, '--network', 'none', '--env', 'HF_HUB_OFFLINE=1',
             '--volume', f'{ecgs}:/ecgs:ro', '--volume', f'{work / "weights"}:/app/weights:ro',
             '--volume', f'{work / "inputs.csv"}:/input.csv:ro', '--volume', f'{folder}:/run-output',
             image, '/checks/ecg_repro_worker.py', 'run', '--mode', 'full_run',
             '--data_path', '/input.csv', '--ecg_signals_path', '/ecgs',
             '--output_folder', '/run-output/outputs', '--preprocessing_folder', '/run-output/preprocessing',
             '--hugging_face_api_key_path', '/run/hf-key.json', '--models', 'signal_models:wcr,efficientnet',
             '--batch_size', '8', '--preprocessing_n_workers', '4',
             '--signal_processing_device', device, '--diagnosis_classifier_device', device], folder / 'pipeline.log')
    if {p.name: sha256(p) for p in files} != provenance['ecgs']:
        raise ValueError('Input ECGs changed during the test')
    if sha256(work / 'inputs.csv') != provenance['input_csv_sha256']:
        raise ValueError('Input CSV changed during the test')
    if sha256(reference_path) != reference_hash or sha256(work / 'reference.json') != reference_hash:
        raise ValueError('Reference changed during the test')
    try:
        report = compare_reference(reference, folder, [p.name for p in files], args.atol)
    except ValueError as error:
        report = {'pass': False, 'error': str(error)}
    report['reference_sha256'] = reference_hash
    (work / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f'Artifacts: {work}')
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
