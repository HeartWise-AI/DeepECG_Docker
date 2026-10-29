import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'png_repro', ROOT / 'scripts/test_png_reproducibility.py'
)
png_repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(png_repro)


def write_input(path, names):
    fields = [
        name
        for task, (file_column, _, _) in png_repro.TASKS.items()
        for name in (task, file_column)
    ]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, name in enumerate(names):
            row = {column: name for column, _, _ in png_repro.TASKS.values()}
            row.update(
                ecg_machine_diagnosis=f'Synthetic report {index}',
                afib_5y=str(index % 2),
                lvef_40='',
                lvef_50='0',
                shd='1',
            )
            writer.writerow(row)


def test_input_validation_uses_one_safe_image_cohort(tmp_path):
    names = ['first.png', 'second.JPEG']
    for name in names:
        (tmp_path / name).write_bytes(b'synthetic image fixture')
    input_csv = tmp_path / 'input.csv'
    write_input(input_csv, names)
    assert png_repro.validate_input_csv(input_csv, tmp_path, 2) == [
        tmp_path / name for name in names
    ]

    for invalid in ('signal.npy', '../first.png', 'nested/first.png'):
        write_input(input_csv, [invalid])
        with pytest.raises(ValueError, match='PNG or JPEG basenames'):
            png_repro.validate_input_csv(input_csv, tmp_path, 1)

    write_input(input_csv, ['first.png', 'first.png'])
    with pytest.raises(ValueError, match='unique ECG pages'):
        png_repro.validate_input_csv(input_csv, tmp_path, 2)

    (tmp_path / 'alias.png').symlink_to(tmp_path / 'first.png')
    write_input(input_csv, ['alias.png'])
    with pytest.raises(ValueError, match='missing ECG page or a symlink'):
        png_repro.validate_input_csv(input_csv, tmp_path, 1)


def write_outputs(folder, names):
    (folder / 'preprocessing').mkdir(parents=True)
    outputs = folder / 'outputs'
    outputs.mkdir()
    (folder / 'environment.json').write_text(json.dumps({'test': True}))
    labels = [f'label_{index}' for index in range(77)]
    columns = (['file_name'] + [f'{label}_bert_model' for label in labels]
               + [f'{label}_sig_model' for label in labels])
    with (outputs / 'deepecg_siglip2_77_classes_now_ecg_machine_diagnosis_probabilities.csv').open(
        'w', newline=''
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row_index, name in enumerate(names):
            writer.writerow([name] + [0.1 + row_index * 0.01] * 77
                            + [0.2 + row_index * 0.01] * 77)
    for task, (_, model, width) in png_repro.TASKS.items():
        if width == 77:
            continue
        with (outputs / f'{model}_now_{task}_probabilities.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['file_name', 'ground_truth', 'predictions'])
            for row_index, name in enumerate(names):
                writer.writerow([name, '', 0.3 + row_index * 0.01])


def test_capture_and_compare_cover_all_png_models(tmp_path):
    names = ['first.png', 'second.png']
    write_outputs(tmp_path, names)
    reference = png_repro.capture_run(tmp_path, names)
    assert set(reference['models']) == {
        png_repro.BERT_MODEL,
        *(model for _, model, _ in png_repro.TASKS.values()),
    }
    report = png_repro.compare_reference(reference, tmp_path, names)
    assert report['pass'] and all(result['pass'] for result in report['models'].values())

    model = 'deepecg_siglip2_shd'
    reference['models'][model]['predictions'][names[0]][0] += 0.01
    report = png_repro.compare_reference(reference, tmp_path, names, atol=0.001)
    assert not report['pass'] and not report['models'][model]['pass']


def test_capture_rejects_signal_preprocessing_artifacts(tmp_path):
    write_outputs(tmp_path, ['first.png'])
    (tmp_path / 'preprocessing' / 'unexpected.base64').write_bytes(b'x')
    with pytest.raises(ValueError, match='signal preprocessing'):
        png_repro.capture_run(tmp_path, ['first.png'])
