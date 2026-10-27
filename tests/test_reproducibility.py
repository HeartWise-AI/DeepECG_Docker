"""Synthetic regression tests for strict comparisons and parallel XML parsing."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('repro', ROOT / 'scripts/test_ecg_reproducibility.py')
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runs = [Path(self.temp.name) / n for n in ('a', 'b')]
        self.names = ['synthetic_1.xml', 'synthetic_2.xml']
        for run in self.runs:
            (run / 'preprocessing').mkdir(parents=True)
            (run / 'outputs' / 'batch_1').mkdir(parents=True)
            (run / 'environment.json').write_text('{"test_environment": true}')
            for name in self.names:
                (run / 'preprocessing' / (Path(name).stem + '.base64')).write_text('synthetic')
            report = run / 'outputs/batch_1/ecg_processing_detailed_report.csv'
            report.write_text('file_id,status\nsynthetic_1,Success\nsynthetic_2,Success\n')
            for task, (_, suffix, width) in repro.TASKS.items():
                for family in ('wcr', 'efficientnetv2'):
                    path = run / 'outputs' / f'{family}_{suffix}_20261025_{task}_probabilities.csv'
                    with path.open('w', newline='') as stream:
                        writer = csv.writer(stream)
                        if width == 77:
                            writer.writerow(['file_name', *[f'p{i}_bert_model' for i in range(77)],
                                             *[f'p{i}_sig_model' for i in range(77)]])
                            writer.writerows([[Path(n).stem + '.base64', *([0.25] * 154)] for n in self.names])
                        else:
                            writer.writerow(['file_name', 'ground_truth', 'predictions'])
                            writer.writerows([[Path(n).stem + '.base64', 0, 0.25] for n in self.names])

        self.reference = repro.capture_run(self.runs[0], self.names)

    def test_complete_identical_outputs_pass(self):
        result = repro.compare_reference(self.reference, self.runs[1], self.names)
        self.assertTrue(result['pass'])
        self.assertEqual(len(result['models']), 9)

    def test_mixed_format_preprocessing_reports_are_checked_together(self):
        batch = self.runs[1] / 'outputs/batch_1'
        (batch / 'ecg_processing_detailed_report.csv').unlink()
        for extension, file_id in [('xml', 'synthetic_1'), ('npy', 'synthetic_2')]:
            (batch / extension).mkdir()
            (batch / extension / 'ecg_processing_detailed_report.csv').write_text(
                f'file_id,status\n{file_id},Success\n'
            )
        self.assertTrue(repro.compare_reference(self.reference, self.runs[1], self.names)['pass'])
        (batch / 'npy/ecg_processing_detailed_report.csv').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete or failed preprocessing report'):
            repro.compare_reference(self.reference, self.runs[1], self.names)

    def test_probability_drift_fails(self):
        path = next((self.runs[1] / 'outputs').glob('*afib*probabilities.csv'))
        path.write_text(path.read_text().replace('0.25', '0.26'))
        self.assertFalse(repro.compare_reference(self.reference, self.runs[1], self.names)['pass'])

    def test_environment_drift_is_reported(self):
        (self.runs[1] / 'environment.json').write_text('{"changed": true}')
        report = repro.compare_reference(self.reference, self.runs[1], self.names)
        self.assertTrue(report['pass'])
        self.assertFalse(report['environment_identical'])

    def test_missing_model_is_rejected(self):
        next((self.runs[1] / 'outputs').glob('*probabilities.csv')).unlink()
        with self.assertRaisesRegex(ValueError, 'eight signal models'):
            repro.compare_reference(self.reference, self.runs[1], self.names)

    def test_changed_preprocessing_is_rejected(self):
        next((self.runs[1] / 'preprocessing').iterdir()).write_text('changed')
        report = repro.compare_reference(self.reference, self.runs[1], self.names)
        self.assertFalse(report['pass'])
        self.assertFalse(report['preprocessing_identical'])

    def test_invalid_rows_are_rejected(self):
        path = Path(self.temp.name) / 'bad.csv'
        for rows in ['a,0,NaN\n', 'a,0,inf\n', 'a,0,1.1\n', 'a,0,0.2\na,0,0.2\n', 'b,0,0.2\n', 'a,0.2,0.2\n', '']:
            with self.subTest(rows=rows):
                path.write_text('file_name,ground_truth,predictions\n' + rows)
                with self.assertRaises(ValueError):
                    repro.read_predictions(path, {'a'}, 1)

    def test_row_order_does_not_change_comparison(self):
        path = next((self.runs[1] / 'outputs').glob('*afib*probabilities.csv'))
        lines = path.read_text().splitlines()
        path.write_text('\n'.join([lines[0], *reversed(lines[1:])]) + '\n')
        self.assertTrue(repro.compare_reference(self.reference, self.runs[1], self.names)['pass'])

    def test_bert_probability_drift_fails(self):
        for path in (self.runs[1] / 'outputs').glob('*77_classes*probabilities.csv'):
            path.write_text(path.read_text().replace('0.25', '0.26', 1))
        report = repro.compare_reference(self.reference, self.runs[1], self.names)
        self.assertFalse(report['pass'])
        self.assertFalse(report['models']['Bert_diagnosis2classification_En_Fr']['pass'])

    def test_inconsistent_bert_outputs_are_rejected(self):
        path = next((self.runs[1] / 'outputs').glob('*77_classes*probabilities.csv'))
        path.write_text(path.read_text().replace('0.25', '0.26', 1))
        with self.assertRaisesRegex(ValueError, 'BERT predictions disagree'):
            repro.compare_reference(self.reference, self.runs[1], self.names)

    def test_failed_comparison_does_not_change_reference(self):
        before = json.dumps(self.reference, sort_keys=True)
        path = next((self.runs[1] / 'outputs').glob('*afib*probabilities.csv'))
        path.write_text(path.read_text().replace('0.25', '0.30'))
        self.assertFalse(repro.compare_reference(self.reference, self.runs[1], self.names)['pass'])
        self.assertEqual(json.dumps(self.reference, sort_keys=True), before)

    def test_invalid_reference_is_rejected(self):
        model = self.reference['models']['wcr_afib_5y']
        model['predictions']['synthetic_1.base64'] = [float('nan')]
        with self.assertRaisesRegex(ValueError, 'Invalid reference'):
            repro.compare_reference(self.reference, self.runs[1], self.names)

    def test_tolerance_is_explicit(self):
        path = next((self.runs[1] / 'outputs').glob('*afib*probabilities.csv'))
        path.write_text(path.read_text().replace('0.25', '0.26'))
        report = repro.compare_reference(self.reference, self.runs[1], self.names, atol=0.02)
        self.assertTrue(report['pass'])


class InputTests(unittest.TestCase):
    def test_normal_template_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'synthetic.xml').write_text('<synthetic/>')
            path = root / 'input.csv'
            fields = [c for task, (name, _, _) in repro.TASKS.items() for c in (task, name)]
            row = {c: 'synthetic.xml' for c in fields}
            row.update(ecg_machine_diagnosis='Synthetic diagnosis', afib_5y='0', lvef_40='0', lvef_50='1')

            def write(values, columns=fields):
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerow(values)

            write(row)
            self.assertEqual(repro.validate_input_csv(path, root, 1), [root / 'synthetic.xml'])
            for update in ({'ecg_machine_diagnosis': ''}, {'afib_5y': '0.2'},
                           {'afib_ecg_file_name': 'different.xml'}):
                with self.subTest(update=update):
                    write({**row, **update})
                    with self.assertRaises(ValueError):
                        repro.validate_input_csv(path, root, 1)
            write(row, [v[0] for v in repro.TASKS.values()])
            with self.assertRaisesRegex(ValueError, 'all diagnosis/outcome'):
                repro.validate_input_csv(path, root, 1)


class PipelineTests(unittest.TestCase):
    def test_parallel_xml_parsers_do_not_share_signal_buffers(self):
        import numpy as np
        import pandas as pd
        from utils.files_handler import XMLProcessor

        barrier = threading.Barrier(3)

        def synthetic_parse(parser, filename):
            number = int(Path(filename).stem)
            parser.full_leads_array = np.full((2500, 12), number, dtype=np.float32)
            barrier.wait(timeout=10)
            return (str(number), 'MHI', 'Success', ''), str(number), parser.full_leads_array

        df = pd.DataFrame({'ecg_path': ['0.xml', '1.xml', '2.xml']})
        with patch.object(XMLProcessor, 'process_single_file', synthetic_parse):
            _, signals = XMLProcessor().process_batch(df, num_workers=3)
        for index, signal in enumerate(signals.ecg_signal):
            self.assertTrue((signal == index).all())

    def test_binary_analysis_uses_eval_and_preserves_labels(self):
        import torch
        from utils.analysis_pipeline import AnalysisPipeline

        class FakeModel:
            def __init__(self):
                self.model = torch.nn.Sequential(torch.nn.Dropout(0.9), torch.nn.Linear(12, 1), torch.nn.Sigmoid())

            def __call__(self, signal):
                return self.model(signal)

        model = FakeModel()
        batch = {'ecg_signal': torch.ones(2, 12), 'file_name': ['one', 'two'], 'diagnosis': torch.tensor([0, 1])}
        with patch('utils.analysis_pipeline.HeartWiseModelFactory.create_model', return_value=model) as factory, \
             patch('utils.analysis_pipeline.create_dataloader', return_value=[batch]):
            args = dict(df=None, batch_size=2, diagnosis_classifier_device='cpu',
                        signal_processing_device='cpu', signal_processing_model_name='wcr_afib_5y',
                        diagnosis_classifier_model_name=None, hugging_face_api_key='')
            metrics, first = AnalysisPipeline._run_analysis_impl(**args)
            _, second = AnalysisPipeline._run_analysis_impl(**args)
        self.assertIn('results', metrics)
        self.assertFalse(model.model.training)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(list(first.columns), ['file_name', 'ground_truth', 'predictions'])
        self.assertEqual(first.ground_truth.tolist(), [0, 1])
        self.assertTrue(first.equals(second))

    def test_multilabel_analysis_runs_bert_and_metrics(self):
        import torch
        from utils.analysis_pipeline import AnalysisPipeline

        class FakeModel:
            def __init__(self):
                self.model = torch.nn.Dropout(0.9)

            def __call__(self, value):
                return self.model(torch.full((len(value), 77), 0.25))

        signal, bert = FakeModel(), FakeModel()
        batch = {'ecg_signal': torch.ones(2, 12), 'file_name': ['one', 'two'],
                 'diagnosis': ['Synthetic normal', 'Synthetic abnormal']}
        with patch('utils.analysis_pipeline.HeartWiseModelFactory.create_model', side_effect=[signal, bert]), \
             patch('utils.analysis_pipeline.create_dataloader', return_value=[batch]), \
             patch('utils.analysis_pipeline.compute_metrics', return_value={'checked': True}) as metrics:
            result, output = AnalysisPipeline._run_analysis_impl(
                df=None, batch_size=2, diagnosis_classifier_device='cpu', signal_processing_device='cpu',
                signal_processing_model_name='efficientnetv2_77_classes',
                diagnosis_classifier_model_name='Bert_diagnosis2classification_En_Fr', hugging_face_api_key='')
        self.assertEqual(result, {'checked': True})
        metrics.assert_called_once()
        self.assertFalse(signal.model.training)
        self.assertFalse(bert.model.training)
        self.assertEqual(output.shape, (2, 155))
        self.assertTrue((output.iloc[:, 1:] == 0.25).all().all())


if __name__ == '__main__':
    unittest.main()
