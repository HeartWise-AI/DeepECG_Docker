"""Synthetic tests for optional credentials and immutable model downloads."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import httpx
from unittest.mock import patch

from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

from utils.credentials import read_api_key
from utils.huggingface_wrapper import HuggingFaceWrapper
from utils.model_manifest import ModelPreparationError, load_manifest, model_entry
import models_setup

ROOT = Path(__file__).resolve().parents[1]


class CredentialsTests(unittest.TestCase):
    def test_create_empty_and_preserve_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'api_key.json'
            self.assertEqual(read_api_key(path), {'HUGGING_FACE_API_KEY': ''})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            original = '{"HUGGING_FACE_API_KEY": "synthetic-test-only", "extra": 1}'
            path.write_text(original)
            self.assertEqual(read_api_key(path)['HUGGING_FACE_API_KEY'], 'synthetic-test-only')
            self.assertEqual(path.read_text(), original)

    def test_malformed_key_is_not_printed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'api_key.json'
            for content in ('synthetic-private-value', '[]', '{"HUGGING_FACE_API_KEY": 2}'):
                path.write_text(content)
                with self.assertRaises(ValueError) as caught:
                    read_api_key(path)
                self.assertNotIn(content, str(caught.exception))
                self.assertEqual(path.read_text(), content)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.local = Path(self.tmp.name) / 'weights'
        self.entry = {'repository': 'test/public', 'revision': 'a' * 40}
        self.manifest = patch('utils.model_manifest.load_manifest', return_value={'public': self.entry})
        self.manifest.start()
        self.addCleanup(self.manifest.stop)
        self.api = patch('utils.huggingface_wrapper.HfApi').start().return_value
        self.api.model_info.side_effect = lambda *args, **kwargs: SimpleNamespace(
            sha=self.entry['revision'], siblings=[SimpleNamespace(rfilename='weights.pt', size=15)])
        self.download = patch('utils.huggingface_wrapper.snapshot_download', side_effect=self.snapshot).start()
        self.addCleanup(patch.stopall)

    @staticmethod
    def snapshot(**kwargs):
        folder = Path(kwargs['local_dir'])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'weights.pt').write_bytes(b'synthetic model')
        return str(folder)

    def get(self, token=''):
        return Path(HuggingFaceWrapper.get_model('test/public', self.local, token))

    def test_empty_token_is_explicitly_anonymous_and_cached_offline(self):
        folder = self.get()
        self.assertIs(self.download.call_args.kwargs['token'], False)
        self.assertIs(self.api.model_info.call_args.kwargs['token'], False)
        self.assertEqual(self.download.call_args.kwargs['revision'], 'a' * 40)
        with patch('utils.huggingface_wrapper.FileLock', side_effect=AssertionError('No writes on cached read')):
            self.assertEqual(self.get(), folder)
        self.download.assert_called_once()

    def test_revision_change_and_legacy_directory_are_isolated(self):
        self.local.mkdir()
        (self.local / 'legacy.pt').write_bytes(b'unverified')
        first = self.get()
        self.entry['revision'] = 'b' * 40
        second = self.get()
        self.assertNotEqual(first, second)
        self.assertEqual(self.download.call_count, 2)
        self.assertTrue((first / 'weights.pt').is_file())
        self.assertFalse((second / 'legacy.pt').exists())

    def test_interrupted_download_retry(self):
        def interrupted(**kwargs):
            self.snapshot(**kwargs)
            raise ConnectionError('synthetic-private-value')
        self.download.side_effect = interrupted
        with self.assertRaises(ModelPreparationError) as caught:
            self.get()
        self.assertNotIn('synthetic-private-value', str(caught.exception))
        self.assertFalse(list(self.local.rglob('.deepecg-complete.json')))
        self.download.side_effect = self.snapshot
        self.assertTrue((self.get() / '.deepecg-complete.json').is_file())
        self.assertEqual(self.download.call_count, 2)

    def test_missing_or_truncated_cached_file_triggers_repair(self):
        folder = self.get()
        (folder / 'weights.pt').write_bytes(b'x')
        self.get()
        self.assertTrue(self.download.call_args.kwargs['force_download'])
        (folder / 'weights.pt').unlink()
        self.get()
        self.assertEqual(self.download.call_count, 3)

    def test_partial_offline_fallback_cannot_publish_completion(self):
        self.api.model_info.side_effect = lambda *args, **kwargs: SimpleNamespace(
            sha=self.entry['revision'], siblings=[
                SimpleNamespace(rfilename='weights.pt', size=15),
                SimpleNamespace(rfilename='config.json', size=2),
            ])
        # Simulate snapshot_download returning a partial directory after losing connectivity.
        with self.assertRaisesRegex(ModelPreparationError, 'incomplete snapshot'):
            self.get()
        self.assertFalse(list(self.local.rglob('.deepecg-complete.json')))
        def recovered(**kwargs):
            folder = self.snapshot(**kwargs)
            (Path(folder) / 'config.json').write_text('{}')
            return folder
        self.download.side_effect = recovered
        folder = self.get()
        self.assertTrue(self.download.call_args.kwargs['force_download'])
        self.assertTrue((folder / '.deepecg-complete.json').is_file())

    def test_inventory_failure_does_not_trust_partial_cache(self):
        self.local.mkdir()
        (self.local / 'partial.pt').write_bytes(b'partial')
        self.api.model_info.side_effect = ConnectionError('synthetic-private-value')
        with self.assertRaises(ModelPreparationError):
            self.get()
        self.download.assert_not_called()
        self.assertFalse(list(self.local.rglob('.deepecg-complete.json')))

    def test_inventory_must_match_requested_revision(self):
        self.api.model_info.side_effect = None
        self.api.model_info.return_value = SimpleNamespace(
            sha='b' * 40, siblings=[SimpleNamespace(rfilename='weights.pt', size=15)])
        with self.assertRaisesRegex(ModelPreparationError, 'cannot verify'):
            self.get()
        self.download.assert_not_called()

    def test_interrupted_repair_remains_forced_on_retry(self):
        folder = self.get()
        (folder / 'weights.pt').write_bytes(b'x')
        self.download.side_effect = ConnectionError('offline')
        with self.assertRaises(ModelPreparationError):
            self.get()
        self.download.side_effect = self.snapshot
        self.get()
        self.assertTrue(self.download.call_args.kwargs['force_download'])
        self.assertEqual((folder / 'weights.pt').read_bytes(), b'synthetic model')

    def test_error_messages_are_actionable_without_exception_details(self):
        for token, error, message in (
            ('', RepositoryNotFoundError('synthetic-private-value', response=httpx.Response(401, request=httpx.Request("GET", "https://example.invalid/model"))), 'API key is empty'),
            ('test-key', GatedRepoError('synthetic-private-value', response=httpx.Response(403, request=httpx.Request("GET", "https://example.invalid/model"))), 'Access denied'),
            ('', RevisionNotFoundError('synthetic-private-value', response=httpx.Response(404, request=httpx.Request("GET", "https://example.invalid/model"))), 'Revision unavailable'),
            ('', RuntimeError('synthetic-private-value'), 'Network/download failure'),
        ):
            with self.subTest(message=message):
                self.download.side_effect = error
                with self.assertRaises(ModelPreparationError) as caught:
                    self.get(token)
                self.assertIn(message, str(caught.exception))
                self.assertNotIn('synthetic-private-value', str(caught.exception))
                self.assertNotIn('test-key', str(caught.exception))

    def test_invalid_manifest_does_not_download(self):
        self.entry['revision'] = 'main'
        with self.assertRaisesRegex(ModelPreparationError, 'exact 40-character'):
            self.get()
        self.download.assert_not_called()

    def test_unselected_inaccessible_entry_does_not_download(self):
        with patch('utils.model_manifest.load_manifest', return_value={
            'public': self.entry,
            'unreleased': {'repository': 'test/unreleased', 'revision': 'b' * 40},
        }):
            self.get()
        self.assertEqual(self.download.call_args.kwargs['repo_id'], 'test/public')
        self.download.assert_called_once()


class SetupTests(unittest.TestCase):
    def test_preload_continues_after_access_failure(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(models_setup, 'load_manifest', return_value={
                 'private': {'repository': 'test/private'},
                 'public': {'repository': 'test/public'},
             }), \
             patch.object(HuggingFaceWrapper, 'get_model', side_effect=[
                 ModelPreparationError('Access denied'), '/cache/public',
             ]) as download:
            self.assertEqual(models_setup.prepare_models(Path(tmp) / 'key.json'),
                             {'private': 'unavailable', 'public': 'available'})
            self.assertEqual(download.call_count, 2)

    def test_checked_in_revisions_match_reference(self):
        fixtures = ROOT / 'tests/fixtures/reproducibility'
        for filename in (
            'data_rows_reproducibility_50.reference.json',
            'data_rows_png_reproducibility_25.reference.json',
        ):
            reference = json.loads((fixtures / filename).read_text())
            for name, entry in reference['model_manifest'].items():
                self.assertEqual(load_manifest()[name], {
                    key: entry[key] for key in ('repository', 'revision')
                })
                self.assertEqual(model_entry(entry['repository'])['revision'], entry['revision'])

    def test_factory_preserves_load_error_and_constructs_only_selected_model(self):
        spec = importlib.util.spec_from_file_location('isolated_factory', ROOT / 'models/heartwise_model_factory.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        factory = module.HeartWiseModelFactory
        class Public(factory):
            name = 'public'
            def __init__(self, **kwargs):
                self.loaded = True
        class Private(factory):
            name = 'private'
            def __init__(self, **kwargs):
                raise ValueError('Model weights cannot be loaded')
        self.assertTrue(factory.create_model({'model_name': 'public'}).loaded)
        with self.assertRaisesRegex(ValueError, "Model 'unknown' not found"):
            factory.create_model({'model_name': 'unknown'})
        with self.assertRaisesRegex(ValueError, 'Model weights cannot be loaded'):
            factory.create_model({'model_name': 'private'})


if __name__ == '__main__':
    unittest.main()
