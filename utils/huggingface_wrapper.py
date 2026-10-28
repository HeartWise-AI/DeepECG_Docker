import os
import json
from pathlib import Path
from filelock import FileLock
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
from utils.model_manifest import ModelPreparationError, model_entry
from huggingface_hub import snapshot_download, HfApi

from utils.log_config import get_logger

logger = get_logger(__name__)


class HuggingFaceWrapper:
    """Download and upload Hugging Face Hub models using an API key."""

    def __init__(self, hugging_face_api_key):
        self.hugging_face_api_key = hugging_face_api_key
        self.api = HfApi()

    @staticmethod
    def get_model(repo_id, local_dir, hugging_face_api_key=''):
        """Return a complete, revision-specific snapshot; never trust legacy directories."""
        entry = model_entry(repo_id)
        revision = entry['revision']
        # Repository identity and revision both participate in cache isolation.
        folder = Path(local_dir) / repo_id.replace('/', '--') / revision
        marker = folder / '.deepecg-complete.json'
        if _complete(folder, marker, entry):
            logger.info("Using pinned model %s at %s", repo_id, revision)
            return str(folder)
        token = hugging_face_api_key.strip() if hugging_face_api_key else False
        try:
            folder.parent.mkdir(parents=True, exist_ok=True)
            # Lock outside the snapshot so concurrent downloads cannot publish partial files.
            with FileLock(str(folder) + '.lock'):
                if _complete(folder, marker, entry):
                    return str(folder)
                repair = folder.exists()
                info = HfApi().model_info(repo_id, revision=revision, token=token, files_metadata=True)
                expected = {file.rfilename: file.size for file in (info.siblings or [])}
                if (info.sha != revision or not expected
                        or any(not isinstance(size, int) or size < 0 for size in expected.values())):
                    raise ModelPreparationError(f'{repo_id}: cannot verify the pinned repository file inventory.')
                marker.unlink(missing_ok=True)
                logger.info("Downloading %s at %s", repo_id, revision)
                snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(folder),
                                  repo_type='model', token=token, force_download=repair)
                files = {str(p.relative_to(folder)): p.stat().st_size
                         for p in folder.rglob('*') if p.is_file()
                         and '.cache' not in p.relative_to(folder).parts
                         and p.name not in {marker.name, marker.name + '.tmp'}}
                if files != expected:
                    raise ModelPreparationError(
                        f'{repo_id}: incomplete snapshot; files do not match the pinned repository inventory. Retry the download.'
                    )
                temporary = marker.with_name(marker.name + '.tmp')
                temporary.write_text(json.dumps({**entry, 'schema_version': 1, 'files': expected}, sort_keys=True))
                temporary.replace(marker)
        except ModelPreparationError:
            raise
        except Exception as error:
            # Hub errors may include request details. Never expose the exception/token.
            raise ModelPreparationError(_download_error(repo_id, revision, token, error)) from None
        return str(folder)

    def upload_model(self, repo_id, local_dir, commit_message="Update model"):
        """
        Upload a model to Hugging Face.
        
        :param repo_id: The ID of the repository (e.g., 'username/repo-name')
        :param local_dir: The local directory containing the model files
        :param commit_message: The commit message for this update
        """
        try:
            # Ensure the repository exists (create if it doesn't)
            try:
                self.api.create_repo(repo_id=repo_id, token=self.hugging_face_api_key, exist_ok=True)
            except Exception as e:
                logger.debug("create_repo: %s", e)

            # Upload all files in the directory
            for root, _, files in os.walk(local_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    repo_path = os.path.relpath(file_path, local_dir)
                    
                    self.api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        token=self.hugging_face_api_key
                    )
                    logger.debug("Uploaded %s to %s", file, repo_id)
            self.api.create_commit(
                repo_id=repo_id,
                operations="push",
                commit_message=commit_message,
                token=self.hugging_face_api_key
            )
            logger.info("Uploaded and committed model to %s", repo_id)
        except Exception as e:
            logger.error("Model upload to %s failed: %s", repo_id, e, exc_info=True)

def _complete(folder, marker, entry):
    """Check publication metadata and every file's presence/size, including read-only caches."""
    try:
        saved = json.loads(marker.read_text())
        if saved.get('schema_version') != 1 or any(saved.get(key) != value for key, value in entry.items()):
            return False
        files = saved['files']
        return bool(files) and all(
            not Path(name).is_absolute() and '..' not in Path(name).parts
            and (folder / name).is_file() and (folder / name).stat().st_size == size
            for name, size in files.items()
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _download_error(repo, revision, token, error):
    prefix = f'{repo} at {revision}: model was not downloaded. '
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    if isinstance(error, RevisionNotFoundError):
        return prefix + 'Revision unavailable; verify the commit in models/manifest.json.'
    if isinstance(error, (GatedRepoError, RepositoryNotFoundError)) or status in (401, 403, 404):
        if not token:
            return prefix + ('The API key is empty and anonymous access failed. Verify the repository; '
                             'if private or gated, supply HUGGING_FACE_API_KEY with read access.')
        return prefix + ('Access denied or repository unavailable. Verify the repository, token read '
                         'permissions, and any required access approval.')
    if isinstance(error, OSError) and not hasattr(error, 'response'):
        return prefix + 'Download/cache I/O failed; check network, cache permissions and free disk space, then retry.'
    return prefix + 'Network/download failure; check connectivity and retry. No complete cache was published.'
