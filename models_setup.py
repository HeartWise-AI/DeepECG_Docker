"""Optionally preload pinned model files without blocking builds on unavailable models."""
from pathlib import Path

from utils.credentials import read_api_key
from utils.huggingface_wrapper import HuggingFaceWrapper
from utils.model_manifest import ModelPreparationError, load_manifest
from utils.log_config import get_logger

logger = get_logger(__name__)


def prepare_models(key_path='api_key.json'):
    token = read_api_key(key_path)['HUGGING_FACE_API_KEY']
    results = {}
    for name, entry in load_manifest().items():
        try:
            HuggingFaceWrapper.get_model(entry['repository'], Path('weights') / name, token)
            results[name] = 'available'
        except ModelPreparationError as error:
            results[name] = 'unavailable'
            logger.warning('%s', error)
    logger.info('Model preload summary: %d available, %d unavailable',
                sum(value == 'available' for value in results.values()),
                sum(value == 'unavailable' for value in results.values()))
    for name, status in results.items():
        logger.info('  %s: %s', name, status)
    return results


if __name__ == '__main__':
    prepare_models()
