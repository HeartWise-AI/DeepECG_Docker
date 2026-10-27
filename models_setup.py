import torch
import models
from models.heartwise_model_factory import HeartWiseModelFactory
from utils.files_handler import read_api_key
from utils.log_config import get_logger

logger = get_logger(__name__)
hugging_face_api_key = read_api_key('api_key.json')['HUGGING_FACE_API_KEY']

for model_class in models.__all__:
    try:
        cls = getattr(models, model_class)
        logger.info("Loading model class: %s", cls.name)
        HeartWiseModelFactory.create_model(
            model_config={
                'model_name': cls.name,
                'map_location': torch.device('cpu'),
                'hugging_face_api_key': hugging_face_api_key
            }
        )
    except Exception as e:
        logger.error("Failed to load model %s: %s", model_class, e, exc_info=True)
