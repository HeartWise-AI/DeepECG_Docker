
import torch
import models
from models.heartwise_model_factory import HeartWiseModelFactory

from utils.files_handler import read_api_key

hugging_face_api_key = read_api_key('api_key.json')['HUGGING_FACE_API_KEY']

try:
    for model_class in models.__all__:
        cls = getattr(models, model_class)
        print(f"Class: {cls.name}")
        model = HeartWiseModelFactory.create_model(
            model_config={
                'model_name': cls.name,
                'map_location': torch.device('cpu'),
                'hugging_face_api_key': hugging_face_api_key
            }
        )
except Exception as e:
    print(f"Error importing model {model_class}: {e}")
