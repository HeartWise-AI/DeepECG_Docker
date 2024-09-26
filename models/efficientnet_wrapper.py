
import os
import sys
import torch

from utils.huggingface_wrapper import HuggingFaceWrapper
from models.heartwise_model_factory import HeartWiseModelFactory
sys.path.append('models/modules')

class EfficientNetWrapper(HeartWiseModelFactory):
    
    name = 'efficientnetv2'
    
    def __init__(
        self, 
        model_name: str, 
        map_location: torch.device,
        hugging_face_api_key: str
    ):
        self.device = map_location
        self._load_model(
            model_path=HuggingFaceWrapper.get_model(
                model_name=model_name, 
                repo_id="heartwise/DeepECG_EfficientNetV2", 
                local_dir=os.path.join("weights", model_name),
                hugging_face_api_key=hugging_face_api_key
            ),
            map_location=map_location
        )
        print(f"Model {model_name} loaded to {map_location}")

    def _load_model(self, model_path: str, map_location: torch.device) -> None:       
        pt_file = next((f for f in os.listdir(model_path) if f.endswith('.pt')), None)
        if not pt_file:
            raise ValueError("No .pt file found in the directory")
        model_path = os.path.join(model_path, pt_file)
        self.model = torch.load(model_path, map_location=map_location, weights_only=False)

    def __call__(self, signal):
        signal = signal.to(self.device)
        return torch.sigmoid(self.model(signal))
