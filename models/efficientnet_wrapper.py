
import os
import sys
import torch

from utils.huggingface_wrapper import HuggingFaceWrapper
from models.heartwise_model_factory import HeartWiseModelFactory


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
                repo_id=f"heartwise/{model_name}", 
                local_dir=os.path.join("weights", model_name),
                hugging_face_api_key=hugging_face_api_key
            ),
            map_location=map_location
        )
        print(f"Model {model_name} loaded to {map_location}")
        self.mhi_factor = 1/0.0048

    def _load_model(self, model_path: str, map_location: torch.device) -> None:       
        pt_file = next((f for f in os.listdir(model_path) if f.endswith('.pt')), None)
        if not pt_file:
            raise ValueError("No .pt file found in the directory")
        model_path = os.path.join(model_path, pt_file)
        self.model = torch.jit.load(model_path, map_location=map_location)

    def __call__(self, signal):
        signal *= self.mhi_factor
        return torch.sigmoid(self.model(signal))


class EfficientNetV2_77_classes(EfficientNetWrapper, HeartWiseModelFactory):
    name = 'efficientnetv2_77_classes'
    
    def __init__(
        self, 
        model_name: str,
        map_location: torch.device,
        hugging_face_api_key: str
    ):
        super().__init__(
            model_name=model_name, 
            map_location=map_location, 
            hugging_face_api_key=hugging_face_api_key
        )

class EfficientNetV2_LVEF_Equal_Under_40(EfficientNetWrapper, HeartWiseModelFactory):
    name = 'efficientnetv2_lvef_equal_under_40'
    
    def __init__(
        self, 
        model_name: str,
        map_location: torch.device,
        hugging_face_api_key: str
    ):
        super().__init__(
            model_name=model_name, 
            map_location=map_location, 
            hugging_face_api_key=hugging_face_api_key
        )

class EfficientNetV2_LVEF_Under_50(EfficientNetWrapper, HeartWiseModelFactory):
    name = 'efficientnetv2_lvef_under_50'
    
    def __init__(
        self, 
        model_name: str,
        map_location: torch.device,
        hugging_face_api_key: str
    ):
        super().__init__(
            model_name=model_name, 
            map_location=map_location, 
            hugging_face_api_key=hugging_face_api_key
        )

class EfficientNetV2_AFIB_5Y(EfficientNetWrapper, HeartWiseModelFactory):
    name = 'efficientnetv2_afib_5y'
    
    def __init__(
        self, 
        model_name: str,
        map_location: torch.device,
        hugging_face_api_key: str
    ):
        super().__init__(
            model_name=model_name, 
            map_location=map_location, 
            hugging_face_api_key=hugging_face_api_key
        )

