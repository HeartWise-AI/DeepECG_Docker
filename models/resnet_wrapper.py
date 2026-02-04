import os
import torch
from utils.huggingface_wrapper import HuggingFaceWrapper
from utils.log_config import get_logger
from models.heartwise_model_factory import HeartWiseModelFactory

logger = get_logger(__name__)


class ResNetWrapper(HeartWiseModelFactory):
    
    name = 'resnet'
    
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
        logger.info("Model %s loaded on %s", model_name, map_location)

    def _load_model(self, model_path: str, map_location: torch.device) -> None:       
        pt_file = next((f for f in os.listdir(model_path) if f.endswith('.pt')), None)
        if not pt_file:
            raise ValueError("No .pt file found in the directory")
        model_path = os.path.join(model_path, pt_file)
        self.model = torch.jit.load(model_path, map_location=map_location)

    def __call__(self, signal):
        signal = signal.to(self.device)
        return torch.sigmoid(self.model(signal))
    
class ResNet_77_classes(ResNetWrapper, HeartWiseModelFactory):
    name = 'resnet_77_classes'
    
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