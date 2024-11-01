
import os
import sys
import torch
import importlib.util

from utils.huggingface_wrapper import HuggingFaceWrapper
from models.heartwise_model_factory import HeartWiseModelFactory

project_dir = os.path.abspath("./fairseq-signals")
root_dir = project_dir
if not root_dir in sys.path:
    sys.path.append(root_dir)

spec = importlib.util.spec_from_file_location("checkpoint_utils", f"{project_dir}/fairseq_signals/utils/checkpoint_utils.py")
checkpoint_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint_utils)

class WCREcgTransformer(HeartWiseModelFactory):
    name = 'wrc_ecg_transformer'
    
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
        
    def _load_model(self, model_path: str, map_location: torch.device) -> None:       
        overrides = {"model_path": os.path.join(model_path, "base_ssl.pt")}
        if not os.path.exists(overrides["model_path"]):
            raise ValueError("No base_ssl.pt file found in the directory")
        
        pt_file = next((f for f in os.listdir(model_path) if f.endswith('.pt') and f != "base_ssl.pt"), None)
        if not pt_file:
            raise ValueError("No .pt file found in the directory") 
        model_path = os.path.join(model_path, pt_file)
        
        model, _, _ = checkpoint_utils.load_model_and_task(
            model_path,
            arg_overrides=overrides,
            suffix=""
        )        
        self.model = model.to(map_location)

    def __call__(self, x, padding_mask=None):
        net_input = { "source": x, "padding_mask": padding_mask}
        net_output = self.model(**net_input)
        return torch.sigmoid(self.model.get_logits(net_output))


class WCR_77_classes(WCREcgTransformer, HeartWiseModelFactory):
    name = 'wcr_77_classes'
    
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
        
class WCR_LVEF_Equal_Under_40(WCREcgTransformer, HeartWiseModelFactory):
    name = 'wcr_lvef_equal_under_40'
    
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

class WCR_LVEF_Under_50(WCREcgTransformer, HeartWiseModelFactory):
    name = 'wcr_lvef_under_50'
    
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

class WCR_AFIB_5Y(WCREcgTransformer, HeartWiseModelFactory):
    name = 'wcr_afib_5y'
    
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