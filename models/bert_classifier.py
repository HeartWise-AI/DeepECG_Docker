

import os
import torch

from utils.huggingface_wrapper import HuggingFaceWrapper
from utils.log_config import get_logger
from models.heartwise_model_factory import HeartWiseModelFactory
from transformers import BertTokenizer, BertForSequenceClassification

logger = get_logger(__name__)

class BaseBertClassifier(HeartWiseModelFactory):
    name = 'base_bert_classifier'
    
    def __init__(
        self, 
        model_name: str, 
        map_location: torch.device, 
        hugging_face_api_key: str,
        num_classes: int = 77
    ) -> None:
        self.device = map_location
        self._load_model(
            model_path=HuggingFaceWrapper.get_model(
                repo_id=f"heartwise/{model_name}", 
                local_dir=os.path.join("weights", model_name),
                hugging_face_api_key=hugging_face_api_key
            ), 
            map_location=map_location, 
            num_classes=num_classes
        )
        logger.info("Model %s loaded on %s", model_name, map_location)

    def _load_model(self, model_path: str, map_location: torch.device, num_classes: int) -> None:
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_classes,
        ).to(map_location)

        self.processor = BertTokenizer.from_pretrained(
            model_path,
        )

    def preprocessing(self, text: str) -> dict:
        return self.processor(
            text,
            padding='max_length', 
            max_length=512, 
            truncation=True,
            return_tensors='pt', 
        )

    def __call__(self, text: str) -> torch.Tensor:
        batch_t = self.preprocessing(text)
        input_ids = batch_t['input_ids'].to(self.device)
        token_type_ids = batch_t['token_type_ids'].to(self.device)
        attention_mask = batch_t['attention_mask'].to(self.device)
        logits = self.model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )['logits']
        return torch.sigmoid(logits)   

class BertClassifier(BaseBertClassifier):
    
    name = 'Bert_diagnosis2classification'
    
    def __init__(
        self, 
        model_name: str, 
        map_location: torch.device, 
        hugging_face_api_key: str,
        num_classes: int = 77
    ) -> None:
        super().__init__(
            model_name=model_name, 
            map_location=map_location, 
            hugging_face_api_key=hugging_face_api_key,
            num_classes=num_classes
        )

class BertClassifier_En_Fr(BaseBertClassifier):
    
    name = 'Bert_diagnosis2classification_En_Fr'
    
    def __init__(
        self, 
        model_name: str, 
        map_location: torch.device, 
        hugging_face_api_key: str,
        num_classes: int = 77
    ) -> None:
        super().__init__(
            model_name=model_name, 
            map_location=map_location, 
            hugging_face_api_key=hugging_face_api_key,
            num_classes=num_classes
        )

