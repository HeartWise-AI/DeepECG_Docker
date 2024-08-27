

import os
import torch

from models.model_factory import ModelFactory
from utils.huggingface_wrapper import HuggingFaceWrapper
from transformers import BertTokenizer, BertForSequenceClassification

class BertClassifier(ModelFactory):
    
    name = 'bert_diagnosis2classification'
    
    def __init__(
        self, 
        model_name: str, 
        map_location: torch.device, 
        num_classes: int = 77
    ) -> None:
        self.device = map_location
        self._load_model(
            model_path=HuggingFaceWrapper.get_model(
                model_name=model_name, 
                repo_id="heartwise/Bert_diagnosis2classification", 
                local_dir=os.path.join("weights", model_name)
            ), 
            map_location=map_location, 
            num_classes=num_classes
        )
        print(f"Model {model_name} loaded to {map_location}")

    def _load_model(self, model_path: str, map_location: torch.device, num_classes: int) -> None:
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_classes,
        ).to(map_location)

        self.processor = BertTokenizer.from_pretrained(
            model_path,
        )

    def preprocessing(self, text: str) -> dict:
        # Process inputs and transfer to device
        batch_t = self.processor(
            text,
            padding='max_length', 
            max_length=128, 
            truncation=True,
            return_tensors='pt', 
        )
        
        return batch_t


    def __call__(self, text: str) -> torch.Tensor:
        batch_t = self.preprocessing(text)
        input_ids = batch_t['input_ids'].to(self.device)
        token_type_ids = batch_t['token_type_ids'].to(self.device)
        attention_mask = batch_t['attention_mask'].to(self.device)
        return self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)['logits']

