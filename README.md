# DeepECG_Deploy

DeepECG_Deploy is a repository designed for deploying deep learning models for ECG signal analysis. This repository includes implementations of various models, such as BERT for sequence classification and EfficientNet for signal processing.

### Models

1. **BertClassifier**: 
   - Located in `models/bert_classifier.py`
   - Utilizes the BERT architecture for sequence classification tasks.
   - Pre-trained model is loaded from HuggingFace, and the model is fine-tuned for diagnosing and classifying ECG signals.
   - The model processes input text and outputs classification logits.

2. **EfficientNetWrapper**:
   - Located in `models/efficientnet_wrapper.py`
   - Utilizes the EfficientNetV2 architecture for processing ECG signals.
   - Pre-trained model is loaded from a specified directory, and the model is used to process ECG signal tensors.
   - The model takes an input signal tensor and outputs the processed result.

### Usage

To use the models, you can refer to the `main.py` script which demonstrates how to load and use the models for inference.
