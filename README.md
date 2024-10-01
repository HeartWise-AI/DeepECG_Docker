# DeepECG_Deploy

DeepECG_Deploy is a repository designed for deploying deep learning models for ECG signal analysis and comparing their performance over a Bert Classifier model. The pipeline can be run locally or in a docker container.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Models](#models)
- [Configuration](#configuration)
- [Usage](#usage)
- [Docker](#docker)
- [Contributing](#contributing)
- [Citation](#citation)

## 🚀 Features

- BERT-based multilabel classification for ECG diagnosis (77 classes)
- EfficientNet-based multilabel classification for ECG signals (77 classes)
- WCR-based multilabel classification for ECG signals (77 classes)
- WCR-based binary classification for ECG signals (LVEF <= 40%)
- WCR-based binary classification for ECG signals (LVEF < 50%)
- WCR-based binary classification for ECG signals (Incident AFIB at 5 years)
- EfficientNet-based binary classification for ECG signals (LVEF <= 40%)
- EfficientNet-based binary classification for ECG signals (LVEF < 50%)
- EfficientNet-based binary classification for ECG signals (Incident AFIB at 5 years)
- Dockerized deployment for easy setup and execution
- Configurable pipeline for flexible usage
- GPU support for accelerated processing

## 🛠️ Installation

1. Clone the repository:
   ```
   git clone https://github.com/HeartWise-AI/DeepECG_Deploy.git
   cd DeepECG_Deploy
   ```

2. Set up your HuggingFace API key:
   - Create a file named `api_key.json` in the root directory
   - Add your API key in the following format:
     ```json
     {
       "huggingface_api_key": "your_api_key_here"
     }
     ```

3. If running locally, optionally, create a new virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate
   ```

4. If running locally, install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
DeepECG_Deploy/
│
├── models/
|   ├── modules
|   |   ├── EfficientNetv2.py
│   ├── bert_classifier.py
│   └── efficientnet_wrapper.py
│
├── inputs/
│   └── data_rows.csv
│
├── outputs/
│   └── (output files will be generated here)
│
├── preprocessing/
│   └── (preprocessed files will be saved here)
│
├── Dockerfile
├── heartwise.config
├── api_key.json
├── requirements.txt
└── README.md
```

## Models

1. **BertClassifier**: 
   - model_name: `bert_diagnosis2classification`
   - Utilizes the BERT architecture for sequence classification tasks.
   - Pre-trained model is loaded from HuggingFace, and the model is fine-tuned for diagnosing and classifying ECG signals.
   - The model processes input text and outputs classification logits.

2. **EfficientV2_77_classes**:
   - model_name: `efficientnetv2_77_classes`
   - Utilizes the EfficientNetV2 architecture for processing ECG signals.
   - Pre-trained model is loaded from a specified directory, and the model is used to process ECG signal tensors.
   - The model takes an input signal tensor and outputs the logits for the 77 diagnosis classes.

3. **EfficientV2_LVEF_Equal_Under_40**:
   - model_name: `efficientnetv2_lvef_equal_under_40`
   - Utilizes the EfficientNetV2 architecture for processing ECG signals.
   - Pre-trained model is loaded from a specified directory, and the model is used to process ECG signal tensors.
   - The model takes an input signal tensor and outputs the logits for the binary classification of LVEF <= 40%.

4. **EfficientV2_LVEF_Less_Than_50**:
   - model_name: `efficientnetv2_lvef_less_than_50`
   - Utilizes the EfficientNetV2 architecture for processing ECG signals.
   - Pre-trained model is loaded from a specified directory, and the model is used to process ECG signal tensors.
   - The model takes an input signal tensor and outputs the logits for the binary classification of LVEF < 50%.

5. **EfficientV2_Incident_AFIB_At_5_Years**:
   - model_name: `efficientnetv2_incident_afib_at_5_years`
   - Utilizes the EfficientNetV2 architecture for processing ECG signals.
   - Pre-trained model is loaded from a specified directory, and the model is used to process ECG signal tensors.
   - The model takes an input signal tensor and outputs the logits for the binary classification of incident AFIB at 5 years.


## 📄 Usage

1. Prepare your input data:
   - Create a CSV file with the following template in inputs/data_rows_template.csv:
     - `diagnosis`: Text Diagnosis of the ECG signal 
     - `ECG_file`: The ECG signal file name
   - Place your input CSV file in the `inputs/` directory

2. Configure the pipeline:
   - Edit the `heartwise.config` file to set the desired configuration
   ```yaml
   diagnosis_classifier_device: cuda:0
   signal_processing_device: cuda:0
   batch_size: 32
   output_folder: /outputs
   hugging_face_api_key_path: /app/api_key.json
   output_file_name: results # Do not add file extension
   signal_processing_model_name: efficientnetv2
   diagnosis_classifier_model_name: bert_diagnosis2classification
   data_path: /inputs/data_rows_template.csv
   ecg_signals_path: /inputs/ecg_signals_template.csv
   ```
   - Edit `heartwise.config` file contains the configuration settings for the pipeline. Below is a description of each configuration parameter:

     - `diagnosis_classifier_device`: Specifies the device to be used for the diagnosis classifier model. Example: `cuda:1` for using the second GPU.
     - `signal_processing_device`: Specifies the device to be used for the signal processing model. Example: `cuda:1` for using the second GPU.
     - `batch_size`: Defines the batch size for processing the data. Example: `32`.
     - `output_folder`: The directory where the output files will be saved. Example: `/outputs`.
     - `hugging_face_api_key_path`: The path to the file containing the HuggingFace API key. Example: `/app/api_key.json`.
     - `output_file_name`: The name of the output file (without extension) where results will be saved. Example: `results`.
     - `signal_processing_model_name`: The name of the signal processing model to be used. Example: `efficientnetv2`.
     - `diagnosis_classifier_model_name`: The name of the diagnosis classifier model to be used. Example: `bert_diagnosis2classification`.
     - `data_path`: The path to the input CSV file containing the data. Example: `/inputs/data_rows_template.csv`.
     - `ecg_signals_path`: The path to the ecg signals files parsed in docker command line. Example: `/ecg_signals_folder`.
     - `mode`: The mode of the pipeline. Example: `analysis` | `preprocessing` | `full_run`.
     - `preprocessing_folder`: The path to the folder where the preprocessed files will be saved. Example: `/preprocessing`.
     - `preprocessing_n_workers`: The number of workers to be used for the preprocessing. Example: `16`.

   - Ensure that the paths and model names in the `heartwise.config` file are correctly set according to your setup and requirements.

3. Run the pipeline:
   - If using Docker, follow the Docker instructions below
   - If running locally:
     Option 1: execute the main script with the correct arguments:
     ```
     python main.py --diagnosis_classifier_device cuda:1 --signal_processing_device cuda:1 --batch_size 32 --output_folder /outputs --hugging_face_api_key_path /app/api_key.json --output_file_name results --signal_processing_model_name efficientnetv2 --diagnosis_classifier_model_name bert_diagnosis2classification --data_path /inputs/data_rows_template.csv --ecg_signals_path /ecg_signals_folder
     ```
     Option 2: execute the bash script:
     ```
     bash run_pipeline.sh
     ```

4. Retrieve results:
   - Check the `outputs/` directory for the generated results file
   - Check the `preprocessing/` directory for the generated preprocessed files

## 🐳 Docker

### Building the Docker Image

To build the Docker image, run the following command in the root directory of the project:
Note that heartwise.config should be changed based on the mode (preprocessing, analysis or full run)

```
docker build -t deepecg-deploy .
```

### Running the Docker Container

To run the Docker container, use one of the following commands based on your hardware:

**For preprocessing:**
```
docker run -v local_path_to_outputs:/app/outputs -v local_path_to_ecg_signals:/app/ecg_signals -v local_path_to_preprocessing:/app/preprocessing -i deepecg-deploy preprocessing
```

**For analysis:**
```
docker run --gpus "device=0" -v local_path_to_outputs:/app/outputs -v local_path_to_preprocessing:/app/preprocessing -i deepecg-deploy analysis
```

**For full run:**
```
docker run --gpus "device=0" -v local_path_to_outputs:/app/outputs -v local_path_to_ecg_signals:/app/ecg_signals -v local_path_to_preprocessing:/app/preprocessing -i deepecg-deploy full_run
```

**Without GPU (CPU only):**
Note that models device in heartwise.config should be set to "cpu"
```
docker run -v local_path_to_outputs:/outputs -v local_path_to_ecg_signals:/ecg_signals -v local_path_to_preprocessing:/preprocessing -i deepecg-deploy full_run
```

These commands mount the `outputs/` and `ecg_signals/` directories from your local machine to the container, allowing you to easily provide input data and retrieve results.

## 🤝 Contributing

Contributions to DeepECG_Deploy are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with clear, descriptive messages
4. Push your changes to your fork
5. Submit a pull request to the main repository

## 📚 Citation

If you find this repository useful, please consider citing our work:

```
@article{,
  title={},
  author={},
  journal={},
  year={},
  publisher={}
}
```

