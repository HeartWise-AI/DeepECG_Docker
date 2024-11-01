# DeepECG_Docker

DeepECG_Docker is a repository designed for deploying deep learning models for ECG signal analysis and comparing their performance over a Bert Classifier model or a specified ground truth. The pipeline can be run locally or in a docker container.
This pipeline offers 3 modes of processing:
- **Preprocessing**: Preprocess the ecg signals and save them in the `preprocessing/` folder
- **Analysis**: Analyze the ecg signals and save the results in the `outputs/` folder (using the preprocessed data)
- **Full run**: Preprocess the ecg signals and analyze them, saving both the preprocessed and analyzed data in the `preprocessing/` and `outputs/` folders respectively

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

- BERT-based multilabel classification model for ECG diagnosis (77 classes)
- EfficientNet-based multilabel classification model for ECG signals (77 classes)
- WCR-based multilabel classification model for ECG signals (77 classes)
- WCR-based binary classification model for ECG signals (LVEF <= 40%)
- WCR-based binary classification model for ECG signals (LVEF < 50%)
- WCR-based binary classification model for ECG signals (Incident AFIB at 5 years)
- EfficientNet-based binary classification model for ECG signals (LVEF <= 40%)
- EfficientNet-based binary classification model for ECG signals (LVEF < 50%)
- EfficientNet-based binary classification model for ECG signals (Incident AFIB at 5 years)
- Dockerized deployment for easy setup and execution
- Configurable pipeline for flexible usage
- CPU & GPU support for accelerated processing

## 🛠️ Installation 

1. 📥 Clone the repository:
   ```
   git clone https://github.com/HeartWise-AI/DeepECG_Deploy.git
   cd DeepECG_Deploy
   ```

2. 🔑 Set up your HuggingFace API key:
   - Create a HuggingFace account if you don't have one yet
   - Ask for access to the DeepECG models need in the [heartwise-ai/DeepECG](https://huggingface.co/collections/heartwise/deepecg-models-66ce09c7d620749ad819fa0d) repository
   - Create an API key in the HuggingFace website in `User Settings` -> `API Keys` -> `Create API Key` -> `Read`
   - Add your API key in the following format in the `api_key.json` file in the root directory:
     ```json
     {
       "huggingface_api_key": "your_api_key_here"
     }
     ```
3. 📄 Populate a csv file containing the data to be processed, example: inputs/data_rows_template.csv (see [Usage](#usage) for more details)

4. 🐳 Build the docker image:
   ```
   docker build -t deepecg-docker .
   ```

5. 🚀 Run the docker container: (see [Docker](#docker) for more details)
   ```
   docker run --gpus "device=0" -v local_path_to_outputs:/app/outputs -v local_path_to_ecg_signals:/app/ecg_signals -v local_path_to_preprocessing:/app/preprocessing -i deepecg-docker full_run
   ```

## Project Structure

```
DeepECG_Docker/
│
├── models/
│   ├── bert_classifier.py
│   ├── efficientnet_wrapper.py
│   ├── heartwise_models_factory.py
│   └── resnet_wrapper.py
│
├── inputs/
│   └── data_rows_template.csv
│
├── outputs/
│   └── (output files will be generated here)
│
├── preprocessing/
│   └── (preprocessed files will be saved here)
│
├── utils/
│   └── ...
│
├── dockerfile
├── heartwise.config
├── api_key.json
├── main.py
├── README.md
├── requirements.txt
└── run_pipeline.sh
```

## Models

1. **BertClassifier**: 
   - model_name: `bert_diagnosis2classification`
   - Utilizes the BERT architecture for sequence classification tasks.
   - Pre-trained model is loaded from HuggingFace, and the model is fine-tuned for diagnosing and classifying ECG diagnosis.
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

4. **EfficientV2_Under_50**:
   - model_name: `efficientnetv2_lvef_under_50`
   - Utilizes the EfficientNetV2 architecture for processing ECG signals.
   - Pre-trained model is loaded from a specified directory, and the model is used to process ECG signal tensors.
   - The model takes an input signal tensor and outputs the logits for the binary classification of LVEF < 50%.

5. **EfficientV2_Incident_AFIB_At_5_Years**:
   - model_name: `efficientnetv2_afib_5y`
   - Utilizes the EfficientNetV2 architecture for processing ECG signals.
   - Pre-trained model is loaded from a specified directory, and the model is used to process ECG signal tensors.
   - The model takes an input signal tensor and outputs the logits for the binary classification of incident AFIB at 5 years.

6. **WCR_77_classes**:
   - model_name: `wcr_77_classes`
   - Utilizes the WCR architecture for processing ECG signals.
   - Pre-trained model is loaded from a specified directory, and the model is used to process ECG signal tensors.
   - The model takes an input signal tensor and outputs the logits for the 77 diagnosis classes.

7. **WCR_LVEF_Equal_Under_40**:
   - model_name: `wcr_lvef_equal_under_40`
   - Utilizes the WCR architecture for processing ECG signals.
   - Pre-trained model is loaded from a specified directory, and the model is used to process ECG signal tensors.
   - The model takes an input signal tensor and outputs the logits for the binary classification of LVEF <= 40%.

8. **WCR_LVEF_Under_50**:
   - model_name: `wcr_lvef_under_50`
   - Utilizes the WCR architecture for processing ECG signals.
   - Pre-trained model is loaded from a specified directory, and the model is used to process ECG signal tensors.
   - The model takes an input signal tensor and outputs the logits for the binary classification of LVEF < 50%.

9. **WCR_Incident_AFIB_At_5_Years**:
   - model_name: `wcr_afib_5y`
   - Utilizes the WCR architecture for processing ECG signals.
   - Pre-trained model is loaded from a specified directory, and the model is used to process ECG signal tensors.
   - The model takes an input signal tensor and outputs the logits for the binary classification of incident AFIB at 5 years.

## 📄 Usage

1. Prepare your input data:
   - Create a CSV file with the following template in inputs/data_rows_template.csv:
   - For each model, add two columns with the following format:
     ```
     'ecg_machine_diagnosis': '77_classes_ecg_file_name',
     'afib_5y': 'afib_ecg_file_name',
     'lvef_40': 'lvef_40_ecg_file_name',
     'lvef_50': 'lvef_50_ecg_file_name'
     ```
     - `ecg_machine_diagnosis`: Diagnosis from the ECG machine
     - `77_classes_ecg_file_name`: The ECG signal **file names** machine ecg diagnosis
     - `afib_5y`: Binary classification of incident AFIB at 5 years
     - `afib_ecg_file_name`: The ECG signal **file names** incident AFIB at 5 years
     - `lvef_40`: Binary classification of LVEF <= 40%
     - `lvef_40_ecg_file_name`: The ECG signal **file names** LVEF <= 40%
     - `lvef_50`: Binary classification of LVEF < 50%
     - `lvef_50_ecg_file_name`: The ECG signal **file names** LVEF < 50%
   - Place your input CSV file in the `inputs/` directory

2. Configure the pipeline:
   - Edit the `heartwise.config` file to set the desired configuration - Most of the time, you'll need to change only the `data_path`.
   ```yaml
   diagnosis_classifier_device: cuda:0
   signal_processing_device: cuda:0
   batch_size: 32
   output_folder: /app/outputs
   hugging_face_api_key_path: /app/api_key.json
   use_efficientnet: True
   use_wcr: True   
   data_path: /app/inputs/data_rows_template.csv # Need to be changed for the actual csv filename
   mode: full_run
   ecg_signals_path: /app/ecg_signals
   preprocessing_folder: /app/preprocessing
   preprocessing_n_workers: 16   
   ```
   - `heartwise.config` file contains the configuration settings for the pipeline. Below is a description of each configuration parameter:

     - `diagnosis_classifier_device`: Specifies the device to be used for the diagnosis classifier model. Example: `cuda:0` for using the first GPU.
     - `signal_processing_device`: Specifies the device to be used for the signal processing model. Example: `cuda:0` for using the first GPU.
     - `batch_size`: Defines the batch size for processing the data. Example: `32`.
     - `output_folder`: The directory where the output files will be saved. Example: `/app/outputs`.
     - `hugging_face_api_key_path`: The path to the file containing the HuggingFace API key. Example: `/app/api_key.json`.
     - `use_efficientnet`: Boolean value to specify if the EfficientNet model should be used. Example: `True`.
     - `use_wcr`: Boolean value to specify if the WCR model should be used. Example: `True`.
     - `data_path`: The path to the input CSV file containing the data. Example: `/app/inputs/data_rows_template.csv`.
     - `mode`: The mode of the pipeline (overwriten by docker command line). Example: `analysis` | `preprocessing` | `full_run`.
     - `ecg_signals_path`: The path to the ecg signals files parsed in docker command line. Example: `/app/ecg_signals`.
     - `preprocessing_folder`: The path to the folder where the preprocessed files will be saved. Example: `/preprocessing`.
     - `preprocessing_n_workers`: The number of workers to be used for the preprocessing. Example: `16`.

   - Ensure that the paths and model names in the `heartwise.config` file are correctly set according to your setup and requirements.

3. Run the pipeline:
   - If using Docker, follow the [Docker](#docker) instructions below

4. Retrieve results:
   - Check the `outputs/` directory for the generated results file
   - Check the `preprocessing/` directory for the generated preprocessed files

## 🐳 Docker

### Running the Docker Container

To run the Docker container, use one of the following commands based on your hardware:

**For full run:**
Run both preprocessing and analysis:
```
docker run --gpus "device=0" -v local_path_to_outputs:/app/outputs -v local_path_to_ecg_signals:/app/ecg_signals -v local_path_to_preprocessing:/app/preprocessing -i deepecg-docker full_run
```

**For preprocessing:**
Run only preprocessing:
```
docker run -v local_path_to_outputs:/app/outputs -v local_path_to_ecg_signals:/app/ecg_signals -v local_path_to_preprocessing:/app/preprocessing -i deepecg-docker preprocessing
```

**For analysis:**
Run only analysis:
```
docker run --gpus "device=0" -v local_path_to_outputs:/app/outputs -v local_path_to_preprocessing:/app/preprocessing -i deepecg-docker analysis
```

**Without GPU (CPU only):**
Note recommanded for WCR models. Note that models device in heartwise.config should be set to "cpu"
```
docker run -v local_path_to_outputs:/outputs -v local_path_to_ecg_signals:/ecg_signals -v local_path_to_preprocessing:/preprocessing -i deepecg-docker full_run
```

These commands mount the `outputs/`, `ecg_signals/` and `preprocessing/` directories from your local machine to the container, allowing you to easily provide input data and retrieve results.

## 💻 Local run

1. Create a virtual environment:
   ```
   python -m venv deploy-venv
   source deploy-venv/bin/activate
   ```

2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

3. Install the following package manually:
   ```
   git clone https://github.com/HeartWise-AI/fairseq-signals && \
   cd fairseq-signals && \
   pip install --editable ./
   ```

4. Run the pipeline:
   Option 1: execute the main script with the correct arguments:
     ```
     python main.py --diagnosis_classifier_device cuda:1 --signal_processing_device cuda:1 --batch_size 32 --output_folder /outputs --hugging_face_api_key_path /app/api_key.json --output_file_name results --use_efficientnet True --use_wcr True --data_path /inputs/data_rows_template.csv --ecg_signals_path /ecg_signals_folder --mode full_run --preprocessing_folder /preprocessing --preprocessing_n_workers 16
     ```
   Option 2: execute the bash script:
     ```
     bash run_pipeline.bash
     ```

To run the pipeline locally, see [Usage](#usage)

## 🤝 Contributing

Contributions to DeepECG_Docker repository are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with clear, descriptive messages
4. Push your changes to your fork
5. Submit a pull request to the main repository

## 📚 Citation

If you find this repository useful, please cite our work:

```
@article{,
  title={},
  author={},
  journal={},
  year={},
  publisher={}
}
```

