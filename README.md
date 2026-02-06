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
- [Testing](#testing)
- [Docker](#docker)
- [Output Folder Structure](#-output-folder-structure)
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
   git clone https://github.com/HeartWise-AI/DeepECG_Docker.git
   cd DeepECG_Docker
   ```

2. 🔑 Set up your HuggingFace API key:
   - Create a HuggingFace account if you don't have one yet
   - Ask for access to the DeepECG models needed in the [heartwise-ai/DeepECG](https://huggingface.co/collections/heartwise/deepecg-models-66ce09c7d620749ad819fa0d) repository
   - Create an API key in the HuggingFace website in `User Settings` -> `API Keys` -> `Create API Key` -> `Read`
   - Add your API key in the following format in the `api_key.json` file in the root directory:
     ```json
     {
       "huggingface_api_key": "your_api_key_here"
     }
     ```
3. 📄 Populate a csv file containing the data to be processed, example: inputs/data_rows_template.csv (see [Usage](#usage) for more details)
   - If using DICOMs, update the root path in [extract_metada_from_dicoms.py](utils/extract_metada_from_dicoms.py) then run the script to extract the metadata from the DICOMs
      ```
      python utils/extract_metada_from_dicoms.py
      ```

4. 🐳 Build the docker image:
   ```
   docker build -t deepecg-docker .
   ```

5. 🚀 Run the docker container: (see [Docker](#docker) for more details)
   ```
   docker run --gpus all -v $(pwd)/inputs:/app/inputs -v $(pwd)/outputs:/app/outputs -v $(pwd)/ecg_signals:/app/ecg_signals:ro -v $(pwd)/preprocessing:/app/preprocessing -i deepecg-docker
   ```

6. Connect to the container
   ```
   docker exec -it deepecg_docker bash
   ```

7. Run pipeline
   ```
   bash run_pipeline.bash --mode full_run --csv_file_name data_rows_template.csv
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
│   ├── batch_1/                    # Preprocessing reports per batch
│   │   ├── ecg_processing_detailed_report.csv
│   │   └── ecg_processing_summary_report.csv
│   ├── {model}_{date}_{task}.json           # Metrics (JSON)
│   ├── {model}_{date}_{task}.csv            # Metrics (CSV)
│   ├── {model}_{date}_{task}_probabilities.csv  # Per-file predictions
│   └── missing_files_{date}.csv             # Files not found (if any)
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
   - Utilizes the BERT architecture fine-tuned to classify ECG diagnosis into 77 classes.
   - More information [here](https://huggingface.co/heartwise/bert_diagnosis2classification)

2. **EfficientV2_77_classes**:
   - Utilizes the EfficientNetV2 architecture to classify ECG signals into 77 classes.
   - More information [here](https://huggingface.co/heartwise/efficientnetv2_77_classes)

3. **EfficientV2_LVEF_Equal_Under_40**:
   - Utilizes the EfficientNetV2 architecture to classify ECG signals into binary classification of LVEF <= 40%.
   - More information [here](https://huggingface.co/heartwise/efficientnetv2_lvef_equal_under_40)

4. **EfficientV2_Under_50**:
   - Utilizes the EfficientNetV2 architecture to classify ECG signals into binary classification of LVEF < 50%.
   - More information [here](https://huggingface.co/heartwise/efficientnetv2_lvef_under_50)

5. **EfficientV2_Incident_AFIB_At_5_Years**:
   - Utilizes the EfficientNetV2 architecture to classify ECG signals into binary classification of incident AFIB at 5 years.
   - More information [here](https://huggingface.co/heartwise/efficientnetv2_afib_5y)

6. **WCR_77_classes**:
   - Utilizes the WCR architecture to classify ECG signals into 77 classes.
   - More information [here](https://huggingface.co/heartwise/wcr_77_classes)

7. **WCR_LVEF_Equal_Under_40**:
   - Utilizes the WCR architecture to classify ECG signals into binary classification of LVEF <= 40%.
   - More information [here](https://huggingface.co/heartwise/wcr_lvef_equal_under_40)

8. **WCR_LVEF_Under_50**:
   - Utilizes the WCR architecture to classify ECG signals into binary classification of LVEF < 50%.
   - More information [here](https://huggingface.co/heartwise/wcr_lvef_under_50)

9. **WCR_Incident_AFIB_At_5_Years**:
   - Utilizes the WCR architecture to classify ECG signals into binary classification of incident AFIB at 5 years.
   - More information [here](https://huggingface.co/heartwise/wcr_afib_5y)

## 📄 Usage

1. Prepare your input data:
   - Create a CSV file with the following template in [inputs/data_rows_template.csv](inputs/data_rows_template.csv):
   - For each model, add two columns with the following format:
     ```
     'ecg_machine_diagnosis': '77_classes_ecg_file_name',
     'afib_5y': 'afib_ecg_file_name',
     'lvef_40': 'lvef_40_ecg_file_name',
     'lvef_50': 'lvef_50_ecg_file_name'
     ```
   - `ecg_machine_diagnosis` (string): Diagnosis from the ECG machine
   - `77_classes_ecg_file_name` (string): The ECG signal **file names** machine ecg diagnosis
   - `afib_5y` (int): Binary classification of incident AFIB at 5 years
   - `afib_ecg_file_name` (string): The ECG signal **file names** incident AFIB at 5 years
   - `lvef_40` (int): Binary classification of LVEF <= 40%
   - `lvef_40_ecg_file_name` (string): The ECG signal **file names** LVEF <= 40%
   - `lvef_50` (int): Binary classification of LVEF < 50%
   - `lvef_50_ecg_file_name` (string): The ECG signal **file names** LVEF < 50%
   - Place your input CSV file in the `inputs/` directory
   - Change the `data_rows_template.csv` filename in the `heartwise.config` file

2. Pipeline configuration:
   - When using docker, you only need to change the actual csv filename. Edit the [heartwise.config](heartwise.config) file to set the desired configuration:

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
     - `preprocessing_folder`: The path to the folder where the preprocessed files will be saved. Example: `/app/preprocessing`.
     - `preprocessing_n_workers`: The number of workers to be used for the preprocessing. Example: `16`.

3. Notes:
   - **Single ECG processing**: When running the pipeline with only one ECG file, metrics computation (AUC, F1, etc.) is automatically skipped since these metrics require multiple samples. Predictions are still generated and saved normally.

## Testing

Run the error-collector unit tests (no GPU or data required):

```bash
python tests/test_error_collector.py
```

To verify that errors are collected and printed at the end (no data or GPU needed):

```bash
python main.py \
  --mode analysis \
  --data_path /nonexistent.csv \
  --output_folder /tmp/out \
  --preprocessing_folder /tmp/pre \
  --hugging_face_api_key_path api_key.json \
  --use_wcr False \
  --use_efficientnet False \
  --ecg_signals_path /tmp
```

You should see `Errors encountered:` followed by a clear message (e.g. file not found) instead of a raw traceback.

Run the full pipeline from the project root (requires `heartwise.config` and data). From inside the container or after installing dependencies locally:

```bash
bash run_pipeline.bash --mode full_run --csv_file_name data_rows_template.csv
```

To run `main.py` directly with explicit arguments:

```bash
python main.py \
  --mode analysis \
  --data_path inputs/your_data.csv \
  --output_folder outputs \
  --preprocessing_folder preprocessing \
  --hugging_face_api_key_path api_key.json \
  --use_wcr True \
  --use_efficientnet True \
  --ecg_signals_path ecg_signals
```

If any step fails, the pipeline collects error messages and prints them at the end under `Errors encountered:`.

## 🐳 Docker

### Interactive shell (recommended for Cursor / IDE terminals)

If `docker run -it ...` hangs or shows a blank screen in Cursor’s terminal, start the container in the background and attach a shell with `docker exec -it`. The image keeps the container running by default.

**1. Start the container (no `-it`):**
```bash
docker run -d --gpus all --name deepecg \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/ecg_signals:/app/ecg_signals:ro \
  -v $(pwd)/preprocessing:/app/preprocessing \
  deepecg-docker
```

**2. Open an interactive shell:**
```bash
docker exec -it deepecg bash
```

You’ll get a prompt inside the container. Run the pipeline manually when you’re ready, e.g.:
```bash
./run_pipeline.bash --mode full_run --csv_file_name data_rows_template.csv
```

When you’re done, exit the shell (`exit`) and stop the container: `docker stop deepecg`. Remove it before the next run if you reuse the name: `docker rm deepecg` (or use `docker rm -f deepecg` to remove a running container).

## 📂 Output Folder Structure

After running the pipeline, the `outputs/` folder contains the following files:

### Preprocessing Reports (per batch)

Located in `outputs/batch_X/` where X is the batch number:

**`ecg_processing_detailed_report.csv`** - Per-file processing status:

| Column | Description |
|--------|-------------|
| `file_id` | ECG file identifier (without extension) |
| `xml_type` | Detected XML format (e.g., `CLSA`, `MHI`) |
| `status` | `Success` or `Failed` |
| `message` | Error message if failed, empty if successful |

**`ecg_processing_summary_report.csv`** - Aggregate statistics:

| Metric | Value |
|--------|-------|
| Total Files | Number of files processed |
| Successful Files | Number of files successfully processed |
| Failed Files | Number of files that failed |
| XML Type: {type} | Count per XML format detected |

### Model Predictions and Metrics

Generated in the root `outputs/` folder with naming pattern `{model}_{datetime}_{task}`:

| File Pattern | Description |
|--------------|-------------|
| `{model}_{datetime}_{task}.json` | Performance metrics in JSON format |
| `{model}_{datetime}_{task}.csv` | Same metrics in CSV format |
| `{model}_{datetime}_{task}_probabilities.csv` | Per-file prediction probabilities |
| `missing_files_{datetime}.csv` | List of ECG files not found on disk |

**Example filenames** (format: `{model}_{YYYYMMDD_HHMMSS}_{task}`):
- `efficientnetv2_77_classes_20260206_143052_ecg_machine_diagnosis.json`
- `efficientnetv2_77_classes_20260206_143052_ecg_machine_diagnosis_probabilities.csv`
- `wcr_afib_5y_20260206_143052_afib_5y.json`

### Probabilities CSV Columns

For **77-class models** (`ecg_machine_diagnosis`), the CSV contains 155 columns:

| Column Pattern | Count | Description |
|----------------|-------|-------------|
| `file_name` | 1 | ECG file identifier |
| `{pattern}_bert_model` | 77 | BERT classifier probability for each ECG pattern |
| `{pattern}_sig_model` | 77 | Signal model (EfficientNet/WCR) probability for each pattern |

Example patterns: `Sinusal`, `Afib`, `Left bundle branch block`, `Left ventricular hypertrophy`, etc.

For **binary models** (`afib_5y`, `lvef_40`, `lvef_50`):

| Column | Description |
|--------|-------------|
| `file_name` | ECG file identifier |
| `ground_truth` | Label from input CSV (0 or 1) |
| `predictions` | Model prediction probability (0.0 to 1.0) |

### Metrics JSON Structure

The JSON contains metrics grouped by **diagnostic category** and **individual patterns**:

**Category-level metrics** (e.g., "Rhythm Disorders", "Conduction Disorder"):

```json
{
  "Rhythm Disorders": {
    "macro_auc": 0.967,
    "macro_auprc": 0.670,
    "macro_f1": 0.431,
    "micro_auc": 0.997,
    "micro_auprc": 0.991,
    "micro_f1": 0.984,
    "threshold": 0.156,
    "prevalence_gt %": 16.07,
    "prevalence_pred %": 16.96
  }
}
```

**Individual pattern metrics** (e.g., "Sinusal", "Afib", "Left bundle branch block"):

```json
{
  "Sinusal": {
    "auc": 0.936,
    "auprc": 0.996,
    "threshold": 0.994,
    "f1": 0.952,
    "prevalence_gt %": 96.28,
    "prevalence_pred %": 88.38
  }
}
```

| Metric | Description |
|--------|-------------|
| `auc` / `macro_auc` / `micro_auc` | Area Under ROC Curve |
| `auprc` / `macro_auprc` / `micro_auprc` | Area Under Precision-Recall Curve |
| `f1` / `macro_f1` / `micro_f1` | F1 Score |
| `threshold` | Optimal classification threshold |
| `prevalence_gt %` | Ground truth prevalence percentage |
| `prevalence_pred %` | Predicted prevalence percentage |

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
@article {Nolin-Lapalme2025.03.02.25322575,
	author = {Nolin-Lapalme, Alexis and Sowa, Achille and Delfrate, Jacques and Tastet, Olivier and Corbin, Denis and Kulbay, Merve and Ozdemir, Derman and No{\"e}l, Marie-Jeanne and Marois-Blanchet, Fran{\c c}ois-Christophe and Harvey, Fran{\c c}ois and Sharma, Surbhi and Ansari, Minhaj and Chiu, I-Min and Dsouza, Valentina and Friedman, Sam F. and Chass{\'e}, Micha{\"e}l and Potter, Brian J. and Afilalo, Jonathan and Elias, Pierre Adil and Jabbour, Gilbert and Bahani, Mourad and Dub{\'e}, Marie-Pierre and Boyle, Patrick M. and Chatterjee, Neal A. and Barrios, Joshua and Tison, Geoffrey H. and Ouyang, David and Maddah, Mahnaz and Khurshid, Shaan and Cadrin-Tourigny, Julia and Tadros, Rafik and Hussin, Julie and Avram, Robert},
	title = {Foundation models for generalizable electrocardiogram interpretation: comparison of supervised and self-supervised electrocardiogram foundation models},
	elocation-id = {2025.03.02.25322575},
	year = {2025},
	doi = {10.1101/2025.03.02.25322575},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2025/03/05/2025.03.02.25322575},
	eprint = {https://www.medrxiv.org/content/early/2025/03/05/2025.03.02.25322575.full.pdf},
	journal = {medRxiv}
}

```

