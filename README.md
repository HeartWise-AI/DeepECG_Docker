# DeepECG_Docker

DeepECG_Docker is a repository designed for deploying deep learning models for ECG signal analysis and comparing their performance over a Bert Classifier model or a specified ground truth. The pipeline have to be run in a docker container for backward library compatibility.
This pipeline offers 3 modes of processing:
- **Preprocessing**: Preprocess the ecg signals and save them in the `preprocessing/` folder
- **Analysis**: Analyze the ecg signals and save the results in the `outputs/` folder (using the preprocessed data)
- **Full run**: Preprocess the ecg signals and analyze them, saving both the preprocessed and analyzed data in the `preprocessing/` and `outputs/` folders respectively

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Models](#models)
- [Optimal Thresholds](#-optimal-thresholds)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Docker](#docker)
- [Output Folder Structure](#-output-folder-structure)
- [Adding Support for New XML Formats](#-adding-support-for-new-xml-formats)
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
       "HUGGING_FACE_API_KEY": "your_api_key_here"
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
   docker run --gpus all -v $(pwd)/api_key.json:/app/api_key.json:ro -v $(pwd)/inputs:/app/inputs -v $(pwd)/outputs:/app/outputs -v $(pwd)/ecg_signals:/app/ecg_signals:ro -v $(pwd)/preprocessing:/app/preprocessing -i deepecg-docker
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
│   ├── data_rows_template.csv          # Mixed NPY/XML example
│   ├── data_rows_template_npy.csv      # NPY-only example
│   └── data_rows_template_xml.csv      # XML-only example
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
├── thresholds/
│   └── wcr_77_classes_ecg_machine_diagnosis.json  # WCR optimal thresholds (MHI training)
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

## 📊 Optimal Thresholds

Optimal classification thresholds are stored in `utils/constants.py` and in JSON files under `thresholds/`. During inference, these thresholds are used to binarize model predictions and log diagnostic warnings.

### Threshold Status by Model

| Model | Task | Thresholds | Source | Location |
|-------|------|------------|--------|----------|
| BERT Classifier | 77 classes | Available (77 labels) | Manual tuning | `utils/constants.py` → `BERT_THRESHOLDS` |
| WCR | 77 classes | Available (76 labels, Brugada N/A) | MHI training data (Youden index) | `utils/constants.py` → `WCR_THRESHOLDS`, `thresholds/wcr_77_classes_ecg_machine_diagnosis.json` |
| EfficientNetV2 | 77 classes | **Missing** | — | — |
| WCR | LVEF <= 40% | Available | MHI training data | [thresholds/digital_biomarker_final.json](https://github.com/HeartWise-AI/DeepECG_Docker/blob/main/thresholds/digital_biomarker_final.json) |
| WCR | LVEF < 50% | Available | MHI training data | [thresholds/digital_biomarker_final.json](https://github.com/HeartWise-AI/DeepECG_Docker/blob/main/thresholds/digital_biomarker_final.json) |
| WCR | AFIB 5Y | Available | MHI training data | [thresholds/digital_biomarker_final.json](https://github.com/HeartWise-AI/DeepECG_Docker/blob/main/thresholds/digital_biomarker_final.json) |
| WCR | LQTS | Available | MHI training data | [thresholds/digital_biomarker_final.json](https://github.com/HeartWise-AI/DeepECG_Docker/blob/main/thresholds/digital_biomarker_final.json) |
| WCR | LQTS TYPE | Available | MHI training data | [thresholds/digital_biomarker_final.json](https://github.com/HeartWise-AI/DeepECG_Docker/blob/main/thresholds/digital_biomarker_final.json) |
| EfficientNetV2 | LVEF <= 40% | **Missing** | — | — |
| EfficientNetV2 | LVEF < 50% | **Missing** | — | — |
| EfficientNetV2 | AFIB 5Y | **Missing** | — | — |
| EfficientNetV2 | LQTS | **Missing** | — | — |
| EfficientNetV2 | LQTS TYPE | **Missing** | — | — |

### Notes
- **WCR 77-class thresholds** were computed on the MHI training set using the Youden index (maximizing sensitivity + specificity - 1). The full JSON includes per-label AUC, AUPRC, F1, threshold, and prevalence metrics.
- **Brugada** has no threshold (no positive samples in training data).
- During WCR 77-class inference, the pipeline logs the number of predictions exceeding each label's threshold for monitoring purposes.
- Binary models (LVEF, AFIB 5Y) compute thresholds dynamically at evaluation time using the Youden index when ground truth is available.

## 📄 Usage

You can run XML inputs alone, NPY inputs alone, or both formats together in one
CSV and one pipeline run. Choose the corresponding template:

| Input formats | Template |
| --- | --- |
| NPY and XML together | [data_rows_template.csv](inputs/data_rows_template.csv) — first row NPY, second row XML |
| NPY only | [data_rows_template_npy.csv](inputs/data_rows_template_npy.csv) |
| XML only | [data_rows_template_xml.csv](inputs/data_rows_template_xml.csv) |

The pipeline chooses each file's decoder from its extension; no format flag is
needed. XML and NPY signals are processed together, with one combined probabilities
CSV and metrics JSON/CSV per selected model and task.

1. Prepare your input data:
   - Copy the appropriate template and replace the example filenames, diagnosis text, and binary labels with your own data. The example labels are placeholders, and the dummy ECG files are not included.
   - Use distinct filename stems for different ECGs: `dummy_npy.npy` and `dummy_xml.xml` can coexist, but `dummy.npy` and `dummy.xml` would both map to `dummy.base64` and are rejected.
   - For each task, add two columns with the following format:
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
   - Select your CSV with `--csv_file_name`, for example: `bash run_pipeline.bash --mode full_run --csv_file_name data_rows_template.csv`. Use `data_rows_template_npy.csv` or `data_rows_template_xml.csv` to run either format alone.

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

Run the complete pipeline once, including BERT and all eight signal models, on
50 approved de-identified ECGs (25 XML and 25 equivalent decoded NPY files) and
compare with saved reference predictions:

```bash
python scripts/test_ecg_reproducibility.py \
  --ecg-dir /mnt/data1/datasets/unit_tests/ECGs \
  --input-csv "$PWD/tests/fixtures/reproducibility/data_rows_reproducibility_50.csv" \
  --reference "$PWD/tests/fixtures/reproducibility/data_rows_reproducibility_50.reference.json" \
  --api-key-file "$PWD/api_key.json" \
  --work-dir /path/to/new/private/test-results \
  --gpu 0
```

The default ECG folder is `/mnt/data1/datasets/unit_tests/ECGs`; use `--ecg-dir`
to override it on another machine. The CSV selects exactly 25 XMLs and 25 NPYs;
extra files in that folder are not included. The NPYs contain the same decoded
signals as their original XMLs, before shared scaling and cleaning.

The work directory must be new. Omit `--gpu` to use CPU. The test builds once with
`--pull --no-cache` from a source-only context, downloads BERT and the eight
signal models at the reference's Hugging Face commit revisions, verifies their
file hashes, then runs the actual `main.py --mode full_run` pipeline once with
network disabled. The run starts with empty preprocessing and output folders;
the source ECG directory and downloaded model weights are mounted read-only.
No ECG data or API keys are copied into the image.

The test requires every ECG in every model output, finite probabilities in
`[0, 1]`, successful preprocessing reports, identical preprocessed signal hashes,
and identical raw BERT and signal-model probabilities. It rejects missing or duplicate rows and missing
model outputs. `--atol` explicitly permits an absolute probability tolerance;
the default is zero. A nonzero exit means the test did not pass. The input CSV
and ECG hashes must match the reference. Environment differences (package versions,
fairseq commit, Python, CUDA, and GPU model) are reported separately.

The fixed reference is `tests/fixtures/reproducibility/data_rows_reproducibility_50.reference.json`
(the default for `--reference`). It contains raw predictions for all 50 ECGs:
BERT's 77 probabilities, both signal models' 77 probabilities, and six binary
model probabilities per ECG. It was saved from the validated full pipeline run.
The test never regenerates or overwrites it. The reference and input CSV live in
`tests/fixtures/reproducibility/`; actual ECG files remain outside git. Replacing
the reference requires reviewing and accepting new expected results.

Artifacts include `report.json`, input/source hashes in `provenance.json`, model
revisions and file hashes in `models.json`, package versions, the fairseq commit,
image metadata, a copy of the reference, and logs and outputs under `run/`.
`run/outputs/` contains the model CSVs; `report.json` contains overall pass/fail,
the maximum absolute difference and count of differing values for each model,
and preprocessing/environment comparisons. Keep the entire
private work directory to preserve the evidence and downloaded snapshots.
The test checks a fresh build against the saved results. Package
versions and upstream Git repositories in the general Docker build are still
floating; this is not yet a guarantee of identical results across future rebuilds
or different hardware.

The input CSV follows `inputs/data_rows_template.csv`: diagnosis text,
binary outcome labels, and the four corresponding filename columns. Each row
must use the same ECG for every task. The prepared 50-record CSV contains XML
diagnosis text and synthetic binary labels obtained by thresholding the saved
WCR probabilities at 0.5. These labels are test inputs, not observed clinical
outcomes. The pipeline calculates metrics as usual; metrics against synthetic
labels must not be interpreted as clinical performance. The regression check
compares raw probabilities, not metrics or thresholded predictions. Preprocessing
scales signals using the whole cohort, so keep the same ECG selection when
comparing results.

Synthetic regression checks (run inside the image with `tests/` and `scripts/`
mounted under `/app`) cover strict output validation, XML worker isolation, and
evaluation mode:

```bash
python -m unittest discover -s tests -p test_reproducibility.py -v
python tests/test_error_collector.py
```

## 🐳 Docker

API keys must be mounted at runtime; they are excluded from the build context.
Create `api_key.json` using the case-sensitive `HUGGING_FACE_API_KEY` property
shown in the installation instructions. Both container commands mount this file.
For optional model preloading, use a BuildKit secret instead of copying the key:

```bash
docker build --build-arg RUN_MODELS_SETUP=true \
  --secret id=hf_api_key,src=api_key.json -t deepecg-docker .
```

### Interactive shell (recommended for Cursor / IDE terminals)

If `docker run -it ...` hangs or shows a blank screen in Cursor’s terminal, start the container in the background and attach a shell with `docker exec -it`. The image keeps the container running by default.

**1. Start the container (no `-it`):**
```bash
docker run -d --gpus all --name deepecg \
  -v $(pwd)/api_key.json:/app/api_key.json:ro \
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

### Mixed XML and NPY inputs

A task's file column can contain both XML and NPY paths, including uppercase
extensions. Each file is decoded with its own loader. Within each batch of up to
10,000 unique signal files, successful records are restored to their input order
before shared scaling and cleaning. The batch size and cohort-dependent scaling
are unchanged: changing the cohort or batch boundaries can change predictions.

XML and NPY predictions are combined per model/task: one `*_probabilities.csv`,
one metrics JSON, and one metrics CSV. Metrics cover the combined successfully
analysed records. Failed files are reported, excluded from inference in `full_run`
(even if old preprocessed files exist), and cause a nonzero exit status.

Only `.xml` and `.npy` are accepted by signal preprocessing. Sources must have
unique filename stems because preprocessed files retain the existing `<stem>.base64`
naming. For example, `record.xml` and `record.npy` must be renamed to distinct
stems; the pipeline rejects this collision before preprocessing rather than
overwriting one signal.

### Preprocessing Reports (per batch)

Located in `outputs/batch_X/` where X is the batch number:

For batches containing both formats, reports are stored separately under
`outputs/batch_X/xml/` and `outputs/batch_X/npy/`. Homogeneous batches retain the
existing location. NPY reports have no `xml_type` column.

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

## 🔧 Adding Support for New XML Formats

ECG machines from different vendors export XML files with varying structures. The pipeline currently supports:

- **CLSA** (Canadian Longitudinal Study on Aging format)
- **MHI** (Montreal Heart Institute format)

To add support for a new XML format, modify the `XMLProcessor` class in `utils/files_handler.py`:

### Step 1: Identify the XML Structure

First, examine your XML file to understand its structure. Use the built-in helper to flatten the XML:

```python
from utils.files_handler import XMLProcessor

processor = XMLProcessor()
data_dict = processor.xml_to_dict('path/to/your/ecg.xml')

# Print all keys to understand the structure
for key in sorted(data_dict.keys()):
    print(f"{key}: {data_dict[key][:50] if isinstance(data_dict[key], str) else data_dict[key]}")
```

### Step 2: Add Detection Logic

In the `process_single_file` method (~line 322), add a condition to detect your XML format:

```python
def process_single_file(self, file_path: str):
    # ... existing code ...
    
    if 'RestingECGMeasurements.MeasurementTable.LeadOrder' in data_dict:
        xml_type = 'CLSA'
        self._process_clsa_xml(data_dict, file_id)
    elif any(f'Waveform.1.LeadData.{j}.LeadID' in data_dict for j in range(12)):
        xml_type = 'MHI'
        self._process_mhi_xml(data_dict, file_id)
    # Add your format here:
    elif 'YourVendor.UniqueKey' in data_dict:
        xml_type = 'YOUR_VENDOR'
        self._process_your_vendor_xml(data_dict, file_id)
    else:
        xml_type = 'Unknown'
        return (file_id, xml_type, 'Failed', 'Unknown XML format'), None, None
```

### Step 3: Implement the Processing Method

Add a new method to extract the 12-lead ECG data. The output must be a numpy array with shape `(samples, 12)` where samples is typically 2500 (10 seconds at 250Hz):

```python
def _process_your_vendor_xml(self, data_dict: dict, file_id: str) -> None:
    """Process YOUR_VENDOR XML format."""
    try:
        # Standard 12-lead order
        correct_lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        leads = {lead: None for lead in correct_lead_order}
        
        # Extract lead data from your XML structure
        # Adapt these keys to match your XML format
        for i in range(12):
            lead_id_key = f'YourVendor.Lead.{i}.ID'
            lead_data_key = f'YourVendor.Lead.{i}.WaveformData'
            
            if lead_id_key in data_dict and lead_data_key in data_dict:
                lead_name = data_dict[lead_id_key]
                # Decode waveform data (base64, comma-separated, etc.)
                waveform_str = data_dict[lead_data_key]
                leads[lead_name] = np.array([float(x) for x in waveform_str.split(',')])
        
        # Handle missing leads
        non_empty_lead_dim = next((leads[l].shape[0] for l in leads if leads[l] is not None), 2500)
        for lead in leads:
            if leads[lead] is None:
                logger.warning("YOUR_VENDOR file %s: lead %s missing, filling with NaN", file_id, lead)
                leads[lead] = np.full(non_empty_lead_dim, np.nan)
        
        # Stack leads into array (samples, 12)
        self.full_leads_array = np.vstack([leads[lead] for lead in correct_lead_order])
        
    except Exception as e:
        raise ValueError(f"Error processing YOUR_VENDOR XML for file {file_id}: {str(e)}") from e
```

### Key Requirements

| Requirement | Description |
|-------------|-------------|
| Lead order | Must be: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 |
| Output shape | `(samples, 12)` - will be transposed/resampled automatically if needed |
| Sample rate | Pipeline resamples to 2500 samples (10s at 250Hz) |
| Data type | Numeric values (float/int), decoded from base64 or text as needed |

### Common XML Vendors

Different ECG machine manufacturers use different XML schemas:

| Vendor | Common Format | Notes |
|--------|---------------|-------|
| GE Healthcare | MUSE XML | Often uses base64-encoded waveforms |
| Philips | Philips XML | May have different lead naming |
| Mortara | Mortara XML | Check for ELI format |
| Schiller | SEMA XML | European standard format |

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
@article{NolinLapalme2026,
  author = {Nolin-Lapalme, Alexis and Sowa, Achille and Delfrate, Jacques and Tastet, Olivier and Corbin, Denis and Kulbay, Merve and Ozdemir, Derman and No{\"e}l, Marie-Jeanne and Marois-Blanchet, Fran{\c{c}}ois-Christophe and Harvey, Fran{\c{c}}ois and Sharma, Surbhi and Ansari, Minhaj and Chiu, I Min and D'souza, Valentina and Friedman, Sam F. and Chass{\'e}, Micha{\"e}l and Potter, Brian J. and Afilalo, Jonathan and Elias, Pierre Adil and Jabbour, Gilbert and Bahani, Mourad and Dub{\'e}, Marie-Pierre and Boyle, Patrick M. and Chatterjee, Neal A. and Barrios, Joshua and Tison, Geoffrey H. and Ouyang, David and Maddah, Mahnaz and Khurshid, Shaan and Cadrin-Tourigny, Julia and Tadros, Rafik and Hussin, Julie and Avram, Robert},
  title = {Foundation models for electrocardiogram interpretation: clinical implications},
  journal = {European Heart Journal},
  year = {2026},
  pages = {ehaf1119},
  doi = {10.1093/eurheartj/ehaf1119},
  URL = {https://doi.org/10.1093/eurheartj/ehaf1119},
  note = {Online ahead of print}
}

```

