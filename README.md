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
- [DeepECG-PNG (ECG images)](#️-deepecg-png-ecg-images)
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

2. 🔑 Initialize the local credential file (Python 3):
   ```bash
   python -m utils.credentials
   ```
   This creates an ignored `api_key.json` with an empty `HUGGING_FACE_API_KEY`
   and preserves an existing file. `api_key.example.json` is the tracked template.
   Public, ungated models work with the key left empty. For private or gated
   models, obtain read access on Hugging Face and enter your token in the local
   `api_key.json`. Never commit this file. Setup and inference also create an empty
   file automatically when the configured path is missing.

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
│   ├── manifest.json                 # Pinned Hugging Face revisions
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

The combined template contains the exact rows from the NPY-only and XML-only
templates, preserving their filenames, diagnosis text, and outcome labels.

The pipeline chooses each file's decoder from its extension; no format flag is
needed. XML and NPY signals are processed together, with one combined probabilities
CSV and metrics JSON/CSV per selected model and task.

1. Prepare your input data:
   - Copy the appropriate template and use each row with its matching ECG file, diagnosis text, and outcome labels. When substituting another ECG, update those values together.
   - The supplied rows use `dummy_ecg.npy` and `dummy_ecg.xml`. These share a filename stem, so the current pipeline rejects running them together: both would map to `dummy_ecg.base64`. To run this example as a mixed cohort, give the files distinct stems and update their corresponding CSV filenames, keeping each file's diagnosis and labels unchanged.
   - For each task, add two columns with the following format:
     ```
     'ecg_machine_diagnosis': '77_classes_ecg_file_name',
     'afib_5y': 'afib_ecg_file_name',
     'lvef_40': 'lvef_40_ecg_file_name',
     'lvef_50': 'lvef_50_ecg_file_name',
     'shd': 'shd_ecg_file_name'          # image models only
     ```
   - A file column may list **signals** (`.xml`, `.npy`) or **ECG page images** (`.png`, `.jpg`, `.jpeg`), mixed
     in the same CSV: the input type of each file is detected from its extension (there is no flag), and each
     task runs the models selected in `heartwise.config` for that input type (see [DeepECG-PNG](#️-deepecg-png-ecg-images)).
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
     - `signal_models`: Models run on signal files (`.xml`, `.npy`), comma-separated, among `efficientnet`, `wcr`. Example: `efficientnet,wcr`.
     - `image_models`: Models run on ECG page images (`.png`, `.jpg`, `.jpeg`), among `deepecg_siglip2`. Example: `deepecg_siglip2`.
     - Optional per-task choice, overriding the two lines above for one task: `<task>_signal_models` / `<task>_image_models`, with `<task>` one of `ecg_machine_diagnosis`, `afib_5y`, `lvef_40`, `lvef_50`, `shd`. Example: `lvef_40_signal_models: efficientnet`. An empty value runs no model for that task and input type; rows left without a model are reported as errors. (The former `use_efficientnet` / `use_wcr` options are still accepted by `main.py` but deprecated.)
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

The default ECG root is `/mnt/data1/datasets/unit_tests/ECGs`, with XML fixtures
under `xml/` and NPY fixtures under `npy/`; use `--ecg-dir` to override the root
on another machine. The CSV keeps basename-only references and selects exactly
25 XMLs and 25 NPYs; extra files under either folder are not included. The NPYs
contain the same decoded signals as their original XMLs, before shared scaling
and cleaning.

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
evaluation mode. Credential and download tests use synthetic snapshots:

```bash
python -m unittest discover -s tests -p test_reproducibility.py -v
python -m pytest tests/test_png_reproducibility.py -q
python tests/test_error_collector.py
python -m unittest discover -s tests -p test_model_downloads.py -v
```

### PNG reproducibility test

The PNG pipeline has the same pinned, offline regression workflow for the 25
approved ECG pages under `/mnt/data1/datasets/unit_tests/ECGs/png`:

```bash
python scripts/test_png_reproducibility.py \
  --png-dir /mnt/data1/datasets/unit_tests/ECGs/png \
  --input-csv "$PWD/tests/fixtures/reproducibility/data_rows_png_reproducibility_25.csv" \
  --reference "$PWD/tests/fixtures/reproducibility/data_rows_png_reproducibility_25.reference.json" \
  --api-key-file "$PWD/api_key.json" \
  --work-dir /path/to/new/private/png-test-results
```

The test performs a clean source-only Docker build, verifies the pinned BERT,
base encoder, and five classifier snapshots and their file hashes, then disables
networking for inference. It passes no PNG mode or image-model flag: the normal
`.png` extension routing and `heartwise.config` select SigLIP2. It requires all
five prediction files, checks every probability is finite and in `[0, 1]`, and
fails if a PNG was missing, unreadable, or sent through signal preprocessing.
The default tolerance is zero. The checked-in reference contains file hashes,
model hashes, environment metadata, and raw probabilities only; ECG pages remain
outside git.

## 🐳 Docker

The image contains an empty `api_key.json`. Public models can run without a
credential mount. To use a token, initialize the host file as described above,
then mount it read-only as shown below. Actual credentials are excluded from the
build context and must never be copied into the image.

Optional preloading downloads the pinned model files during the build. Without a key:

```bash
docker build --build-arg RUN_MODELS_SETUP=true -t deepecg-docker .
```

To preload models requiring access, pass the local key as a temporary BuildKit secret:

```bash
docker build --build-arg RUN_MODELS_SETUP=true \
  --secret id=hf_api_key,src=api_key.json -t deepecg-docker .
```

Preloading prints each model's availability and a summary. Denied access,
unavailable revisions, and download failures are reported without blocking the
image build. It downloads files without instantiating the models. Runtime inference
loads only models selected by the CSV tasks and the model settings. An unavailable
unselected model does not block inference; a selected model that cannot be downloaded
or loaded produces an error and a nonzero exit status.

### Pinned models and cached downloads

`models/manifest.json` records each registered model's Hugging Face repository
and exact 40-character commit SHA. Docker preloading and runtime downloads use
this same manifest. The nine reproducibility models retain the revisions in the
saved reference; the legacy BERT model has its own pinned revision.

Snapshots are stored under `weights/<model>/<owner>--<repository>/<commit>/`.
Before publishing a completion marker, setup fetches the pinned repository file
list and verifies that every file is present with the expected size. An offline
fallback to a partial download cannot become a complete cache. Cache hits check
that all recorded files exist with the expected sizes; the reproducibility test
additionally verifies their SHA-256 hashes against the reference. Concurrent
downloads are locked, interrupted downloads can be retried, and changing a
revision uses a separate directory. Old unversioned `weights/<model>/` files are
left untouched and are not trusted or reused automatically. Mount `/app/weights`
to a persistent host directory if downloads should survive container removal.

An empty key explicitly uses anonymous access. If access fails, the message
explains whether to supply credentials/check permissions, verify a revision,
or check connectivity/cache permissions. Hugging Face may hide private repositories
as unavailable, so an anonymous failure does not prove that a repository is missing.

To update a model, obtain its exact Hugging Face commit SHA, change its entry in
`models/manifest.json`, and review the resulting predictions. The reproducibility
test rejects a manifest revision that differs from its reference. For an intentional
model update, separately review and approve new reference files and expected
probabilities; the test never overwrites the reference automatically. A model
revision change does not pin Python packages, CUDA, or other image dependencies.

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

## 🖼️ DeepECG-PNG (ECG images)

DeepECG-PNG reads ECG **page images (PNG)** instead of signals: a SigLIP-2 ViT-B/16 image encoder
(run ux5uajqm) with linear probes for **77 classes, SHD, LVEF ≤ 40 %, LVEF < 50 % and incident AFib
at 5 years**. Images are preprocessed exactly as in training (grayscale, stretched to 224×224,
CLIP normalization); it expects ECG pages rendered like the training images.

**1. Weights (Hugging Face).** The SigLIP2 models use the same pinned manifest download path as the
signal models. `models/manifest.json` identifies one shared base encoder
([`heartwise/SigLIP-2_BaseEncoder`](https://huggingface.co/heartwise/SigLIP-2_BaseEncoder)) and five
independent classifier repositories: 77 classes, SHD, LVEF ≤ 40%, LVEF < 50%, and incident AFib at
5 years. Each immutable snapshot is stored under `weights/<manifest-key>/<repository>/<commit>/`,
either by `models_setup.py` or on first use. The encoder is loaded once per process/device; each task
loads its own `classifier.safetensors`, metadata, and thresholds from its pinned repository.

**2. Input CSV.** The same per-task CSV as for signals, with image file names in the file columns
(template: [inputs/data_rows_template_png.csv](inputs/data_rows_template_png.csv)). Signals and images
can be mixed in one CSV; images are never sent to the signal preprocessing.

| column | content |
|---|---|
| `<task>_ecg_file_name` (e.g. `shd_ecg_file_name`) | image file name (relative to `ecg_signals_path`) or absolute path (with Docker: a path inside the container) |
| `shd`, `lvef_40`, `lvef_50`, `afib_5y` | 0/1 labels, optional for images (metrics are computed on labelled rows) |
| `ecg_machine_diagnosis` | report text, optional for images: 77-class ground truth through the BERT classifier |

Image rows without labels get predictions only. Labels may be 0/1, `true`/`false` or `yes`/`no`; any
other value is treated as missing (with a warning). Images that are missing or cannot be decoded are
skipped and listed (`missing_files_<ts>.csv`, `unreadable_images_<ts>.csv`); the other images still run.
The image probes use the image only (no age or sex input).

Each image is opened with its own format's decoder only, after a header check: a `.png` must be a real,
non-animated PNG and a `.jpg` / `.jpeg` a real single-picture JPEG (no MPO), both at most 50 megapixels
(ECG pages are ~1–3). Anything else, e.g. a TIFF renamed `.png`, is refused and listed as unreadable, so
no other Pillow decoder is ever reached. The model was trained on PNG renders: on JPEG copies of the same validation pages (quality 75–95)
the binary-task probabilities moved by 0.002 on average (at most 0.019), and 1–3 of 192 calls flipped.

**3. Run.** `image_models: deepecg_siglip2` is the default in `heartwise.config`; no other setting is needed:
```bash
docker run -d --gpus all --name deepecg \
  -v $(pwd)/api_key.json:/app/api_key.json:ro \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/outputs:/app/outputs \
  -v /path/to/pngs:/app/ecg_signals:ro \
  -v $(pwd)/preprocessing:/app/preprocessing \
  deepecg-docker
docker exec -it deepecg ./run_pipeline.bash --mode full_run --csv_file_name my_pngs.csv
```

**4. Outputs** (in `output_folder`), as for the other models: per task,
`<model>_<ts>_<task>_probabilities.csv` (e.g. `deepecg_siglip2_lvef_equal_under_40_<ts>_lvef_40_probabilities.csv`,
with the same columns as the signal models) and the metrics `<model>_<ts>_<task>.json` / `.csv` (empty
when the task has no labels). The 77-class classifier returns all 77 probabilities.
The validation Youden thresholds of the binary probes are in the snapshot's `thresholds.json`.

With a GPU (`signal_processing_device: cuda:0`) the encoder runs once per minibatch and all five
tasks reuse those features. Images are decoded only as their minibatch is read and no derived image
or Base64 file is persisted. The in-process feature cache is capped at 500,000 rows (about 1.43 GiB
for 768-dimensional float32 embeddings); older minibatches are evicted and recomputed if a larger
cohort is processed. Split very large CSVs to retain the five-head reuse without reaching that cap.

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

