# External-validation ingestion (Icentia-11k)

Scores DeepECG-SSL (`wcr_77_classes`) on a single-lead Holter dataset that the main pipeline can't read.
It reads the raw WFDB records, cuts them into 10 s windows, labels each window from the annotations,
puts the lead into a 12-lead input (`--lead`), maps dataset labels to model classes (`--mapping`), runs
the model and computes window-level metrics. BERT, `main.py` and the Docker preprocessing are not used,
and no existing file is changed.

## Run it

The model needs torch, which is only in the DeepECG Docker image, so use the launch script:

```bash
./run_icentia.sh --gpu 2 \
  --data <icentia root> \
  --lead 12 \
  --mapping configs/icentia_mapping.csv \
  --thresholds <thresholds.csv> \
  --patients p00000-p00009 \
  --out <out dir>
```

`run_icentia.sh` takes the same arguments as `python -m ingestion.run` with host paths, plus:

| Option | Meaning |
|---|---|
| `--gpu N` | GPU to use (default 0). The script sets `--device` itself. |
| `--image NAME` | Docker image (default `deepecg-docker`, built from the repo's `dockerfile`). |
| `--self-test` | Run `tests_ingestion/` inside the image, then exit. |

What it does:
- mounts `--data` **read-only**, so the dataset can't be written to, and `--out` read-write;
- mounts `ingestion/`, `configs/` and `thresholds/` read-only, because the image contains none of them;
- keeps the model weights in the repo's `weights/` (gitignored). If `weights/wcr_77_classes` is missing,
  they are downloaded on first use with the image's `api_key.json`;
- runs as your user, so outputs aren't owned by root, and removes the container when it finishes.

Without Docker (labels only, no torch needed):

```bash
python -m ingestion.run --data <root> --lead 12 --mapping configs/icentia_mapping.csv \
  --patients p00000-p00009 --out <out> --labels-only
```

## Arguments (`ingestion/run.py`)

| Argument | Meaning |
|---|---|
| `--data` | Dataset root, laid out as `p00/p00000/p00000_s00.{hea,dat,atr}`. Only read. |
| `--lead` | `12` copies the lead into all 12 positions (default for Icentia). A lead name (`I`, `II`, …, `V6`) puts it in that one position and leaves the others at zero. |
| `--mapping` | CSV `dataset_label,deepecg_class`. `A\|B` in `deepecg_class` scores one task from two outputs as 1 − (1 − p_A)(1 − p_B). Class names are checked against the model's 77. |
| `--patients` | `p00000-p00009`, `p00000,p00020`, or a mix. Only `pNNNNN` IDs are accepted; a patient listed twice is processed once. Missing patients are logged and skipped. |
| `--out` | Output folder. Must be empty, and must not be `--data` or inside it (symlinks are resolved). |
| `--overwrite` | Reuse a non-empty `--out`. Once the model has loaded, this pipeline's own output files there are deleted (other files are kept); the previous results are then gone even if the new run fails. Don't start two runs on the same folder at once. |
| `--thresholds` | CSV `task,threshold`, used for sensitivity and specificity. Values must be numbers between 0 and 1, each task listed once, and task names must match the mapping's tasks (an unknown name is an error; a missing one gives a warning). Without it, only AUROC and AUPRC are reported. |
| `--bootstrap` | Patient-bootstrap resamples for the 95% CIs (default 1000, 0 = none). |
| `--batch-size`, `--device` | Inference settings (defaults 512, `cuda:0`). The GPU computes slightly differently per batch size (probabilities move by up to about 6e-4), so keep 512 to reproduce earlier runs. |
| `--labels-only` | Stop after the labels: no model, no metrics. |
| `--save-example-input` | Save one 12 × 2500 model input per patient, to check `--lead`. |

No thresholds file is included; pick thresholds on separate tuning data. The format is one row per
task of the mapping (the values below are placeholders, not recommendations):

```csv
task,threshold
Afib|Atrial flutter,0.5
Sinusal,0.5
Premature ventricular complex,0.5
```

## Window labels

Rules follow the task table of the reference Icentia-11k validation and are in `ingestion/labels.py` (`RULES`):

| Task | Positive | Negative |
|---|---|---|
| AF / flutter | whole window AF or flutter | whole window normal rhythm |
| Sinus | normal rhythm | AF or flutter |
| Bradycardia | clean sinus, rate < 60 | clean sinus, rate 60–100 |
| Tachycardia | clean sinus, rate > 100 | clean sinus, rate 60–100 |
| PVC | at least one PVC, no AF | clean sinus, every beat reviewed |
| PAC | at least one PAC, no PVC, no AF | clean sinus, every beat reviewed |
| Bigeminy | ≥ 6 beats alternating N/V | clean sinus |

Clean sinus = normal rhythm with no early beats. "No AF" means the window doesn't overlap any AF/flutter
stretch, so a PVC or PAC in unannotated time counts as positive; this reproduces the reference counts (requiring
confirmed normal rhythm does not). Windows that match neither side are left out of that
task. Rhythm boundaries get a ±2-sample tolerance, which reproduces the reference window counts exactly.
Windows with an invalid sample (-32768, WFDB's missing-sample value) or a saturated one (32767) are
skipped; flat windows (std < 0.01 mV) are flagged but kept. Annotations are put in time order first; a
rhythm interval that is never closed runs to the end of the segment, and one interrupted by a new rhythm
marker ends there.

## Outputs

| File | Contents |
|---|---|
| `<patient>_windows.parquet` | One row per window: segment, start sample, skip reason, flat flag, beat counts, heart rate, the 8 dataset labels and the mapped ground truth (`gt::<task>`). In segments without annotations, beat counts and `clean_sinus` are missing (NA), not 0 / False. |
| `<patient>_probs.npy` | float32 `[n_windows, 77]` model probabilities in `ECG_PATTERNS` order; NaN for skipped windows. |
| `label_summary.csv` | Positive / negative / unknown counts per label and task. |
| `metrics.csv`, `metrics.json` | Pooled per task: counts, AUROC, AUPRC, sensitivity, specificity, bootstrap CIs. `metrics.json` is strict JSON: values that can't be computed are `null`. |
| `metrics_per_patient.csv` | The same per patient (one row per patient and task). |
| `run_manifest.json` | Written first with `"complete": false` and rewritten with `"complete": true` at the end, so an interrupted run is visible. Arguments, lead slots, mapping, thresholds, label rules, git commit, skipped/flat windows, segments without annotations, missing patients, unreadable segments, timings. |

AUROC and AUPRC need both positive and negative windows, so they are blank for a patient (or a set of
patients) with no positives, e.g. AF on patients who never have AF.

## Validation

Checked against the per-patient results of an internal reference validation (batch 1), using the same
per-task thresholds:

- p00000–p00009 (9 of them in the reference; p00002 was a threshold-tuning patient): window counts
  identical for every patient and task; 156 metric values within 0.0016.
- 6 AF patients (p00020, p00084, p00237, p00239, p00765, p05104): window counts identical; AF/flutter
  and sinus AUROC identical to 4 decimals; 126 of 132 values within 0.001. The largest gap is p00084
  bigeminy sensitivity, 0.2121 vs 0.2424: 2 of 66 windows sit right at a very low threshold.
- Probabilities are identical across repeated runs and GPUs.
- The remaining gaps are numeric precision: the reference used TF32, and its numbers match
  float16-rounded scores better than float32 ones (e.g. p00020 PVC AUPRC: 0.1804 from float16, as in
  the reference, vs 0.2081 from float32). This pipeline keeps float32.

About 43–80 s per patient on one A6000.

## Tests

```bash
python -m pytest tests_ingestion     # host, no torch needed
./run_icentia.sh --self-test         # inside the image
```

All tests use synthetic data; none needs the real dataset.
