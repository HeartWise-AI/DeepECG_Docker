# Icentia-11k external validation

Scores a DeepECG 77-class model (`wcr_77_classes` by default) on Icentia-11k single-lead Holter records against
the beat and rhythm annotations. BERT is not used: the ground truth comes from the annotations.

```
.hea/.dat ─► 10 s windows ─► 12-lead input (--single_lead) ─► pipeline preprocessing ─► model ─► 77 probabilities ─┐
.atr ──────► same windows ─► window labels (labels.py) ─► --mapping ─► ground truth ─────────────────────────────┴─► metrics
```

The 12-lead input, the preprocessing (PTB-XL power scaling and line-noise cleaning) and the model calls are the
main pipeline's own (`utils/files_handler.py`, `utils/analysis_pipeline.py`). Only the Icentia parts are here:

| File | Role |
|---|---|
| `reader.py` | Read WFDB records with `wfdb`; cut 10 s windows; skip windows with invalid or saturated samples. |
| `labels.py` | Window labels from the annotations (`RULES`). |
| `mapping.csv` | Icentia label → DeepECG class (`A\|B` = one task scored from two classes). |
| `run.py` | The runner. |

`ingestion/mapping.py` and `ingestion/metrics.py` hold the dataset-independent parts (label mapping, metrics).

## Run it

```bash
python -m ingestion.icentia11k.run --data <icentia root> --patients p00000-p00009 --out <new folder> \
    [--single_lead all] [--thresholds <thresholds.json>] [--model wcr|efficientnet] [--device cuda:0]
```

In the Docker image (the dataset mounted read-only, outputs owned by you, weights cached in `weights/`):

```bash
docker run --rm --gpus '"device=0"' --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v <icentia root>:/data/icentia:ro -v <out folder>:/out -v "$PWD/weights:/app/weights" \
    --entrypoint python deepecg-docker -m ingestion.icentia11k.run \
    --data /data/icentia --patients p00000-p00009 --out /out
```

| Argument | Meaning |
|---|---|
| `--data` | Dataset root, laid out as `p00/p00000/p00000_s00.{hea,dat,atr}`. Only read. |
| `--patients` | `p00000-p00009`, `p00000,p00020`, or a mix. Only `pNNNNN` IDs. A missing patient is logged and skipped. |
| `--out` | New or empty folder, not inside `--data`. |
| `--single_lead` | `all` (default) copies the lead into all 12 leads; a lead name (`I` … `V6`) puts it in that lead only. |
| `--mapping` | CSV `dataset_label,deepecg_class` (default: `mapping.csv` here). Class names are checked against the 77. |
| `--thresholds` | JSON in the `thresholds/` format (`{"<task>": {"threshold": 0.5}, ...}`), keyed by mapping task. Used for sensitivity and specificity; without it only AUROC and AUPRC are reported. Pick thresholds on separate tuning data. |
| `--model` | `wcr` (DeepECG-SSL, default) or `efficientnet` (DeepECG-SL). |
| `--bootstrap` | Patient-bootstrap resamples for the 95% CIs (default 1000, 0 = none). |
| `--batch_size`, `--device`, `--preprocessing_n_workers`, `--hugging_face_api_key_path` | As in the main pipeline. |

## Window labels

| Task | Positive | Negative |
|---|---|---|
| AF / flutter | whole window AF or flutter | whole window normal rhythm |
| Sinus | whole window normal rhythm | whole window AF or flutter |
| Bradycardia | clean sinus, rate < 60 | clean sinus, rate 60–100 |
| Tachycardia | clean sinus, rate > 100 | clean sinus, rate 60–100 |
| PVC | at least one PVC, no AF | clean sinus, every beat reviewed |
| PAC | at least one PAC, no PVC, no AF | clean sinus, every beat reviewed |
| Bigeminy | ≥ 6 beats alternating N/V | clean sinus |

- "Whole window" = the rhythm covers at least 99.9% of the window (at most 2 of 2,500 samples differ).
- Clean sinus = whole window normal rhythm, no PVC and no PAC. "Every beat reviewed" = no Q beat.
- "No AF" = no part of the window overlaps an AF or flutter stretch, so a PVC or PAC in unlabelled time counts.
- Heart rate = 60 / median gap between beats; it needs 3 or more beats.
- Windows that match neither side are left out of that task, and so are windows of segments without a `.atr` file.
- Windows with an invalid (-32768) or saturated (32767) sample are skipped; flat windows (std < 0.01 mV) are
  flagged but kept.

## Outputs

| File | Contents |
|---|---|
| `<patient>_windows.parquet` | One row per window: segment, start sample, skip reason, flat flag, beat counts, heart rate, the 8 Icentia labels and the mapped ground truth (`gt::<task>`). |
| `<patient>_probs.npy` | float32 `[n_windows, 77]` probabilities in `ECG_PATTERNS` order; NaN for skipped windows. |
| `label_summary.csv` | Positive / negative / unknown counts per label and task. |
| `metrics.csv` | Pooled per task: counts, AUROC, AUPRC, sensitivity, specificity, bootstrap CIs. |
| `metrics_per_patient.csv` | The same per patient. |
| `run_manifest.json` | Arguments, model, lead slots, mapping, thresholds, label rules, skipped windows, unreadable segments, missing patients, timings. |

## Validation

Checked on the host (not in the Docker image) against the internal reference validation (batch 1, lead copied to
all 12 leads, raw mV, TF32, float16 logits), on p00000–p00009 and six AF patients (p00020, p00084, p00237, p00239,
p00765, p05104): 16 patients, 335,200 windows.

- Labels: every window has the same label as the reference, for all 7 tasks; the scored windows are the same.
- Metrics: same windows and labels, pooled over the 16 patients, at the reference (S1) thresholds:

| Task | Positives | AUROC (reference) | AUPRC (reference) | Sensitivity (reference) | Specificity (reference) |
|---|---|---|---|---|---|
| AF / flutter | 28,859 | 0.9861 (0.9860) | 0.9028 (0.9026) | 0.944 (0.944) | 0.943 (0.942) |
| Sinus | 185,626 | 0.9808 (0.9809) | 0.9970 (0.9970) | 0.903 (0.903) | 0.958 (0.957) |
| Bradycardia | 38,539 | 0.9886 (0.9888) | 0.9727 (0.9730) | 0.946 (0.947) | 0.960 (0.959) |
| PVC | 15,304 | 0.9514 (0.9519) | 0.7676 (0.7695) | 0.740 (0.740) | 0.934 (0.935) |
| PAC | 14,888 | 0.9723 (0.9725) | 0.8414 (0.8413) | 0.847 (0.849) | 0.953 (0.953) |
| Tachycardia | 2,578 | 0.9944 (0.9942) | 0.9550 (0.9532) | 0.936 (0.936) | 0.994 (0.994) |
| Bigeminy | 2,268 | 0.502 (0.493) | 0.014 (0.014) | 0.284 (0.284) | 0.796 (0.792) |

  The small differences come from the pipeline preprocessing (the reference used raw mV) and from the reference's
  TF32 / float16 outputs; they were not separated. All are well inside the reference's patient-bootstrap CIs.
- The single-lead path of `main.py` (`--single_lead all` and `V1`, 10,000 windows of p00000 as 10 s NPY files)
  gives the same probabilities as this runner, bit for bit.
- Speed: about 60–80 s per patient (20,950 windows) on one A6000. When the cleaning step finds line-noise peaks
  (p00003), it runs on all 12 leads and the patient takes about 190 s.

## Tests

```bash
python -m pytest tests/test_icentia11k.py tests/test_single_lead.py
```
