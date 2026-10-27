# Design doc: external-validation ingestion for DeepECG_Docker

**Status:** built and validated · **Author:** Omar Khedr · **Last updated:** 2026-10-19 · **First dataset:** Icentia-11k
**Constraint:** existing DeepECG_Docker code isn't modified. The new route lives in new files and imports what it needs.
**Code:** `ingestion/`, `tests_ingestion/`, `run_icentia.sh`, `configs/`. Usage: `ingestion/README.md`.

---

## 1. Purpose

DeepECG_Docker validates a model against **12-lead ECGs that come with report text**, which BERT turns into ground truth. External datasets like Icentia have **fewer leads** and **class labels instead of reports**, so today each one needs ad-hoc code.

**Goal:** run an external validation from two flags, and **bypass BERT**:

```
--lead <lead_specification>   --mapping <mapping.csv>
```

**Priority:** Icentia results first; a reusable pipeline for other datasets second.

### Non-goals
- Modifying existing files (`main.py`, `utils/`, `models/`, `run_pipeline.bash`, `dockerfile`, `heartwise.config`).
- Searching for the optimal lead combination.
- A full 11,000-patient run.

---

## 2. Today's route vs. the new one

**Today's route** (unchanged):

```
ECG file + report ──► preprocessing ──► model ──► 77 probabilities ─┐
       └──► BERT(report) ──► 77 labels (ground truth) ──────────────┴─► compare ─► metrics
```

**New route:**

```
ECG signal ──► 10 s windows ──► --lead (build 12 × 2500) ──► model ──► 77 probabilities ─┐
annotations ──► window labels ──► --mapping (dataset label → 77-class name) ────────────┴─► compare ─► metrics
```

**For Icentia, file by file, with the module that does each step:**

```
.hea ─┐ icentia.py         windows.py         leads.py           infer.py
.dat ─┴─► mV signal ─► 10 s windows ─► 12 × 2500 input ─► model ─► 77 probabilities ─┐
.atr ───► markers ───► same windows ─► labels.py ─► mapping.py ─► ground truth ─────┴─► metrics.py
                                                                       run.py drives it all, per patient
```

- **`.hea`:** the settings: 1 lead, 250 Hz, 1,048,577 samples, gain 74.36 units/mV.
- **`.dat`:** the signal, stored as int16. Divided by the gain to get mV.
- **`.atr`:** the beat and rhythm markers. The time-skip field is signed.
- **One window, both paths:** the same 10 s window feeds both paths, so each prediction and its answer refer to the exact same samples.
- **Read-only:** the dataset is never modified; everything is transformed in memory. In Docker the dataset is mounted read-only.

| | Today's route | New route |
|---|---|---|
| Leads | 12 required | Built into 12 by `--lead` |
| Ground truth | BERT on report text | Dataset labels via `--mapping` |
| Classes scored | All 77 | Only the tasks named in `mapping.csv` (all 77 outputs are saved) |
| Unit of work | One preprocessed file per ECG | 10 s windows from long recordings, processed patient by patient |

---

## 3. The two flags

### 3.1 `--lead <lead_specification>`

The model takes 12 channels in the order I, II, III, aVR, aVL, aVF, V1–V6, each 2500 samples (10 s at 250 Hz). `--lead` says how to build those 12 from the dataset's single lead (`ingestion/leads.py`).

| Value | Behaviour |
|---|---|
| A lead name, e.g. `--lead I` | Put the signal in that slot; **zero the other 11** |
| `--lead 12` | **Copy the signal into all 12 slots** (used for Icentia) |

In the reference Icentia validation, all-12 was equivalent to V1–V6 copied (the best combination) except PVC AUPRC (−0.041); one slot with zeros was clearly worse.

### 3.2 `--mapping <mapping.csv>` (required)

A two-column file mapping each dataset label to a DeepECG class name (`ingestion/mapping.py`). Several dataset labels may map to the same class, and `A|B` scores one task from two model outputs.

```csv
dataset_label,deepecg_class
AFIB,Afib|Atrial flutter
AFL,Afib|Atrial flutter
N_RHYTHM,Sinusal
PVC,Premature ventricular complex
PAC,Premature atrial complex
BRADY,Bradycardia
TACHY,Atrial tachycardia (>= 100 BPM)
BIGEMINY,Regularly irregular
```

- **Task:** each distinct `deepecg_class` value is one task (7 for Icentia).
- **Ground truth:** a window is positive for a task if any dataset label mapped to it is 1, negative if all are 0, otherwise left out.
- **Combined score (`A|B`):** 1 − (1 − p_A)(1 − p_B), the chance that either is present. Used for AF/flutter, as in the reference validation.
- **BERT:** never loaded or run on this route.
- **Validation of the file:** the run stops with an error if the file is missing or names a class that isn't one of the 77. The class list is read from `utils/constants.py` without importing torch.

---

## 4. Components

All in `ingestion/`. Each step is a small module so another dataset only needs its own reader and labeller.

### 4.1 Reader: `icentia.py`
- **What it reads:** WFDB `.hea`/`.dat`/`.atr`, with plain numpy (no wfdb dependency).
- **Signal:** 1 lead, 250 Hz, format 16 (little-endian int16), converted to mV with the header gain. A `.dat` whose length doesn't match the header is an error.
- **Annotations:** MIT format. The SKIP field is a **signed** 32-bit value (reading it unsigned shifts annotations by 2³²). Beat codes N=1, S=9 (PAC), V=5 (PVC), Q=13 (unclassified); rhythm changes are code 28 with aux text `(N`, `(AFIB`, `(AFL` or `)`.
- **Missing annotations:** a segment without `.atr` gets every label unknown, and its beat counts and clean-sinus flag are missing (NA) rather than 0 / False (e.g. p00008_s49).

### 4.2 Windowing: `windows.py`
- **Windows:** non-overlapping 2500-sample (10 s) windows, 419 per Icentia segment; a trailing partial window is dropped. No resampling.
- **Skipped windows:** any sample at −32768 (WFDB's invalid-sample value) → `invalid`; at 32767 → `saturated`. Skipped windows get no prediction and no metrics.
- **Flat windows:** std < 0.01 mV is flagged but kept.

### 4.3 Window labels: `labels.py`
Icentia labels **individual beats** and **rhythm stretches**, not 10 s windows. This step gives each window a label in Icentia's terms, before `--mapping` renames it. Rules are the task table of the reference validation; all parameters are in `RULES` and saved in the run manifest.

#### Stretches, windows and labels

Cardiologists marked up the long recording in two ways, each tied to a **sample number** (250 samples per second):
1. **Beat marks:** one per heartbeat, e.g. "sample 10,412: normal beat", "sample 10,630: PVC".
2. **Rhythm stretches:** a start and an end mark, e.g. "AF from sample 50,000 to sample 90,000". Time with no stretch around it is **unannotated**.

A **stretch** and a **window** are both pieces of the recording, but they come from different places:
- **Stretch:** drawn by the **cardiologist**, wherever the rhythm actually was. Any length, any position.
- **Window:** cut by **us**, mechanically, every 10 s from the start (0–10 s, 10–20 s, …), ignoring the content, because the model only accepts 10 s inputs.

They don't line up, because the heart doesn't change rhythm on our 10 s grid:

```
time (s):     0    10    20    30    40    50    60    70
our windows:  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |      ← fixed grid, every 10 s
cardiologist: |---- normal ----|--------- AF ---------|-- normal --
                              23 s                   61 s       ← stretches, drawn where the rhythm changed
```

A **label** is the answer we derive for one window and one task, from the marks inside it: **yes**, **no**, or **left out** (not used for that task). The model's score for the window is compared with it. For AF/flutter in the picture above:

| Window | Sits in | AF/flutter label |
|---|---|---|
| 1, 2 (0–20 s) | entirely normal | **no** |
| 3 (20–30 s) | normal until 23 s, then AF | **left out** (mixed) |
| 4, 5, 6 (30–60 s) | entirely AF | **yes** |
| 7 (60–70 s) | AF until 61 s, then normal | **left out** (mixed) |

Beat-based tasks work the same way, on the beat marks inside the window. For example, a window entirely in normal rhythm with 12 normal beats is clean sinus (PVC = no, PAC = no, brady/tachy from its heart rate); one PVC beat makes it PVC = yes.

#### The 2-sample tolerance

Rules that need the **whole window** in one rhythm would, strictly, require the stretch to start at or before the window's first sample and end at or after its last. Sometimes a stretch boundary lands *almost* on a window edge:

```
window 3:  |20.000 s ................................ 30 s|
AF:          |20.008 s .................................
              ↑ 8 ms = 2 samples late
```

- **Strictly:** window 3 has 8 ms of normal rhythm, then AF, so it is mixed and left out.
- **The reference code:** counts it as AF.

Applied strictly, our counts differed from the reference by 0–3 windows per patient (out of about 20,000). Treating a stretch as covering the whole window if it misses an edge by **2 samples (8 ms) or less** makes every count match exactly; 0 or 1 sample doesn't. 8 ms is clinically irrelevant. This is most likely a boundary detail in that code (e.g. rounding between seconds and samples, or an inclusive end sample), so we don't know the reason, only that 2 reproduces him.

#### The rules

"Clean sinus" = whole window in normal rhythm, at least one beat, no PVC and no PAC.

| Task | Positive window | Negative window |
|---|---|---|
| AF / flutter | Whole window AF or flutter | Whole window normal rhythm |
| Sinus rhythm | Whole window normal rhythm | Whole window AF or flutter |
| Bradycardia | Clean sinus, rate < 60 | Clean sinus, rate 60–100 |
| Tachycardia | Clean sinus, rate > 100 | Clean sinus, rate 60–100 |
| PVC | ≥ 1 PVC, no AF | Clean sinus, every beat reviewed (no Q) |
| PAC | ≥ 1 PAC, no PVC, no AF | Clean sinus, every beat reviewed (no Q) |
| Bigeminy | ≥ 6 beats alternating normal and PVC | Clean sinus |

Every other window is left out of that task; unlabelled time is never used as a negative. Details:
- **Rhythm intervals:** built from `(X` … `)` markers. Annotations are put in time order first. A new `(X` while one is open closes the open one there; an interval never closed runs to the end of the segment. Zero-length `(N)` markers are ignored. (The last two cases don't occur in the 800 segments checked.)
- **Whole-window tolerance:** an interval may start or end up to **2 samples** inside the window (see "The 2-sample tolerance" above). This reproduces the reference counts exactly on every patient checked; the reason is unknown.
- **"No AF"** (a condition in the PVC and PAC rules, not a label): the window doesn't overlap any AF/flutter interval, so a PVC or PAC in unannotated time counts as positive. Confirmed by the reference counts: this reading matches all 30 PVC/PAC patient counts checked; requiring confirmed normal rhythm instead matches only 8 (e.g. p00009: 116 of the 296 reference PVC windows are in unannotated time).
- **Heart rate:** 60 / median gap between consecutive beats (any type), only if the window has ≥ 4 beats and no gap > 3 s; otherwise unknown. Any zero gap (duplicate marks) leaves it unknown.

### 4.4 Inference: `infer.py`
- **Model:** `wcr_77_classes`, loaded through the existing `HeartWiseModelFactory` (same Hugging Face weights). The token is only read if the weights aren't cached, and never logged.
- **Run mode:** `.eval()` on the loaded network and `torch.no_grad()`. The existing wrapper leaves dropout on, which changes outputs on every run.
- **Input:** raw mV, no Docker preprocessing. The reference validation found the model amplitude-invariant, and filtering gave no net gain.
- **Output:** 77 probabilities per window (the wrapper applies the sigmoid), reordered to `ECG_PATTERNS` with the existing `WCR_COLUMN_CONVERSION`. Batches of 512.

### 4.5 Thresholds
A threshold turns a probability into yes/no, needed for sensitivity and specificity. AUROC and AUPRC don't use one.
- **Where they come from:** they should be picked on separate tuning data and those patients excluded from scoring. The reference validation picked them per task by **Youden's index** (maximise sensitivity + specificity − 1) on a 520-window tuning set. Docker's built-in thresholds don't transfer to Holter data.
- **Implemented:** a fixed thresholds file, `--thresholds <thresholds.csv>` (format in `ingestion/README.md`; none is shipped). Task names must match the mapping's tasks: an unknown name is an error, a missing one a warning.
- **Not built:** `--thresholds calibration:<manifest>` (fit by Youden inside the pipeline), useful for a new dataset or lead setting.

### 4.6 Metrics: `metrics.py`
- **Per task, on labelled, non-skipped windows:** positive/negative counts, AUROC, AUPRC, and sensitivity/specificity at the fixed threshold. AUROC/AUPRC are blank when a set has no positives or no negatives (e.g. AF for a patient with no AF).
- **Pooled and per patient.** 95% CIs for pooled AUROC/AUPRC come from a bootstrap that resamples **whole patients** (default 1000).

### 4.7 Driver and outputs: `run.py`
For each patient: read every segment → windows → labels → ground truth → 12-lead input → model. Then it pools all patients for metrics.
- **Robustness:** only `pNNNNN` patient IDs are accepted (no path tricks); `--out` must be empty unless `--overwrite`, which first deletes only this pipeline's own files there; `run_manifest.json` is written first with `"complete": false` and rewritten with `true` at the end. A missing patient folder, or a segment that can't be read (bad header, truncated `.dat`, sampling rate ≠ 250 Hz), is logged and skipped, not fatal. `--out` equal to or inside `--data` is refused (symlinks resolved).
- **`--labels-only`:** stops before the model (no torch needed). Inference is the default.
- **Output files** (in `--out`):
  - `<patient>_windows.parquet`: one row per window with metadata, beat counts, heart rate, labels and ground truth.
  - `<patient>_probs.npy`: float32, 77 probabilities per window, NaN for skipped windows.
  - `label_summary.csv`, `metrics.csv` / `metrics.json` (pooled), `metrics_per_patient.csv`.
  - `run_manifest.json`: arguments, host paths, lead slots, mapping, thresholds, label rules, git hash, model and mode, skipped/flat windows, segments without `.atr`, missing patients, unreadable segments, timings.

---

## 5. Running it

```bash
./run_icentia.sh --gpu 2 \
  --data <icentia root> \
  --lead 12 \
  --mapping configs/icentia_mapping.csv \
  --thresholds <thresholds.csv> \
  --patients p00000-p00009 \
  --out <out dir>
```

- **Separate entry point:** `run_pipeline.bash` and `main.py` are untouched.
- **Why Docker:** the model needs torch, which is only in the `deepecg-docker` image. `run_icentia.sh` takes the same arguments as `python -m ingestion.run`, with host paths, and:
  - mounts `--data` **read-only** and `--out` read-write; mounts `--mapping`/`--thresholds` files read-only;
  - mounts `ingestion/`, `configs/`, `thresholds/` read-only and `weights/` for the model cache, because the image contains none of them (`utils/constants.py` loads `thresholds/` at import);
  - runs as the calling user (outputs not owned by root) and removes the container afterwards;
  - `--gpu N` picks the GPU; `--self-test` runs the tests inside the image.
- **Weights:** cached in the repo's `weights/` (gitignored). If missing, they download on first use with the image's `api_key.json` (this path is untested).
- **Without torch:** `python -m ingestion.run ... --labels-only` runs on the host.
- **Speed:** about 43–80 s per patient on one A6000 (most recently 43 s). A full run is not planned, so resume and sharding were not built.

### Relationship to the existing code

| Existing piece | Use |
|---|---|
| `models/` (factory, WCR wrapper) | Imported; eval mode set on the loaded object |
| `utils/constants.py` | Imported (`ECG_PATTERNS`, `WCR_COLUMN_CONVERSION`) |
| `thresholds/` | Mounted, only because `utils/constants.py` loads it at import |
| `main.py`, `analysis_pipeline.py`, preprocessing, BERT | Not used |

---

## 6. Validation

| Check | Result |
|---|---|
| **Unit tests** (`tests_ingestion/`, synthetic data only) | 29 pass on the host and in the image: `.atr` signed skip, header/signal, windows and skip reasons, both `--lead` options, labelling rules on a hand-built segment, bigeminy runs, mapping file, combined score, unsorted annotations, reopened/unclosed rhythm, duplicate beats, threshold-file checks, per-patient grouping, the CLI end to end (missing patient, unreadable and wrong-rate segments, `--out` inside `--data`), and the PR-review fixes (patient-ID validation, non-empty `--out` / `--overwrite`, interrupted-run manifest, bad threshold values, a single zero RR gap, missing annotations as NA, `--batch-size` reaching the model and strict `metrics.json`, via a fake model). |
| **Labels vs reference**, p00000–p00009 (9 in the reference; p00002 was a tuning patient) | All 126 per-patient window counts identical. |
| **Metrics vs reference**, same 9 patients, same thresholds | 156 values within 0.0016. |
| **AF patients vs reference**: p00020, p00084, p00237, p00239, p00765, p05104 | Window counts identical for all 7 tasks; AF/flutter and sinus AUROC identical to 4 decimals. |
| **Determinism** | Identical probabilities across runs, GPUs, and data copied into a container vs read through the read-only mount. |
| **After the code-review fixes** (16 patients re-run) | Labels identical to before; float32 probabilities round to the earlier float16 values exactly. |

**Remaining differences are numeric precision, not logic.** The reference used TF32, and its numbers match float16-rounded outputs better than float32 ones, which suggests its metrics were computed from float16 files. Example: p00020 PVC (3 positive windows) has AUPRC 0.1804 from float16 scores, exactly the reference value, and 0.2081 from float32. Over all 16 patients (288 values), 232 match to 4 decimals with float16 scores and 224 with float32. The largest gap either way is p00084 bigeminy sensitivity, 0.2121 vs 0.2424: 2 of 66 windows sit next to a very low threshold. We keep float32 because it is the more exact result.

**Pooled numbers aren't comparable with the reference report**, which covers 5,241 patients; these checks cover 16.

### Code review
`/code-review` on `ingestion/`, `tests_ingestion/`, `run_icentia.sh` found 10 issues; all were fixed and tested:
1. annotations assumed to be in time order → sorted first;
2. metrics computed from float16-rounded scores → float32;
3. `--out` equal to `--data`, or reached through a symlink, not refused → refused;
4. a wrong sampling rate crashed the run → that segment is logged and skipped;
5. unclosed or interrupted rhythm intervals were dropped → kept (see §4.3);
6. a typo in a threshold task name silently dropped sensitivity/specificity → error;
7. duplicate beat marks gave an infinite heart rate → rate unknown;
8. the script resolved relative file paths from the repo, unlike `--data`/`--out` → from the current directory;
9. per-patient grouping was O(patients × windows) → one pass;
10. −32768 reported as "saturated", and an unreachable "nan" reason → "invalid", dead code removed.

None of the data cases behind 1, 4, 5, 7 and 10 occur in the 800 real segments checked, so these fixes don't change any label matched against the reference.

### PR review bots
Sourcery and a Cursor automation reviewed the PR and found 7 more issues; all were fixed, each with a test that fails on the old code:
1. patient IDs could contain path components (`../`) → only `pNNNNN` IDs and ranges;
2. runs sharing `--out`, or an interrupted run, could mix outputs → `--out` must be empty unless `--overwrite` (which deletes only this pipeline's files), and the manifest's `"complete"` flag marks finished runs;
3. blank, non-numeric or out-of-range thresholds were accepted → must be numbers in 0–1, each task once;
4. one zero RR gap among normal ones still gave a heart rate → any zero gap leaves it unknown;
5. segments without annotations showed beat counts 0 and clean sinus False → missing (NA);
6. `metrics.json` contained non-standard `NaN` → `null`;
7. `--batch-size` was recorded but never passed to the model → passed.

Re-run on p00008 (has the segment without annotations) and p00084: labels, heart rates and counts identical; at the default batch size, probabilities and per-patient metrics are bit-identical to before. A different batch size moves probabilities by up to 6e-4 and per-patient metrics by up to 6e-6 (GPU kernels differ by batch size).

A second `/code-review` (all findings) on these fixes found 9 more; 8 were fixed with tests: a patient listed twice was counted twice; `--overwrite` used glob patterns (broke on `[` in the path, and matched non-`pNNNNN` names); old outputs were deleted before the model loaded; an `--out` that is a file gave a traceback; `metrics.json` serialisation missed pandas/numpy types; the column typing was duplicated; and the wiki was out of date. Not changed: masking with the nullable `clean_sinus` column, which works in pandas 1.3.5 and 2.3.3 (NA counts as False). Two runs started on the same folder at the same moment were not guarded against; after the push, Sourcery's remaining open finding ("Output files mix run results") flagged this, and `--out` now holds an exclusive `.ingestion.lock` for the whole run (created with `O_EXCL`, removed on exit, also after an error), so a second run stops at once. A full staging folder with an atomic publish was not adopted: `--out` must already be empty or `--overwrite`, the old outputs are deleted before writing, and the manifest's `"complete"` flag marks an interrupted run.

---

## 7. Open points
1. **Heart rate:** the rules don't define how it is measured. We use 60 / median beat gap (§4.3); it reproduces the reference counts but is not confirmed.
2. **2-sample tolerance:** needed to reproduce the reference counts; the reason is unknown (§4.3).
3. **Calibration mode:** fitting thresholds inside the pipeline (§4.5) is not built.
4. **Weight download:** the first-run download path of `run_icentia.sh` is untested; runs so far used cached weights.
