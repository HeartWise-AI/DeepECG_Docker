---
name: heartwise-merge-gate
description: Pre-merge quality gate for HeartWise-AI inference repos (DeepECG_Docker, DeepECG-PNG, DeepCORO_CLIP and siblings). Use before opening or merging any PR there, when asked to "run the gate", "check against Jacques's comments", or after adding a dataset adapter, input type, model, flag or script to the Docker pipeline. Encodes the maintainer's recurring review comments (one unified Docker, reuse the existing pipeline, configure via heartwise.config, no dead folders) and fails the PR on any violation.
---

# HeartWise merge gate

The maintainers repeat the same few review comments across PRs. This gate turns them into checks so they are fixed before review, not after.

## Where the rules come from

Recurring maintainer review themes on this repo: one unified Docker (no second image or entry point per input type or project), reuse the existing pipeline instead of rebuilding it, control behavior from `heartwise.config` rather than new flags, remove code the shipped pipeline does not use (PR #81), and merge only with reproducibility evidence (PRs #85, #92, #94). Add a line here when a review repeats a comment.

## Gate checklist (every item must pass; report each as PASS or FAIL with evidence)

### A. Structure and recycling
1. **No parallel pipeline.** New work enters through `main.py` / `utils/analysis_pipeline.py` / `utils/files_handler.py`. A new top-level `__main__` or `run_*.sh` that re-implements load, preprocess or predict is a FAIL. Check: `grep -rn "if __name__" --include=*.py . | grep -v "tests/\|scripts/\|fairseq"`.
2. **Reuse before write.** For each new function, name the existing function it could have extended. Duplicated model loading, threshold loading, lead ordering or batching is a FAIL. Check the diff against `utils/` and `models/`.
3. **One Docker.** No second Dockerfile, no new `COPY` of a folder that is not needed at inference time. Every `COPY` line must map to code the container runs. Check: `git diff origin/main -- dockerfile`.
4. **No unused folders.** For every new top-level directory or package, show the import or call from the shipped path. If it is only used by an offline study, it does not belong in the image and probably not in this repo (move it, or keep it out of the `COPY`). Unreferenced = FAIL.
5. **Right place.** Experiment scripts, dataset adapters and one-off analyses go to the lab repo that owns them. Only inference, preprocessing, model loading and their tests live here.

5b. **Extension agnostic.** No flag that tells the pipeline what kind of input it got (signal, image, PDF, XML, NPY). It detects from the file and the CSV. A flag is allowed only for a property that cannot be detected, and then it lives in `heartwise.config`.
5c. **Branch, not fork.** Work is on a branch of the existing repo; no copied repo or second Dockerfile.
5d. **Single scope.** One issue per PR. If the diff mixes a feature with experiments, sweeps or scripts for another issue, split it.
5e. **Check main first.** List the existing keys, helpers and modules the change could have used; new names for an existing concept are a FAIL.

5f. **Baby steps.** Each commit is one idea that was validated before the commit; independent steps are not combined.

### B. Configuration
6. **heartwise.config is the control surface.** Any new setting is a key in `heartwise.config`, read by `run_pipeline.bash` `get_param`, documented in README. No new Docker `-e`/flag. An argparse flag is acceptable only as the internal plumbing `run_pipeline.bash` fills from the config; it must have a default that preserves old behavior.
7. **Defaults preserve behavior.** With the new key absent or at its default, outputs are byte-identical to `origin/main`.

### C. Correctness and reproducibility
8. **Reproducibility run.** `scripts/test_ecg_reproducibility.py` (and `test_png_reproducibility.py` if images are touched) in a fresh `--platform linux/amd64` build: preprocessing hashes and probabilities match the unchanged reference at tolerance 0. Paste the summary line. Not run = FAIL. Run the PNG and ECG checks WITHOUT `--gpu`: the saved references were recorded on CPU (`environment.gpu: None`), and a GPU run differs by about 1e-6 (main fails the same way). On CPU, `origin/main` passes the PNG check at tolerance 0 (7 of 7). If the report says `environment_identical: false` (different GPU or libs than the reference) and diffs are about 1e-6, run the same check on `origin/main` on the same host: identical diffs on main and the PR mean the PR changes nothing (PASS with a note); different diffs mean a real change (FAIL). The PNG check needs a Hugging Face key with read access to the `heartwise` org, supplied through `--api-key-file` and never committed.
9. **Tests run in the Docker image**, not only on the host. Paste the count (N tests, M subtests). Skipped tests are listed by name.
10. **New input type has a real sample test** (real file, not only synthetic), including the failure path (bad file, wrong shape) and row alignment with the input list.
10b. **Benchmark run.** The fixed benchmark cohort (the 50-ECG reference, 25-image PNG fixture) is re-run for any behavior-relevant change.
10c. **Computed claims.** Any number printed or written in docs (overlap, prevalence, counts) comes from code or says "not checked". Unknown or malformed config keys error instead of being dropped; NaN or non-binary labels are rejected.
11. **Clinical parameters are cited.** Any threshold, window length, label rule or tolerance has a source paper or a config entry owned by Robert. Invented values = FAIL (ask, do not guess).
12. **Pinned dependencies.** Model revisions go through `models/manifest.json`; new Python deps are pinned in `requirements.txt`; no `:latest` tags.

### D. Hygiene
13. No secrets, no data, no `.DS_Store`, `__pycache__`, outputs committed. `.gitignore` exceptions are justified.
14. README updated for every new config key; no em dashes.
15. PR description: what changed, what was verified (executed vs inspected), what is intentionally out of scope. Reviewer requested (the maintainer).
16. Bot findings (Sourcery, Cursor, Codex, Bugbot) each answered: fixed with commit hash, or rebutted with a pointer.

## Procedure

1. `git fetch origin main`; warn and stop if `refname 'origin/main' is ambiguous`.
2. `git diff --stat origin/main...HEAD`; list new top-level paths and every changed `dockerfile`, `heartwise.config`, `run_pipeline.bash`, `utils/parser.py` line.
3. Walk A to D in order. For each FAIL write the smallest fix (delete, reuse, move to config), apply it, re-run the affected check.
4. Re-run the full gate after fixes. Iterate until all PASS or a FAIL needs a decision from Robert (clinical value, repo move, scope).
5. Report a table: item, PASS/FAIL, evidence (command output or file:line), tag executed / inspected / assumed. Never write "tests pass" if any were skipped.
6. Merge rule: DeepECG_Docker and siblings are NOT self-merge repos. Open or update the PR, request Jacques, stop.

## Gate runs log

- 2026-11-10 DeepECG_Docker PR #101 (PDF inputs): PASS A1-A5, 5b, 6 (no new flag/Docker change). FAIL then fixed: item 12 (`scipy` unpinned, now 1.15.3), item 14 (README line 355 still `.png`-only routing), item 16 (Sourcery `/UserUnit` finding real, now refused + test). Host run: 84 tests pass. Still open: item 8/9 reproducibility and in-image test run (needs Docker build), item 15 reviewer already requested.

## Improving this gate

After each Jacques review or meeting that adds a new repeated comment, add one line to "Where the rules come from" and one checklist item. Delete items that have not caught anything in five PRs. Keep it under ~120 lines.
