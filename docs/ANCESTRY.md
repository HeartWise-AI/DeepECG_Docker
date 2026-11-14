# DeepECG-Ancestry

The external-validation layer of DeepECG_Docker: hand it a partner site's ECGs, genotypes and
ancestry calls, and it returns one audited zip of **aggregate tables and figures**. No waveform,
no genotype, no measurement and no identifier leaves the site, and the image needs no network.

It uses the same Python, torch and pinned `fairseq-signals` commit as the main image, while its
genetics dependencies and command surface stay isolated in `dockerfile.ancestry`.

---

## Why this is one pipeline and not two

Two containers were merged into `ancestry/`:

| was | did | now |
|---|---|---|
| `deepecg-ancestry` | the validation suite shipped to two external sites: one inference pass, the pre-specified analysis plan, no genotypes needed | `validate`, `metrics`, `admixed`, `ensemble` |
| `deepecg-mgh` | the genetic replication and the fairness audit: the discovery GWAS loci, nine trait polygenic scores, the 77-finding audit, the deprivation arm | `prepare`, `score`, `geno`, `loci`, `pgs`, `scan`, `fairness`, `ses`, … |

They were not independent. The inference core was **byte-identical** between them, the
preprocessing had only diverged by added robustness, and the newer disclosure-control module was
a strict superset of the older one. Merging therefore removed duplication rather than creating a
wrapper: there is now one loader, one `prepare`, one privacy module and one selftest.

The loader uses the same `fairseq-signals` checkpoint contract as the released WCR model:
`checkpoint_utils.load_model_and_task(head, arg_overrides={'model_path': backbone})`. The
ancestry fold heads are exactly that shape. `ancestry.infer` owns this standalone inference path,
and `ancestry/model_manifest.json` supplies its immutable snapshot metadata without registering a new
task in `main.py`.

### Weights and the frozen reference tables

Both come from Hugging Face by pinned revision through the existing `HuggingFaceWrapper`; nothing
is stored in git:

| artefact | repository | note |
|---|---|---|
| the five ancestry fold heads | `heartwise/DeepECG-SSL_Ancestry` | **access-gated.** Derived from UK Biobank, MHI and CLSA recordings, so access is granted per project. Note that a gated repository still serves its model card, file names and file sizes publicly -- only the bytes are withheld |
| the frozen discovery tables | the same snapshot, under `reference/` | cohort aggregates, so they travel with the weights and one access grant covers both |
| the shared SSL backbone | `heartwise/wcr_77_classes` (public, already pinned) | `base_ssl.pt` there is byte-identical to the backbone the folds were fine-tuned from (sha256 `39985e90...54383c`), so there is nothing extra to upload |
| the released 77-finding head | `heartwise/wcr_77_classes` (public) | audited unchanged; no re-implementation |

### Hugging Face, or a local install -- as a choice, not a guess

`weights_source` in `ancestry.config` (or `ANCESTRY_WEIGHTS_SOURCE`, or
`--weights-source` on the `weights` mode):

| | what it does | who wants it |
|---|---|---|
| `auto` | a path given to the mode, then the environment variable, then a snapshot already fetched, then the pinned revision | the default |
| `local` | **never reaches the network.** Only a path, an environment variable or a snapshot already on disk; a missing mount is an error, not a download | an air-gapped site |
| `huggingface` | ignores the environment variables and uses the pinned revision | being certain a run used the published artefact, not something left in a mount |

A path passed directly to a mode (`--models-root`, `--ssl-ckpt`) always wins: it is the most
specific instruction there is.

### The snapshot can live anywhere

The access-gated repository exists, but the complete snapshot must be uploaded and pinned before
the default manifest route can download it. A staged or project-specific copy is nameable without
editing this repository:

```bash
# ancestry.config
ancestry_model_repo:     someorg/their_copy
ancestry_model_revision: 0123456789abcdef0123456789abcdef01234567
```

The revision must be an exact 40-character commit; a branch name is refused, because it would
make a validation run unreproducible. Only the ancestry snapshot is overridable this way --
repinning the public heads is a change to this repository, reviewed like any other.

### `weights`: the mode for all of it

```bash
python -m ancestry weights                       # where each artefact resolves from, no network
python -m ancestry weights --download            # fetch the pinned snapshot, then verify it
python -m ancestry weights --verify DIR          # check a copy on disk: layout and checksums
python -m ancestry weights --upload DIR          # what WOULD be pushed (a dry run)
python -m ancestry weights --upload DIR --go     # push it; PRIVATE unless --public is given
```

`--download` prints the two exports a validation run needs, and `--upload` refuses a directory
that does not verify, so a half-staged snapshot cannot reach the Hub.

### Who can do the upload, and the one step after it

Creating the repository needs **write or admin on the `heartwise` Hugging Face organisation**,
which is a property of the account, not of the token: a classic token with the `write` role
still gets

```
403 Forbidden: You don't have the rights to create a model under the namespace "heartwise".
```

if the account's role in the organisation is `read`. `python -m ancestry weights --upload DIR
--go` reports that verbatim and uploads nothing, so a failed attempt leaves no partial
repository behind. Check the role first:

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['orgs'])"
```

Then, from a machine that has the staged bundle:

```bash
huggingface-cli login                                        # paste a write token
python -m ancestry weights --upload DIR                      # dry run: lists what would go
python -m ancestry weights --upload DIR --go                 # private by default
```

The upload is not finished until the commit is pinned. `ancestry/model_manifest.json` ships
`"revision": "PENDING-UPLOAD"`, which every loader refuses by design — only an exact
40-character commit is accepted, so no run can quietly depend on a moving branch. Replace it
with the sha the upload prints, then confirm the pin resolves:

```bash
python -m ancestry weights                                   # the pin and where it resolves from
python -m ancestry weights --download                        # fetch it back and verify checksums
```

Until that is done, a site runs from mounted weights (`MODELS_ROOT`, `REFERENCE_DIR`), which is
a supported route in its own right and the only one an air-gapped site ever uses.

---

## What it asks

**1. Does the genetic signature replicate?** The ECG-predicted ancestry score behaves like a
heritable trait: in the genetically European participants of three discovery cohorts it has a
non-zero SNP-heritability and a set of genome-wide-significant loci, most of them already-known
ECG-trait loci. `loci` looks those leads up directly, `scan` runs a genome-wide scan per ancestry stratum, `pgs`
tests the nine ECG-trait polygenic scores and the ancestry-score PGS.

**2. Is the released 77-finding model unfair across ancestry groups — and can a hospital see that
without genotyping anyone?** `fairness` audits it twice: on true genetic ancestry, and on
ancestry inferred from the ECG itself.

**3. Does the whole external-validation plan reproduce?** `validate` runs the eight pre-specified
analyses from one inference pass: the gated primary AUROC, dose-response on everyone, a gate
sweep, a threshold-free AUROC, the age/sex-matched subset, fairness by subgroup, ensemble
agreement with a 1→5 size sweep, and intrapatient stability (ICC) where a person has repeats.

`docs/ANCESTRY_QUESTIONS.md` maps every sub-question to the one file that answers it, and lists
what a single site cannot answer. Read it before interpreting anything.

---

## Install

```bash
docker build -f dockerfile.ancestry -t deepecg-ancestry:1.0.0 .
```

For an air-gapped partner site, bake the pinned weights in so the image reaches the network for
nothing at all:

```bash
docker build -f dockerfile.ancestry --build-arg RUN_MODELS_SETUP=true \
    --secret id=hf_api_key,src=api_key.json -t deepecg-ancestry:1.0.0 .
```

The build fails if the package does not import or a mode does not parse its arguments, and the
plink binaries are verified against `licenses/plink.SHA256SUMS.txt` before use.

`docs/ANCESTRY_DEPLOY.md` covers shipping the image to a site that cannot build it.

---

## Run

Three inputs, described in `inputs/README_ancestry_inputs.md` and named in `ancestry.config`:
the RFMix ancestry proportions, the ECG inventory, and the genotypes.

```bash
bash run_ancestry.bash --mode weights     # where the weights come from; touches no network
bash run_ancestry.bash --mode survey      # READ-ONLY. Run this first and send the output back.
bash run_ancestry.bash --mode selftest    # a synthetic cohort, end to end. Run this second.
bash run_ancestry.bash --mode all         # the whole study
```

`run_ancestry.bash` reads `ancestry.config` exactly as `run_pipeline.bash` reads
`heartwise.config`, and forwards anything else to the mode. Equivalently, without the wrapper:

```bash
python -m ancestry <mode> ...
python -m ancestry <mode> --help          # that mode's full option list
```

**`survey` before anything else.** It opens a sample of every file you intend to pass and reports
what it found — units, genome build, columns, how many people per ancestry stratum, whether there
is ground truth. It writes nothing, and it is the cheapest place to find a format problem.

### Every mode, what it needs, what it writes, and whether it comes back

One row per mode. "Returns" means the stage writes into `results/`, which is the aggregate-only
tree `package` zips; a stage that writes only into `work/` is feeding a later one, and its output
stays at the site. Times are for a ~25,000-ECG cohort (see ANCESTRY_USER_GUIDE.md § Compute).

| # | mode | what it does | needs | writes | returns | time |
|---|---|---|---|---|---|---|
| 0 | `weights` | says where every artefact resolves from, with no network. `--download`, `--verify DIR` | nothing | — | no | seconds |
| 0 | `survey` | read-only: opens a sample of every file you mean to pass and reports units, build, columns, people per stratum, whether there is ground truth | the raw inputs | a report (`--out`) | send it to us | minutes |
| 0 | `selftest` | a whole synthetic cohort end to end, with the privacy assertions. Must print `ALL CHECKS PASSED` with 0 skipped | the weights + plink | its own scratch dir | no | ~15 min |
| 1 | `convert` | VCF/BCF → plink2 pgen, once (inserted automatically when `--geno` is a VCF) | `--vcf` | `work/geno/pgen` | no | 20–90 min |
| 2 | `prepare` | RFMix + the ECG inventory → the roster, the ancestry strata, the matched subset, model-ready arrays | `--rfmix`, `--ecg-meta`, optional `--linkage` | `work/`, `prepared/` | `results/cohort/` | 10–40 min |
| 3 | `score` | DeepECG-Pop (five folds) and the 77-finding head over every ECG, once each | `prepare`, the weights | `work/` scores | `results/score/` | ~25 min GPU, ~6 h CPU (`--shard`) |
| 4 | `traits` | NeuroKit2 ECG measurements off your own waveforms — **run it even if you have a cart panel**, so `pgs` can compare the two rulers | `prepare` | `work/measurements.csv` | no | ~2 h at `--jobs 16` |
| 5 | `geno` | one variant subset, within-stratum PCs, kinship pruning, plink-format phenotypes | `--geno`, `prepare` | `work/geno/` | `results/geno/` | 20–60 min |
| 6 | `loci` | the discovery leads, per stratum. **The headline genetic result** | `geno`, `score` | — | `results/loci/` | minutes |
| 7 | `pgs` | the trait-PGS positive control on both measurement rulers (`--measurement-source both\|device\|neurokit`, default **both**; `--measurement-map` for your own column names, and a missing trait is filled from the other ruler and labelled), the decile curves, and the ancestry-score PGS | `geno`, `score`, `traits` and/or `--measurements` | `work/geno/*/pgs/` | `results/pgs/` | 10–30 min |
| 7b | `scan` | optional, expensive: the genome-wide scan and its Manhattan, per stratum. Summary statistics stay in `work/` | `geno`, `score` | `work/scan/` | `results/scan/` | 2–6 h per stratum |
| 8 | `fairness` | the 77-finding audit on true genetic ancestry and on ancestry inferred from the ECG, matched and unmatched | `score`, `--labels` or `--statements` | — | `results/fairness/` | ~10 min |
| 8b | `ses` | optional: is the signal explained by socioeconomic deprivation? Four arms against coupling-matched shams, at the median **and** the mean cut | `score`, `--ses` | — | `results/ses/` | ~15 min |
| 9 | `figures` | every plate, drawn from the tables above and computing nothing — it takes no `--work` at all | `results/` | — | `results/figures/` | seconds |
| 10 | `package` | **run last.** Re-audits every file, searches each for your own identifiers, and writes one zip | `work/`, `results/` | — | **the zip** | minutes |
| — | `validate` | the eight pre-specified external-validation analyses from a single inference pass (primary AUROC, dose–response, gate sweep, threshold-free split, matched subset, fairness by sex/age/batch/round, 1→5 ensemble sweep, intrapatient ICC) | `prepare`, the weights | — | `--output`, point it inside `results/` | ~30 min GPU |
| — | `metrics`, `admixed`, `ensemble` | the same pieces as stand-alone modes, for running one on its own | `prepare` | — | `--output` | minutes |
| — | `submit` | optional: write (and with `--go`, submit) the whole graph as Slurm job arrays | the same paths | sbatch scripts | no | seconds |
| — | `all` | `prepare → score → traits → geno → loci → pgs → fairness → figures → package`, in order (add `--with-scan`, `--ses FILE`), resumable with `--from-mode`, `--skip`, `--keep-going` | everything above | — | all of the above | a day |

Every flag of every mode carries its own `--help` text — at a site with no network that is the
documentation, so there is no undocumented flag in the package (one test asserts it).

`validate` is deliberately **not** in `all`. It does its own inference pass from
`all_genotyped/`, so putting it in the chain would score every ECG twice; a site that wants the
external-validation plan runs `prepare` and then `validate`, and a site that wants the genetics
runs `all`. Both share the same `prepare` output.

`submit` will not guess the site's account, partition or GPU. The graph is wired with `afterok`,
so one rejected stage cancels up to 147 descendants; it asks for all the missing values in one
message and writes nothing until it has them.

---

## Nothing per-patient ever leaves

`work/` carries the ids, the scores, the genotype subsets and the summary statistics, and stays
at the site. `results/` is aggregate by construction — see `ancestry/privacy.py`:

- every exported table is written through `write_table`, which **refuses** an identifier column,
  refuses a table long enough to be per-person, and masks counts of 1–9;
- no metric is computed on fewer than 20 people per class, no ancestry stratum below 100 people
  is analysed at all, and a clinical finding needs 15 positives in both groups;
- `package` re-audits every file — including the bytes of every figure, against the site's own
  identifier list — and refuses to write the zip if anything trips.

---

## Verifying a build

```bash
pytest tests/test_ancestry_cli.py tests/test_ancestry_privacy.py tests/test_ancestry_registry.py
docker run --rm deepecg-ancestry:1.0.0 selftest --out /tmp/selftest
```

The pytest suites need no weights, no plink and no network. The selftest needs all three and is
the real gate: **109 checks, 0 failed, 0 skipped**. A skip means the weights or plink are not where
the image expects them, and those are the only end-to-end tests of the genetic path and of the
checkpoints — so a skip is a packaging fault, not a pass.

---

## Where this departs from the main pipeline, and why

Everything above follows this repository's conventions because they already had a precedent.
These four do not, and each is a choice rather than an oversight.

| | the main pipeline | here | why |
|---|---|---|---|
| **output** | `print()` 3 times, `utils.log_config` 71 | `print()` 334, logger 0 | these modes emit a formatted operator report -- banners, aligned tables, `[PASS]`/`[FAIL]` lines -- that a `%(asctime)s [INFO] ancestry.loci:` prefix on every line would destroy. The report IS the interface a partner site reads |
| **directories** | one `output_folder` | `work_folder` and `results_folder` | nine stages need scratch that stays at the site, kept apart from the audited package that leaves it. One directory cannot carry that distinction, and the distinction is the disclosure control |
| **entry point** | `heartwise.config` + `main.py`, `mode: full_run` | a 20-mode CLI, `mode: all` | a site runs this over days, resumes after a failed stage, and sometimes submits it to Slurm. The separate entry point prevents genetics dependencies and input assumptions from changing the normal ECG pipeline |
| **batching** | `data/ECGSignalDataset` + DataLoader | memmapped `(N, 2500, 12)` arrays | ~90k ECGs are scored at once; the arrays do not fit in RAM and there is no DataFrame of paths at that point |
| **reproducibility** | `tests/fixtures/reproducibility/*.reference.json`, compared at `atol=0.0` with an environment block | `scripts/test_ancestry_npy_reproducibility.py`, image-to-image on the approved 25-NPY NAS cohort | hashes every input, records both environments, and compares all five fold probabilities plus the ensemble. Default `atol=1e-6` covers cross-runtime roundoff; `--atol 0` exposes bitwise differences |

What is NOT a departure: the ECG files. `.npy` and the `.base64` that
`utils/files_handler.ECGFileHandler` writes are both read here, with that module's own shape
rules -- squeeze, transpose, integer-step decimation, refuse anything short of 2500 samples --
so a site already running the main pipeline has nothing to convert. XML is refused by name:
`XMLProcessor` parses CLSA and MHI, but its cohort scaling happens further along and an
unverified scale would silently change every score rather than fail.
