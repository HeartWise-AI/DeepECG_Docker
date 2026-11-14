# DeepECG-Ancestry — user guide

> **What each output is for:** `docs/QUESTIONS.md` maps every
> sub-question to the one file that answers it, and lists what a
> single site cannot answer. Read it before interpreting anything.

One container. Everything runs inside your infrastructure. You send back a single zip of
aggregate tables and figures. No waveform, no genotype, no measurement and no identifier ever
leaves, and the container has no network access of any kind.

It follows an earlier external run at another hospital site, and reuses that image's SSL
backbone and its five ancestry fold checkpoints unchanged, so the ancestry scores it produces are
the same scores — and adds the two things that run was not built for.

---

## What is being asked

**1. Does the genetic signature replicate?** The ECG-predicted ancestry score behaves like a
heritable trait: in the genetically European participants of three discovery cohorts it has a
non-zero SNP-heritability and a set of genome-wide-significant loci, most of them already-known
ECG-trait loci. Your site is in no part of that analysis. Four questions, in increasing order of cost:

| | what | cost |
|---|---|---|
| `loci` | the discovery leads looked up directly, per ancestry stratum | minutes |
| `pgs` | the nine ECG-trait polygenic scores, and the ancestry score's own PGS | minutes |
| `scan` | a whole-genome scan and its Manhattan, per stratum — **optional** | hours per stratum |
| | | |

**2. Can a hospital audit a vendor model for ancestry disparity without genotypes?** The released
77-finding diagnostic head flags findings at unequal rates in European and non-European patients.
`fairness` measures that gap twice — once splitting patients on their RFMix ancestry, once on the
ancestry DeepECG-Pop infers from the ECG itself — and asks whether the two agree.

**The ancestry strata.** Everything genetic is run separately in five strata, assigned at the
0.70 RFMix gate: **EUR**, **AFR**, **EAS**, **SAS**, and **ADMIXED** (no single ancestry ≥ 0.70).
EUR reproduces the discovery setup exactly. The other three are asking the European-discovered
leads a question they have never been asked, and the package reports the expected power beside
every one of their results — see *How to read a non-European stratum* below.

---

## What you need

| | | |
|---|---|---|
| **ECGs** | 12-lead, 10 s, 500 Hz or 250 Hz, **in millivolts**, as `.npy` arrays or as the `.base64` files the main pipeline writes | required |
| **RFMix** | the weighted-sum ancestry matrices you used in October | required |
| **ECG metadata** | one table per extraction round, with the patient id and acquisition date | required |
| **genotypes** | imputed dosages, chr 1–22, pgen / bed / bgen / VCF with `DS`, **GRCh38** | for everything genetic |

| **finding labels** | 0/1 per recording for any subset of the 77 findings, **or** the free-text reads | for the fairness audit |
| **ECG measurements** | your cart's PR/QRS/QT/… panel, if you have it | optional; measured here too, and `pgs` reads **both** by default |

> **If your genotypes are VCF or BCF.** Pass them as `--geno '/data/chr@.vcf.gz'` and the run
> inserts a `convert` stage that writes plink2 pgen once. plink reads a VCF directly, but it
> re-parses it on every call and a full run makes about a hundred; converting once is the
> difference between hours and days. It also fixes the four things an imputed VCF needs —
> `--double-id` sample ids, rebuilt `.` variant ids, biallelic-only records, half-calls treated as
> missing — and imports `DS` dosages when the header has them, recording which happened. Opt out
> with `--no-convert`; run it alone with
> `convert --work /out/work --vcf '/data/chr@.vcf.gz'`. A single whole-genome VCF works too: pass
> it without `@` and it is split by chromosome.

**Units are the one input error that does not raise.** A microvolt array scores like noise and
produces a plausible-looking number. `survey` checks this first and `score` refuses to run on it.

**Build.** The reference tables are GRCh38. If your release is GRCh37, pass `--build grch37`:
the discovery leads and the nine trait scores ship in both builds. The *ancestry-score* PGS ships in
GRCh38 only — see `docs/BUILD.md` for the one extra step it needs on a GRCh37 release.


### File formats: csv, tsv, or whatever your export actually wrote

Every table you supply is read by **content, not by extension**. The delimiter is sniffed from
the header line, `.gz` is followed, and the ECG inventory is accepted as parquet *or* as
delimited text. A tab-separated file named `.csv` — which is what an Excel or SQL export
usually produces — reads correctly, and so does a parquet file named `.csv`.

| you supply | formats accepted | delimiter | column names |
|---|---|---|---|
| `--ecg-meta` | parquet, csv, tsv, txt, `.gz` | sniffed | resolved case-insensitively; override with `--ecg-id-col` / `--gender-col` / `--age-col` / `--path-col` |
| `--rfmix` | csv, tsv, txt, `.gz` | sniffed (`--rfmix-sep` to force) | `--sample-col`, `--ancestries` |
| `--linkage` | csv, tsv, txt, `.gz` | sniffed | `--barcode-col`, `--pseudo-col` |
| `--labels` | csv, tsv, txt, `.gz` | sniffed | finding names from `ref/findings77.tsv` |
| `--statements` | csv, tsv, txt, `.gz` | sniffed | any id column (`participant_id`, `EMPI`, `MRN`, …) plus any text columns |
| `--measurements` | csv, tsv, txt, `.gz` | sniffed | `participant_id` plus measurement columns |
| `--ses` | csv, tsv, txt, `.gz` | sniffed | `--ses-col`, `--ses-id-col` |
| `--geno` | pgen, bed, bgen, VCF/BCF | — | `--geno-id-col` for the linkage side |
| ECG waveforms | `.npy` or `.base64` under the prepared directory | — | — |

**Label values** need not be `0`/`1`. `yes/no`, `Y/N`, `true/false`, `positive/negative`,
`present/absent` and `oui/non` are all understood, in any case, and the run prints which
mapping it used. A column it cannot map is named, with its first few values, rather than
silently becoming missing data.

A genuinely one-column file (a plain id list) is accepted as one column. The guard above fires
only when the single column *name* still has a delimiter in it, which means the header was
never split.

**The errors name the fix.** Each of these stops the run immediately and says what to change:

| what is wrong | what you are told |
|---|---|
| file does not exist | `--ses: /data/sdi.csv does not exist` |
| zero-byte file | `--ses: /data/sdi.csv is an empty file (0 bytes)` |
| header but no rows | `--ses: /data/sdi.csv has a header but no rows.` — *not* an id-mismatch message |
| a delimiter we do not know (`:`, fixed width) | `parsed to a single column, named 'participant_id:sdi'. its header still contains: ':' — save the file as comma- or tab-separated` |
| id column not recognised | `no participant id column found. columns present: ['MRN_hash', 'sdi'] — pass --ses-id-col <name>`, then `--ses-id-col MRN_hash   <- looks like the one` |
| metadata column missing | `--ecg-meta: ... is missing column(s) ['age']. columns present: [...] — name them with --age-col` |
| unmappable label values | `3 finding column(s) hold no value this image can read as 0/1 -- e.g. [...]. first few values: ['maybe', ...]. accepted: 0/1, yes/no, y/n, true/false, positive/negative, present/absent.` |
| stage run out of order | the missing file, the stage that produces it, and the full stage order — never a `FileNotFoundError` traceback |

Run `survey` before anything else: it reads a sample of every file you intend to pass, reports
what it found in each, and is the cheapest place to discover a format problem.

---

## Run it

```bash
IMG=deepecg-ancestry:1.0.0
DATA=/path/to/your/data          # read-only
OUT=/path/to/output              # work/ stays here; results/ is what you send

docker run --rm --gpus all -v $DATA:/data:ro -v $OUT:/out $IMG <mode> ...
```

### Step 0a — weights (read-only, seconds). Does the image have what it needs?

```bash
docker run --rm $IMG weights
```

Prints where each artefact will be resolved from — the pinned Hugging Face revision, or a path
you mounted — and reaches the network for nothing. If the image was built with the weights baked
in, every line says so and you are done. If you mounted them instead, add
`-v /path/to/weights:/weights:ro` and set `MODELS_ROOT`/`SSL_CKPT`/`WCR_CKPT`, or pass
`--models-root`/`--ssl-ckpt`/`--wcr-ckpt` to the mode that needs them. `weights --verify DIR`
checks a copy you already hold against its manifest; `weights --download` fetches what is
missing.

### Step 0 — survey (read-only, 1 minute). Send us this output.

```bash
docker run --rm -v $DATA:/data:ro $IMG survey \
    --rfmix /data/RFMIX/RFmix.weightedSum.BATCH1.txt \
    --ecg-meta /data/ecgs/round1/ecg_table.with_metadata.parquet \
    --geno '/data/genotypes/chr@' \
    --prepared /data/prepared_oct2026 \
    --labels /data/mgb_findings.csv
```

It prints what exists, whether the waveforms are millivolts, how many people fall in each
stratum, how many of the 77 findings your label table covers, and whether plink2 can read your
genotypes. It writes nothing but its own report.

### Step 1 — selftest (10 minutes). It must say ALL CHECKS PASSED.

```bash
docker run --rm -v $OUT:/out $IMG selftest --out /out/selftest
```

A synthetic cohort of 5,000 people with **known truth injected** — genotypes whose effect on the
phenotype we set, and a clinical finding whose false-positive gap between the two ancestry groups
we set — run through every stage. It asserts the pipeline recovers what was put in. Send back
`/out/selftest/selftest_report.txt`.

### Step 2 — the whole run

```bash
docker run --rm --gpus all -v $DATA:/data:ro -v $OUT:/out $IMG all \
    --work /out/work --results /out/results \
    --rfmix /data/RFMIX/RFmix.weightedSum.*.txt \
    --ecg-meta /data/ecgs/*/ecg_table.with_metadata.parquet \
    --geno '/data/genotypes/chr@' \
    --labels /data/mgb_findings.csv \
    --reuse-prepared /data/prepared_oct2026 \
    --patient-id-col PatientID --date-col AcquisitionDate \
    --run-name YOUR-SITE --threads 16
```

`--reuse-prepared` points at October's prepared directory and skips re-preparing — faster, and it
removes even the possibility of this run's cohort differing from that one's. Add `--with-scan`
for the genome-wide Manhattans (hours per stratum; run it after the rest has come back clean).

Or one stage at a time, which is what we suggest the first time:

```bash
$RUN prepare  --work /out/work --results /out/results --rfmix ... --ecg-meta ...
$RUN score    --work /out/work --results /out/results          # both model heads, one pass each
$RUN traits   --work /out/work                                  # run this EVEN IF you have a cart panel
$RUN geno     --work /out/work --results /out/results --geno '/data/genotypes/chr@'
$RUN loci     --work /out/work --results /out/results           # the headline genetic result
$RUN pgs      --work /out/work --results /out/results [--measurements /data/cart_panel.csv]
$RUN fairness --work /out/work --results /out/results --labels /data/mgb_findings.csv
$RUN figures  --results /out/results
$RUN package  --work /out/work --results /out/results --run-name YOUR-SITE
```

### Step 2a — which ECG measurements the positive control uses

`pgs` opens with a positive control: each of the nine trait polygenic scores has to predict **its
own measured ECG trait** at your site before it is allowed to say anything about ancestry. That
needs measurements, and there are two independent rulers for them:

* **your cart's precomputed panel**, passed as `--measurements /data/cart_panel.csv`;
* **NeuroKit2**, run here on the same waveforms by `$RUN traits` (writes `work/measurements.csv`).

Pass both and `pgs` runs the control on both, one row per ruler per trait, and prints them side by
side. **Do that on the first pass** — it is the default and the only setting that separates the two
failures that look alike: a trait that clears t = 3 on one ruler and not the other is the two
instruments disagreeing (expected on PR and on the amplitudes, where the two rulers agree
least), while a
trait that fails on *both* is genotyping, liftover or allele matching and does invalidate that
trait's results.

| flag on `pgs` | effect |
|---|---|
| *(nothing)* / `--measurement-source both` | both rulers, side by side, with a verdict per disagreement |
| `--measurement-source device` | your panel only — and any trait it does not carry is filled from NeuroKit, marked `neurokit (fill)` in `source` with the reason in `note` |
| `--measurement-source neurokit` | NeuroKit only, your panel used as the same per-trait fill |
| `--measurement-map FILE.json` | your panel's own column names, and/or a trait pinned to one ruler |

```json
{ "HR":   "VENT_RATE",
  "Tamp": { "column": "T_AMPLITUDE_V6", "source": "device" },
  "QT":   { "source": "neurokit" } }
```

The map describes **your** file only — the NeuroKit table always uses this image's column names —
and it refuses a trait name it does not know rather than quietly ignoring the line. `QRSTangle`
(the spatial QRS-T angle) has no local measurement anywhere: it needs a vectorcardiogram, and the
table says so instead of substituting the frontal-plane angle.

### Step 2b — external validation (if that is the arm you are running)

The genetics route above (`all`) and the external-validation route are **two different asks that
share one `prepare`**. If you are validating the ancestry model rather than replicating the
genetics, you need no genotypes beyond the ancestry calls themselves:

```bash
$RUN prepare  --work /out/work --results /out/results --rfmix ... --ecg-meta ...
$RUN validate --input /out/prepared/all_genotyped --output /out/results/validate
```

`validate` runs all eight pre-specified analyses from a **single** inference pass: the primary
AUROC on the training definition, the continuous dose–response against the RFMix European
fraction, a sweep of the ancestry gate, a threshold-free split, the age/sex-matched subset,
fairness by sex / age band / genotyping batch / ECG round and per non-European group, fold
agreement with a 1→5 ensemble-size sweep, and — where a person has more than one recording —
intrapatient stability as ICC(2,1) with a bootstrap interval. Every metric carries a 95%
confidence interval, and any group too small to report is marked suppressed rather than
estimated.

It is deliberately **not** part of `all`: it scores `all_genotyped/` itself, so chaining it
would score every ECG twice. `metrics`, `admixed` and `ensemble` are the same pieces as
stand-alone modes if you want one of them on its own.

### Step 3 — send back exactly one file

`$OUT/results_package_<run-name>_<date>.zip`, and the selftest report. Nothing else. `$OUT/work/`
holds the identifiers, the per-recording model outputs, the genotype subsets and the summary
statistics; it stays with you.

---

## The finding labels

The audit needs your ground truth for the 77 findings. Two ways, and the package records which
was used:

**A table** (preferred) — `participant_id` plus one 0/1 column per finding, named exactly as the
`finding` column of `ref/findings77.tsv`. Any subset is fine; 20 well-populated findings is a
good audit. `survey --labels` tells you how many of your columns were recognised.

**The free text** — `participant_id` plus one or more text columns holding the cardiologist's or
MUSE's statements. The container matches them against the regular expressions in
`findings77.tsv` (56 of the 77 have one). This is a weaker instrument and every number produced
from it is labelled as such.

A finding enters the audit only with **at least 15 positives in both ancestry groups**, and the
head's own AUROC for it at your site is reported first: a gap in a finding the model cannot
detect at all at MGB is a statement about the label, not about fairness.

---

## How to read a non-European stratum

Two things work against the discovery leads in AFR, EAS and SAS, and **neither is evidence about the
biology**:

1. **Size.** Your non-European strata are hundreds to low thousands of people, against a
   discovery meta-analysis tens of times larger. `loci/replication_summary.csv` carries `expected_n_bonferroni` — how many leads a
   sample this size *should* have detected if every discovery effect were exact. Compare it with
   `n_bonferroni` before reading a shortfall as a failure.
2. **Linkage disequilibrium.** A European lead SNP tags its causal variant through European LD,
   which does not transfer — least of all to African-ancestry genomes. An attenuated slope in AFR
   is fully consistent with the signal being real and the instrument being wrong for that
   population.

The headline for every stratum is therefore the **aggregate**: the share of leads pointing the
same way (with its binomial *P*), and the inverse-variance slope of your effects on ours through
the origin. A slope of 1 means the discovery effects reproduce at full size; below 1 with
concordant signs means they reproduce attenuated, which is what winner's curse predicts.

---

## Privacy, concretely

- `results/` is aggregate **by construction**: every table is written through a check that refuses
  an identifier column and refuses a table long enough to be per-person.
- Counts of 1–9 people are written `<10`. No metric is computed for a group with fewer than 20
  people per class. No stratum below 100 people is analysed at all. No finding with fewer than 15
  positives per group enters the audit.
- No score thresholds are exported, and curves are thinned (a fine enough ROC curve is a list of
  scores).
- `package` re-reads **every file** before zipping and searches each one for your own participant
  identifiers, read from your roster and never printed. A file that fails does not go in the zip.
- The genome-wide summary statistics are **not** in the package. They stay in `work/`. A full
  summary statistic is an export request in its own right and this image does not pre-empt it; if
  you are willing to release one, tell us and we will ask for it separately.

---

## Compute

| stage | GPU | CPU-only |
|---|---|---|
| `score` (both heads, ~25k ECGs) | ~25 min | ~6 h — use `--shard i/N` across jobs |
| `traits` (NeuroKit2) | n/a | ~2 h at `--jobs 16`, shardable |
| `geno` | n/a | 20–60 min |
| `loci` | n/a | minutes |
| `pgs` | n/a | 10–30 min |
| `fairness` | n/a | ~10 min |
| `scan` (optional) | n/a | 2–6 h per stratum at `--threads 16` |

Every long stage takes `--shard i/N` and a `--merge` pass, so it fans out over a scheduler. There
is no GPU requirement: with `--device cpu` the numbers are identical, only slower.

## If something stops

Every mode prints the image version first and writes `FAILED.txt` into `results/` on a crash,
carrying the message or the traceback. Send that file. A results directory is never left
half-new: anything already there is moved aside with a timestamp before a run starts.

---

## Optional: running it across Slurm

Slurm is never required — `python -m ancestry all ...` runs the whole study on one machine. If the site
has a scheduler, `submit` lays the same work out as job arrays instead:

```
python -m ancestry submit --work /out/work --results /out/results \
    --rfmix RF.txt --ecg-meta meta.parquet --geno '/data/chr@' --labels labels.csv \
    --image /opt/sif/deepecg-ancestry_1.0.0.sif --with-scan \
    --account <your slurm account> --partition <queue> --gpu-partition <gpu queue>
```

### The four values it will not guess

`submit` resolves the account, the partition and the GPU question before it writes anything, and
if it cannot, it stops and names every missing one in a single message. That is deliberate: these
jobs are wired with `afterok`, so one rejected `prepare` cancels all 147 descendants, and the two
most common rejections are a missing `--account` and a `--gres=gpu:1` on a cluster that has no
GPU — which is a *rejection*, not a fallback to CPU.

| value | how to find it | if the cluster does not use it |
|---|---|---|
| `--account` | `sacctmgr -nP show assoc user=$USER format=Account,Partition,QOS` | `--no-account` |
| `--partition` | `sinfo -s` (the one marked `*` is the default) | `--no-partition` |
| `--gpu-partition` | `sinfo -o "%P %G"` — a partition with `gpu:N` | `--no-gpu`, and `score` runs on CPU: same numbers, roughly twice the walltime |
| `--qos` | optional everywhere; never required | omit it |

Run from a login node, `submit` looks all three up itself and adopts anything unambiguous — one
account, one partition, a partition advertising a GPU — and says which it adopted and why. Run
from inside the container, where Slurm is not visible, it has to ask.

That writes `work/slurm/*.sbatch` and `work/slurm/submit_all.sh` and stops. Adding `--go` submits
them, with the dependencies already wired, **if `sbatch` is reachable from where you ran it** — it
usually is not from inside a container, because Slurm lives on the host, so the normal route is to
run `submit` (no `--go`) and then `bash work/slurm/submit_all.sh` from a login node.

What fans out: the genome-wide scan over stratum × chromosome (110 tasks for five strata), scoring
over GPU shards, the NeuroKit measurements over CPU shards, and `geno` over strata. `loci`, `pgs`
and `scan` do not wait for each other. Dependencies are `afterok`, so a failed stage stops its
descendants rather than packaging a half-finished result.

Other useful flags: `--max-concurrent N` (caps simultaneous array tasks on a shared queue),
`--score-shards`, `--trait-shards`, `--launcher docker|none`.

`survey` reports whether Slurm is visible, and lists the accounts and QoS it can see for you,
before you decide.

If `sbatch` rejects a stage, `submit_all.sh` stops there and prints the ids it had already
submitted together with the `scancel` line for them — cancel those before retrying, or the early
stages run twice.

The job scripts hold paths and parameters only. Slurm's logs go to `work/slurm/logs/`, which stays
at the site like the rest of `work/`.

---

## Socioeconomic deprivation (optional, `--ses`)


The threshold is **not** a single choice you have to get right. `--cut` defaults to the national
median (SDI 50, since the index is a national percentile rank) and also accepts `cohort-median`,
`cohort-mean`, `national-mean`, `q80` or a number — but whichever you pick,
`results/ses/ses_cut_grid.csv` reports the ancestry AUROC in both halves at the national median,
the cohort median, the **cohort mean** and the 80th percentile, beside a full nine-decile sweep.
Median and mean are different splits whenever the index is skewed, so a conclusion that holds at
one and not the other is a statement about the threshold rather than about deprivation, and the
grid is where you can see which it is.
If you can attach an area-level deprivation index — the Social Deprivation Index is the intended
one — the container will ask how much of the ECG ancestry signal survives adjusting for it.

```bash
$RUN ses --work /out/work --results /out/results \
    --ses /data/sdi_by_patient.csv \
    --ses-geography 'census tract' --ses-vintage 'ACS 2015-2019'
```

or add `--ses /data/sdi_by_patient.csv` to `all`. The file is `participant_id` plus one numeric
column, higher = more deprived.

**Read `ses_coupling.csv` before anything else.** In a US hospital catchment, census-tract
deprivation tracks genetic ancestry closely. Where it does, adjusting for deprivation partly
adjusts for ancestry, so every arm is reported against a *coupling-matched sham* — a meaningless
variable carrying deprivation's own ancestry correlation. The real number is read against that
sham, never against zero. Simulated at a cohort of this size with no socioeconomic path in it at
all, the score-level adjustment still returned a non-zero cost — and the sham reproduced that
cost, which is the whole reason the pair is reported together.

**Arm B is the one to report.** At a cohort of this size it is the arm that can actually bound the
socioeconomic contribution within the pre-specified margin. Arm C is unbiased but noisy and arm D
deletes a single direction, so neither carries much weight; both are included for comparability
with the manuscript's existing table.

Add `--save-embeddings` to `score` for the stronger versions of arms A and D. Without it arm A
uses the measured trait panel and arm D residualises the score, and both say so in their rows.
