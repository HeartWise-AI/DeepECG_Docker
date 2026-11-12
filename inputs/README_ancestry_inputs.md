# Inputs the ancestry pipeline takes

The main pipeline reads one CSV of ECG file names per task (`data_rows_template*.csv`). The
ancestry pipeline needs three site files instead, because the thing being validated is the
relationship between an ECG and a genome, and no CSV of file names carries that.

| `ancestry.config` key | what it is | format |
|---|---|---|
| `rfmix` | RFMix weighted-sum ancestry proportions, one row per genotyped person | TSV/CSV: an id column plus one column per ancestry (`AFR`, `EAS`, `SAS`, `EUR`, ...). Several files are fine -- give them space-separated, one per genotyping batch |
| `ecg_meta` | the ECG inventory | parquet, CSV or TSV: ECG id, sex, age, and the path to each waveform. `--patient-id-col` and `--date-col` as well, if repeat recordings are to be used |
| `geno` | the genotypes | plink2 pgen/bed stem or VCF/BCF, with `@` standing for the chromosome number, e.g. `/data/geno/chr@` |

Optional, each enabling one arm and skipped when absent: `labels` (the site's own 77-finding
ground truth), `statements` (free-text ECG statements instead), `measurements` (the site's cart
measurement panel), `ses` (area-level deprivation per patient).

**Run `survey` before anything else.** It opens a sample of every file named above and reports
what it found -- units, build, columns, how many people per ancestry stratum, whether there is
ground truth. It writes nothing and it is the cheapest place to find a format problem:

```bash
bash run_ancestry.bash --mode survey
```

## The waveform files themselves

The path column of `ecg_meta` may point at either format the main pipeline deals in:

| | |
|---|---|
| `.npy` | 12-lead, in millivolts. Either orientation, `(2500, 12)` or `(12, 2500)`; a longer recording is decimated by an integer step, so 10 s at 500 Hz becomes 2500 samples exactly as `utils/files_handler` does it |
| `.base64` | what the main pipeline's preprocessing writes -- raw float32, sample-major, the format defined by `ECGFileHandler.save_ecg_signal`. A test asserts this pipeline decodes it identically |

**XML is not read here.** The main pipeline parses CLSA and MHI XML in
`utils/files_handler.XMLProcessor`, but its cohort scaling happens further along, and an
unverified scale would silently change every score rather than fail. So run the main pipeline's
preprocessing first and pass the `.base64` files it produces -- which is also why a site already
running DeepECG_Docker has nothing to convert.

Anything shorter than 2500 samples, or not 12 leads, is refused rather than padded: a
part-filled waveform would still score, it would just score something else. Units are checked,
never corrected -- see `infer.check_units`.
