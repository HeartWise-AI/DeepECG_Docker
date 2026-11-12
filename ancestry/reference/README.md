# `ancestry/reference/` -- the frozen discovery tables

**The tables are not in this repository, on purpose.** This repository is public. The tables are
aggregates of the discovery cohorts -- UK Biobank, MHI and CLSA -- and include meta-analysis
allele frequencies at 21,055 variants and per-cohort ROC/PR curves. Whether those may be
released is a data-access question for each cohort, not a packaging convenience, so they travel
with the weights instead of with the code.

They ship under `reference/` in the pinned ancestry snapshot named in
`ancestry/model_manifest.json`. One access grant therefore covers both the model and the numbers it is
compared against, and the two can never drift apart.

What is tracked here:

| file | why it is safe to publish |
|---|---|
| `MANIFEST.txt` | md5, size and filename of each table. Fingerprints disclose nothing, and they are the integrity contract: every run verifies the tables it was given against this list. |
| `_build_reference.py`, `_build_external_reference.py` | how the tables were produced. Provenance only -- their inputs are the discovery cohorts and the fold checkpoints, neither of which is here, so they cannot run in this repository. |

## Getting the tables

Any one of these, in the order the code tries them:

```bash
export REFERENCE_DIR=/path/to/reference        # a copy you were given, or a mount
# or: put them in this directory (a baked-in or developer copy)
# or: let the pinned snapshot supply them, with a token that can read the repository
docker build -f dockerfile.ancestry --build-arg RUN_MODELS_SETUP=true \
    --secret id=hf_api_key,src=api_key.json -t deepecg-ancestry:1.0.0 .
```

`python -m ancestry <mode> --help` works without them. Any mode that reads a table fails at
startup, by name, saying which of the above to do -- it never downloads silently, and never
reaches the network when the weights are already on disk.
