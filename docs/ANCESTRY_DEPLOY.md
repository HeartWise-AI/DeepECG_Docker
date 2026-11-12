# Deploying the standalone Ancestry image

`dockerfile.ancestry` produces a separate image. It does not replace the default DeepECG image,
copy `main.py`, or add the genetics stack to the default runtime.

## Build and identify the image

```bash
docker build -f dockerfile.ancestry -t deepecg-ancestry:1.0.0 .
docker image inspect deepecg-ancestry:1.0.0 --format '{{.Id}} {{.Size}}'
docker run --rm --network none deepecg-ancestry:1.0.0 --help
```

The unweighted image built during this integration was 8.86 GB according to
`docker image inspect`. Treat that as a measured reference, not a release promise: record the
actual image ID and size for every build. The five fold heads, SSL backbone, 77-finding head and
reference tables are not stored in git and are not included in that figure.

The gated `heartwise/DeepECG-SSL_Ancestry` snapshot must contain the complete bundle and its exact
40-character revision must replace `PENDING-UPLOAD` in `ancestry/model_manifest.json` before an
online build can preload it. Once pinned, preload it with a BuildKit secret rather than copying a
token into a layer:

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_api_key,src=/secure/path/api_key.json \
  --build-arg RUN_MODELS_SETUP=true \
  -f dockerfile.ancestry -t deepecg-ancestry:1.0.0-bundled .
```

Air-gapped installations should use the ordinary unweighted image and mount the approved model
bundle read-only. This is also the supported route while the gated snapshot is awaiting upload.

## Verify before shipping

Run the unit suite, the full self-test with the real approved assets, and the release audit. The
self-test must end with `ALL CHECKS PASSED` and zero failed or skipped checks.

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -e MODELS_ROOT=/models \
  -e SSL_CKPT=/assets/checkpoint_last.pt \
  -e WCR_CKPT=/assets/wcr_77_classes.pt \
  -e REFERENCE_DIR=/models/reference \
  -v /approved/ancestry_bundle:/models:ro \
  -v /approved/shared_assets:/assets:ro \
  -v /private/validation:/validation \
  deepecg-ancestry:1.0.0 selftest --out /validation/selftest

docker run --rm --network none --entrypoint python \
  -v "$PWD:/repo:ro" -v /private/validation:/validation:ro -w /repo \
  deepecg-ancestry:1.0.0 tests/release_audit.py /validation/selftest
```

Before replacing an earlier Ancestry image, run the approved 25-NPY regression described in the
main README. Inputs and models are mounted read-only, both images run without network access, and
the report includes input hashes, model paths, runtime versions, all five fold probabilities and
the ensemble comparison. A tolerance of `1e-6` covers cross-runtime floating-point roundoff; use
`--atol 0` for a bitwise diagnostic.

## Export and transfer

```bash
docker save deepecg-ancestry:1.0.0 | gzip > deepecg-ancestry-1.0.0.tar.gz
sha256sum deepecg-ancestry-1.0.0.tar.gz > deepecg-ancestry-1.0.0.tar.gz.sha256
ls -lh deepecg-ancestry-1.0.0.tar.gz
```

Send the image archive, its checksum through a separate trusted channel, this guide, the user
guide, and the approved model bundle plus its checksum/provenance record. Never include an API
token, `api_key.json`, participant data, NPY comparison outputs, or per-person results.

At the receiving site:

```bash
sha256sum -c deepecg-ancestry-1.0.0.tar.gz.sha256
docker load < deepecg-ancestry-1.0.0.tar.gz
docker run --rm --network none deepecg-ancestry:1.0.0 --help
```

Record the compressed archive size and loaded image size when planning disk space; do not reuse
the measurements from the legacy image. Apptainer conversion is site-specific and is not
automated by this repository.

## Runtime boundary and outputs

Mount ECGs, genotypes, labels, weights and reference tables read-only. Mount only the work and
results directories read-write. A site run needs no network after its assets are present.

`work/` can contain identifiers, scores and genotype-derived intermediates and must remain at the
site. `results/` contains aggregates. The `package` mode re-audits those outputs and writes the
single zip intended to travel back; see `docs/ANCESTRY_USER_GUIDE.md` for the complete runbook.

The earlier deployed image used Python 3.8 and Torch 2.4. This image follows the main repository's
Python 3.11, Torch 2.8 and pinned `fairseq-signals` commit. The 25-NPY image-to-image regression is
therefore the release evidence for prediction equivalence, rather than an assertion that the two
software environments are identical.
