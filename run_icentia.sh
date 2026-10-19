#!/usr/bin/env bash
# Run the external-validation ingestion pipeline (python -m ingestion.run) inside the DeepECG Docker image.
#
#   ./run_icentia.sh --data <icentia root> --lead 12 \
#       --mapping configs/icentia_mapping.csv [--thresholds <thresholds.csv>] \
#       --patients p00000-p00009 --out <out dir> [--gpu 2] [--image deepecg-docker]
#
#   ./run_icentia.sh --self-test        # run tests_ingestion/ inside the image
#
# Takes the same arguments as ingestion/run.py, with host paths. The script:
#   - mounts --data read-only (the dataset is never written to) and --out read-write;
#   - mounts --mapping / --thresholds files read-only wherever they are (relative paths are resolved
#     from the current directory, as for --data and --out);
#   - mounts ingestion/, configs/ and thresholds/ from this repo read-only (the image has none of them);
#   - caches the model weights in this repo's weights/ (downloaded on first use with the image's api_key.json);
#   - runs as the calling user, so outputs aren't owned by root, and removes the container afterwards.
# Pass --gpu N to choose the GPU (default 0); --device is set by the script.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image=deepecg-docker
gpu=0
data="" out=""
args=() mounts=()

die() { echo "run_icentia.sh: $*" >&2; exit 1; }
abspath() { realpath -m -- "$1"; }

common=(--rm --user "$(id -u):$(id -g)" -e HOME=/tmp -w /app
        -v "$repo/ingestion:/app/ingestion:ro" -v "$repo/configs:/app/configs:ro"
        -v "$repo/thresholds:/app/thresholds:ro")

if [[ "${1:-}" == "--self-test" ]]; then
    exec docker run "${common[@]}" -v "$repo/tests_ingestion:/app/tests_ingestion:ro" \
        --entrypoint python "$image" -m pytest -q -p no:cacheprovider tests_ingestion
fi

n_extra=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)  [[ $# -ge 2 ]] || die "--data needs a value"; data="$(abspath "$2")"; shift 2 ;;
        --out)   [[ $# -ge 2 ]] || die "--out needs a value"; out="$(abspath "$2")"; shift 2 ;;
        --gpu)   [[ $# -ge 2 ]] || die "--gpu needs a value"; gpu="$2"; shift 2 ;;
        --image) [[ $# -ge 2 ]] || die "--image needs a value"; image="$2"; shift 2 ;;
        --device|--device=*) die "use --gpu N instead of --device" ;;
        --data=*|--out=*|--mapping=*|--thresholds=*) die "use '${1%%=*} VALUE' (space, not '=')" ;;
        --mapping|--thresholds)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            f="$(abspath "$2")"   # relative to where you run the script, like --data and --out
            [[ -f "$f" ]] || die "$1: file not found: $2 (looked for $f)"
            n_extra=$((n_extra + 1)); dest="/app/cli_inputs/$n_extra/$(basename "$f")"
            mounts+=(-v "$(abspath "$f"):$dest:ro"); args+=("$1" "$dest"); shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done

[[ -n "$data" ]] || die "--data is required"
[[ -n "$out" ]] || die "--out is required"
[[ -d "$data" ]] || die "--data: not a directory: $data"
[[ "$out/" != "$data/"* ]] || die "--out must not be inside --data (the dataset is read-only)"
docker image inspect "$image" >/dev/null 2>&1 || die "Docker image '$image' not found (build it from the repo's dockerfile)"

mkdir -p "$out" "$repo/weights"
exec docker run "${common[@]}" --gpus "\"device=$gpu\"" --name "icentia_run_$$" \
    -e HF_HOME=/app/weights/.hf_cache \
    -e ICENTIA_HOST_DATA="$data" -e ICENTIA_HOST_OUT="$out" \
    -v "$data:/data/icentia:ro" -v "$out:/out" -v "$repo/weights:/app/weights" "${mounts[@]}" \
    --entrypoint python "$image" -m ingestion.run --data /data/icentia --out /out --device cuda:0 "${args[@]}"
