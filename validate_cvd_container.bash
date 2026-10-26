#!/usr/bin/env bash
# Build and test only on the explicitly selected local Docker daemon.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo 'Usage: bash validate_cvd_container.bash WEIGHTS_DIR ENCODER_PT OUTPUT_DIR' >&2
    exit 2
fi
weights_dir=$(realpath "$1")
encoder_path=$(realpath "$2")
output_dir=$(realpath -m "$3")
python - "$weights_dir" "$encoder_path" "$output_dir" <<'PY'
from pathlib import Path
import sys
w,e,o=map(Path,sys.argv[1:])
if o.is_relative_to(w) or w.is_relative_to(o) or e.is_relative_to(o):
    raise SystemExit('Use a dedicated output directory that does not overlap model inputs.')
if o.exists() and any(o.iterdir()):
    raise SystemExit('Use an empty output directory so previous validation reports cannot be mistaken for this run.')
PY
mkdir -p "$output_dir"
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_dir"
docker_local=(env -u DOCKER_HOST -u DOCKER_CONTEXT -u DOCKER_TLS -u DOCKER_TLS_VERIFY -u DOCKER_CERT_PATH docker --host=unix:///var/run/docker.sock)
"${docker_local[@]}" info --format '{{.ServerVersion}}' > "$output_dir/docker_server_version.txt"
python - "$output_dir/config.json" <<'PY'
import json,sys
c={'weights_dir':'/models/weights','base_checkpoint':'/models/encoder.pt','params_path':'/app/clsa_cvd/model_params.json','format':'clsa','device':'cuda:0','precision':'float32','batch_size':32}
open(sys.argv[1],'w').write(json.dumps(c,indent=2)+'\n')
PY
build_args=(build -f Dockerfile.clsa-cvd --iidfile "$output_dir/image.id")
if [[ -n "${CODEX_PROXY_CERT:-}" ]]; then
    build_args+=(--secret "id=proxy_ca,src=$CODEX_PROXY_CERT")
fi
if [[ -n "${CVD_BASE_IMAGE:-}" ]]; then
    build_args+=(--build-arg "CVD_BASE_IMAGE=$CVD_BASE_IMAGE")
fi
"${docker_local[@]}" "${build_args[@]}" . > "$output_dir/build.log" 2>&1
image_id=$(cat "$output_dir/image.id")
metadata_format='{"Id":{{json .Id}},"RepoDigests":{{json .RepoDigests}},"Created":{{json .Created}},"Architecture":{{json .Architecture}},"Os":{{json .Os}}}'
"${docker_local[@]}" image inspect --format "$metadata_format" "$image_id" > "$output_dir/image_metadata.json"
"${docker_local[@]}" image inspect --format "$metadata_format" "${CVD_BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.08-py3}" > "$output_dir/base_image_metadata.json"
"${docker_local[@]}" run --rm --gpus 'device=0' --read-only --tmpfs /tmp:rw,nosuid,size=512m \
    --mount "type=bind,src=$weights_dir,dst=/models/weights,readonly" \
    --mount "type=bind,src=$encoder_path,dst=/models/encoder.pt,readonly" \
    --mount "type=bind,src=$output_dir,dst=/validation" \
    --entrypoint python "$image_id" /app/container_validate_cvd.py \
    --config /validation/config.json --out /validation/container_validation.json \
    > "$output_dir/run.log" 2>&1
python - "$output_dir" <<'PY'
from pathlib import Path
import hashlib,json,sys
p=Path(sys.argv[1]);r=json.loads((p/'container_validation.json').read_text())
assert r['model_execution_verified'] and r['artifact_hashes_match']
(p/'container_validation_envelope.json').write_text(json.dumps({'image_id':(p/'image.id').read_text().strip(),'result_sha256':hashlib.sha256((p/'container_validation.json').read_bytes()).hexdigest(),'base_image_metadata':json.loads((p/'base_image_metadata.json').read_text()),'status':'synthetic_container_validation_passed'},indent=2)+'\n')
print('Synthetic container validation passed; the image ID and aggregate evidence are recorded.')
PY
