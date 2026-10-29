"""Export the DeepECG-PNG model (SigLIP-2 encoder ux5uajqm + linear probes) into a small,
self-contained weights folder for the Docker image (models/siglip2_wrapper.py).

Run it ONCE on the machine that holds the training checkpoints, with the DeepECG-PNG
repository's Python environment (it reuses that repository's validated loaders, so the
probes are checked to belong to the encoder and each threshold to its probe weights).
It needs the DeepECG-PNG branch with utils/siglip2_probe_bank.py (the ECG PDF pipeline,
issue #215) and the Hugging Face cache of openai/clip-vit-base-patch16 (or network access):

    /path/to/DeepECG-PNG/.venv/bin/python tools/export_deepecg_png.py \\
        --deepecg-png-repo /path/to/DeepECG-PNG \\
        --config config/ecg_pdf/siglip2_probes.yaml \\
        --out weights/deepecg_png --push-to-hub

--push-to-hub uploads the exported folder to the private Hugging Face repository
heartwise/deepecg_png (created if needed; token from `huggingface-cli login` or HF_TOKEN),
from which the Docker image downloads it like every other model (models/siglip2_wrapper.py).

Writes to --out:
    encoder.safetensors   the CLIP vision tower (float32), keys relative to the tower
    config.json           CLIP vision config, 2D axial RoPE settings, image preprocessing
    heads.safetensors     probe weights (float64), heads.json their outputs
    thresholds.json       validation Youden thresholds of the binary probes
    provenance.json       source files (sha256), epochs, validation AUROCs, versions

--reference-out FILE additionally writes, for --reference-n validation PNGs per cohort,
the features and probabilities computed by the DeepECG-PNG code itself, for the parity
test of the Docker implementation (tests/test_deepecg_png_parity.py). That file lists PNG
paths: keep it out of git.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import save_file

VISION_KEYS = ("hidden_size", "intermediate_size", "projection_dim", "num_hidden_layers",
               "num_attention_heads", "num_channels", "image_size", "patch_size", "hidden_act",
               "layer_norm_eps", "attention_dropout", "initializer_range", "initializer_factor")
HEAD_NAMES = {"shd_lvef_3head": "shd_lvef", "iaf5": "iaf5", "cls77": "cls77"}
COHORT_CACHES = ("shdlvef3", "iaf5", "cls77")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(repo: str):
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain"], capture_output=True, text=True).stdout
        return out.stdout.strip() + (" +uncommitted changes" if dirty.strip() else "")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deepecg-png-repo", required=True)
    ap.add_argument("--config", default="config/ecg_pdf/siglip2_probes.yaml",
                    help="probe-bank config, relative to --deepecg-png-repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--features-root", default="/volume/DeepECG-PNG/checkpoints/linprobe_ux5uajqm/features",
                    help="cached validation features of the encoder (for --reference-out)")
    ap.add_argument("--reference-out", default=None)
    ap.add_argument("--reference-n", type=int, default=128)
    ap.add_argument("--push-to-hub", action="store_true",
                    help="upload the exported folder to --hub-repo (private) after the export")
    ap.add_argument("--hub-repo", default="heartwise/deepecg_png")
    args = ap.parse_args()

    # absolute BEFORE the chdir below (relative paths are meant from the caller's directory)
    final_out = os.path.abspath(args.out)
    reference_out = os.path.abspath(args.reference_out) if args.reference_out else None
    repo = os.path.abspath(args.deepecg_png_repo)
    sys.path.insert(0, repo)
    os.chdir(repo)
    from models.vit_clip import clip_vision_tower
    from utils.constants import ECG_PATTERNS
    from utils.siglip2_probe_bank import Siglip2ProbeBank

    torch.set_grad_enabled(False)
    bank = Siglip2ProbeBank(args.config, device="cpu")
    model = bank.encoder
    tower = clip_vision_tower(model.backbone)
    vc = model.backbone.config
    rope = model.rope
    grid = int(vc.image_size) // int(vc.patch_size)
    head_dim = int(vc.hidden_size) // int(vc.num_attention_heads)
    # the RoPE settings written below must rebuild the encoder's own rotation tables
    from models.self_learning.rope import AxialRoPE
    again = AxialRoPE(head_dim, grid, theta=10000.0, has_cls=True, mode=rope.mode)
    for buf in ("cos_r", "sin_r", "cos_c", "sin_c"):
        if not torch.equal(getattr(again, buf), getattr(rope, buf).cpu()):
            raise SystemExit(f"RoPE {buf}: the exported settings do not rebuild the encoder's tables")

    # every check passed: write into a temporary folder, moved into place at the end, so a
    # failure never leaves --out with a mix of old and new files
    out = f"{final_out}.tmp-{os.getpid()}"
    os.makedirs(out)
    try:
        write_export(out, final_out, bank, tower, vc, rope, grid, head_dim, repo, ECG_PATTERNS)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)   # never leave a half-written temporary folder
        raise
    print(f"exported to {final_out}: " + ", ".join(sorted(os.listdir(final_out))))
    if args.push_to_hub:
        push_to_hub(final_out, args.hub_repo)
    if reference_out:
        write_reference(bank, args.features_root, args.reference_n, reference_out)


def write_export(out, final_out, bank, tower, vc, rope, grid, head_dim, repo, ECG_PATTERNS):
    """Write every file into `out`, then move them into `final_out` (0644)."""

    # ---- encoder: the vision tower only (no classifier / projection; the probes read
    # the pooled CLS), keys relative to the tower so any transformers layout can load it
    state = {k: v.detach().to(torch.float32).contiguous() for k, v in tower.state_dict().items()
             if not k.endswith("position_ids")}
    save_file(state, os.path.join(out, "encoder.safetensors"))
    config = {
        "model": "DeepECG-PNG SigLIP-2 ViT-B/16 + 2D axial RoPE (run ux5uajqm)",
        "vision_config": {k: getattr(vc, k) for k in VISION_KEYS},
        "rope": {"mode": rope.mode, "grid": grid, "head_dim": head_dim, "theta": 10000.0},
        "preprocessing": {"resize": [bank.resize, bank.resize], "resize_mode": "stretch",
                          "color_mode": bank.normalization["color_mode"],
                          "mean": bank.normalization["mean"], "std": bank.normalization["std"]},
        "feature": "pooler_output (768-d pooled CLS)",
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---- probes
    tensors, heads_meta, thresholds = {}, {}, {}
    for p in bank.probes:
        name = HEAD_NAMES[p.name]
        tensors[f"{name}.weight"] = torch.from_numpy(np.ascontiguousarray(p.weight, dtype=np.float64))
        tensors[f"{name}.bias"] = torch.from_numpy(np.ascontiguousarray(p.bias, dtype=np.float64))
        meta = {"outputs": list(p.outputs), "standardize": p.mean is not None, "source": p.name}
        if p.mean is not None:
            tensors[f"{name}.mean"] = torch.from_numpy(np.ascontiguousarray(p.mean, dtype=np.float64))
            tensors[f"{name}.scale"] = torch.from_numpy(np.ascontiguousarray(p.scale, dtype=np.float64))
        heads_meta[name] = meta
        if p.binary_calls:
            thresholds.update({o: float(t) for o, t in zip(p.outputs, p.thresholds)})
    save_file(tensors, os.path.join(out, "heads.safetensors"))
    with open(os.path.join(out, "heads.json"), "w") as f:
        json.dump({"heads": heads_meta, "labels_77": list(ECG_PATTERNS)}, f, indent=2)
    with open(os.path.join(out, "thresholds.json"), "w") as f:
        json.dump({"thresholds": thresholds,
                   "rule": "validation Youden threshold of the epoch the probe weights come from; "
                           "prediction = probability >= threshold",
                   "sources": {p.name: p.sources.get("threshold_files") for p in bank.probes if p.binary_calls}},
                  f, indent=2)

    # ---- provenance
    prov = bank.provenance()
    enc = bank.encoder_checkpoint
    prov["encoder"]["sha256"] = sha256(enc)
    prov["encoder"]["size_bytes"] = os.path.getsize(enc)
    for p in bank.probes:
        src = p.sources.get("checkpoint") or p.sources.get("bundle")
        prov["probes"][p.name]["sha256"] = sha256(src)
    prov.update({"exported": datetime.datetime.now().isoformat(timespec="seconds"),
                 "deepecg_png_commit": git_commit(repo),
                 "versions": {"torch": torch.__version__,
                              "transformers": __import__("transformers").__version__}})
    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2, default=str)
    os.makedirs(final_out, exist_ok=True)
    for name in sorted(os.listdir(out)):
        # readable by any container user (safetensors writes 0600)
        os.chmod(os.path.join(out, name), 0o644)
        os.replace(os.path.join(out, name), os.path.join(final_out, name))
    os.rmdir(out)


def push_to_hub(folder, repo_id):
    """Upload the exported weights to a private Hugging Face model repository. Only the
    exported files are sent (never the --reference-out file, which lists patient PNG paths)."""
    from huggingface_hub import HfApi
    files = ("encoder.safetensors", "config.json", "heads.safetensors", "heads.json",
             "thresholds.json", "provenance.json")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=folder,
                               allow_patterns=list(files),
                               commit_message=f"DeepECG-PNG export {datetime.date.today().isoformat()}")
    print(f"uploaded {', '.join(files)} to https://huggingface.co/{repo_id} ({commit.oid})")


def write_reference(bank, features_root, n, reference_out):
    """Features and probabilities of the DeepECG-PNG code on `n` validation PNGs per cohort,
    for tests/test_deepecg_png_parity.py."""
    from PIL import Image
    outputs = [o for p in bank.probes for o in p.outputs]
    files, cached = [], []
    for cohort in COHORT_CACHES:
        d = torch.load(os.path.join(features_root, cohort, "val.pt"), map_location="cpu",
                       weights_only=False, mmap=True)
        idx = np.linspace(0, len(d["files"]) - 1, n).round().astype(int)
        files += [str(d["files"][i]) for i in idx]
        cached.append(d["features"][torch.as_tensor(idx)].float().numpy())
    feats = np.concatenate([bank.features([Image.open(f) for f in files[i:i + 16]])
                            for i in range(0, len(files), 16)])
    probs = np.concatenate([p.probabilities(feats) for p in bank.probes], axis=1)
    pixels = np.stack([bank.preprocess(Image.open(f)).numpy() for f in files[:16]])
    np.savez_compressed(reference_out, files=np.array(files), features=feats,
                        cached_features=np.concatenate(cached), probabilities=probs,
                        outputs=np.array(outputs), pixels=pixels)
    cos = (feats * np.concatenate(cached)).sum(1) / (
        np.linalg.norm(feats, axis=1) * np.linalg.norm(np.concatenate(cached), axis=1))
    print(f"reference: {len(files)} PNGs -> {reference_out}; DeepECG-PNG code vs cached "
          f"features: min cosine {cos.min():.6f}")


if __name__ == "__main__":
    main()
