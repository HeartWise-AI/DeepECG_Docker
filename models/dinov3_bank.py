"""DeepECG DINOv3-SigLIP2 encoder, probe heads and single-image predictor for ECG page images.

The DINOv3 ViT-B/16 image tower of DeepECG-PNG's DINOv3-SigLIP2 contrastive pretraining
(W&B mhi_ai/deepecg_dinov3_siglip2 run zo59vdxk, epoch 30) and its frozen linear probes. One
repository, pinned in ``models/manifest.json``, holds both:

    heartwise/DeepECG-PNG_DINOv3-SigLIP2_epoch030
      base/epoch_030.pt                  encoder checkpoint (image tower under "image_encoder")
      probes/config.json                 encoder and preprocessing description
      probes/<head>/config.json          identical to probes/config.json
      probes/<head>/head.json            outputs, encoder {repo, file, sha256}, classifier metadata
      probes/<head>/thresholds.json      {"thresholds": {output: validation Youden cutoff}} ({} = none)
      probes/<head>/classifier.safetensors   weight, bias, mean, scale (float32)

Every head comes in two versions: ``<task>_clean`` (fitted on features of clean rendered pages)
and ``<task>_aug`` (fitted on a 25/25/25/25 mix of clean, photo-distorted, multi-layout and both).
In the CSV pipeline, heartwise.config picks one with the image model family (``deepecg_dinov3``
for clean, ``deepecg_dinov3_aug``; models/dinov3_wrapper.py); here, with ``variant``.

Feature: DeepECG-PNG ``DINOv3CLIPEncoder.extract_features`` = concat[CLS, mean of the patch
tokens] of the final-norm output (1536-d, before the CLIP projection), computed under bfloat16
autocast as the probes' training features were. Probability: z = (feature - mean) / scale, then
sigmoid(z @ weight.T + bias) per output, or softmax over the outputs for the HCM 3-class head.

The architecture is built from the DINOv3 ViT-B/16 configuration below (the values of
facebook/dinov3-vitb16-pretrain-lvd1689m/config.json), so the gated Meta repository is never
contacted; every weight comes from base/epoch_030.pt, loaded strictly.

This module needs only torch, transformers, safetensors and utils/image_preprocessing.py (plus
utils/log_config.py), so other services can vendor it unchanged (the PACS-AI container does).
"""

import argparse
import hashlib
import json
import os
import sys

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from utils.log_config import get_logger

logger = get_logger(__name__)

REPO_ID = "heartwise/DeepECG-PNG_DINOv3-SigLIP2_epoch030"
ENCODER_FILE = "base/epoch_030.pt"
ENCODER_SHA256 = "335aaec39064d009c31710a96fe17d98450d85611c5bb0271ce3aad118149ae6"
ENCODER_STATE_KEY = "image_encoder"
# DINOv3 ViT-B/16 (facebook/dinov3-vitb16-pretrain-lvd1689m config.json). pos_embed_rescale is a
# training-time RoPE jitter (DeepECG-PNG sets "off" for these runs); it has no effect in eval mode.
DINOV3_VITB16 = {
    "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12, "num_attention_heads": 12,
    "hidden_act": "gelu", "attention_dropout": 0.0, "drop_path_rate": 0.0, "layer_norm_eps": 1e-05,
    "layerscale_value": 1.0, "image_size": 224, "patch_size": 16, "num_channels": 3,
    "num_register_tokens": 4, "rope_theta": 100.0, "pos_embed_shift": None, "pos_embed_jitter": None,
    "pos_embed_rescale": None, "query_bias": True, "key_bias": False, "value_bias": True,
    "proj_bias": True, "mlp_bias": True, "use_gated_mlp": False,
}
PROJECTION_DIM = 512    # the CLIP projection stored in the checkpoint (loaded, never used for features)
# Real-world pages: photos carry an EXIF orientation and exports may be transparent. The
# training pages were opaque and upright, so both steps leave them unchanged.
PAGE_FIXES = {"exif_transpose": True, "alpha_background": "white"}



# Outputs of every probe task, in head.json order
ALL_77 = "77_classes"   # pseudo output: every 77-class label, in ECG_PATTERNS order
PROBE_OUTPUTS = {
    "ecg77": ALL_77,
    "afib_5y": ("iAFib5",),
    "lvef_le40": ("LVEF<=40",),
    "lvef_lt50": ("LVEF<50",),
    "shd": ("SHD",),
    "hcm": ("confirmed HCM (vs rest)",),
    "hcm_3class": ("rest", "probable HCM", "confirmed HCM"),
    "aco": ("ACO",),
    "acco_vs_aico": ("ACCO (vs AICO; occlusion ECGs only)",),
}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str) -> dict:
    try:
        with open(path) as stream:
            value = json.load(stream)
    except Exception as e:
        raise RuntimeError(f"cannot read {path}: {type(e).__name__}: {e}") from e
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


# ---------------------------------------------------------------------------- encoder
class DeepECG_DINOv3_Encoder(nn.Module):
    """DINOv3 ViT-B/16 backbone + the CLIP projection, as DeepECG-PNG DINOv3CLIPEncoder."""

    def __init__(self):
        super().__init__()
        config = DINOv3ViTConfig(**DINOV3_VITB16)
        config._attn_implementation = "sdpa"     # as from_pretrained(..., attn_implementation="sdpa")
        self.backbone = DINOv3ViTModel(config)
        self.n_prefix = 1 + int(config.num_register_tokens)
        self.patch_size = int(config.patch_size)
        hidden = int(config.hidden_size)
        self.feature_dim = 2 * hidden
        self.classifier = nn.Sequential(nn.Dropout(p=0.0), nn.Linear(self.feature_dim, PROJECTION_DIM))

    def load_weights(self, path: str) -> None:
        """Load the image tower strictly. transformers <= 5.3 names the blocks ``layer.N``, later
        versions (and the checkpoint) ``model.layer.N``; the names are mapped to this install's."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not isinstance(state.get(ENCODER_STATE_KEY), dict):
            raise RuntimeError(f"{path}: no '{ENCODER_STATE_KEY}' state dict")
        state = state[ENCODER_STATE_KEY]
        own = set(self.state_dict())
        mapped = {}
        for key, value in state.items():
            if key not in own and key.startswith("backbone.model.layer."):
                key = "backbone.layer." + key[len("backbone.model.layer."):]
            elif key not in own and key.startswith("backbone.layer."):
                key = "backbone.model.layer." + key[len("backbone.layer."):]
            mapped[key] = value
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        if missing or unexpected:
            # not ValueError: HeartWiseModelFactory would swallow it as "model not found"
            raise RuntimeError(f"{path}: missing {sorted(missing)[:5]}, unexpected {sorted(unexpected)[:5]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled feature concat[CLS, mean(patch tokens)], (B, 1536), registers dropped."""
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise RuntimeError(f"input {tuple(x.shape[-2:])} is not a multiple of the {self.patch_size}-pixel patch")
        hidden = self.backbone(pixel_values=x).last_hidden_state
        return torch.cat([hidden[:, 0], hidden[:, self.n_prefix:].mean(dim=1)], dim=-1)


# ---------------------------------------------------------------------------- encoder + probes
class _DeepECG_DINOv3_Bank:
    """One shared encoder plus lazily loaded probe heads of the same pinned snapshot."""

    MAX_CACHED_ROWS = 500_000

    def __init__(self, folder: str, device: torch.device):
        self.folder = os.path.abspath(folder)
        config_path = self._required("probes/config.json")
        encoder_path = self._required(ENCODER_FILE)
        self.config = _read_json(config_path)
        checkpoint = self.config.get("checkpoint", {})
        if checkpoint.get("file") != ENCODER_FILE or checkpoint.get("sha256") != ENCODER_SHA256:
            raise RuntimeError(f"{config_path}: encoder checkpoint does not match {ENCODER_FILE} ({ENCODER_SHA256})")
        if checkpoint.get("state_dict_key") != ENCODER_STATE_KEY or checkpoint.get("strict") is not True:
            raise RuntimeError(f"{config_path}: unsupported checkpoint loading settings")
        if self.config.get("model_name") != "dinov3_vitb16_clip":
            raise RuntimeError(f"{config_path}: model_name is not dinov3_vitb16_clip")
        self.encoder_sha256 = _sha256(encoder_path)
        if self.encoder_sha256 != ENCODER_SHA256:
            raise RuntimeError(f"{encoder_path}: SHA-256 {self.encoder_sha256} is not the pinned encoder")
        preprocessing = self.config.get("preprocessing", {})
        feature_dim = int(self.config.get("feature", {}).get("dim", 0))
        if (tuple(int(v) for v in preprocessing.get("resize", ())) != tuple(self.config["vision_config"]["image_size"])
                or preprocessing.get("resize_mode") != "stretch"):
            raise RuntimeError(f"{config_path}: preprocessing does not match the encoder input size")
        self.device = device
        self.encoder = DeepECG_DINOv3_Encoder()
        if feature_dim != self.encoder.feature_dim:
            raise RuntimeError(f"{config_path}: feature dim {feature_dim}, encoder gives {self.encoder.feature_dim}")
        self.encoder.load_weights(encoder_path)
        self.encoder.eval().to(device)
        self.preprocessing = {**preprocessing, **PAGE_FIXES}
        self.heads = {}
        self._features = {}
        self._cached_rows = 0

    def _required(self, relative: str) -> str:
        path = os.path.join(self.folder, relative)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"DINOv3 weights not found: {path} (expected from {REPO_ID}; "
                f"delete {self.folder} to download it again)")
        return path

    def load_head(self, head_key: str, expected_outputs) -> dict:
        """Load and validate probes/<head_key>/, caching it."""
        if head_key in self.heads:
            return self.heads[head_key]
        name = f"{REPO_ID}/probes/{head_key}"
        base = f"probes/{head_key}"
        config_path = self._required(f"{base}/config.json")
        metadata_path = self._required(f"{base}/head.json")
        classifier_path = self._required(f"{base}/classifier.safetensors")
        thresholds_path = self._required(f"{base}/thresholds.json")

        if _read_json(config_path) != self.config:
            raise RuntimeError(f"{config_path}: head configuration differs from probes/config.json")
        metadata = _read_json(metadata_path)
        threshold_document = _read_json(thresholds_path)
        outputs = metadata.get("outputs")
        if not isinstance(outputs, list) or not outputs or not all(isinstance(v, str) for v in outputs):
            raise RuntimeError(f"{metadata_path}: outputs must be a non-empty string list")
        if expected_outputs == ALL_77:
            if len(outputs) != 77 or len(set(outputs)) != 77:
                raise RuntimeError(f"{metadata_path}: expected 77 unique outputs, found {len(outputs)}")
        elif outputs != list(expected_outputs):
            raise RuntimeError(f"{metadata_path}: outputs {outputs} do not match {list(expected_outputs)}")
        if metadata.get("name") != name:
            raise RuntimeError(f"{metadata_path}: head name does not match {name}")
        encoder = metadata.get("encoder", {})
        if (encoder.get("repo") != REPO_ID or encoder.get("file") != ENCODER_FILE
                or encoder.get("sha256") != self.encoder_sha256):
            raise RuntimeError(f"{metadata_path}: classifier does not match the loaded encoder")

        classifier = metadata.get("classifier", {})
        dim = self.encoder.feature_dim
        standardize = classifier.get("standardize") is True
        activation = classifier.get("activation")
        expected_shapes = {"weight": (len(outputs), dim), "bias": (len(outputs),)}
        if standardize:
            expected_shapes.update({"mean": (dim,), "scale": (dim,)})
        declared_shapes = {
            key: tuple(shape) for key, shape in classifier.get("tensors", {}).items() if isinstance(shape, list)
        }
        if (classifier.get("file") != "classifier.safetensors"
                or classifier.get("type") != "linear"
                or classifier.get("input_dim") != dim
                or classifier.get("dtype") != "float32"
                or activation not in ("sigmoid", "softmax")
                or (activation == "softmax" and len(outputs) < 2)
                or declared_shapes != expected_shapes):
            raise RuntimeError(f"{metadata_path}: unsupported or inconsistent classifier metadata")
        try:
            tensors = load_file(classifier_path)
        except Exception as e:
            raise RuntimeError(f"cannot read {classifier_path}: {type(e).__name__}: {e}") from e
        if set(tensors) != set(expected_shapes):
            raise RuntimeError(f"{classifier_path}: tensors {sorted(tensors)} do not match {sorted(expected_shapes)}")
        wrong_shapes = {k: tuple(tensors[k].shape) for k, s in expected_shapes.items() if tuple(tensors[k].shape) != s}
        if wrong_shapes:
            raise RuntimeError(f"{classifier_path}: tensor shapes {wrong_shapes}, expected {expected_shapes}")
        if any(t.dtype != torch.float32 for t in tensors.values()):
            raise RuntimeError(f"{classifier_path}: classifier tensors must be float32")

        raw_thresholds = threshold_document.get("thresholds")
        if not isinstance(raw_thresholds, dict):
            raise RuntimeError(f"{thresholds_path}: thresholds must be an object")
        try:
            thresholds = {key: float(value) for key, value in raw_thresholds.items()}
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"{thresholds_path}: thresholds must be numeric") from e
        if thresholds and (set(thresholds) != set(outputs) or any(not 0 <= v <= 1 for v in thresholds.values())):
            raise RuntimeError(f"{thresholds_path}: expected one [0, 1] cutoff per output, or none")
        if (expected_outputs == ALL_77 or activation == "softmax") and thresholds:
            raise RuntimeError(f"{thresholds_path}: a probability-only head must not define cutoffs")

        head = {
            "task": str(metadata.get("task") or head_key),
            "outputs": outputs,
            "thresholds": thresholds,
            "tensors": {key: tensor.float().cpu() for key, tensor in tensors.items()},
            "standardize": standardize,
            "activation": activation,
            "name": name,
        }
        self.heads[head_key] = head
        logger.info("DINOv3 probe head loaded: %s", head_key)
        return head

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Pooled features (B, 1536), float32 on the CPU, under bfloat16 autocast as at probe training."""
        x = images.detach().to("cpu", torch.float32).contiguous()
        key = hashlib.blake2b(x.numpy().tobytes(), digest_size=16).hexdigest() + str(tuple(x.shape))
        if key not in self._features:
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                feats = self.encoder(x.to(self.device))
            feats = feats.float().cpu()
            while self._features and self._cached_rows + len(feats) > self.MAX_CACHED_ROWS:
                oldest = next(iter(self._features))
                self._cached_rows -= len(self._features.pop(oldest))
            self._features[key] = feats
            self._cached_rows += len(feats)
        return self._features[key]

    def probabilities(self, images: torch.Tensor, head: dict) -> torch.Tensor:
        """Apply one head to a minibatch's shared encoder features."""
        return head_probabilities(self.features(images), head)


def head_probabilities(features: torch.Tensor, head: dict) -> torch.Tensor:
    """z = (feature - mean) / scale; sigmoid(z @ weight.T + bias), or softmax over the outputs."""
    tensors = head["tensors"]
    features = features.to(torch.float32)
    if head["standardize"]:
        features = (features - tensors["mean"]) / tensors["scale"]
    logits = features @ tensors["weight"].T + tensors["bias"]
    return torch.softmax(logits, dim=-1) if head["activation"] == "softmax" else torch.sigmoid(logits)


# ---------------------------------------------------------------------------- single image
# Heads of predict_image by default (decision of 2026-11-26). Off by default, available with
# heads=...: hcm_3class (softmax, probabilities only, not reviewed for use) and acco_vs_aico
# (test AUROC 0.59-0.61; only meaningful for ECGs with an acute coronary occlusion).
DEFAULT_HEADS = ("ecg77", "shd", "lvef_le40", "lvef_lt50", "afib_5y", "hcm", "aco")
GATED_HEADS = {"acco_vs_aico": "aco"}   # only interpreted when the named head is positive
PROBE_TASKS = set(PROBE_OUTPUTS)


class DeepECG_DINOv3_Predictor:
    """Every selected head of one ECG page in one encoder pass, outside the CSV pipeline.

    ``predict_image(page)`` returns {"variant", "encoder_sha256", "heads": [...]} with one entry per
    output: {"id", "label", "probability", "threshold", "positive", "applicable"}. ``threshold`` is
    the validation Youden cutoff or None (77 classes, HCM 3-class: probabilities only);
    ``positive`` is None without a threshold or when the head is not applicable; ``applicable`` is
    False for acco_vs_aico when aco is negative (it is only meaningful for occlusion ECGs).
    Single-output heads use the probe task as id (e.g. "aco"); multi-output heads use
    "<task>:<output>" (e.g. "ecg77:Afib").
    """

    def __init__(self, variant: str = "clean", device: str = "cpu", hugging_face_api_key: str = "",
                 snapshot_folder: str | None = None, heads=DEFAULT_HEADS):
        if variant not in ("clean", "aug"):
            raise ValueError("variant must be 'clean' or 'aug'")
        unknown = sorted(set(heads) - PROBE_TASKS)
        if unknown:
            raise ValueError(f"unknown head(s) {unknown}; available: {sorted(PROBE_TASKS)}")
        self.variant = variant
        device = torch.device(device)
        if snapshot_folder is None:
            from models.dinov3_wrapper import _get_bank   # the pipeline's pinned download
            self.bank = _get_bank(device, hugging_face_api_key)
        else:
            self.bank = _DeepECG_DINOv3_Bank(snapshot_folder, device)
        needed = list(heads) + [GATED_HEADS[h] for h in heads if h in GATED_HEADS and GATED_HEADS[h] not in heads]
        self.heads = list(heads)
        self.loaded = {task: self.bank.load_head(f"{task}_{variant}", PROBE_OUTPUTS[task]) for task in needed}

    def predict_tensor(self, page: torch.Tensor) -> dict:
        """page: one preprocessed page (3, H, W), e.g. from utils.image_preprocessing."""
        features = self.bank.features(page.unsqueeze(0))
        probabilities = {task: head_probabilities(features, head)[0].tolist() for task, head in self.loaded.items()}
        positives = {}
        rows = []
        for task in sorted(self.loaded, key=lambda t: t in GATED_HEADS):   # gates before gated heads
            head = self.loaded[task]
            gate = GATED_HEADS.get(task)
            applicable = gate is None or positives.get(gate) is True
            for output, probability in zip(head["outputs"], probabilities[task]):
                threshold = head["thresholds"].get(output)
                positive = None if threshold is None or not applicable else bool(probability >= threshold)
                if len(head["outputs"]) == 1:
                    positives[task] = None if threshold is None else bool(probability >= threshold)
                if task in self.heads:
                    single = len(head["outputs"]) == 1
                    rows.append({"id": task if single else f"{task}:{output}",
                                 "label": head["task"] if single else output,
                                 "probability": float(probability), "threshold": threshold,
                                 "positive": positive, "applicable": applicable})
        return {"variant": self.variant, "encoder_sha256": self.bank.encoder_sha256, "heads": rows}

    def predict_image(self, page) -> dict:
        """page: a PNG / JPEG / PDF path (checked and decoded as the pipeline does) or a PIL image."""
        from utils.image_preprocessing import load_ecg_page, preprocess_ecg_image

        if isinstance(page, (str, os.PathLike)):
            tensor = load_ecg_page(str(page), self.bank.preprocessing)
        else:
            tensor = preprocess_ecg_image(page, self.bank.preprocessing)
        return self.predict_tensor(tensor)


def main(argv=None) -> int:
    """python -m models.dinov3_bank --image page.png [--variant clean|aug] [--device cuda:0]
    [--snapshot DIR | --api-key-file api_key.json] [--heads aco shd ...]: JSON lines, one per image."""
    parser = argparse.ArgumentParser(description="DeepECG DINOv3 probes on single ECG page images")
    parser.add_argument("--image", nargs="+", required=True, help="PNG, JPEG or PDF ECG pages")
    parser.add_argument("--variant", default="clean", choices=("clean", "aug"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--snapshot", help=f"a local copy of {REPO_ID} (base/ and probes/)")
    parser.add_argument("--api-key-file", default="api_key.json", help="used when --snapshot is not given")
    parser.add_argument("--heads", nargs="+", default=list(DEFAULT_HEADS), choices=sorted(PROBE_TASKS))
    args = parser.parse_args(argv)
    key = ""
    if not args.snapshot:
        from utils.credentials import read_api_key
        key = read_api_key(args.api_key_file)["HUGGING_FACE_API_KEY"]
    predictor = DeepECG_DINOv3_Predictor(args.variant, args.device, key, args.snapshot, args.heads)
    for path in args.image:
        print(json.dumps({"image": os.path.basename(path), **predictor.predict_image(path)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
