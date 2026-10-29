"""DeepECG-PNG: an image model for ECG pages (PNG).

A SigLIP-2 ViT-B/16 image encoder (DeepECG-PNG run ux5uajqm) and the linear probes fitted
on its frozen 768-d features. Like the other models, the weights are downloaded from the
Hugging Face repository ``heartwise/deepecg_png``, at the commit pinned in models/manifest.json
(entry ``deepecg_png``; without it the image models stop with an error), into
``weights/deepecg_png/heartwise--deepecg_png/<commit>`` (by models_setup.py at build time, or on
first use). The repository holds the folder written by
``tools/export_deepecg_png.py --push-to-hub``; the five task models share it:

* ``encoder.safetensors`` + ``config.json``: the CLIP vision tower, its config, the 2D axial
  RoPE it was trained with and the image preprocessing (grayscale, 224x224, CLIP mean/std);
* ``heads.safetensors`` + ``heads.json``: the probes (SHD / LVEF<=40 / LVEF<50, iAFib5,
  77 classes) as plain weight matrices;
* ``thresholds.json``: the validation Youden thresholds of the binary probes.

Every task model takes a batch of preprocessed images (B, 3, 224, 224) from
``utils.image_preprocessing.load_ecg_image`` and returns probabilities (B, k).
"""
import hashlib
import json
import os
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import CLIPVisionConfig, CLIPVisionModel

from models.heartwise_model_factory import HeartWiseModelFactory
from utils.huggingface_wrapper import HuggingFaceWrapper
from utils.log_config import get_logger

logger = get_logger(__name__)

WEIGHTS_NAME = "deepecg_png"
HF_REPO_ID = f"heartwise/{WEIGHTS_NAME}"
WEIGHTS_DIR = os.path.join("weights", WEIGHTS_NAME)
ALL_77 = "77_classes"   # pseudo output: every 77-class label, in ECG_PATTERNS order


def download_weights(hugging_face_api_key: str) -> str:
    """The folder holding the DeepECG-PNG weights, downloaded from Hugging Face if absent."""
    return HuggingFaceWrapper.get_model(
        repo_id=HF_REPO_ID,
        local_dir=WEIGHTS_DIR,
        hugging_face_api_key=hugging_face_api_key
    )


# ---------------------------------------------------------------------------- encoder
# 2D axial RoPE on top of CLIP's learned absolute position embedding, exactly as the
# encoder was trained (DeepECG-PNG models/self_learning/rope.py): each head is split in
# a row half and a column half of the 14x14 patch grid; the CLS token is not rotated.
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class _AxialRoPE(nn.Module):
    def __init__(self, head_dim: int, grid: int, theta: float = 10000.0, mode: str = "2d"):
        super().__init__()
        n = grid * grid + 1
        idx = torch.arange(grid * grid)
        rows, cols = torch.zeros(n), torch.zeros(n)
        rows[1:] = (idx // grid).float()
        cols[1:] = (idx % grid).float()
        if mode == "1d":
            seqpos = torch.zeros(n)
            seqpos[1:] = idx.float()
            rows, cols = seqpos, seqpos.clone()
        d_ax = head_dim // 2
        inv = 1.0 / (theta ** (torch.arange(0, d_ax, 2).float() / d_ax))

        def cos_sin(pos):
            f = torch.outer(pos, inv)
            emb = torch.cat([f, f], dim=-1)
            return emb.cos()[None, None], emb.sin()[None, None]

        cr, sr = cos_sin(rows)
        cc, sc = cos_sin(cols)
        for name, t in (("cos_r", cr), ("sin_r", sr), ("cos_c", cc), ("sin_c", sc)):
            self.register_buffer(name, t, persistent=False)

    def apply_rope(self, t: torch.Tensor) -> torch.Tensor:
        d = t.shape[-1] // 2
        tr, tc = t[..., :d], t[..., d:]
        tr = tr * self.cos_r.to(t.dtype) + _rotate_half(tr) * self.sin_r.to(t.dtype)
        tc = tc * self.cos_c.to(t.dtype) + _rotate_half(tc) * self.sin_c.to(t.dtype)
        return torch.cat([tr, tc], dim=-1)


def _rope_attention_forward(self, hidden_states, attention_mask=None, **kwargs):
    """CLIPAttention.forward with RoPE on Q/K and a plain SDPA (as in training)."""
    b, s, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    if s == self._rope.cos_r.shape[2]:
        q = self._rope.apply_rope(q)
        k = self._rope.apply_rope(k)
    o = F.scaled_dot_product_attention(
        q, k, v, dropout_p=(self.dropout if self.training else 0.0), scale=self.scale)
    o = o.transpose(1, 2).reshape(b, s, -1)
    return self.out_proj(o), None


def _vision_tower(backbone: nn.Module) -> nn.Module:
    """The module owning embeddings / encoder / post_layernorm (transformers < 5.17
    nests it under ``vision_model``; 5.17 flattened it)."""
    nested = getattr(backbone, "vision_model", None)
    if nested is not None and hasattr(nested, "encoder"):
        return nested
    if hasattr(backbone, "encoder") and hasattr(backbone, "embeddings"):
        return backbone
    raise AttributeError(f"{type(backbone).__name__}: transformers CLIP layout not recognised")


class DeepECG_SigLIP2_Encoder(nn.Module):
    """CLIP ViT-B/16 vision tower + 2D axial RoPE; returns the pooled CLS (768-d)."""

    def __init__(self, config: dict):
        super().__init__()
        vision, rope = config["vision_config"], config["rope"]
        # RoPE is applied only when the sequence matches its table; a table that does not fit
        # the patch grid would silently switch it off, so the settings must agree exactly.
        grid = int(vision["image_size"]) // int(vision["patch_size"])
        head_dim = int(vision["hidden_size"]) // int(vision["num_attention_heads"])
        if (int(rope["grid"]), int(rope["head_dim"]), rope["mode"]) != (grid, head_dim, "2d"):
            raise RuntimeError(f"RoPE settings {rope} do not fit the encoder (grid {grid}, head_dim {head_dim}, 2d)")
        self.backbone = CLIPVisionModel(CLIPVisionConfig(**vision))
        tower = _vision_tower(self.backbone)
        self.rope = _AxialRoPE(head_dim, grid, float(rope["theta"]), rope["mode"])
        for layer in tower.encoder.layers:
            layer.self_attn._rope = self.rope
            layer.self_attn.forward = types.MethodType(_rope_attention_forward, layer.self_attn)

    def load_weights(self, path: str) -> None:
        """Load the exported vision tower; any missing or unexpected tensor is an error
        (``position_ids`` is a buffer some transformers versions persist)."""
        try:
            state = load_file(path)
        except Exception as e:
            raise RuntimeError(f"cannot read {path}: {type(e).__name__}: {e}") from e
        not_fp32 = sorted(k for k, v in state.items() if v.dtype != torch.float32)
        if not_fp32:
            raise RuntimeError(f"{path}: encoder tensors must be float32 ({not_fp32[:3]} are not)")
        missing, unexpected = _vision_tower(self.backbone).load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("position_ids")]
        unexpected = [k for k in unexpected if not k.endswith("position_ids")]
        if missing or unexpected:
            # not ValueError: HeartWiseModelFactory would swallow it as "model not found"
            raise RuntimeError(f"{path}: missing {missing[:5]}, unexpected {unexpected[:5]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values=x).pooler_output


# ---------------------------------------------------------------------------- encoder + probes
class _DeepECG_SigLIP2_Bank:
    """The encoder and every probe, loaded once per (folder, device) and shared by the
    task models. Features of a batch are kept (float32, oldest batches dropped first), so
    the five tasks run the encoder once per batch rather than five times."""

    MAX_CACHED_ROWS = 500_000

    def __init__(self, folder: str, device: torch.device):
        for name in ("config.json", "encoder.safetensors", "heads.json", "heads.safetensors", "thresholds.json"):
            if not os.path.exists(os.path.join(folder, name)):
                raise FileNotFoundError(
                    f"DeepECG-PNG weights not found: {os.path.join(folder, name)} (expected from the "
                    f"Hugging Face repository {HF_REPO_ID}; delete {folder} to download it again)")
        with open(os.path.join(folder, "config.json")) as f:
            self.config = json.load(f)
        with open(os.path.join(folder, "heads.json")) as f:
            self.heads_meta = json.load(f)
        with open(os.path.join(folder, "thresholds.json")) as f:
            self.thresholds = {k: float(v) for k, v in json.load(f)["thresholds"].items()}
        self.device = device
        self.encoder = DeepECG_SigLIP2_Encoder(self.config)
        self.encoder.load_weights(os.path.join(folder, "encoder.safetensors"))
        self.encoder.eval().to(device)
        heads_path = os.path.join(folder, "heads.safetensors")
        try:
            tensors = load_file(heads_path)
        except Exception as e:
            raise RuntimeError(f"cannot read {heads_path}: {type(e).__name__}: {e}") from e
        dim = int(self.config["vision_config"]["hidden_size"])
        self.heads = {}
        self.where = {}   # output name -> (head, row)
        for head, meta in self.heads_meta["heads"].items():
            k = len(meta["outputs"])
            h = {name: tensors[f"{head}.{name}"].double() for name in ("weight", "bias")}
            shapes = {"weight": (k, dim), "bias": (k,)}
            if meta.get("standardize"):
                h["mean"] = tensors[f"{head}.mean"].double()
                h["scale"] = tensors[f"{head}.scale"].double()
                shapes.update({"mean": (dim,), "scale": (dim,)})
            wrong = {n: tuple(h[n].shape) for n, shape in shapes.items() if tuple(h[n].shape) != shape}
            if wrong:
                raise RuntimeError(f"{heads_path}: head {head} has shapes {wrong}, expected {shapes}")
            self.heads[head] = h
            for j, out in enumerate(meta["outputs"]):
                self.where[out] = (head, j)
        if tuple(int(v) for v in self.config["preprocessing"]["resize"]) != (int(self.config["vision_config"]["image_size"]),) * 2:
            raise RuntimeError(f"{folder}/config.json: preprocessing resize does not match the encoder's image_size")
        self.labels_77 = list(self.heads_meta["labels_77"])
        self._features = {}
        self._cached_rows = 0

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Pooled CLS features (B, 768), float64 on the CPU, as the probes were fitted."""
        x = images.detach().to("cpu", torch.float32).contiguous()
        key = hashlib.blake2b(x.numpy().tobytes(), digest_size=16).hexdigest() + str(tuple(x.shape))
        if key not in self._features:
            feats = self.encoder(x.to(self.device)).float().cpu()
            while self._features and self._cached_rows + len(feats) > self.MAX_CACHED_ROWS:
                oldest = next(iter(self._features))
                self._cached_rows -= len(self._features.pop(oldest))
            self._features[key] = feats
            self._cached_rows += len(feats)
        return self._features[key].double()

    def probabilities(self, images: torch.Tensor, outputs: list) -> torch.Tensor:
        """(B, len(outputs)) sigmoid probabilities, float64; NaN for an output with no head."""
        z = self.features(images)
        cache = {}
        cols = []
        for out in outputs:
            if out not in self.where:
                cols.append(torch.full((len(z),), float("nan"), dtype=torch.float64))
                continue
            head, j = self.where[out]
            if head not in cache:
                h = self.heads[head]
                x = (z - h["mean"]) / h["scale"] if "mean" in h else z
                cache[head] = torch.sigmoid(x @ h["weight"].T + h["bias"])
            cols.append(cache[head][:, j])
        return torch.stack(cols, dim=1)


_BANKS = {}


def _get_bank(device: torch.device, hugging_face_api_key: str) -> _DeepECG_SigLIP2_Bank:
    folder = os.path.abspath(download_weights(hugging_face_api_key))
    key = (folder, str(device))
    if key not in _BANKS:
        _BANKS[key] = _DeepECG_SigLIP2_Bank(folder, device)
        logger.info("DeepECG-PNG weights loaded from %s on %s", folder, device)
    return _BANKS[key]


# ---------------------------------------------------------------------------- task models
class DeepECG_SigLIP2_Wrapper(HeartWiseModelFactory):
    """Base of the DeepECG-PNG task models. Inputs are images, not signals."""

    name = 'deepecg_siglip2'
    input_type = 'image'          # utils.constants.IMAGE: fed ECG page images, not signals
    outputs = ()

    def __init__(
        self,
        model_name: str,
        map_location: torch.device,
        hugging_face_api_key: str
    ):
        if not self.outputs:
            raise RuntimeError(f"{model_name} is the DeepECG-PNG base class; use one of its task models")
        self.device = map_location
        try:
            self.bank = _get_bank(map_location, hugging_face_api_key)
        except ValueError as e:
            # e.g. a corrupt json: HeartWiseModelFactory would swallow a ValueError as "model not found"
            raise RuntimeError(f"invalid DeepECG-PNG weights ({HF_REPO_ID}, under {os.path.abspath(WEIGHTS_DIR)}): "
                               f"{type(e).__name__}: {e}") from e
        self.output_names = list(self.bank.labels_77) if self.outputs == (ALL_77,) else list(self.outputs)
        if self.outputs == (ALL_77,):
            missing = [] if any(o in self.bank.where for o in self.output_names) else ["the 77-class probe"]
        else:
            missing = [o for o in self.output_names if o not in self.bank.where]
        if missing:
            raise RuntimeError(f"{model_name}: no probe for {missing} in {HF_REPO_ID}")
        # validation Youden threshold of a binary task, from the weights' thresholds.json
        # (None for the 77 classes, which are reported as probabilities)
        self.decision_threshold = (self.bank.thresholds[self.output_names[0]]
                                   if len(self.output_names) == 1 else None)
        self.preprocessing = self.bank.config["preprocessing"]
        logger.info("Model %s loaded on %s", model_name, map_location)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.bank.probabilities(images, self.output_names)


class DeepECG_SigLIP2_77_classes(DeepECG_SigLIP2_Wrapper):
    name = 'deepecg_siglip2_77_classes'
    outputs = (ALL_77,)


class DeepECG_SigLIP2_AFIB_5Y(DeepECG_SigLIP2_Wrapper):
    name = 'deepecg_siglip2_afib_5y'
    outputs = ('iAFib5',)


class DeepECG_SigLIP2_LVEF_Equal_Under_40(DeepECG_SigLIP2_Wrapper):
    name = 'deepecg_siglip2_lvef_equal_under_40'
    outputs = ('LVEF_le40',)


class DeepECG_SigLIP2_LVEF_Under_50(DeepECG_SigLIP2_Wrapper):
    name = 'deepecg_siglip2_lvef_under_50'
    outputs = ('LVEF_lt50',)


class DeepECG_SigLIP2_SHD(DeepECG_SigLIP2_Wrapper):
    name = 'deepecg_siglip2_shd'
    outputs = ('SHD',)
