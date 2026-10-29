"""DeepECG SigLIP2 models for ECG page images.

A SigLIP-2 ViT-B/16 image encoder (DeepECG-PNG run ux5uajqm) and five linear probes fitted
on its frozen 768-d features. Like the signal models, every repository and immutable revision
comes from ``models/manifest.json``. The shared encoder is downloaded from
``heartwise/SigLIP-2_BaseEncoder``; the 77-class, SHD, LVEF and iAFib5 classifiers are each
downloaded from their own pinned repository.

The encoder snapshot supplies ``image_encoder.safetensors`` and ``config.json``. Each classifier
snapshot supplies ``classifier.safetensors``, ``head.json``, the same ``config.json`` and its
``thresholds.json``. All five task models share one in-process encoder.

Every task model takes a batch of preprocessed images (B, 3, 224, 224) from
``utils.image_preprocessing.load_ecg_image`` and returns probabilities (B, k).
"""
import hashlib
import json
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import CLIPVisionConfig, CLIPVisionModel

from models.heartwise_model_factory import HeartWiseModelFactory
from utils.huggingface_wrapper import HuggingFaceWrapper
from utils.log_config import get_logger

logger = get_logger(__name__)

ALL_77 = "77_classes"   # pseudo output: every 77-class label, in ECG_PATTERNS order

MODEL_REPOSITORIES = {
    "siglip2_base_encoder": "heartwise/SigLIP-2_BaseEncoder",
    "siglip2_77_classes": "heartwise/SigLIP2_77_classes",
    "siglip2_afib_5y": "heartwise/SigLIP2_AFIB_5Y",
    "siglip2_lvef_equal_under_40": "heartwise/SigLIP2_LVEF_equal_under_40",
    "siglip2_lvef_under_50": "heartwise/SigLIP2_LVEF_under_50",
    "siglip2_shd": "heartwise/SigLIP2_SHD",
}
BASE_MODEL_KEY = "siglip2_base_encoder"
BASE_REPO_ID = MODEL_REPOSITORIES[BASE_MODEL_KEY]


def download_weights(model_key: str, hugging_face_api_key: str) -> str:
    """Download one manifest-pinned SigLIP2 repository using the standard model cache."""
    try:
        repo_id = MODEL_REPOSITORIES[model_key]
    except KeyError:
        raise RuntimeError(f"Unknown SigLIP2 model key: {model_key}") from None
    return HuggingFaceWrapper.get_model(
        repo_id=repo_id,
        local_dir=os.path.join("weights", model_key),
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
        """Load the published vision tower and reject unmapped or incomplete tensors."""
        try:
            published = load_file(path)
        except Exception as e:
            raise RuntimeError(f"cannot read {path}: {type(e).__name__}: {e}") from e
        not_fp32 = sorted(k for k, v in published.items() if v.dtype != torch.float32)
        if not_fp32:
            raise RuntimeError(f"{path}: encoder tensors must be float32 ({not_fp32[:3]} are not)")
        prefix = "backbone.vision_model."
        projection = {
            "classifier.1.weight", "classifier.1.bias", "_fc_layer.weight", "_fc_layer.bias"
        }
        unmapped = sorted(key for key in published if not key.startswith(prefix) and key not in projection)
        if unmapped:
            raise RuntimeError(f"{path}: unmapped encoder tensors {unmapped[:5]}")
        state = {key[len(prefix):]: value for key, value in published.items() if key.startswith(prefix)}
        if not state:
            raise RuntimeError(f"{path}: no tensors use the documented {prefix!r} prefix")
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
    """One shared encoder plus lazily loaded, independently pinned classifier heads."""

    MAX_CACHED_ROWS = 500_000

    def __init__(self, encoder_folder: str, device: torch.device):
        self.encoder_folder = os.path.abspath(encoder_folder)
        config_path = self._required(self.encoder_folder, "config.json", BASE_REPO_ID)
        encoder_path = self._required(
            self.encoder_folder, "image_encoder.safetensors", BASE_REPO_ID)
        self.config = self._read_json(config_path)
        self.device = device
        self.encoder = DeepECG_SigLIP2_Encoder(self.config)
        self.encoder.load_weights(encoder_path)
        self.encoder.eval().to(device)
        self.encoder_sha256 = self._sha256(encoder_path)
        image_size = int(self.config["vision_config"]["image_size"])
        if tuple(int(v) for v in self.config["preprocessing"]["resize"]) != (image_size, image_size):
            raise RuntimeError(
                f"{config_path}: preprocessing resize does not match the encoder's image_size")
        self.heads = {}
        self._features = {}
        self._cached_rows = 0

    @staticmethod
    def _required(folder: str, name: str, repo_id: str) -> str:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"SigLIP2 weights not found: {path} (expected from {repo_id}; "
                f"delete {folder} to download it again)")
        return path

    @staticmethod
    def _read_json(path: str) -> dict:
        try:
            with open(path) as stream:
                value = json.load(stream)
        except Exception as e:
            raise RuntimeError(f"cannot read {path}: {type(e).__name__}: {e}") from e
        if not isinstance(value, dict):
            raise RuntimeError(f"{path}: expected a JSON object")
        return value

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load_head(self, folder: str, model_key: str, expected_outputs) -> dict:
        """Load and validate one classifier repository, caching it by snapshot path."""
        folder = os.path.abspath(folder)
        if folder in self.heads:
            return self.heads[folder]
        repo_id = MODEL_REPOSITORIES[model_key]
        config_path = self._required(folder, "config.json", repo_id)
        metadata_path = self._required(folder, "head.json", repo_id)
        classifier_path = self._required(folder, "classifier.safetensors", repo_id)
        thresholds_path = self._required(folder, "thresholds.json", repo_id)

        config = self._read_json(config_path)
        metadata = self._read_json(metadata_path)
        threshold_document = self._read_json(thresholds_path)
        if config != self.config:
            raise RuntimeError(f"{config_path}: head configuration differs from {BASE_REPO_ID}")
        outputs = metadata.get("outputs")
        if not isinstance(outputs, list) or not outputs or not all(isinstance(v, str) for v in outputs):
            raise RuntimeError(f"{metadata_path}: outputs must be a non-empty string list")
        if expected_outputs == ALL_77:
            if len(outputs) != 77 or len(set(outputs)) != 77:
                raise RuntimeError(f"{metadata_path}: expected 77 unique outputs, found {len(outputs)}")
        elif outputs != list(expected_outputs):
            raise RuntimeError(f"{metadata_path}: outputs {outputs} do not match {list(expected_outputs)}")
        if metadata.get("name") != repo_id:
            raise RuntimeError(f"{metadata_path}: model name does not match {repo_id}")

        encoder = metadata.get("encoder", {})
        if (encoder.get("repo") != BASE_REPO_ID
                or encoder.get("file") != "image_encoder.safetensors"
                or encoder.get("sha256") != self.encoder_sha256):
            raise RuntimeError(f"{metadata_path}: classifier does not match the pinned base encoder")

        classifier = metadata.get("classifier", {})
        dim = int(self.config["vision_config"]["hidden_size"])
        standardize = classifier.get("standardize") is True
        expected_shapes = {"weight": (len(outputs), dim), "bias": (len(outputs),)}
        if standardize:
            expected_shapes.update({"mean": (dim,), "scale": (dim,)})
        declared_shapes = {
            name: tuple(shape) for name, shape in classifier.get("tensors", {}).items()
            if isinstance(shape, list)
        }
        if (classifier.get("file") != "classifier.safetensors"
                or classifier.get("type") != "linear"
                or classifier.get("input_dim") != dim
                or classifier.get("dtype") != "float32"
                or classifier.get("activation") != "sigmoid"
                or declared_shapes != expected_shapes):
            raise RuntimeError(f"{metadata_path}: unsupported or inconsistent classifier metadata")
        try:
            tensors = load_file(classifier_path)
        except Exception as e:
            raise RuntimeError(f"cannot read {classifier_path}: {type(e).__name__}: {e}") from e
        if set(tensors) != set(expected_shapes):
            raise RuntimeError(
                f"{classifier_path}: tensors {sorted(tensors)} do not match {sorted(expected_shapes)}")
        wrong_shapes = {
            name: tuple(tensors[name].shape) for name, shape in expected_shapes.items()
            if tuple(tensors[name].shape) != shape
        }
        wrong_dtypes = [name for name, tensor in tensors.items() if tensor.dtype != torch.float32]
        if wrong_shapes:
            raise RuntimeError(f"{classifier_path}: tensor shapes {wrong_shapes}, expected {expected_shapes}")
        if wrong_dtypes:
            raise RuntimeError(f"{classifier_path}: classifier tensors must be float32 ({wrong_dtypes})")

        raw_thresholds = threshold_document.get("thresholds")
        if not isinstance(raw_thresholds, dict):
            raise RuntimeError(f"{thresholds_path}: thresholds must be an object")
        try:
            thresholds = {name: float(value) for name, value in raw_thresholds.items()}
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"{thresholds_path}: thresholds must be numeric") from e
        if expected_outputs == ALL_77:
            if thresholds:
                raise RuntimeError(f"{thresholds_path}: the 77-class probability head must not define cutoffs")
        elif set(thresholds) != set(outputs) or any(not 0 <= value <= 1 for value in thresholds.values()):
            raise RuntimeError(f"{thresholds_path}: expected one [0, 1] cutoff for {outputs}")

        head = {
            "outputs": outputs,
            "thresholds": thresholds,
            "tensors": {name: tensor.float().cpu() for name, tensor in tensors.items()},
            "standardize": standardize,
            "repo_id": repo_id,
        }
        self.heads[folder] = head
        logger.info("SigLIP2 classifier loaded from %s", folder)
        return head

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Pooled CLS features (B, 768), float32 on the CPU."""
        x = images.detach().to("cpu", torch.float32).contiguous()
        key = hashlib.blake2b(x.numpy().tobytes(), digest_size=16).hexdigest() + str(tuple(x.shape))
        if key not in self._features:
            feats = self.encoder(x.to(self.device)).float().cpu()
            while self._features and self._cached_rows + len(feats) > self.MAX_CACHED_ROWS:
                oldest = next(iter(self._features))
                self._cached_rows -= len(self._features.pop(oldest))
            self._features[key] = feats
            self._cached_rows += len(feats)
        return self._features[key]

    def probabilities(self, images: torch.Tensor, head: dict) -> torch.Tensor:
        """Apply one classifier head to a minibatch's shared encoder features."""
        features = self.features(images)
        tensors = head["tensors"]
        if head["standardize"]:
            features = (features - tensors["mean"]) / tensors["scale"]
        return torch.sigmoid(features @ tensors["weight"].T + tensors["bias"])


_BANKS = {}


def _get_bank(device: torch.device, hugging_face_api_key: str) -> _DeepECG_SigLIP2_Bank:
    folder = os.path.abspath(download_weights(BASE_MODEL_KEY, hugging_face_api_key))
    key = (folder, str(device))
    if key not in _BANKS:
        _BANKS[key] = _DeepECG_SigLIP2_Bank(folder, device)
        logger.info("SigLIP2 base encoder loaded from %s on %s", folder, device)
    return _BANKS[key]


# ---------------------------------------------------------------------------- task models
class DeepECG_SigLIP2_Wrapper(HeartWiseModelFactory):
    """Base of the SigLIP2 task models. Inputs are ECG page images, not signals."""

    name = 'deepecg_siglip2'
    input_type = 'image'
    model_key = None
    expected_outputs = ()

    def __init__(
        self,
        model_name: str,
        map_location: torch.device,
        hugging_face_api_key: str
    ):
        if self.model_key is None:
            raise RuntimeError(f"{model_name} is the SigLIP2 base class; use one of its task models")
        self.device = map_location
        try:
            self.bank = _get_bank(map_location, hugging_face_api_key)
            head_folder = download_weights(self.model_key, hugging_face_api_key)
            self.head = self.bank.load_head(head_folder, self.model_key, self.expected_outputs)
        except (KeyError, TypeError, ValueError) as e:
            repo_id = MODEL_REPOSITORIES.get(self.model_key, self.model_key)
            raise RuntimeError(
                f"invalid SigLIP2 weights ({repo_id}): {type(e).__name__}: {e}") from e
        self.output_names = list(self.head["outputs"])
        self.decision_threshold = (
            self.head["thresholds"][self.output_names[0]] if len(self.output_names) == 1 else None)
        self.preprocessing = self.bank.config["preprocessing"]
        logger.info("Model %s loaded on %s", model_name, map_location)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.bank.probabilities(images, self.head)


class DeepECG_SigLIP2_77_classes(DeepECG_SigLIP2_Wrapper):
    name = 'deepecg_siglip2_77_classes'
    model_key = 'siglip2_77_classes'
    expected_outputs = ALL_77


class DeepECG_SigLIP2_AFIB_5Y(DeepECG_SigLIP2_Wrapper):
    name = 'deepecg_siglip2_afib_5y'
    model_key = 'siglip2_afib_5y'
    expected_outputs = ('iAFib5',)


class DeepECG_SigLIP2_LVEF_Equal_Under_40(DeepECG_SigLIP2_Wrapper):
    name = 'deepecg_siglip2_lvef_equal_under_40'
    model_key = 'siglip2_lvef_equal_under_40'
    expected_outputs = ('LVEF_le40',)


class DeepECG_SigLIP2_LVEF_Under_50(DeepECG_SigLIP2_Wrapper):
    name = 'deepecg_siglip2_lvef_under_50'
    model_key = 'siglip2_lvef_under_50'
    expected_outputs = ('LVEF_lt50',)


class DeepECG_SigLIP2_SHD(DeepECG_SigLIP2_Wrapper):
    name = 'deepecg_siglip2_shd'
    model_key = 'siglip2_shd'
    expected_outputs = ('SHD',)
