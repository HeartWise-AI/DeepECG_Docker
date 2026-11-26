"""DeepECG DINOv3-SigLIP2 task models of the CSV pipeline.

The encoder, the probe heads and the single-image predictor are in models/dinov3_bank.py. This
module adds the pinned download (models/manifest.json: heartwise/DeepECG-PNG_DINOv3-SigLIP2_epoch030)
and one HeartWiseModelFactory model per task and probe version: ``deepecg_dinov3_<task>`` (probes
fitted on clean rendered pages) and ``deepecg_dinov3_aug_<task>`` (probes fitted on the 25/25/25/25
clean / photo / multi-layout / both mix), selected in heartwise.config with ``image_models:
deepecg_dinov3`` or ``deepecg_dinov3_aug``. They share one encoder pass per batch.
"""
import os

import torch

from models.dinov3_bank import PROBE_OUTPUTS, REPO_ID, _DeepECG_DINOv3_Bank
from models.heartwise_model_factory import HeartWiseModelFactory
from utils.huggingface_wrapper import HuggingFaceWrapper
from utils.log_config import get_logger

logger = get_logger(__name__)

MODEL_KEY = "dinov3_siglip2_epoch030"


def download_weights(hugging_face_api_key: str) -> str:
    """The manifest-pinned snapshot of the encoder and probe repository."""
    return HuggingFaceWrapper.get_model(
        repo_id=REPO_ID,
        local_dir=os.path.join("weights", MODEL_KEY),
        hugging_face_api_key=hugging_face_api_key
    )


_BANKS = {}


def _get_bank(device: torch.device, hugging_face_api_key: str) -> _DeepECG_DINOv3_Bank:
    folder = os.path.abspath(download_weights(hugging_face_api_key))
    key = (folder, str(device))
    if key not in _BANKS:
        _BANKS[key] = _DeepECG_DINOv3_Bank(folder, device)
        logger.info("DINOv3 encoder loaded from %s on %s", folder, device)
    return _BANKS[key]


# ---------------------------------------------------------------------------- task models
class DeepECG_DINOv3_Wrapper(HeartWiseModelFactory):
    """Base of the DINOv3 task models. Inputs are ECG page images, not signals."""

    name = 'deepecg_dinov3'
    input_type = 'image'
    task = None             # probe folder name without its _clean / _aug suffix
    variant = 'clean'
    expected_outputs = ()

    def __init__(self, model_name: str, map_location: torch.device, hugging_face_api_key: str):
        if self.task is None:
            raise RuntimeError(f"{model_name} is the DINOv3 base class; use one of its task models")
        self.device = map_location
        self.head_key = f"{self.task}_{self.variant}"
        try:
            self.bank = _get_bank(map_location, hugging_face_api_key)
            self.head = self.bank.load_head(self.head_key, self.expected_outputs)
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(f"invalid DINOv3 weights ({REPO_ID}, {self.head_key}): {type(e).__name__}: {e}") from e
        self.output_names = list(self.head["outputs"])
        self.decision_threshold = (
            self.head["thresholds"].get(self.output_names[0]) if len(self.output_names) == 1 else None)
        self.preprocessing = self.bank.preprocessing
        logger.info("Model %s loaded on %s (probe %s)", model_name, map_location, self.head_key)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.bank.probabilities(images, self.head)


_TASKS = {
    # task: (class suffix, probe folder task)
    "77_classes": ("77_classes", "ecg77"),
    "afib_5y": ("AFIB_5Y", "afib_5y"),
    "lvef_equal_under_40": ("LVEF_Equal_Under_40", "lvef_le40"),
    "lvef_under_50": ("LVEF_Under_50", "lvef_lt50"),
    "shd": ("SHD", "shd"),
    "hcm": ("HCM", "hcm"),
    "hcm_3class": ("HCM_3class", "hcm_3class"),
    "aco": ("ACO", "aco"),
    "acco_vs_aico": ("ACCO_vs_AICO", "acco_vs_aico"),
}


def _task_class(task: str, variant: str) -> type:
    suffix, probe_task = _TASKS[task]
    variant_suffix = "" if variant == "clean" else "_aug"
    class_name = f"DeepECG_DINOv3_{suffix}{'_Aug' if variant == 'aug' else ''}"
    return type(class_name, (DeepECG_DINOv3_Wrapper,), {
        "name": f"deepecg_dinov3{variant_suffix}_{task}",
        "task": probe_task,
        "variant": variant,
        "expected_outputs": PROBE_OUTPUTS[probe_task],
        "__module__": __name__,
        "__doc__": f"DINOv3 {probe_task} probe ({variant}).",
    })


# deepecg_dinov3_<task> (clean probes) and deepecg_dinov3_aug_<task> (augmented probes)
TASK_MODELS = {(task, variant): _task_class(task, variant) for task in _TASKS for variant in ("clean", "aug")}
globals().update({cls.__name__: cls for cls in TASK_MODELS.values()})
