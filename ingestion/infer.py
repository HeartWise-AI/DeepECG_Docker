"""Run DeepECG-SSL (wcr_77_classes) on 12 x 2500 windows. Needs torch: run inside the DeepECG Docker image.

Uses the existing model wrapper unchanged; only the loaded network is switched to eval mode here.
"""
import os

import numpy as np
import torch

from models.heartwise_model_factory import HeartWiseModelFactory
from models.wcr_ecg_transformer import WCR_77_classes  # noqa: F401  (registers the subclass)
from utils.constants import WCR_COLUMN_CONVERSION

MODEL_NAME = 'wcr_77_classes'


def _api_key(path='api_key.json'):
    """Only needed if the weights aren't cached in weights/. Never logged."""
    if os.path.isdir(os.path.join('weights', MODEL_NAME)):
        return None
    import json
    with open(path) as f:
        return json.load(f)['HUGGING_FACE_API_KEY']


class WCRScorer:
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device)
        self.wrapper = HeartWiseModelFactory.create_model({
            'model_name': MODEL_NAME, 'map_location': self.device, 'hugging_face_api_key': _api_key()})
        self.wrapper.model.eval()  # the wrapper leaves dropout on (R-08)
        self.conversion = torch.tensor(WCR_COLUMN_CONVERSION, device=self.device)

    @torch.no_grad()
    def __call__(self, x, batch_size=512):
        """x [n, 12, 2500] float32 mV -> probabilities [n, 77] float32, in ECG_PATTERNS order."""
        out = np.empty((x.shape[0], 77), dtype=np.float32)
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).to(self.device)
            out[i:i + batch_size] = self.wrapper(xb)[:, self.conversion].float().cpu().numpy()
        return out
