# import useful old code

import os
import sys
import random
import importlib
import math

import torch
import torch.nn as nn

from collections import OrderedDict
from typing import Any, Dict, Optional, Union

#TODO: configure it
project_dir = os.path.abspath("/volume/deepecg/fairseq-signals")
root_dir = project_dir
if not root_dir in sys.path:
    sys.path.append(root_dir)

spec = importlib.util.spec_from_file_location("checkpoint_utils", f"{project_dir}/fairseq_signals/utils/checkpoint_utils.py")
checkpoint_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint_utils)


class WCREcgTransformer(nn.Module):
    def __init__(
        self, 
        model_path: str,
        pretrained_path: str = None,
        overrides: Optional[Dict[str, Any]] = None,
        task=None,
        strict=True,
        suffix="",
        num_shards=1,
        state=None,
    ):
        super().__init__()
        overrides = {} if overrides is None else vars(overrides)
        if pretrained_path is not None:
            overrides.update({"model_path": pretrained_path})
        model, saved_cfg, task = checkpoint_utils.load_model_and_task(
            model_path,
            arg_overrides=overrides,
            suffix=suffix
        )

        self.model = model
        
    def forward(self, x, padding_mask=None):
        net_input = { "source": x, "padding_mask": padding_mask}
        net_output = self.model(**net_input)
        return self.model.get_logits(net_output), net_output
