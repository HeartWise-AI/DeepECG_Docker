#!/usr/bin/env python3
"""
Export dual-head WCR (ACS + Vessel) model to ONNX.

Usage:
    python scripts/export_dual_head_onnx.py \
        --weights checkpoints_acs_dual_head/best_model.pt \
        --output weights/exported/acs_dual_head.onnx
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")
sys.path.insert(0, "/volume/DeepECG_Docker")

NUM_VESSELS = 4
VESSEL_CLASSES = ["LAD", "RCA", "LCX", "Left_Main"]


class DualHeadWCR(nn.Module):
    """Standalone dual-head WCR wrapper for ONNX export."""

    def __init__(self, base_model, hidden_dim: int = 768):
        super().__init__()
        self.base = base_model
        self.vessel_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, NUM_VESSELS),
        )

    def forward(self, source):
        captured = {}

        def hook_fn(module, input, output):
            captured['pooled'] = input[0]

        handle = self.base.proj.register_forward_hook(hook_fn)
        try:
            output = self.base(source=source)
            acs_logits = self.base.get_logits(output)
        finally:
            handle.remove()

        pooled = captured['pooled']
        vessel_logits = self.vessel_head(pooled)

        return torch.sigmoid(acs_logits), torch.sigmoid(vessel_logits)


class DualHeadONNXWrapper(nn.Module):
    """Clean wrapper without hooks for ONNX tracing."""

    def __init__(self, base_model, vessel_head, hidden_dim=768):
        super().__init__()
        self.base = base_model
        self.vessel_head = vessel_head

    def forward(self, x):
        # Replicate the WCR forward pass without hooks
        # The model structure is: base.encoder.feature_extractor, etc.
        enc = self.base.encoder

        # 1. Feature extraction
        features = enc.feature_extractor(x)
        features = features.transpose(1, 2)  # (B, T, C)
        features = enc.layer_norm(features)
        features = enc.post_extract_proj(features)

        # 2. Positional encoding
        features = features + enc.conv_pos(features)

        # 3. Dropout
        features = enc.dropout_input(features)

        # 4. Transformer encoder
        x_enc = enc.encoder(features, padding_mask=None)
        # x_enc is typically a dict with 'x' key or the tensor directly
        if isinstance(x_enc, dict):
            x_enc = x_enc['x']

        # 5. Mean pooling
        pooled = x_enc.mean(dim=1)  # (B, hidden_dim)

        # 6. Final dropout + ACS head
        pooled_drop = self.base.final_dropout(pooled)
        acs_logits = self.base.proj(pooled_drop)  # (B, 1)
        acs_prob = torch.sigmoid(acs_logits)

        # 7. Vessel head
        vessel_logits = self.vessel_head(pooled)  # (B, 4)
        vessel_prob = torch.sigmoid(vessel_logits)

        return acs_prob, vessel_prob


def export(weights_path: str, output_path: str, model_dir: str):
    from fairseq_signals.utils import checkpoint_utils

    print(f"Loading checkpoint: {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Load base WCR model
    base_ssl = os.path.join(model_dir, "base_ssl.pt")
    init_ckpt = "checkpoints_acs_npy_ceiling/checkpoint_best.pt"
    if os.path.exists(init_ckpt):
        base_model, _, _ = checkpoint_utils.load_model_and_task(
            init_ckpt,
            arg_overrides={"model_path": base_ssl},
            suffix=""
        )
    else:
        wcr_path = os.path.join(model_dir, "wcr_77_classes.pt")
        base_model, _, _ = checkpoint_utils.load_model_and_task(
            wcr_path, arg_overrides={"model_path": base_ssl}, suffix=""
        )

    hidden_dim = base_model.proj.in_features
    print(f"  Hidden dim: {hidden_dim}")

    # Build dual-head model and load weights
    model = DualHeadWCR(base_model, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Remove weight_norm before export
    for name, module in model.named_modules():
        if hasattr(module, "weight_g") and hasattr(module, "weight_v"):
            torch.nn.utils.remove_weight_norm(module)

    # Use the hook-free wrapper for ONNX
    wrapper = DualHeadONNXWrapper(model.base, model.vessel_head, hidden_dim)
    wrapper.eval()

    # Verify wrapper matches original
    dummy = torch.randn(1, 12, 2500)
    with torch.no_grad():
        ref_acs, ref_vessel = model(dummy)
        wrap_acs, wrap_vessel = wrapper(dummy)

    diff_acs = (ref_acs - wrap_acs).abs().max().item()
    diff_vessel = (ref_vessel - wrap_vessel).abs().max().item()
    print(f"  Wrapper verification: ACS diff={diff_acs:.6f}, Vessel diff={diff_vessel:.6f}")

    if diff_acs > 1e-3 or diff_vessel > 1e-3:
        print("  WARNING: Large difference — falling back to tracing the hooked model")
        # Trace instead
        traced = torch.jit.trace(model, dummy)
        wrapper = traced

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"  Exporting to ONNX: {output_path}")
    torch.onnx.export(
        wrapper, dummy, output_path,
        input_names=["ecg_12lead"],
        output_names=["acs_probability", "vessel_probability"],
        dynamic_axes={
            "ecg_12lead": {0: "batch_size"},
            "acs_probability": {0: "batch_size"},
            "vessel_probability": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    # Verify ONNX
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)
    onnx_out = sess.run(None, {"ecg_12lead": dummy.numpy()})

    with torch.no_grad():
        ref_acs, ref_vessel = model(dummy)

    diff_acs = np.abs(onnx_out[0] - ref_acs.numpy()).max()
    diff_vessel = np.abs(onnx_out[1] - ref_vessel.numpy()).max()
    print(f"  ONNX verification: ACS max diff = {diff_acs:.6f} {'OK' if diff_acs < 1e-3 else 'WARNING'}")
    print(f"  ONNX verification: Vessel max diff = {diff_vessel:.6f} {'OK' if diff_vessel < 1e-3 else 'WARNING'}")

    # Print checkpoint metadata
    if "best_val_auc" in ckpt:
        print(f"\n  Checkpoint info:")
        print(f"    Best Val AUC: {ckpt['best_val_auc']:.4f}")
        print(f"    Epoch: {ckpt.get('epoch', '?')}")
        print(f"    Step: {ckpt.get('global_step', '?')}")
    if "vessel_aucs" in ckpt and ckpt["vessel_aucs"]:
        print(f"    Vessel AUCs: {ckpt['vessel_aucs']}")

    print(f"\n  Output names: acs_probability (B,1), vessel_probability (B,4)")
    print(f"  Vessel order: {VESSEL_CLASSES}")
    print(f"\nDone!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="checkpoints_acs_dual_head/best_model.pt")
    parser.add_argument("--output", default="weights/exported/acs_dual_head.onnx")
    parser.add_argument("--model_dir", default="weights/wcr_77_classes")
    args = parser.parse_args()

    export(args.weights, args.output, args.model_dir)


if __name__ == "__main__":
    main()
