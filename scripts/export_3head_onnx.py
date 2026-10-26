#!/usr/bin/env python3
"""
Export the 3-head deployment model (ACS + ACCO-vs-rest gate + culprit vessel) to ONNX.

Checkpoint: checkpoints_acs_acco_vessel/best_model.pt
  - model_state_dict: base (WCRv2 encoder + proj=ACS) + vessel_head (768->256->4)
  - acco_head_state_dict: ACCO-vs-rest head (768->128->1)

Inputs : ecg_12lead (B, 12, 2500) float32, raw mV (ADC x0.00488)
Outputs: acs_probability (B,1), acco_probability (B,1), vessel_probability (B,4)
Deploy : show culprit vessel only if acco_probability > gate_thr (stored in ckpt).

Usage: python scripts/export_3head_onnx.py --device cpu
"""
import argparse, os, sys, numpy as np, torch, torch.nn as nn
_o = torch.load
torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})
sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")
sys.path.insert(0, "/volume/DeepECG_Docker")

CKPT = "checkpoints_acs_acco_vessel/best_model.pt"
VESSEL_CLASSES = ["LAD", "RCA", "LCX", "Left_Main"]


def mlp(din, dh, dout):
    return nn.Sequential(nn.Linear(din, dh), nn.ReLU(), nn.Dropout(0.0), nn.Linear(dh, dout))


class ThreeHeadWCR(nn.Module):
    """Hooked reference model (matches training/eval forward)."""
    def __init__(self, base, acco_head, vessel_head):
        super().__init__()
        self.base = base
        self.acco_head = acco_head
        self.vessel_head = vessel_head

    def forward(self, source):
        cap = {}
        h = self.base.proj.register_forward_hook(lambda m, i, o: cap.__setitem__("p", i[0]))
        try:
            out = self.base(source=source)
            acs = self.base.get_logits(out)
        finally:
            h.remove()
        p = cap["p"]
        return torch.sigmoid(acs), torch.sigmoid(self.acco_head(p)), torch.sigmoid(self.vessel_head(p))


class ThreeHeadONNX(nn.Module):
    """Hook-free wrapper replicating the WCR encoder forward for tracing."""
    def __init__(self, base, acco_head, vessel_head):
        super().__init__()
        self.base = base
        self.acco_head = acco_head
        self.vessel_head = vessel_head

    def forward(self, x):
        enc = self.base.encoder
        f = enc.feature_extractor(x)
        f = f.transpose(1, 2)
        f = enc.layer_norm(f)
        f = enc.post_extract_proj(f)
        f = f + enc.conv_pos(f)
        f = enc.dropout_input(f)
        xe = enc.encoder(f, padding_mask=None)
        if isinstance(xe, dict):
            xe = xe["x"]
        pooled = xe.mean(dim=1)
        acs = torch.sigmoid(self.base.proj(self.base.final_dropout(pooled)))
        acco = torch.sigmoid(self.acco_head(pooled))
        vessel = torch.sigmoid(self.vessel_head(pooled))
        return acs, acco, vessel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--output", default="weights/exported/acs_acco_vessel_3head.onnx")
    ap.add_argument("--model_dir", default="weights/wcr_77_classes")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from fairseq_signals.utils import checkpoint_utils

    ck = torch.load(args.ckpt, map_location="cpu")
    base_ssl = os.path.join(args.model_dir, "base_ssl.pt")
    skel = "checkpoints_acs_npy_ceiling/checkpoint_best.pt"
    src = skel if os.path.exists(skel) else os.path.join(args.model_dir, "wcr_77_classes.pt")
    base, _, _ = checkpoint_utils.load_model_and_task(src, arg_overrides={"model_path": base_ssl}, suffix="")
    hidden = base.proj.in_features
    base.proj = nn.Linear(hidden, 1)

    ah = ck["acco_head_arch"]["hidden"]; vh = ck["vessel_head_arch"]["hidden"]
    acco_head = mlp(hidden, ah, 1)
    vessel_head = mlp(hidden, vh, 4)

    model = ThreeHeadWCR(base, acco_head, vessel_head)
    # load base + vessel_head from model_state_dict; acco_head from its own dict
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    acco_head.load_state_dict({k: v for k, v in ck["acco_head_state_dict"].items()})
    model.eval()
    # sanity: only acco_head.* should be "missing" from model_state_dict load
    leftover = [m for m in missing if not m.startswith("acco_head")]
    assert not leftover, f"unexpected missing keys: {leftover[:6]}"
    print(f"loaded: base+vessel from model_state_dict, acco_head({ah}) from acco_head_state_dict")

    # remove weight_norm for export
    for _, mod in model.named_modules():
        if hasattr(mod, "weight_g") and hasattr(mod, "weight_v"):
            torch.nn.utils.remove_weight_norm(mod)

    wrapper = ThreeHeadONNX(model.base, model.acco_head, model.vessel_head).eval()
    dummy = torch.randn(1, 12, 2500)
    with torch.no_grad():
        r = model(dummy); w = wrapper(dummy)
    for name, a, b in zip(["acs", "acco", "vessel"], r, w):
        d = (a - b).abs().max().item()
        print(f"  wrapper vs hooked {name}: max diff {d:.2e}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_kwargs = dict(
        input_names=["ecg_12lead"],
        output_names=["acs_probability", "acco_probability", "vessel_probability"],
        dynamic_axes={"ecg_12lead": {0: "batch"}, "acs_probability": {0: "batch"},
                      "acco_probability": {0: "batch"}, "vessel_probability": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    try:
        torch.onnx.export(wrapper, dummy, args.output, dynamo=False, **export_kwargs)
    except TypeError:
        # older torch without the dynamo kwarg uses the legacy exporter by default
        torch.onnx.export(wrapper, dummy, args.output, **export_kwargs)
    mb = os.path.getsize(args.output) / 1e6
    print(f"exported -> {args.output} ({mb:.1f} MB)")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"ecg_12lead": dummy.numpy()})
        with torch.no_grad():
            ref = model(dummy)
        for name, o, rf in zip(["acs", "acco", "vessel"], out, ref):
            d = np.abs(o - rf.numpy()).max()
            print(f"  ONNX vs torch {name}: max diff {d:.2e} {'OK' if d < 1e-3 else 'WARN'}")
        # batch test
        out2 = sess.run(None, {"ecg_12lead": np.random.randn(4, 12, 2500).astype(np.float32)})
        print(f"  batch shapes: acs{out2[0].shape} acco{out2[1].shape} vessel{out2[2].shape}")
    except ImportError:
        print("  (onnxruntime not installed; skipped runtime verification)")

    print(f"\nVessel order: {VESSEL_CLASSES}")
    print(f"Deploy: show vessel only if acco_probability > {ck.get('gate_thr_spec90_val'):.6f} (val-fit ~90% spec)")
    print("Input: raw 12-lead mV (ADC x0.00488), shape (B,12,2500)")


if __name__ == "__main__":
    main()
