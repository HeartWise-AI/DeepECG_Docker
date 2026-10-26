"""
Evaluate a dual-head WCRv2 (ACS + culprit-vessel) checkpoint on val/test.

- Deduplicates to one row per unique ECG (by digitized-NPY basename).
- ACS head: AUROC + AUPRC on test, with 1000-iter bootstrap 95% CIs.
- Operating point: Youden threshold fit on VAL, evaluated on TEST
  (sensitivity / specificity / PPV / NPV). Never argmax/0.5.
- Vessel head: per-vessel AUROC among ACS-positive samples.

Usage:
    python scripts/eval_acs_dual_head_test.py \
        --checkpoint checkpoints_acs_dual_head_v2_digitized/best_model.pt \
        --linked_parquet data/acs_multilayout_v2_dual_linked.parquet \
        --npy_dir data/acs_digitized_npy --device cuda:3
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# PyTorch 2.6 defaults weights_only=True; fairseq checkpoints contain numpy
# objects, so force the legacy behavior for the trusted local checkpoints.
_orig_torch_load = torch.load
def _torch_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _torch_load

sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")

VESSEL_CLASSES = ["LAD", "RCA", "LCX", "Left_Main"]
NUM_VESSELS = len(VESSEL_CLASSES)


class DualHeadWCR(nn.Module):
    def __init__(self, base_model, hidden_dim=768):
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

        def hook_fn(module, inp, out):
            captured["pooled"] = inp[0]

        handle = self.base.proj.register_forward_hook(hook_fn)
        try:
            output = self.base(source=source)
            acs_logits = self.base.get_logits(output)
        finally:
            handle.remove()
        vessel_logits = self.vessel_head(captured["pooled"])
        return acs_logits, vessel_logits


def load_model(checkpoint, model_dir, device):
    from fairseq_signals.utils import checkpoint_utils

    base_ssl = os.path.join(model_dir, "base_ssl.pt")
    init_ckpt = "checkpoints_acs_npy_ceiling/checkpoint_best.pt"
    src = init_ckpt if os.path.exists(init_ckpt) else os.path.join(model_dir, "wcr_77_classes.pt")
    base_model, _, _ = checkpoint_utils.load_model_and_task(
        src, arg_overrides={"model_path": base_ssl}, suffix=""
    )
    hidden_dim = base_model.proj.in_features
    base_model.proj = nn.Linear(hidden_dim, 1)
    model = DualHeadWCR(base_model, hidden_dim=hidden_dim)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, ckpt


def load_split(linked_df, npy_dir, split):
    df = linked_df[linked_df.Split == split].copy()
    df["b"] = df["npy_path"].apply(os.path.basename)
    df["dig"] = df["b"].apply(lambda b: os.path.join(npy_dir, b))
    df = df[df["dig"].apply(os.path.exists)]
    df = df.drop_duplicates(subset="b")  # one row per unique ECG
    sigs, acs, ves = [], [], []
    for _, r in df.iterrows():
        a = np.load(r["dig"]).astype(np.float32)
        if a.shape == (12, 2500):
            a = a.T
        if a.shape == (2500, 12):
            sigs.append(a)
            acs.append(r["Acute_Obstruction"])
            ves.append([r[v] for v in VESSEL_CLASSES])
    return (np.stack(sigs), np.array(acs, dtype=np.float32),
            np.array(ves, dtype=np.float32))


@torch.no_grad()
def predict(model, X, device, batch_size=256):
    acs_p, ves_p = [], []
    for i in range(0, len(X), batch_size):
        b = np.transpose(X[i:i + batch_size], (0, 2, 1))
        src = torch.from_numpy(b).float().to(device)
        al, vl = model(src)
        acs_p.append(torch.sigmoid(al).cpu().numpy())
        ves_p.append(torch.sigmoid(vl).cpu().numpy())
    acs_p = np.concatenate(acs_p, 0)
    if acs_p.ndim == 2:
        acs_p = acs_p[:, 0]
    return acs_p, np.concatenate(ves_p, 0)


def bootstrap_ci(y, p, fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        vals.append(fn(y[s], p[s]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--linked_parquet", required=True)
    ap.add_argument("--npy_dir", default="data/acs_digitized_npy")
    ap.add_argument("--model_dir", default="weights/wcr_77_classes")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading model: {args.checkpoint}")
    model, ckpt = load_model(args.checkpoint, args.model_dir, device)
    print(f"  stored best_val_auc={ckpt.get('best_val_auc')}, step={ckpt.get('global_step')}")

    linked = pd.read_parquet(args.linked_parquet)
    Xv, Yv, _ = load_split(linked, args.npy_dir, "val")
    Xt, Yt, Vt = load_split(linked, args.npy_dir, "test")
    print(f"  val unique ECGs={len(Yv)} (acs+={int(Yv.sum())})")
    print(f"  test unique ECGs={len(Yt)} (acs+={int(Yt.sum())})")

    pv, _ = predict(model, Xv, device)
    pt, vt = predict(model, Xt, device)

    # ACS metrics on test
    auroc = roc_auc_score(Yt, pt)
    auprc = average_precision_score(Yt, pt)
    auroc_ci = bootstrap_ci(Yt, pt, roc_auc_score)
    auprc_ci = bootstrap_ci(Yt, pt, average_precision_score)

    # Youden threshold fit on val, applied to test
    fpr, tpr, thr = roc_curve(Yv, pv)
    thr_y = float(thr[np.argmax(tpr - fpr)])
    pred = (pt >= thr_y).astype(int)
    tp = int(((pred == 1) & (Yt == 1)).sum()); fp = int(((pred == 1) & (Yt == 0)).sum())
    fn = int(((pred == 0) & (Yt == 1)).sum()); tn = int(((pred == 0) & (Yt == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else 0
    spec = tn / (tn + fp) if tn + fp else 0
    ppv = tp / (tp + fp) if tp + fp else 0
    npv = tn / (tn + fn) if tn + fn else 0

    # Vessel AUROC among ACS-positive test samples
    mask = Yt == 1
    vessel = {}
    for j, name in enumerate(VESSEL_CLASSES):
        yt = Vt[mask, j]; pp = vt[mask, j]
        if yt.sum() >= 5 and (1 - yt).sum() >= 5:
            vessel[name] = {
                "auroc": float(roc_auc_score(yt, pp)),
                "auroc_ci": list(bootstrap_ci(yt, pp, roc_auc_score)),
                "auprc": float(average_precision_score(yt, pp)),
                "n_pos": int(yt.sum()), "n": int(len(yt)),
            }
        else:
            vessel[name] = {"auroc": None, "n_pos": int(yt.sum()), "n": int(len(yt)),
                            "note": "insufficient positives"}

    res = {
        "checkpoint": args.checkpoint,
        "linked_parquet": args.linked_parquet,
        "acs": {
            "n_test": int(len(Yt)), "n_pos": int(Yt.sum()),
            "prevalence": float(Yt.mean()),
            "auroc": float(auroc), "auroc_ci": list(auroc_ci),
            "auprc": float(auprc), "auprc_ci": list(auprc_ci),
            "youden_threshold_val_fit": thr_y,
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "vessel": vessel,
    }
    print("\n" + json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
