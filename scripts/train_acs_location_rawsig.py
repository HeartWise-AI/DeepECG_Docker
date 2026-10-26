"""
End-to-end WCRv2 fine-tune of the culprit-vessel (location) head on RAW signals.

Per request:
  - Binary ACS label kept as-is (Acute_Obstruction = ACCO u AICO).
  - Location head trained on ALL labeled ACCO/AICO cases (2,022 ECGs), not the
    ~411 previously cached. Uses raw MHI signal NPYs (ADC units x 0.00488 -> mV),
    NOT pre-digitized signals.
  - Backbone fine-tuned end-to-end, initialized from the v2 dual-head checkpoint.
  - ACS head is preserved structurally and re-evaluated afterwards to report drift
    (end-to-end vessel fine-tuning shifts the shared backbone).

Usage:
    python scripts/train_acs_location_rawsig.py --device cuda:3
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

_orig = torch.load
def _load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig(*a, **k)
torch.load = _load

sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")

VESSEL_CLASSES = ["LAD", "RCA", "LCX", "Left_Main"]
NUM_VESSELS = 4
ADC_TO_MV = 0.00488


class DualHeadWCR(nn.Module):
    def __init__(self, base_model, hidden_dim=768):
        super().__init__()
        self.base = base_model
        self.vessel_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, NUM_VESSELS),
        )

    def forward(self, source):
        cap = {}
        h = self.base.proj.register_forward_hook(lambda m, i, o: cap.__setitem__("p", i[0]))
        try:
            out = self.base(source=source)
            acs_logits = self.base.get_logits(out)
        finally:
            h.remove()
        return acs_logits, self.vessel_head(cap["p"])


def build_model(ckpt_path, model_dir, device):
    from fairseq_signals.utils import checkpoint_utils
    base_ssl = os.path.join(model_dir, "base_ssl.pt")
    init = "checkpoints_acs_npy_ceiling/checkpoint_best.pt"
    src = init if os.path.exists(init) else os.path.join(model_dir, "wcr_77_classes.pt")
    base_model, _, _ = checkpoint_utils.load_model_and_task(
        src, arg_overrides={"model_path": base_ssl}, suffix="")
    hidden = base_model.proj.in_features
    base_model.proj = nn.Linear(hidden, 1)
    model = DualHeadWCR(base_model, hidden)
    ck = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device), hidden


def load_raw_split(linked, split, labeled_only=True, cache_dir="outputs/acs_location_rawsig"):
    cache = os.path.join(cache_dir, f"raw_{split}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["X"], z["acs"], z["ves"]
    df = linked[linked.Split == split].copy()
    df["key"] = df.npy_path.apply(lambda p: os.path.basename(p).replace(".npy", ""))
    df = df.drop_duplicates("key")
    df = df[df.Acute_Obstruction == 1]
    if labeled_only:
        df = df[df[VESSEL_CLASSES].sum(axis=1) > 0]
    df = df[df.npy_path.apply(os.path.exists)]
    X, acs, ves = [], [], []
    for _, r in df.iterrows():
        a = np.load(r.npy_path).squeeze().astype(np.float32)  # (2500,12)
        if a.shape == (12, 2500):
            a = a.T
        if a.shape != (2500, 12):
            continue
        a = np.nan_to_num(a) * ADC_TO_MV
        X.append(a)
        acs.append(r.Acute_Obstruction)
        ves.append([r[v] for v in VESSEL_CLASSES])
    X = np.stack(X).astype(np.float32)
    acs = np.array(acs, dtype=np.float32)
    ves = np.array(ves, dtype=np.float32)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache, X=X, acs=acs, ves=ves)
    return X, acs, ves


@torch.no_grad()
def predict(model, X, device, bs=64):
    model.eval()
    ap, vp = [], []
    for i in range(0, len(X), bs):
        b = np.transpose(X[i:i + bs], (0, 2, 1))
        al, vl = model(torch.from_numpy(b).float().to(device))
        ap.append(torch.sigmoid(al).cpu().numpy())
        vp.append(torch.sigmoid(vl).cpu().numpy())
    ap = np.concatenate(ap, 0)
    if ap.ndim == 2:
        ap = ap[:, 0]
    return ap, np.concatenate(vp, 0)


def boot_ci(y, p, fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < 2:
            continue
        out.append(fn(y[s], p[s]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else [None, None]


def vessel_report(Yv, Pv, tag):
    print(f"\n  [{tag}] per-vessel AUROC (n={len(Yv)} ACS+):")
    res = {}
    for j, v in enumerate(VESSEL_CLASSES):
        yt = Yv[:, j]
        if yt.sum() >= 5 and (1 - yt).sum() >= 5:
            au = roc_auc_score(yt, Pv[:, j])
            res[v] = {"auroc": float(au), "auroc_ci": boot_ci(yt, Pv[:, j], roc_auc_score),
                      "auprc": float(average_precision_score(yt, Pv[:, j])), "n_pos": int(yt.sum())}
            print(f"    {v}: AUROC {au:.3f} CI{tuple(round(x,3) for x in res[v]['auroc_ci'])} "
                  f"AUPRC {res[v]['auprc']:.3f} (n_pos={int(yt.sum())})")
        else:
            res[v] = {"auroc": None, "n_pos": int(yt.sum())}
            print(f"    {v}: n_pos={int(yt.sum())} insufficient")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_ckpt", default="checkpoints_acs_dual_head_v2_digitized/best_model.pt")
    ap.add_argument("--linked_parquet", default="data/acs_multilayout_v2_dual_linked.parquet")
    ap.add_argument("--model_dir", default="weights/wcr_77_classes")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", default="checkpoints_acs_location_v2_raw")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading model from {args.init_ckpt}")
    model, hidden = build_model(args.init_ckpt, args.model_dir, device)

    print("Loading raw signals (x0.00488 -> mV)...")
    linked = pd.read_parquet(args.linked_parquet)
    Xtr, _, Vtr = load_raw_split(linked, "train")
    Xva, _, Vva = load_raw_split(linked, "val")
    Xte, Ate, Vte = load_raw_split(linked, "test")
    print(f"  train={len(Xtr)} val={len(Xva)} test={len(Xte)} labeled ACS+")

    # Baseline: current checkpoint on the raw test set (before fine-tuning)
    a0, v0 = predict(model, Xte, device)
    base_vessel = vessel_report(Vte, v0, "BEFORE (v2 ckpt on raw test)")

    pos = Vtr.sum(0); neg = len(Vtr) - pos
    pos_weight = torch.tensor(neg / np.clip(pos, 1, None), dtype=torch.float32, device=device)
    print(f"\n  vessel train positives: {dict(zip(VESSEL_CLASSES, pos.astype(int)))}")

    opt = torch.optim.AdamW([
        {"params": model.base.parameters(), "lr": args.lr_backbone},
        {"params": model.vessel_head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.wd)

    val_idx = [0, 1, 2]  # LAD/RCA/LCX (Left Main has 0 val positives)
    Ttr = torch.from_numpy(Vtr).float()
    best, best_state, wait, best_ep = -1, None, 0, 0
    n = len(Xtr)

    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(n)
        t0 = time.time()
        ep_loss = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            b = np.transpose(Xtr[idx], (0, 2, 1))
            src = torch.from_numpy(b).float().to(device)
            tgt = Ttr[idx].to(device)
            opt.zero_grad()
            _, vl = model(src)
            loss = F.binary_cross_entropy_with_logits(vl, tgt, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(idx)

        _, pv = predict(model, Xva, device)
        aucs = [roc_auc_score(Vva[:, j], pv[:, j]) for j in val_idx
                if Vva[:, j].sum() >= 3 and (1 - Vva[:, j]).sum() >= 3]
        vmacro = float(np.mean(aucs)) if aucs else 0.0
        print(f"  ep{ep+1:02d} loss={ep_loss/n:.4f} val_macro_auroc={vmacro:.4f} ({time.time()-t0:.0f}s)")
        if vmacro > best:
            best, best_ep, wait = vmacro, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  early stop @ ep{ep+1}")
                break

    print(f"\nBest val macro-AUROC (LAD/RCA/LCX) = {best:.4f} @ epoch {best_ep+1}")
    model.load_state_dict(best_state)

    a1, v1 = predict(model, Xte, device)
    after_vessel = vessel_report(Vte, v1, "AFTER (raw fine-tune)")

    # ACS drift on the same raw test set (ACS+ only here, so report mean prob shift)
    print(f"\n  ACS head check on raw test (all ACS+, n={len(Ate)}): "
          f"mean P(ACS) before={a0.mean():.3f} after={a1.mean():.3f} "
          f"(recall@0.5 before={(a0>=.5).mean():.3f} after={(a1>=.5).mean():.3f})")

    ckpt_out = os.path.join(args.save_dir, "best_model.pt")
    torch.save({
        "model_state_dict": best_state,
        "vessel_classes": VESSEL_CLASSES,
        "base_model": "WCRv2-amp-preserved",
        "trained_on": "raw signals x0.00488 mV, labeled ACCO/AICO cases",
        "init_from": args.init_ckpt,
        "best_val_macro_auroc": best,
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
    }, ckpt_out)

    summary = {
        "checkpoint": ckpt_out,
        "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "best_val_macro_auroc": best,
        "test_vessel_before": base_vessel,
        "test_vessel_after": after_vessel,
        "acs_mean_prob_before": float(a0.mean()), "acs_mean_prob_after": float(a1.mean()),
    }
    out_json = os.path.join("outputs/acs_location_rawsig", "results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved checkpoint -> {ckpt_out}\nSaved results -> {out_json}")


if __name__ == "__main__":
    main()
