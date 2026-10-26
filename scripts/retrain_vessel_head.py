"""
Retrain ONLY the culprit-vessel head of the dual-head WCRv2 model.

The v2 best_model.pt was saved at step 200, so the randomly-initialized vessel
head was badly undertrained. The backbone + ACS head are good, so we:
  1. Freeze the WCRv2 backbone, precompute pooled embeddings (input to base.proj)
     for the ACS-positive ECGs (cached digitized NPYs, mV / no PSA).
  2. Train a fresh vessel MLP (768->256->4) with BCE + pos_weight, dropout,
     weight decay, early-stopping on val macro-AUROC (LAD/RCA/LCX; Left Main has
     0 val positives).
  3. Splice the new head into a new full checkpoint that eval_acs_dual_head_test.py
     can load unchanged. The ACS head / backbone are untouched.

Usage:
    python scripts/retrain_vessel_head.py --device cuda:3
"""

import argparse
import os
import sys

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
            self.base(source=source)
        finally:
            h.remove()
        return cap["p"]


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
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt, hidden


@torch.no_grad()
def embed_split(model, linked, npy_dir, split, device, acs_only=True, bs=256):
    df = linked[linked.Split == split].copy()
    df["b"] = df.npy_path.apply(os.path.basename)
    df["dig"] = df["b"].apply(lambda b: os.path.join(npy_dir, b))
    df = df[df["dig"].apply(os.path.exists)].drop_duplicates("b")
    if acs_only:
        df = df[df.Acute_Obstruction == 1]
    sigs, ves = [], []
    for _, r in df.iterrows():
        a = np.load(r["dig"]).astype(np.float32)
        if a.shape == (12, 2500):
            a = a.T
        if a.shape == (2500, 12):
            sigs.append(a)
            ves.append([r[v] for v in VESSEL_CLASSES])
    X = np.stack(sigs)
    emb = []
    for i in range(0, len(X), bs):
        b = np.transpose(X[i:i + bs], (0, 2, 1))
        emb.append(model(torch.from_numpy(b).float().to(device)).cpu().numpy())
    return np.concatenate(emb, 0), np.array(ves, dtype=np.float32)


class VesselHead(nn.Module):
    def __init__(self, hidden=768, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Dropout(p), nn.Linear(256, NUM_VESSELS))

    def forward(self, x):
        return self.net(x)


def macro_auroc(y, p, idxs):
    aucs = []
    for j in idxs:
        if y[:, j].sum() >= 3 and (1 - y[:, j]).sum() >= 3:
            aucs.append(roc_auc_score(y[:, j], p[:, j]))
    return float(np.mean(aucs)) if aucs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_acs_dual_head_v2_digitized/best_model.pt")
    ap.add_argument("--linked_parquet", default="data/acs_multilayout_v2_dual_linked.parquet")
    ap.add_argument("--npy_dir", default="data/acs_digitized_npy")
    ap.add_argument("--model_dir", default="weights/wcr_77_classes")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_ckpt", default="checkpoints_acs_dual_head_v2_digitized/best_model_vessel_retrained.pt")
    ap.add_argument("--emb_cache", default="outputs/acs_dual_head_v2_eval/vessel_embeddings.npz")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading model {args.checkpoint}")
    model, ckpt, hidden = build_model(args.checkpoint, args.model_dir, device)
    linked = pd.read_parquet(args.linked_parquet)

    if os.path.exists(args.emb_cache):
        print(f"Loading cached embeddings {args.emb_cache}")
        z = np.load(args.emb_cache)
        Etr, Ytr = z["Etr"], z["Ytr"]
        Eva, Yva = z["Eva"], z["Yva"]
        Ete, Yte = z["Ete"], z["Yte"]
    else:
        print("Extracting frozen-backbone embeddings (ACS-positive ECGs)...")
        Etr, Ytr = embed_split(model, linked, args.npy_dir, "train", device)
        Eva, Yva = embed_split(model, linked, args.npy_dir, "val", device)
        Ete, Yte = embed_split(model, linked, args.npy_dir, "test", device)
        os.makedirs(os.path.dirname(args.emb_cache), exist_ok=True)
        np.savez(args.emb_cache, Etr=Etr, Ytr=Ytr, Eva=Eva, Yva=Yva, Ete=Ete, Yte=Yte)
    print(f"  train ACS+={len(Ytr)}  val ACS+={len(Yva)}  test ACS+={len(Yte)}")

    # Train only on ACS+ samples that carry >=1 vessel label (matches original design)
    tr_mask = Ytr.sum(1) > 0
    Xtr = torch.from_numpy(Etr[tr_mask]).float().to(device)
    Ttr = torch.from_numpy(Ytr[tr_mask]).float().to(device)
    print(f"  trainable vessel-labeled ACS+ = {int(tr_mask.sum())}")
    for j, v in enumerate(VESSEL_CLASSES):
        print(f"    {v}: train_pos={int(Ttr[:, j].sum())}")

    pos = Ttr.sum(0)
    neg = Ttr.shape[0] - pos
    pos_weight = (neg / pos.clamp(min=1)).to(device)

    head = VesselHead(hidden, args.dropout).to(device)
    # warm-start from existing head weights for a fair continuation
    try:
        sd = {k.replace("vessel_head.", "net."): v
              for k, v in ckpt["model_state_dict"].items() if k.startswith("vessel_head.")}
        head.load_state_dict(sd)
        print("  warm-started from existing vessel head")
    except Exception as e:
        print(f"  fresh head init ({e})")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    val_idx = [0, 1, 2]  # LAD/RCA/LCX (Left Main has 0 val positives)

    best, best_state, wait, best_ep = -1, None, 0, 0
    for ep in range(args.epochs):
        head.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(Xtr), Ttr, pos_weight=pos_weight)
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            pv = torch.sigmoid(head(torch.from_numpy(Eva).float().to(device))).cpu().numpy()
        m = macro_auroc(Yva, pv, val_idx)
        if m > best:
            best, best_ep, wait = m, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    print(f"\nBest val macro-AUROC (LAD/RCA/LCX) = {best:.4f} @ epoch {best_ep}")

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pte = torch.sigmoid(head(torch.from_numpy(Ete).float().to(device))).cpu().numpy()

    print("\nTEST per-vessel (among all test ACS+, OvR):")
    old = {"LAD": 0.727, "RCA": 0.603, "LCX": 0.555, "Left_Main": None}
    for j, v in enumerate(VESSEL_CLASSES):
        yt = Yte[:, j]
        if yt.sum() >= 5 and (1 - yt).sum() >= 5:
            au = roc_auc_score(yt, pte[:, j])
            ap_ = average_precision_score(yt, pte[:, j])
            delta = f"(was {old[v]:.3f}, Δ{au-old[v]:+.3f})" if old[v] else ""
            print(f"  {v}: AUROC {au:.3f}  AUPRC {ap_:.3f}  n_pos={int(yt.sum())}  {delta}")
        else:
            print(f"  {v}: n_pos={int(yt.sum())} insufficient")

    # Splice new head into a full checkpoint loadable by eval_acs_dual_head_test.py
    new_sd = dict(ckpt["model_state_dict"])
    for k, v in best_state.items():
        new_sd[k.replace("net.", "vessel_head.")] = v
    out = dict(ckpt)
    out["model_state_dict"] = new_sd
    out["vessel_head_retrained"] = True
    out["vessel_retrain_best_val_macro_auroc"] = best
    torch.save(out, args.out_ckpt)
    print(f"\nSaved retrained checkpoint -> {args.out_ckpt}")


if __name__ == "__main__":
    main()
