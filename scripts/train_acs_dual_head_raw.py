"""
Unified RAW-signal dual-head WCRv2 model: ACS detection + culprit-vessel location.

Per request: a single model trained ONLY from raw signals (no digitization),
initialized from WCRv2 / DeepECG-SSL-v2 (amp-preserved 77-label SSL base), doing
BOTH tasks:
  - ACS head (1 logit): Acute_Obstruction = ACCO u AICO, trained on the full set
    (positives + negatives), BCE with pos_weight.
  - Vessel head (4 logits): LAD/RCA/LCX/Left_Main, trained on ACS-positive cases
    with vessel labels (vessel loss masked to those samples).
  combined loss = acs_loss + vessel_weight * vessel_loss

Raw MHI signal NPYs are ADC units; converted to mV via x0.00488 (matches the
amplitude-preserved WCRv2 input scale). Signals are cached per split as .npy.

Usage:
    python scripts/train_acs_dual_head_raw.py --device cuda:3
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
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

_orig = torch.load
def _load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig(*a, **k)
torch.load = _load

sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")

VESSEL_CLASSES = ["LAD", "RCA", "LCX", "Left_Main"]
NUM_VESSELS = 4
ADC_TO_MV = 0.00488
# Raw SSL-v2 encoder (NOT the 77-label fine-tune) — backbone init for an ACS-only model
SSL_V2_ENCODER = "/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-wcr-amp-preserved-encoder.pt"
# ft77 classifier checkpoint, used ONLY to source the ecg_transformer_classifier cfg/architecture
FT77_ARCH = "/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-amp-preserved-ft77-best.pt"


class DualHeadWCR(nn.Module):
    def __init__(self, base_model, hidden_dim=768):
        super().__init__()
        self.base = base_model
        self.vessel_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, NUM_VESSELS),
        )
        nn.init.xavier_uniform_(self.vessel_head[0].weight)
        nn.init.xavier_uniform_(self.vessel_head[3].weight)

    def forward(self, source):
        cap = {}
        h = self.base.proj.register_forward_hook(lambda m, i, o: cap.__setitem__("p", i[0]))
        try:
            out = self.base(source=source)
            acs_logits = self.base.get_logits(out)
        finally:
            h.remove()
        return acs_logits, self.vessel_head(cap["p"])


def build_model(init_ckpt, model_dir, device):
    """Build a dual-head classifier whose backbone is the RAW SSL-v2 encoder.

    We take the ecg_transformer_classifier cfg/architecture from the ft77
    checkpoint, override model_path to the SSL-v2 *encoder* (raw, not 77-label
    fine-tuned), and build the model WITHOUT loading the ft77 state. build_model
    loads the SSL-v2 encoder via from_pretrained, with a freshly initialized proj
    head -> i.e. raw SSL-v2 init for an ACS-only model.
    """
    from fairseq_signals.utils import checkpoint_utils
    from fairseq_signals import tasks
    state = checkpoint_utils.load_checkpoint_to_cpu(
        FT77_ARCH, arg_overrides={"model_path": SSL_V2_ENCODER, "no_pretrained_weights": False})
    cfg = state["cfg"] if state.get("cfg") is not None else \
        checkpoint_utils.convert_namespace_to_omegaconf(state["args"])
    task = tasks.setup_task(cfg.task, from_checkpoint=True)
    if "task_state" in state:
        task.load_state_dict(state["task_state"])
    base_model = task.build_model(cfg.model)  # loads SSL-v2 encoder; fresh proj; NO ft77 state
    print(f"  backbone init = RAW SSL-v2 encoder ({SSL_V2_ENCODER.split('/')[-1]}); fresh heads")
    hidden = base_model.proj.in_features
    base_model.proj = nn.Linear(hidden, 1)
    nn.init.xavier_uniform_(base_model.proj.weight)
    nn.init.constant_(base_model.proj.bias, 0.0)
    return DualHeadWCR(base_model, hidden).to(device), hidden


def cache_split(linked, split, cache_dir):
    xp = os.path.join(cache_dir, f"raw_{split}_X.npy")
    yp = os.path.join(cache_dir, f"raw_{split}_meta.npz")
    if os.path.exists(xp) and os.path.exists(yp):
        z = np.load(yp)
        return np.load(xp, mmap_mode="r"), z["acs"], z["ves"]
    df = linked[linked.Split == split].copy()
    df["key"] = df.npy_path.apply(lambda p: os.path.basename(p).replace(".npy", ""))
    df = df.drop_duplicates("key")
    df = df[df.npy_path.apply(os.path.exists)]
    n = len(df)
    X = np.zeros((n, 2500, 12), dtype=np.float32)
    acs = np.zeros(n, dtype=np.float32)
    ves = np.zeros((n, NUM_VESSELS), dtype=np.float32)
    k = 0
    t0 = time.time()
    for _, r in df.iterrows():
        a = np.load(r.npy_path).squeeze().astype(np.float32)
        if a.shape == (12, 2500):
            a = a.T
        if a.shape != (2500, 12):
            continue
        X[k] = np.nan_to_num(a) * ADC_TO_MV
        acs[k] = r.Acute_Obstruction
        ves[k] = [r[v] for v in VESSEL_CLASSES]
        k += 1
        if k % 5000 == 0:
            print(f"    {split}: {k}/{n} ({time.time()-t0:.0f}s)", flush=True)
    X = X[:k]; acs = acs[:k]; ves = ves[:k]
    os.makedirs(cache_dir, exist_ok=True)
    np.save(xp, X)
    np.savez(yp, acs=acs, ves=ves)
    return np.load(xp, mmap_mode="r"), acs, ves


@torch.no_grad()
def predict(model, X, device, bs=128):
    model.eval()
    ap, vp = [], []
    for i in range(0, len(X), bs):
        b = np.ascontiguousarray(np.transpose(X[i:i + bs], (0, 2, 1)))
        al, vl = model(torch.from_numpy(b).float().to(device))
        ap.append(torch.sigmoid(al).float().cpu().numpy())
        vp.append(torch.sigmoid(vl).float().cpu().numpy())
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_ckpt", default=SSL_V2_ENCODER,
                    help="raw SSL-v2 encoder used as backbone init (label only; build_model uses the module constants)")
    ap.add_argument("--linked_parquet", default="data/acs_multilayout_v2_dual_linked.parquet")
    ap.add_argument("--model_dir", default="weights/wcr_77_classes")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--vessel_weight", type=float, default=2.0)
    ap.add_argument("--pos_frac", type=float, default=0.35,
                    help="target fraction of ACS-positive samples per batch (oversampling)")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default="outputs/acs_dual_head_raw/cache")
    ap.add_argument("--save_dir", default="checkpoints_acs_dual_head_v2_raw")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Init backbone from RAW SSL-v2 encoder: {args.init_ckpt}", flush=True)
    model, hidden = build_model(args.init_ckpt, args.model_dir, device)

    print("Caching raw signals (x0.00488 -> mV)...", flush=True)
    linked = pd.read_parquet(args.linked_parquet)
    Xtr, Atr, Vtr = cache_split(linked, "train", args.cache_dir)
    Xva, Ava, Vva = cache_split(linked, "val", args.cache_dir)
    Xte, Ate, Vte = cache_split(linked, "test", args.cache_dir)
    print(f"  train={len(Atr)} (ACS+={int(Atr.sum())}) val={len(Ava)} test={len(Ate)}", flush=True)

    # class weights. ACS pos_weight matches the OVERSAMPLED batch distribution
    # (~pos_frac positives) so we don't double-correct the imbalance.
    acs_pos_weight = torch.tensor([(1 - args.pos_frac) / args.pos_frac],
                                  dtype=torch.float32, device=device)
    lab = Vtr.sum(1) > 0
    vp = Vtr[lab].sum(0); vn = lab.sum() - vp
    vessel_pos_weight = torch.tensor(vn / np.clip(vp, 1, None), dtype=torch.float32, device=device)
    print(f"  ACS pos_weight={acs_pos_weight.item():.2f} (oversampled)  vessel pos_weight={vessel_pos_weight.tolist()}", flush=True)

    opt = torch.optim.AdamW([
        {"params": model.base.parameters(), "lr": args.lr_backbone},
        {"params": model.vessel_head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler()

    try:
        import wandb
        os.environ.setdefault("WANDB_MODE", "offline")
        wandb.init(project="deepecg-acs", name="acs-dual-head-v2-raw-unified",
                   dir=".", config=vars(args))
        use_wandb = True
    except Exception as e:
        print(f"  wandb off ({e})"); use_wandb = False

    Atr_t = torch.from_numpy(np.asarray(Atr)).float()
    Vtr_t = torch.from_numpy(np.asarray(Vtr)).float()
    n = len(Atr)
    best, best_state, wait, best_ep = -1, None, 0, 0

    # Oversampling: draw each epoch's indices so ~pos_frac of samples are ACS+
    # (and thus vessel-labeled cases appear far more often than their 3.5% base rate),
    # giving the rare location head enough gradient while ACS still sees all negatives.
    acs_pos_idx = np.where(np.asarray(Atr) == 1)[0]
    acs_neg_idx = np.where(np.asarray(Atr) == 0)[0]
    w = np.ones(n, dtype=np.float64)
    if len(acs_pos_idx):
        # weight positives so expected positive fraction per draw == pos_frac
        w[acs_pos_idx] = (args.pos_frac / (1 - args.pos_frac)) * (len(acs_neg_idx) / len(acs_pos_idx))
    w = w / w.sum()
    print(f"  oversampling: target pos_frac={args.pos_frac} (ACS+ base rate {len(acs_pos_idx)/n:.3f})", flush=True)

    for ep in range(args.epochs):
        model.train()
        perm = np.random.choice(n, size=n, replace=True, p=w)
        t0 = time.time(); run = 0.0
        for s in range(0, n, args.batch_size):
            idx = np.sort(perm[s:s + args.batch_size])
            b = np.ascontiguousarray(np.transpose(Xtr[idx], (0, 2, 1)))
            src = torch.from_numpy(b).float().to(device)
            at = Atr_t[idx].unsqueeze(1).to(device)
            vt = Vtr_t[idx].to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                al, vl = model(src)
                la = F.binary_cross_entropy_with_logits(al, at, pos_weight=acs_pos_weight)
                m = (at.squeeze(1) == 1) & (vt.sum(1) > 0)
                lv = (F.binary_cross_entropy_with_logits(vl[m], vt[m], pos_weight=vessel_pos_weight)
                      if m.any() else torch.tensor(0.0, device=device))
                loss = la + args.vessel_weight * lv
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            run += loss.item() * len(idx)

        pa, pv = predict(model, Xva, device)
        acs_auc = roc_auc_score(Ava, pa)
        vmac = np.mean([roc_auc_score(Vva[:, j], pv[:, j]) for j in range(3)
                        if Vva[:, j].sum() >= 3])
        combined = 0.5 * acs_auc + 0.5 * vmac
        print(f"  ep{ep+1:02d} loss={run/n:.4f} val_ACS_auroc={acs_auc:.4f} "
              f"val_vessel_macro={vmac:.4f} combined={combined:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if use_wandb:
            wandb.log({"epoch": ep + 1, "train_loss": run / n, "val_acs_auroc": acs_auc,
                       "val_vessel_macro": vmac, "val_combined": combined})
        if combined > best:
            best, best_ep, wait = combined, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "vessel_classes": VESSEL_CLASSES,
                        "base_model": "WCRv2/DeepECG-SSL-v2 amp-preserved",
                        "trained_on": "raw signals x0.00488 mV (both heads)",
                        "init_from": args.init_ckpt, "epoch": ep + 1,
                        "val_acs_auroc": acs_auc, "val_vessel_macro": vmac},
                       os.path.join(args.save_dir, "best_model.pt"))
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  early stop @ ep{ep+1}", flush=True); break

    print(f"\nBest combined val={best:.4f} @ ep{best_ep+1}", flush=True)
    model.load_state_dict(best_state)

    # ---- Test evaluation ----
    pa, pv = predict(model, Xte, device)
    pav, _ = predict(model, Xva, device)
    acs_auc = roc_auc_score(Ate, pa); acs_ap = average_precision_score(Ate, pa)
    fpr, tpr, thr = roc_curve(Ava, pav)
    thy = float(thr[np.argmax(tpr - fpr)])
    pred = (pa >= thy).astype(int)
    tp = int(((pred == 1) & (Ate == 1)).sum()); fp = int(((pred == 1) & (Ate == 0)).sum())
    fn = int(((pred == 0) & (Ate == 1)).sum()); tn = int(((pred == 0) & (Ate == 0)).sum())
    acs = {"auroc": acs_auc, "auroc_ci": boot_ci(Ate, pa, roc_auc_score),
           "auprc": acs_ap, "auprc_ci": boot_ci(Ate, pa, average_precision_score),
           "youden_thr_val": thy, "sens": tp / (tp + fn), "spec": tn / (tn + fp),
           "ppv": tp / (tp + fp) if tp + fp else 0, "npv": tn / (tn + fn) if tn + fn else 0,
           "n_test": len(Ate), "n_pos": int(Ate.sum())}

    m = Ate == 1
    vessel = {}
    for j, v in enumerate(VESSEL_CLASSES):
        yt = Vte[m, j]; pp = pv[m, j]
        if yt.sum() >= 5 and (1 - yt).sum() >= 5:
            vessel[v] = {"auroc": float(roc_auc_score(yt, pp)),
                         "auroc_ci": boot_ci(yt, pp, roc_auc_score),
                         "auprc": float(average_precision_score(yt, pp)), "n_pos": int(yt.sum())}
        else:
            vessel[v] = {"auroc": None, "n_pos": int(yt.sum())}

    print("\n=== TEST (unified raw dual-head) ===")
    print(f"  ACS: AUROC {acs['auroc']:.4f} CI{tuple(round(x,3) for x in acs['auroc_ci'])} "
          f"AUPRC {acs['auprc']:.4f} | Youden Sens {acs['sens']:.3f} Spec {acs['spec']:.3f} NPV {acs['npv']:.3f}")
    for v in VESSEL_CLASSES:
        d = vessel[v]
        if d["auroc"] is not None:
            print(f"  {v}: AUROC {d['auroc']:.3f} CI{tuple(round(x,3) for x in d['auroc_ci'])} "
                  f"AUPRC {d['auprc']:.3f} (n_pos={d['n_pos']})")
        else:
            print(f"  {v}: n_pos={d['n_pos']} insufficient")

    res = {"checkpoint": os.path.join(args.save_dir, "best_model.pt"),
           "best_combined_val": best, "acs": acs, "vessel": vessel}
    out = "outputs/acs_dual_head_raw/results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    if use_wandb:
        wandb.summary.update({"test_acs_auroc": acs["auroc"], "test_acs_auprc": acs["auprc"]})
        for v in VESSEL_CLASSES:
            if vessel[v]["auroc"] is not None:
                wandb.summary[f"test_{v}_auroc"] = vessel[v]["auroc"]
        wandb.finish()
    print(f"\nSaved -> {res['checkpoint']} and {out}", flush=True)


if __name__ == "__main__":
    main()
