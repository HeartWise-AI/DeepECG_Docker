"""3-class WCRv2 fine-tune with age + sex covariates concatenated to the pooled
encoder embedding before the classifier head.

Architecture:
    ECG (B,12,2500) -> WCRv2 encoder -> pooled features (B, 768)
    demo  (B,3) = [age_z, sex_male, sex_female]
    concat -> Linear(768+3, 3)  (head replaces the original 77-way proj)

Uses the winning hyperparams from the no-demographics sweep: lr=1.56e-5,
head_dropout=0.1, uniform class weights, no encoder freeze, AdamW.

Output: checkpoints_stress_echo/cohort_3class_demo/{best.pt, history.json}
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")

WCRV2_CKPT = "/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-amp-preserved-ft77-best.pt"
COHORT_DIR = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class"
SAVE_DIR = "/volume/DeepECG_Docker/checkpoints_stress_echo/cohort_3class_demo"


class DemoWCRv2(nn.Module):
    """Wraps the WCRv2 encoder and concatenates a covariate vector before the head."""

    def __init__(self, base, hidden_dim: int, num_classes: int, demo_dim: int = 3,
                 head_dropout: float = 0.1):
        super().__init__()
        self.base = base
        self.hidden_dim = hidden_dim
        self.demo_dim = demo_dim
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim + demo_dim, num_classes),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        # Replace the base proj with Identity so base.get_logits returns pooled features
        # (we then concat with demo and run our own head).
        self.base.proj = nn.Identity()

    def forward(self, source, demo):
        out = self.base(source=source)
        feats = self.base.get_logits(out).float()  # (B, 768) since proj=Identity
        if feats.dim() == 3:
            feats = feats.mean(dim=1)
        z = torch.cat([feats, demo], dim=1)
        return self.head(z)


def load_data():
    def load(split):
        x = np.load(os.path.join(COHORT_DIR, f"{split}_x.npy"))    # (N,2500,12,1)
        y = np.load(os.path.join(COHORT_DIR, f"{split}_y.npy"))    # (N,)
        d = np.load(os.path.join(COHORT_DIR, f"{split}_demo.npy")) # (N,3) float32
        return (torch.from_numpy(x.squeeze(-1)).permute(0, 2, 1).contiguous(),
                torch.from_numpy(y).long(),
                torch.from_numpy(d).float())
    out = {s: load(s) for s in ["train", "val", "test"]}
    for s, (xs, ys, ds) in out.items():
        print(f"  {s}: x={tuple(xs.shape)}  demo={tuple(ds.shape)}  "
              f"class dist={np.bincount(ys.numpy(), minlength=3).tolist()}")
    return out


def load_base_wcrv2():
    from fairseq_signals.utils import checkpoint_utils
    base, _, _ = checkpoint_utils.load_model_and_task(
        WCRV2_CKPT, arg_overrides={"no_pretrained_weights": True}, suffix=""
    )
    hidden_dim = base.proj.in_features
    print(f"  loaded WCRv2 ft77 (hidden={hidden_dim}, orig labels={base.proj.out_features})")
    return base, hidden_dim


def iter_batches(x, y, d, bs, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        yield x[sel], y[sel], d[sel]


@torch.no_grad()
def run_eval(model, data_tuple, device, bs=64):
    model.eval()
    x, y, d = data_tuple
    probs = []
    for xb, yb, db in iter_batches(x, y, d, bs, shuffle=False):
        xb = xb.to(device, non_blocking=True)
        db = db.to(device, non_blocking=True)
        logits = model(source=xb, demo=db).float()
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    p = np.concatenate(probs); t = y.numpy()
    aurocs = {}
    for c in range(3):
        tb = (t == c).astype(int)
        aurocs[f"auroc_c{c}"] = float(roc_auc_score(tb, p[:, c])) if tb.sum() and tb.sum() != len(tb) else float("nan")
    valid = [v for v in aurocs.values() if not np.isnan(v)]
    metrics = {**aurocs, "auroc_macro": float(np.mean(valid)) if valid else float("nan")}
    metrics["auprc_latent"] = float(average_precision_score((t == 2).astype(int), p[:, 2])) if (t == 2).sum() else float("nan")
    return metrics, p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.56e-5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--head_dropout", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda:3")
    args = p.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    data = load_data()
    device = torch.device(args.device)

    base, hidden_dim = load_base_wcrv2()
    model = DemoWCRv2(base, hidden_dim=hidden_dim, num_classes=3, demo_dim=3,
                      head_dropout=args.head_dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98),
                                  eps=1e-8, weight_decay=0.0)

    best_score = -1.0
    best_epoch = 0
    patience_left = args.patience
    history = []
    save_path = os.path.join(SAVE_DIR, "best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        tx, ty, td = data["train"]
        for xb, yb, db in iter_batches(tx, ty, td, args.batch_size, shuffle=True):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            db = db.to(device, non_blocking=True)
            logits = model(source=xb, demo=db)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        vm, _ = run_eval(model, data["val"], device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_auroc_macro": vm["auroc_macro"],
               "val_auroc_latent": vm["auroc_c2"],
               "val_auroc_hcm": vm["auroc_c1"],
               "val_auroc_other": vm["auroc_c0"],
               "val_auprc_latent": vm["auprc_latent"],
               "time_s": time.time() - t0}
        history.append(row)
        marker = ""
        if not np.isnan(vm["auroc_macro"]) and vm["auroc_macro"] > best_score:
            best_score = vm["auroc_macro"]; best_epoch = epoch
            patience_left = args.patience
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_auroc_macro": best_score, "args": vars(args)}, save_path)
            marker = " *"
        else:
            patience_left -= 1
        print(f"  ep{epoch:02d}: loss={row['train_loss']:.4f}  val_macro={vm['auroc_macro']:.4f}  "
              f"latent={vm['auroc_c2']:.4f}  hcm={vm['auroc_c1']:.4f}  other={vm['auroc_c0']:.4f}  "
              f"({row['time_s']:.1f}s){marker}")
        if patience_left <= 0:
            print(f"  early stop at epoch {epoch} (best val_macro={best_score:.4f})")
            break

    state = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    tm, _ = run_eval(model, data["test"], device)
    print(f"\n  TEST: macro={tm['auroc_macro']:.4f}  latent={tm['auroc_c2']:.4f}  "
          f"hcm={tm['auroc_c1']:.4f}  other={tm['auroc_c0']:.4f}  "
          f"auprc_latent={tm['auprc_latent']:.4f}  (best epoch={state['epoch']})")
    with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
        json.dump({"history": history, "test": tm, "best_epoch": state["epoch"],
                   "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
