"""3-class WCRv2 fine-tune on stress-echo cohort.

Labels: 0=other, 1=HCM no latent, 2=latent+
Metrics: per-class OvR AUROC + macro AUROC + latent+ AUPRC (class-2 binary).

Usage:
    python scripts/train_stress_echo_wcrv2_3class.py
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
BASE_SSL_V1 = "/volume/DeepECG_Docker/weights/wcr_77_classes/base_ssl.pt"
COHORT_DIR = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class"
SAVE_DIR = "/volume/DeepECG_Docker/checkpoints_stress_echo/cohort_3class"


def load_cohort():
    def load(split):
        x = np.load(os.path.join(COHORT_DIR, f"{split}_x.npy"))
        y = np.load(os.path.join(COHORT_DIR, f"{split}_y.npy"))
        return torch.from_numpy(x.squeeze(-1)).permute(0, 2, 1).contiguous(), torch.from_numpy(y).long()
    out = {s: load(s) for s in ["train", "val", "test"]}
    for s, (xs, ys) in out.items():
        print(f"  {s}: {xs.shape}  class dist={np.bincount(ys.numpy(), minlength=3).tolist()}")
    return out


def load_wcrv2_multiclass(num_classes: int):
    from fairseq_signals.utils import checkpoint_utils
    model, _, _ = checkpoint_utils.load_model_and_task(
        WCRV2_CKPT, arg_overrides={"model_path": BASE_SSL_V1}, suffix=""
    )
    hidden_dim = model.proj.in_features
    print(f"  loaded WCRv2 ft77 (hidden={hidden_dim}, orig labels={model.proj.out_features}) -> {num_classes}")
    model.proj = nn.Linear(hidden_dim, num_classes)
    nn.init.xavier_uniform_(model.proj.weight)
    nn.init.constant_(model.proj.bias, 0.0)
    return model


def iter_batches(x, y, bs, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        sel = idx[i : i + bs]
        yield x[sel], y[sel]


@torch.no_grad()
def run_eval(model, x, y, device, bs=64):
    model.eval()
    probs, targets = [], []
    for xb, yb in iter_batches(x, y, bs, shuffle=False):
        xb = xb.to(device, non_blocking=True)
        out = model(source=xb)
        logits = model.get_logits(out).float()
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        targets.append(yb.numpy())
    p = np.concatenate(probs)
    t = np.concatenate(targets)
    classes_present = np.unique(t)
    metrics = {"prevalence": np.bincount(t, minlength=3).tolist()}
    aurocs = {}
    for c in range(3):
        tbin = (t == c).astype(int)
        if tbin.sum() == 0 or tbin.sum() == len(tbin):
            aurocs[f"auroc_c{c}"] = float("nan")
        else:
            aurocs[f"auroc_c{c}"] = float(roc_auc_score(tbin, p[:, c]))
    metrics.update(aurocs)
    valid = [v for v in aurocs.values() if not np.isnan(v)]
    metrics["auroc_macro"] = float(np.mean(valid)) if valid else float("nan")
    tpos = (t == 2).astype(int)
    metrics["auprc_latent"] = float(average_precision_score(tpos, p[:, 2])) if tpos.sum() and tpos.sum() != len(tpos) else float("nan")
    if len(classes_present) == 3:
        try:
            metrics["auroc_ovo_macro"] = float(roc_auc_score(t, p, multi_class="ovo", average="macro"))
        except Exception:
            metrics["auroc_ovo_macro"] = float("nan")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:3")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="stress-echo-latent-obstruction")
    args = p.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    data = load_cohort()
    device = torch.device(args.device)
    print(f"  device: {device}")

    model = load_wcrv2_multiclass(3).to(device)

    # Class weights inverse to frequency (on train)
    train_y = data["train"][1].numpy()
    counts = np.bincount(train_y, minlength=3).astype(float)
    weights = counts.sum() / (3 * np.clip(counts, 1, None))
    print(f"  train class counts: {counts.astype(int).tolist()}  weights: {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8)

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name="cohort_3class",
            reinit="finish_previous",
            config={
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "patience": args.patience, "base_checkpoint": WCRV2_CKPT,
                "scaling": "mv_direct_no_psa",
                "class_weights": weights.tolist(),
                "num_classes": 3,
            },
        )

    best_score = -1.0
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        tx, ty = data["train"]
        for xb, yb in iter_batches(tx, ty, args.batch_size, shuffle=True):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(source=xb)
            logits = model.get_logits(out)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        vm = run_eval(model, *data["val"], device)
        train_loss = float(np.mean(losses))
        row = {
            "epoch": epoch, "train_loss": train_loss,
            "val_auroc_macro": vm["auroc_macro"],
            "val_auroc_latent": vm["auroc_c2"],
            "val_auroc_hcm_nl": vm["auroc_c1"],
            "val_auroc_other": vm["auroc_c0"],
            "val_auprc_latent": vm["auprc_latent"],
            "time_s": time.time() - t0,
        }
        history.append(row)
        print(f"  ep{epoch:02d}: loss={train_loss:.4f}  "
              f"val_auroc_macro={vm['auroc_macro']:.4f}  latent={vm['auroc_c2']:.4f}  "
              f"hcm_nl={vm['auroc_c1']:.4f}  other={vm['auroc_c0']:.4f}  "
              f"({row['time_s']:.1f}s)")
        if wandb_run is not None:
            wandb_run.log(row)

        score = vm["auroc_macro"]
        if not np.isnan(score) and score > best_score:
            best_score = score
            patience_left = args.patience
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auroc_macro": best_score},
                       os.path.join(SAVE_DIR, "best.pt"))
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at epoch {epoch} (best val_auroc_macro={best_score:.4f})")
                break

    state = torch.load(os.path.join(SAVE_DIR, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    tm = run_eval(model, *data["test"], device)
    print(f"\n  TEST: macro={tm['auroc_macro']:.4f}  latent={tm['auroc_c2']:.4f}  "
          f"hcm_nl={tm['auroc_c1']:.4f}  other={tm['auroc_c0']:.4f}  "
          f"auprc_latent={tm['auprc_latent']:.4f}  (best epoch={state['epoch']})")

    with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
        json.dump({"history": history, "test": tm, "best_epoch": state["epoch"]}, f, indent=2)
    if wandb_run is not None:
        wandb_run.log({f"test/{k}": v for k, v in tm.items() if not isinstance(v, list)})
        wandb_run.log({"test/best_epoch": state["epoch"]})
        wandb_run.finish()


if __name__ == "__main__":
    main()
