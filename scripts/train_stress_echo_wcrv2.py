"""Fine-tune WCRv2 (amp-preserved, MHI scaling) on stress-echo latent-obstruction cohorts.

Loads the 77-class ft checkpoint, replaces proj head with a binary head, and trains
on the stacked NPY manifests produced by build_stress_echo_manifests.py.

No PSA / no re-normalization — WCRv2 expects raw mV signals as stored in the MHI NPYs.

Usage:
    python scripts/train_stress_echo_wcrv2.py --cohort cohort_latent_all
    python scripts/train_stress_echo_wcrv2.py --cohort cohort_latent_vs_hcm
    python scripts/train_stress_echo_wcrv2.py --cohort both
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")

WCRV2_CKPT = "/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-amp-preserved-ft77-best.pt"
BASE_SSL_V1 = "/volume/DeepECG_Docker/weights/wcr_77_classes/base_ssl.pt"
FT_DATA_ROOT = "/volume/DeepECG_Docker/finetune_data/stress_echo"
CKPT_ROOT = "/volume/DeepECG_Docker/checkpoints_stress_echo"


def load_cohort(cohort_dir: str):
    def load(split):
        x = np.load(os.path.join(cohort_dir, f"{split}_x.npy"))  # (N, 2500, 12, 1)
        y = np.load(os.path.join(cohort_dir, f"{split}_y.npy"))  # (N, 1)
        return torch.from_numpy(x.squeeze(-1)).permute(0, 2, 1).contiguous(), torch.from_numpy(y).float()
    train_x, train_y = load("train")
    val_x, val_y = load("val")
    test_x, test_y = load("test")
    print(f"  train: {train_x.shape} pos={int(train_y.sum())}/{len(train_y)}")
    print(f"  val:   {val_x.shape} pos={int(val_y.sum())}/{len(val_y)}")
    print(f"  test:  {test_x.shape} pos={int(test_y.sum())}/{len(test_y)}")
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def load_wcrv2_binary(ckpt_path: str, base_ssl_override: str):
    from fairseq_signals.utils import checkpoint_utils
    model, cfg, task = checkpoint_utils.load_model_and_task(
        ckpt_path,
        arg_overrides={"model_path": base_ssl_override},
        suffix=""
    )
    hidden_dim = model.proj.in_features
    print(f"  loaded WCRv2 ft77 checkpoint (hidden={hidden_dim}, orig labels={model.proj.out_features})")
    model.proj = nn.Linear(hidden_dim, 1)
    nn.init.xavier_uniform_(model.proj.weight)
    nn.init.constant_(model.proj.bias, 0.0)
    return model


def iter_batches(x, y, batch_size, shuffle=True):
    n = len(x)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, batch_size):
        sel = idx[i : i + batch_size]
        yield x[sel], y[sel]


def run_eval(model, x, y, device, batch_size=64):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size, shuffle=False):
            xb = xb.to(device, non_blocking=True)
            out = model(source=xb)
            logits = model.get_logits(out).float().squeeze(-1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(yb.squeeze(-1).numpy())
    p = np.concatenate(probs)
    t = np.concatenate(targets)
    if t.sum() == 0 or t.sum() == len(t):
        return {"auroc": float("nan"), "auprc": float("nan"), "prevalence": float(t.mean())}
    return {
        "auroc": float(roc_auc_score(t, p)),
        "auprc": float(average_precision_score(t, p)),
        "prevalence": float(t.mean()),
    }


def train_cohort(cohort: str, args, wandb_run=None):
    cohort_dir = os.path.join(FT_DATA_ROOT, cohort)
    save_dir = os.path.join(CKPT_ROOT, cohort)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Cohort: {cohort} ===")
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_cohort(cohort_dir)

    device = torch.device(args.device)
    print(f"  device: {device}")

    model = load_wcrv2_binary(WCRV2_CKPT, BASE_SSL_V1).to(device)

    # Class weight for BCE to handle imbalance
    n_pos = int(train_y.sum())
    n_neg = len(train_y) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"  pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8)

    best_auroc = -1.0
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(source=xb)
            logits = model.get_logits(out).squeeze(-1)
            loss = criterion(logits, yb.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        val_metrics = run_eval(model, val_x, val_y, device)
        train_loss = float(np.mean(losses))
        epoch_time = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_prevalence": val_metrics["prevalence"],
            "time_s": epoch_time,
        }
        history.append(row)
        print(f"  ep{epoch:02d}: train_loss={train_loss:.4f}  "
              f"val_auroc={val_metrics['auroc']:.4f}  val_auprc={val_metrics['auprc']:.4f}  "
              f"({epoch_time:.1f}s)")

        if wandb_run is not None:
            wandb_run.log(row)

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            patience_left = args.patience
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_auroc": best_auroc},
                os.path.join(save_dir, "best.pt"),
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at epoch {epoch} (best val_auroc={best_auroc:.4f})")
                break

    # Reload best and evaluate on test
    state = torch.load(os.path.join(save_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test_metrics = run_eval(model, test_x, test_y, device)
    print(f"  TEST: auroc={test_metrics['auroc']:.4f}  auprc={test_metrics['auprc']:.4f}  "
          f"prev={test_metrics['prevalence']:.4f}  (best epoch={state['epoch']})")

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump({"history": history, "test": test_metrics, "best_epoch": state["epoch"]}, f, indent=2)

    if wandb_run is not None:
        wandb_run.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb_run.log({"test/best_epoch": state["epoch"]})

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["cohort_latent_all", "cohort_latent_vs_hcm", "both"],
                        default="both")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="stress-echo-latent-obstruction")
    args = parser.parse_args()

    cohorts = ["cohort_latent_all", "cohort_latent_vs_hcm"] if args.cohort == "both" else [args.cohort]
    for c in cohorts:
        wandb_run = None
        if not args.no_wandb:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=c,
                reinit="finish_previous",
                config={
                    "cohort": c,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "patience": args.patience,
                    "base_checkpoint": WCRV2_CKPT,
                    "scaling": "mv_direct_no_psa",
                },
            )
        try:
            train_cohort(c, args, wandb_run=wandb_run)
        finally:
            if wandb_run is not None:
                wandb_run.finish()


if __name__ == "__main__":
    main()
