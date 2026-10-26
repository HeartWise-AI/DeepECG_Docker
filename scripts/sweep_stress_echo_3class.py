"""Random-search sweep over WCRv2 fine-tune hyperparams on the 3-class cohort.

Selection metric = val macro AUROC (data-leakage-clean).
Reports test only for the winning config.

Usage:
    python scripts/sweep_stress_echo_3class.py --n_trials 10
"""

import argparse
import json
import os
import random
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
OUT_ROOT = "/volume/DeepECG_Docker/checkpoints_stress_echo/sweep_3class"


def load_cohort():
    def load(split):
        x = np.load(os.path.join(COHORT_DIR, f"{split}_x.npy"))
        y = np.load(os.path.join(COHORT_DIR, f"{split}_y.npy"))
        return torch.from_numpy(x.squeeze(-1)).permute(0, 2, 1).contiguous(), torch.from_numpy(y).long()
    return {s: load(s) for s in ["train", "val", "test"]}


def maybe_normalize(x: torch.Tensor, mode: str) -> torch.Tensor:
    # x: (N, 12, 2500)
    if mode == "none":
        return x
    if mode == "zscore_per_lead":
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - m) / s
    return x


def load_wcrv2(num_classes: int, head_dropout: float):
    from fairseq_signals.utils import checkpoint_utils
    model, _, _ = checkpoint_utils.load_model_and_task(
        WCRV2_CKPT, arg_overrides={"model_path": BASE_SSL_V1}, suffix=""
    )
    hidden = model.proj.in_features
    head_layers = []
    if head_dropout > 0:
        head_layers.append(nn.Dropout(head_dropout))
    head_layers.append(nn.Linear(hidden, num_classes))
    head = nn.Sequential(*head_layers)
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
    model.proj = head
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
        out = model(source=xb.to(device, non_blocking=True))
        logits = model.get_logits(out).float()
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        targets.append(yb.numpy())
    p = np.concatenate(probs); t = np.concatenate(targets)
    aurocs = {}
    for c in range(3):
        tb = (t == c).astype(int)
        aurocs[f"auroc_c{c}"] = float(roc_auc_score(tb, p[:, c])) if tb.sum() and tb.sum() != len(tb) else float("nan")
    valid = [v for v in aurocs.values() if not np.isnan(v)]
    metrics = {**aurocs, "auroc_macro": float(np.mean(valid)) if valid else float("nan")}
    metrics["auprc_latent"] = float(average_precision_score((t == 2).astype(int), p[:, 2])) if (t == 2).sum() else float("nan")
    return metrics


def train_one(cfg, data, device, trial_dir):
    os.makedirs(trial_dir, exist_ok=True)
    # Normalize once (free — we have the full tensors in memory already)
    tx = maybe_normalize(data["train"][0], cfg["norm"])
    ty = data["train"][1]
    vx = maybe_normalize(data["val"][0], cfg["norm"]).to(device)
    vy = data["val"][1]

    model = load_wcrv2(3, cfg["head_dropout"]).to(device)

    # Freeze encoder for first N epochs
    if cfg["freeze_epochs"] > 0:
        for n, p in model.named_parameters():
            if not n.startswith("proj."):
                p.requires_grad = False

    counts = np.bincount(ty.numpy(), minlength=3).astype(float)
    if cfg["class_weights"] == "inv_freq":
        w = counts.sum() / (3 * np.clip(counts, 1, None))
    else:
        w = np.ones(3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.98),
                                  eps=1e-8, weight_decay=cfg["weight_decay"])

    best_score = -1.0
    best_epoch = 0
    patience_left = cfg["patience"]
    history = []
    save_path = os.path.join(trial_dir, "best.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        if epoch == cfg["freeze_epochs"] + 1 and cfg["freeze_epochs"] > 0:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.98),
                                          eps=1e-8, weight_decay=cfg["weight_decay"])

        t0 = time.time()
        model.train()
        losses = []
        for xb, yb in iter_batches(tx, ty, cfg["batch_size"], shuffle=True):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(source=xb)
            logits = model.get_logits(out)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            losses.append(loss.item())

        vm = run_eval(model, vx, vy, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               **{f"val_{k}": v for k, v in vm.items()}, "time_s": time.time() - t0}
        history.append(row)
        score = vm["auroc_macro"]
        marker = ""
        if not np.isnan(score) and score > best_score:
            best_score = score; best_epoch = epoch; patience_left = cfg["patience"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": score}, save_path)
            marker = " *"
        else:
            patience_left -= 1
        print(f"    ep{epoch:02d}: loss={row['train_loss']:.3f}  val_macro={score:.4f}  "
              f"latent={vm['auroc_c2']:.4f}  hcm={vm['auroc_c1']:.4f}  other={vm['auroc_c0']:.4f}  "
              f"({row['time_s']:.1f}s){marker}")
        if patience_left <= 0:
            break

    return {"best_val_auroc_macro": best_score, "best_epoch": best_epoch,
            "history": history, "save_path": save_path}


def sample_cfg(rng: random.Random):
    return {
        "lr": 10 ** rng.uniform(np.log10(5e-6), np.log10(3e-4)),
        "head_dropout": rng.choice([0.0, 0.1, 0.2, 0.3]),
        "class_weights": rng.choice(["inv_freq", "uniform"]),
        "freeze_epochs": rng.choice([0, 2, 5]),
        "weight_decay": rng.choice([0.0, 1e-5, 1e-4]),
        "grad_clip": rng.choice([0.5, 1.0, 2.0]),
        "norm": rng.choice(["none", "zscore_per_lead"]),
        "batch_size": 32,
        "patience": 5,
        "epochs": 20,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:3")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="stress-echo-latent-obstruction")
    args = p.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    data = load_cohort()
    for s, (xs, ys) in data.items():
        print(f"  {s}: {xs.shape}  class dist={np.bincount(ys.numpy(), minlength=3).tolist()}")
    device = torch.device(args.device)
    rng = random.Random(args.seed)

    results = []
    best = {"val": -1.0, "idx": None, "cfg": None, "path": None}

    for i in range(args.n_trials):
        cfg = sample_cfg(rng)
        print(f"\n=== Trial {i+1}/{args.n_trials} ===")
        print(f"  cfg: {json.dumps(cfg, default=str)}")
        trial_dir = os.path.join(OUT_ROOT, f"trial_{i:02d}")

        wandb_run = None
        if not args.no_wandb:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, name=f"sweep3_trial_{i:02d}",
                                    config=cfg, reinit="finish_previous")
        try:
            r = train_one(cfg, data, device, trial_dir)
        finally:
            if wandb_run is not None:
                wandb_run.log({"best_val_auroc_macro": r["best_val_auroc_macro"],
                               "best_epoch": r["best_epoch"]})
                wandb_run.finish()
        results.append({"trial": i, "cfg": cfg, **{k: v for k, v in r.items() if k != "history"}})
        with open(os.path.join(OUT_ROOT, "sweep_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  -> best val_macro={r['best_val_auroc_macro']:.4f} @ ep{r['best_epoch']}")

        if r["best_val_auroc_macro"] > best["val"]:
            best = {"val": r["best_val_auroc_macro"], "idx": i, "cfg": cfg, "path": r["save_path"]}
            print(f"  *** NEW BEST trial {i} (val_macro={best['val']:.4f}) ***")

    # Evaluate winning config on test
    print(f"\n=== Sweep done. Winner: trial {best['idx']}  val_macro={best['val']:.4f} ===")
    print(f"    cfg: {json.dumps(best['cfg'], default=str)}")
    model = load_wcrv2(3, best["cfg"]["head_dropout"]).to(device)
    state = torch.load(best["path"], map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    tm = run_eval(model, *data["test"], device)
    print(f"    TEST: macro={tm['auroc_macro']:.4f}  latent={tm['auroc_c2']:.4f}  "
          f"hcm={tm['auroc_c1']:.4f}  other={tm['auroc_c0']:.4f}  auprc_latent={tm['auprc_latent']:.4f}")

    with open(os.path.join(OUT_ROOT, "winner.json"), "w") as f:
        json.dump({"winner_trial": best["idx"], "winner_cfg": best["cfg"],
                   "winner_val": best["val"], "winner_test": tm}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
