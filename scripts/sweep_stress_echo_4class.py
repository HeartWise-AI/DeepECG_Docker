"""Autoresearch sweep over the 4-class WCRv2 fine-tune.

Labels (cohort_4class):
    0 = Other, 1 = HCM_no_obs, 2 = Manifest_obs, 3 = Latent+

Architectural variants explored in a single random search:
  - demo_mode      : {none, concat, mlp_concat, film}    age+sex injection
  - loss           : {ce, focal1, focal2}                gamma=1 or 2 focal
  - class_weights  : {uniform, inv_freq, sqrt_inv,
                      latent_boost_2x, latent_boost_5x}
  - head_dropout   : {0.0, 0.1, 0.2, 0.3}
  - freeze_epochs  : {0, 2}
  - lr             : log-uniform [5e-6, 1e-4]
  - weight_decay   : {0.0, 1e-5, 1e-4}
  - grad_clip      : {0.5, 1.0, 2.0}
  - demo_hidden    : {8, 16, 32}    (ignored when demo_mode == none)

Selection metric: val macro AUROC. Each trial -> its own wandb run
(group: wcrv2-4class-sweep).
"""

import argparse
import json
import os
import random
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
COHORT_DIR = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_4class"
OUT_ROOT = "/volume/DeepECG_Docker/checkpoints_stress_echo/sweep_4class"
DEMO_DIM = 3
NUM_CLASSES = 4
CLASS_NAMES = ["Other", "HCM_no_obs", "Manifest_obs", "Latent_pos"]


class WCRv2Multiclass(nn.Module):
    """WCRv2 encoder + optional demo injection + linear head."""

    def __init__(self, base, hidden_dim, demo_mode="none", demo_hidden=16,
                 head_dropout=0.1):
        super().__init__()
        self.base = base
        self.base.proj = nn.Identity()
        self.hidden_dim = hidden_dim
        self.demo_mode = demo_mode

        if demo_mode == "none":
            in_dim = hidden_dim
            self.demo_net = None
        elif demo_mode == "concat":
            in_dim = hidden_dim + DEMO_DIM
            self.demo_net = None
        elif demo_mode == "mlp_concat":
            self.demo_net = nn.Sequential(
                nn.Linear(DEMO_DIM, demo_hidden), nn.ReLU(),
                nn.Linear(demo_hidden, demo_hidden), nn.ReLU())
            in_dim = hidden_dim + demo_hidden
        elif demo_mode == "film":
            self.demo_net = nn.Sequential(
                nn.Linear(DEMO_DIM, demo_hidden), nn.ReLU(),
                nn.Linear(demo_hidden, 2 * hidden_dim))
            in_dim = hidden_dim
        else:
            raise ValueError(demo_mode)

        self.head = nn.Sequential(nn.Dropout(head_dropout),
                                  nn.Linear(in_dim, NUM_CLASSES))
        mods = list(self.head.modules())
        if self.demo_net is not None:
            mods += list(self.demo_net.modules())
        for m in mods:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, source, demo=None):
        out = self.base(source=source)
        feats = self.base.get_logits(out).float()
        if feats.dim() == 3:
            feats = feats.mean(dim=1)
        if self.demo_mode == "none":
            z = feats
        elif self.demo_mode == "concat":
            z = torch.cat([feats, demo], dim=1)
        elif self.demo_mode == "mlp_concat":
            z = torch.cat([feats, self.demo_net(demo)], dim=1)
        else:  # film
            gb = self.demo_net(demo)
            gamma, beta = gb[:, :self.hidden_dim], gb[:, self.hidden_dim:]
            z = feats * (1 + gamma) + beta
        return self.head(z)


def focal_loss(logits, target, gamma, weight=None):
    """Multi-class focal loss with optional per-class weight."""
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    target_onehot = F.one_hot(target, num_classes=NUM_CLASSES).float()
    pt = (p * target_onehot).sum(dim=-1).clamp(min=1e-8)
    logpt = (logp * target_onehot).sum(dim=-1)
    focal = -((1 - pt) ** gamma) * logpt
    if weight is not None:
        w = (weight[None, :] * target_onehot).sum(dim=-1)
        focal = focal * w
    return focal.mean()


def load_data():
    def load(split):
        x = np.load(os.path.join(COHORT_DIR, f"{split}_x.npy"))
        y = np.load(os.path.join(COHORT_DIR, f"{split}_y.npy"))
        d = np.load(os.path.join(COHORT_DIR, f"{split}_demo.npy"))
        return (torch.from_numpy(x.squeeze(-1)).permute(0, 2, 1).contiguous(),
                torch.from_numpy(y).long(),
                torch.from_numpy(d).float())
    return {s: load(s) for s in ["train", "val", "test"]}


def load_base():
    from fairseq_signals.utils import checkpoint_utils
    base, _, _ = checkpoint_utils.load_model_and_task(
        WCRV2_CKPT, arg_overrides={"no_pretrained_weights": True}, suffix="")
    return base, base.proj.in_features


def iter_batches(x, y, d, bs, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        yield x[sel], y[sel], d[sel]


@torch.no_grad()
def run_eval(model, tup, device, bs=64, uses_demo=True):
    model.eval()
    x, y, d = tup
    probs = []
    for xb, yb, db in iter_batches(x, y, d, bs, shuffle=False):
        xb = xb.to(device, non_blocking=True)
        db_in = db.to(device, non_blocking=True) if uses_demo else None
        logits = model(source=xb, demo=db_in).float()
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    p = np.concatenate(probs); t = y.numpy()
    a = {}
    for c in range(NUM_CLASSES):
        tb = (t == c).astype(int)
        a[f"auroc_c{c}"] = float(roc_auc_score(tb, p[:, c])) if tb.sum() and tb.sum() != len(tb) else float("nan")
    valid = [v for v in a.values() if not np.isnan(v)]
    a["auroc_macro"] = float(np.mean(valid)) if valid else float("nan")
    # AUPRC for the rare classes
    for c in (2, 3):
        tb = (t == c).astype(int)
        a[f"auprc_c{c}"] = float(average_precision_score(tb, p[:, c])) if tb.sum() else float("nan")
    a["class_n"] = np.bincount(t, minlength=NUM_CLASSES).tolist()
    return a


def class_weights(strategy, train_y):
    counts = np.bincount(train_y, minlength=NUM_CLASSES).astype(float)
    if strategy == "uniform":
        return np.ones(NUM_CLASSES)
    if strategy == "inv_freq":
        return counts.sum() / (NUM_CLASSES * np.clip(counts, 1, None))
    if strategy == "sqrt_inv":
        raw = counts.sum() / (NUM_CLASSES * np.clip(counts, 1, None))
        return np.sqrt(raw)
    if strategy == "latent_boost_2x":
        w = np.ones(NUM_CLASSES); w[3] = 2.0; return w
    if strategy == "latent_boost_5x":
        w = np.ones(NUM_CLASSES); w[3] = 5.0; return w
    raise ValueError(strategy)


def sample_cfg(rng):
    return {
        "lr": float(10 ** rng.uniform(np.log10(5e-6), np.log10(1e-4))),
        "head_dropout": rng.choice([0.0, 0.1, 0.2, 0.3]),
        "demo_mode": rng.choice(["none", "concat", "mlp_concat", "film"]),
        "demo_hidden": int(rng.choice([8, 16, 32])),
        "loss": rng.choice(["ce", "focal1", "focal2"]),
        "class_weights": rng.choice(["uniform", "inv_freq", "sqrt_inv",
                                      "latent_boost_2x", "latent_boost_5x"]),
        "weight_decay": float(rng.choice([0.0, 1e-5, 1e-4])),
        "grad_clip": float(rng.choice([0.5, 1.0, 2.0])),
        "freeze_epochs": int(rng.choice([0, 2])),
        "batch_size": 32,
        "patience": 5,
        "epochs": 20,
    }


def train_one(cfg, data, device, trial_dir, wandb_run=None):
    os.makedirs(trial_dir, exist_ok=True)
    base, hidden = load_base()
    model = WCRv2Multiclass(base, hidden, demo_mode=cfg["demo_mode"],
                            demo_hidden=cfg["demo_hidden"],
                            head_dropout=cfg["head_dropout"]).to(device)
    uses_demo = cfg["demo_mode"] != "none"

    if cfg["freeze_epochs"] > 0:
        for n, p in model.named_parameters():
            if n.startswith("base."):
                p.requires_grad = False

    w = torch.tensor(class_weights(cfg["class_weights"], data["train"][1].numpy()),
                     dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=w)

    def loss_fn(logits, target):
        if cfg["loss"] == "ce":
            return ce(logits, target)
        gamma = 1.0 if cfg["loss"] == "focal1" else 2.0
        return focal_loss(logits, target, gamma=gamma, weight=w)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-8,
                            weight_decay=cfg["weight_decay"])

    best, best_ep, patience = -1.0, 0, cfg["patience"]
    save_path = os.path.join(trial_dir, "best.pt")
    for ep in range(1, cfg["epochs"] + 1):
        if ep == cfg["freeze_epochs"] + 1 and cfg["freeze_epochs"] > 0:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                    betas=(0.9, 0.98), eps=1e-8,
                                    weight_decay=cfg["weight_decay"])
        t0 = time.time()
        model.train()
        losses = []
        tx, tyt, td = data["train"]
        for xb, yb, db in iter_batches(tx, tyt, td, cfg["batch_size"], shuffle=True):
            xb = xb.to(device); yb = yb.to(device)
            db_in = db.to(device) if uses_demo else None
            loss = loss_fn(model(source=xb, demo=db_in), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step(); losses.append(loss.item())

        vm = run_eval(model, data["val"], device, uses_demo=uses_demo)
        score = vm["auroc_macro"]
        row = {
            "epoch": ep, "train_loss": float(np.mean(losses)),
            "val_auroc_macro": score,
            "val_auroc_other": vm["auroc_c0"],
            "val_auroc_hcm_no_obs": vm["auroc_c1"],
            "val_auroc_manifest": vm["auroc_c2"],
            "val_auroc_latent": vm["auroc_c3"],
            "val_auprc_manifest": vm["auprc_c2"],
            "val_auprc_latent": vm["auprc_c3"],
            "time_s": time.time() - t0,
        }
        if wandb_run is not None:
            wandb_run.log(row)
        mark = ""
        if not np.isnan(score) and score > best:
            best, best_ep, patience = score, ep, cfg["patience"]
            torch.save({"model": model.state_dict(), "epoch": ep, "cfg": cfg}, save_path)
            mark = " *"
        else:
            patience -= 1
        print(f"    ep{ep:02d}: loss={row['train_loss']:.3f} val_macro={score:.4f} "
              f"oth={vm['auroc_c0']:.3f} hcm={vm['auroc_c1']:.3f} "
              f"man={vm['auroc_c2']:.3f} lat={vm['auroc_c3']:.3f}"
              f" ({row['time_s']:.0f}s){mark}")
        if patience <= 0:
            break

    state = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test_m = run_eval(model, data["test"], device, uses_demo=uses_demo)
    return {"best_val_auroc_macro": best, "best_epoch": best_ep,
            "test": test_m, "save_path": save_path,
            "uses_demo": uses_demo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:3")
    ap.add_argument("--wandb_project", type=str, default="stress-echo-latent-obstruction")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    data = load_data()
    for s, (xs, ys, ds) in data.items():
        print(f"  {s}: x={tuple(xs.shape)} demo={tuple(ds.shape)} "
              f"dist={np.bincount(ys.numpy(), minlength=NUM_CLASSES).tolist()}")
    device = torch.device(args.device)
    rng = random.Random(args.seed)

    results, best = [], {"val": -1.0, "idx": None, "cfg": None, "path": None}
    for i in range(args.n_trials):
        cfg = sample_cfg(rng)
        print(f"\n=== Trial {i+1}/{args.n_trials} ===\n  cfg: {json.dumps(cfg, default=str)}")
        wandb_run = None
        if not args.no_wandb:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project,
                                   name=f"wcrv2_4class_trial_{i:02d}",
                                   group="wcrv2-4class-sweep",
                                   config=cfg, reinit="finish_previous")
        try:
            r = train_one(cfg, data, device,
                          os.path.join(OUT_ROOT, f"trial_{i:02d}"),
                          wandb_run=wandb_run)
        finally:
            if wandb_run is not None:
                wandb_run.log({"best_val_auroc_macro": r["best_val_auroc_macro"],
                               "best_epoch": r["best_epoch"],
                               **{f"test/{k}": v for k, v in r["test"].items()
                                  if not isinstance(v, list)}})
                wandb_run.finish()
        results.append({"trial": i, "cfg": cfg,
                        **{k: v for k, v in r.items() if k != "save_path"}})
        with open(os.path.join(OUT_ROOT, "sweep_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        t = r["test"]
        print(f"  -> best val_macro={r['best_val_auroc_macro']:.4f} @ ep{r['best_epoch']}"
              f" | test macro={t['auroc_macro']:.4f} "
              f"man={t['auroc_c2']:.4f} lat={t['auroc_c3']:.4f}")
        if r["best_val_auroc_macro"] > best["val"]:
            best = {"val": r["best_val_auroc_macro"], "idx": i,
                    "cfg": cfg, "path": r["save_path"]}
            print(f"  *** NEW BEST trial {i} (val_macro={best['val']:.4f}) ***")

    print(f"\n=== Sweep done. Winner trial {best['idx']} val_macro={best['val']:.4f} ===")
    print(f"    cfg: {json.dumps(best['cfg'], default=str)}")
    winner_test = [r for r in results if r["trial"] == best["idx"]][0]["test"]
    with open(os.path.join(OUT_ROOT, "winner.json"), "w") as f:
        json.dump({"winner_trial": best["idx"], "winner_cfg": best["cfg"],
                   "winner_val": best["val"], "winner_test": winner_test},
                  f, indent=2, default=str)
    print(f"Saved winner.json + sweep_results.json -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
