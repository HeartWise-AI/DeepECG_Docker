"""Re-evaluate the 4-class sweep winner with bootstrap 95% CIs and a
Youden operating point for Latent+ (class 3) fit on val, evaluated on test.

Inputs:
  - winner = T13 by default (highest val macro AUROC)
  - cohort_4class/{val,test}_{x,y,demo}.npy

Outputs:
  outputs/stress_echo_4class/winner_eval.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")

SWEEP_ROOT = "/volume/DeepECG_Docker/checkpoints_stress_echo/sweep_4class"
WINNER_TRIAL = 13
COHORT_DIR = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_4class"
WCRV2_CKPT = "/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-amp-preserved-ft77-best.pt"
OUT_JSON = Path("/volume/DeepECG_Docker/outputs/stress_echo_4class/winner_eval.json")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 4
CLASS_NAMES = ["Other", "HCM_no_obs", "Manifest_obs", "Latent_pos"]
DISPLAY = ["Other", "Manifest_minus_Latent_minus", "Manifest_pos", "Latent_pos"]


class WCRv2Multiclass(nn.Module):
    def __init__(self, base, hidden_dim, demo_mode="none", demo_hidden=16, head_dropout=0.1):
        super().__init__()
        self.base = base
        self.base.proj = nn.Identity()
        self.hidden_dim = hidden_dim
        self.demo_mode = demo_mode
        if demo_mode == "none":
            in_dim = hidden_dim; self.demo_net = None
        elif demo_mode == "concat":
            in_dim = hidden_dim + 3; self.demo_net = None
        elif demo_mode == "mlp_concat":
            self.demo_net = nn.Sequential(nn.Linear(3, demo_hidden), nn.ReLU(),
                                          nn.Linear(demo_hidden, demo_hidden), nn.ReLU())
            in_dim = hidden_dim + demo_hidden
        elif demo_mode == "film":
            self.demo_net = nn.Sequential(nn.Linear(3, demo_hidden), nn.ReLU(),
                                          nn.Linear(demo_hidden, 2 * hidden_dim))
            in_dim = hidden_dim
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(in_dim, NUM_CLASSES))

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
        else:
            gb = self.demo_net(demo)
            z = feats * (1 + gb[:, :self.hidden_dim]) + gb[:, self.hidden_dim:]
        return self.head(z)


def load_split(split):
    x = np.load(os.path.join(COHORT_DIR, f"{split}_x.npy"))
    y = np.load(os.path.join(COHORT_DIR, f"{split}_y.npy"))
    d = np.load(os.path.join(COHORT_DIR, f"{split}_demo.npy"))
    return (torch.from_numpy(x.squeeze(-1)).permute(0, 2, 1).contiguous(),
            torch.from_numpy(y).long(),
            torch.from_numpy(d).float())


@torch.no_grad()
def predict(model, tup, device, uses_demo, bs=64):
    model.eval()
    x, y, d = tup
    probs = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(device)
        db = d[i:i + bs].to(device) if uses_demo else None
        logits = model(source=xb, demo=db).float()
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs), y.numpy()


def bootstrap_ci(score_fn, y, p, n_iter=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    vals = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        try:
            vals.append(score_fn(y[idx], p[idx]))
        except Exception:
            continue
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    # Load winner trial cfg + ckpt
    sweep = json.load(open(os.path.join(SWEEP_ROOT, "sweep_results.json")))
    winner = [t for t in sweep if t["trial"] == WINNER_TRIAL][0]
    cfg = winner["cfg"]
    print(f"Winner T{WINNER_TRIAL}: val={winner['best_val_auroc_macro']:.4f}, cfg={cfg}")

    from fairseq_signals.utils import checkpoint_utils
    base, _, _ = checkpoint_utils.load_model_and_task(
        WCRV2_CKPT, arg_overrides={"no_pretrained_weights": True}, suffix="")
    hidden = base.proj.in_features

    device = torch.device("cuda:3")
    model = WCRv2Multiclass(base, hidden, demo_mode=cfg["demo_mode"],
                            demo_hidden=cfg["demo_hidden"],
                            head_dropout=cfg["head_dropout"]).to(device)
    state = torch.load(os.path.join(SWEEP_ROOT, f"trial_{WINNER_TRIAL:02d}", "best.pt"),
                       map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    uses_demo = cfg["demo_mode"] != "none"
    print(f"  best epoch: {state['epoch']}")

    val_p, val_y = predict(model, load_split("val"), device, uses_demo)
    test_p, test_y = predict(model, load_split("test"), device, uses_demo)
    print(f"  val n={len(val_y)} class_n={np.bincount(val_y, minlength=4).tolist()}")
    print(f"  test n={len(test_y)} class_n={np.bincount(test_y, minlength=4).tolist()}")

    # Per-class metrics on test + 95% CIs
    per_class = {}
    for c, key in zip(range(NUM_CLASSES), DISPLAY):
        tb = (test_y == c).astype(int)
        if tb.sum() == 0:
            continue
        auroc = float(roc_auc_score(tb, test_p[:, c]))
        auprc = float(average_precision_score(tb, test_p[:, c]))
        prev = float(tb.mean())
        ci_auroc = bootstrap_ci(
            lambda y, p: roc_auc_score(y, p) if y.sum() and y.sum() != len(y) else float("nan"),
            tb, test_p[:, c])
        ci_auprc = bootstrap_ci(
            lambda y, p: average_precision_score(y, p) if y.sum() else float("nan"),
            tb, test_p[:, c])
        per_class[key] = {
            "n_pos": int(tb.sum()), "prevalence": prev,
            "auroc": auroc, "auroc_ci95": [ci_auroc[0], ci_auroc[1]],
            "auprc": auprc, "auprc_ci95": [ci_auprc[0], ci_auprc[1]],
        }
        print(f"  {key:32s} prev={prev*100:.2f}%  "
              f"AUROC {auroc:.3f} [{ci_auroc[0]:.3f}-{ci_auroc[1]:.3f}]  "
              f"AUPRC {auprc:.3f} [{ci_auprc[0]:.3f}-{ci_auprc[1]:.3f}]")

    # Macro AUROC + bootstrap
    def macro_fn(y, p):
        vals = []
        for c in range(NUM_CLASSES):
            tb = (y == c).astype(int)
            if tb.sum() == 0 or tb.sum() == len(tb):
                continue
            vals.append(roc_auc_score(tb, p[:, c]))
        return float(np.mean(vals)) if vals else float("nan")
    macro = macro_fn(test_y, test_p)
    macro_ci = bootstrap_ci(macro_fn, test_y, test_p)
    print(f"  Macro AUROC {macro:.3f} [{macro_ci[0]:.3f}-{macro_ci[1]:.3f}]")

    # Operating point: Youden for Latent+ fit on val
    val_lat = (val_y == 3).astype(int)
    fpr, tpr, thr = roc_curve(val_lat, val_p[:, 3])
    j = tpr - fpr
    t_lat = float(thr[int(np.argmax(j))])
    pred = (test_p[:, 3] >= t_lat).astype(int)
    test_lat = (test_y == 3).astype(int)
    tp = int(((test_lat == 1) & (pred == 1)).sum())
    fp = int(((test_lat == 0) & (pred == 1)).sum())
    fn = int(((test_lat == 1) & (pred == 0)).sum())
    tn = int(((test_lat == 0) & (pred == 0)).sum())
    op = {
        "threshold": t_lat,
        "sens": tp / max(tp + fn, 1),
        "spec": tn / max(tn + fp, 1),
        "ppv":  tp / max(tp + fp, 1),
        "npv":  tn / max(tn + fn, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    print(f"\nLatent+ Youden t={t_lat:.5f}  Sens {op['sens']*100:.1f}%  "
          f"Spec {op['spec']*100:.1f}%  PPV {op['ppv']*100:.1f}%  NPV {op['npv']*100:.1f}%")

    # Rest-obstructed-excluded sensitivity check on this new model too
    cohort = __import__("pandas").read_parquet(os.path.join(COHORT_DIR, "cohort.parquet"))
    test_cohort = cohort[cohort["split"] == "test"].reset_index(drop=True)
    rest_obs = (test_cohort["rest_grad"] >= 30).fillna(False).values
    keep = ~rest_obs
    sens_metrics = {}
    for c, key in zip(range(NUM_CLASSES), DISPLAY):
        tb = ((test_y == c) & keep).astype(int)
        if tb.sum() == 0:
            continue
        sub_y = test_y[keep]; sub_p = test_p[keep]
        tb_k = (sub_y == c).astype(int)
        if tb_k.sum() == 0 or tb_k.sum() == len(tb_k):
            continue
        sens_metrics[key] = {
            "auroc": float(roc_auc_score(tb_k, sub_p[:, c])),
            "n_pos": int(tb_k.sum()),
        }
    print("\nRest-obstructed-excluded test (n=%d): %s" %
          (int(keep.sum()),
           {k: round(v["auroc"], 3) for k, v in sens_metrics.items()}))

    out = {
        "winner_trial": WINNER_TRIAL,
        "winner_cfg": cfg,
        "winner_val_macro_auroc": winner["best_val_auroc_macro"],
        "winner_best_epoch": state["epoch"],
        "test": {
            "n": int(len(test_y)),
            "class_n": np.bincount(test_y, minlength=4).tolist(),
            "per_class": per_class,
            "macro_auroc": macro,
            "macro_auroc_ci95": list(macro_ci),
            "latent_operating_point_youden": op,
        },
        "sensitivity_rest_obstructed_excluded": {
            "n": int(keep.sum()),
            "per_class": sens_metrics,
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
