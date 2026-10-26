"""Rule-IN threshold optimization for DeepECG-HCM.

Goal: find an operating point where a POSITIVE model prediction is actionable
(high PPV) so the clinic can prioritize who to send for provocative echo / CMR.

Strategies (all FIT on val, EVALUATED on test):
  - spec@95 / spec@99 / spec@995 : highest sensitivity at fixed minimum specificity
  - ppv_max                      : threshold that maximizes PPV on val
  - f_half                       : F0.5 (precision weighted 2x recall)
  - youden                       : kept for reference (rule-out point)
  - cost_5to1                    : kept for reference (mild FN bias)

Outputs:
  - JSON : outputs/stress_echo_fig/rulein_thresholds.json
  - CSV  : outputs/stress_echo_fig/rulein_metrics.csv
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_curve, fbeta_score

PKG_ROOT = "/media/data1/models/DeepECG-HCM"
CFG = os.path.join(PKG_ROOT, "config", "inference.yaml")
VAL_X = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class/val_x.npy"
VAL_Y = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class/val_y.npy"
TEST_X = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class/test_x.npy"
TEST_Y = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class/test_y.npy"

OUT = Path("/volume/DeepECG_Docker/outputs/stress_echo_fig")
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Other", "HCM_no_latent", "Latent_pos"]


def get_probs(npy_x: str, npy_y: str, device: torch.device, cfg: dict):
    sys.path.insert(0, cfg["inference"].get("fairseq_signals_path",
                                             "/volume/DeepECG_Docker/fairseq-signals"))
    from fairseq_signals.utils import checkpoint_utils

    m = cfg["model"]
    base_ckpt = m["base_checkpoint"]
    base_ckpt = base_ckpt if os.path.isabs(base_ckpt) else os.path.join(PKG_ROOT, base_ckpt)
    ft_ckpt = m["finetuned_checkpoint"]
    ft_ckpt = ft_ckpt if os.path.isabs(ft_ckpt) else os.path.join(PKG_ROOT, ft_ckpt)

    base, _, _ = checkpoint_utils.load_model_and_task(
        base_ckpt, arg_overrides={"no_pretrained_weights": True}, suffix="")
    base.proj = nn.Sequential(nn.Dropout(0.1), nn.Linear(m["hidden_dim"], m["num_classes"]))
    state = torch.load(ft_ckpt, map_location=device, weights_only=False)
    base.load_state_dict(state["model"])
    base = base.to(device).eval()

    x = np.load(npy_x); y = np.load(npy_y)
    x = torch.from_numpy(x.squeeze(-1)).permute(0, 2, 1).contiguous()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            xb = x[i:i + 64].to(device, non_blocking=True)
            out = base(source=xb)
            logits = base.get_logits(out).float()
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0), y


def metrics_at(y_true, y_score, t):
    pred = y_score >= t
    tp = int(((y_true == 1) & pred).sum()); fp = int(((y_true == 0) & pred).sum())
    tn = int(((y_true == 0) & ~pred).sum()); fn = int(((y_true == 1) & ~pred).sum())
    n_pos = tp + fn; n_neg = fp + tn
    return {
        "threshold": float(t),
        "sens": tp / max(n_pos, 1),
        "spec": tn / max(n_neg, 1),
        "ppv":  tp / max(tp + fp, 1),
        "npv":  tn / max(tn + fn, 1),
        "f1":   (2 * tp) / max(2 * tp + fp + fn, 1),
        "f0_5": (1.25 * tp) / max(1.25 * tp + 0.25 * fn + fp, 1),
        "youden_j": tp / max(n_pos, 1) + tn / max(n_neg, 1) - 1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_flagged_positive": tp + fp,
    }


def fit_spec_at(y_true, y_score, target_spec):
    """Highest sensitivity threshold s.t. specificity >= target."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    spec = 1 - fpr
    mask = spec >= target_spec
    if not mask.any():
        return float(thresholds[-1])
    # pick the threshold that has highest sens among those meeting spec target
    idx = np.where(mask)[0]
    best = idx[np.argmax(tpr[idx])]
    return float(thresholds[best])


def fit_ppv_max(y_true, y_score, min_recall=0.05):
    """Threshold that maximizes PPV on val, subject to a minimum recall floor
    to avoid degenerate `1 TP, 0 FP` solutions."""
    grid = np.unique(np.concatenate([[0.0, 1.0], np.linspace(0, 1, 2001)]))
    best_t, best_ppv = 0.5, -1
    for t in grid:
        m = metrics_at(y_true, y_score, t)
        if m["sens"] < min_recall:
            continue
        if m["ppv"] > best_ppv:
            best_ppv, best_t = m["ppv"], t
    return float(best_t)


def fit_fbeta(y_true, y_score, beta=0.5):
    grid = np.unique(np.concatenate([[0.0, 1.0], np.linspace(0, 1, 2001)]))
    best_t, best = 0.5, -1
    for t in grid:
        pred = (y_score >= t).astype(int)
        try:
            f = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        except Exception:
            continue
        if f > best:
            best, best_t = f, t
    return float(best_t)


def main():
    cfg = yaml.safe_load(open(CFG))
    device = torch.device("cuda:3")

    val_probs, val_y = get_probs(VAL_X, VAL_Y, device, cfg)
    test_probs, test_y = get_probs(TEST_X, TEST_Y, device, cfg)
    print(f"val n={len(val_y)} dist={np.bincount(val_y, minlength=3).tolist()}")
    print(f"test n={len(test_y)} dist={np.bincount(test_y, minlength=3).tolist()}")

    results = {}; rows = []
    for c, name in enumerate(CLASSES):
        val_bin = (val_y == c).astype(int)
        test_bin = (test_y == c).astype(int)
        val_sc = val_probs[:, c]; test_sc = test_probs[:, c]

        strategies = {
            "youden": (lambda: youden_fit(val_bin, val_sc)),
            "spec@0.95": (lambda: fit_spec_at(val_bin, val_sc, 0.95)),
            "spec@0.99": (lambda: fit_spec_at(val_bin, val_sc, 0.99)),
            "spec@0.995": (lambda: fit_spec_at(val_bin, val_sc, 0.995)),
            "ppv_max":   (lambda: fit_ppv_max(val_bin, val_sc, min_recall=0.05)),
            "f0.5":      (lambda: fit_fbeta(val_bin, val_sc, beta=0.5)),
        }
        results[name] = {}
        for sname, fn in strategies.items():
            t = fn()
            test_m = metrics_at(test_bin, test_sc, t)
            val_m = metrics_at(val_bin, val_sc, t)
            results[name][sname] = {"threshold": t, "val": val_m, "test": test_m}
            rows.append({
                "class": name, "strategy": sname,
                "threshold": round(t, 5),
                "test_sens": round(test_m["sens"], 3),
                "test_spec": round(test_m["spec"], 3),
                "test_ppv":  round(test_m["ppv"],  3),
                "test_npv":  round(test_m["npv"],  3),
                "test_f1":   round(test_m["f1"],   3),
                "test_f0.5": round(test_m["f0_5"], 3),
                "test_flagged_pos": test_m["n_flagged_positive"],
                "test_tp": test_m["tp"], "test_fp": test_m["fp"],
                "test_fn": test_m["fn"], "test_tn": test_m["tn"],
            })

    df = pd.DataFrame(rows)
    print("\n=== Rule-IN operating points (fit on val, eval on test) ===")
    print(df.to_string(index=False))

    with open(OUT / "rulein_thresholds.json", "w") as f:
        json.dump(results, f, indent=2)
    df.to_csv(OUT / "rulein_metrics.csv", index=False)
    print(f"\nSaved: {OUT/'rulein_thresholds.json'}")
    print(f"Saved: {OUT/'rulein_metrics.csv'}")


def youden_fit(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


if __name__ == "__main__":
    main()
