"""Optimize per-class thresholds for DeepECG-HCM 3-class model.

Fits thresholds on the **validation** split (no test leakage), then evaluates
sensitivity, specificity, PPV, NPV, F1, Youden J on the **test** split.

Strategies:
  1. Youden index   : argmax of (sensitivity + specificity - 1) on val.
  2. Cost-aware     : minimize expected misclassification cost on val with a
                      per-class FN:FP cost ratio (default 5:1 for Latent+,
                      3:1 for HCM_no_latent, 1:1 for Other).

Output:
  - JSON: /volume/DeepECG_Docker/outputs/stress_echo_fig/thresholds.json
  - CSV : /volume/DeepECG_Docker/outputs/stress_echo_fig/threshold_metrics.csv
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
from sklearn.metrics import roc_curve

PKG_ROOT = "/media/data1/models/DeepECG-HCM"
CFG = os.path.join(PKG_ROOT, "config", "inference.yaml")
COHORT_PARQUET = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class/cohort.parquet"
VAL_X = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class/val_x.npy"
VAL_Y = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class/val_y.npy"
TEST_X = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class/test_x.npy"
TEST_Y = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class/test_y.npy"

OUT = Path("/volume/DeepECG_Docker/outputs/stress_echo_fig")
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Other", "HCM_no_latent", "Latent_pos"]
# Default cost ratios: FN cost / FP cost (per class). Latent obstruction missed
# is the most clinically costly because it's a treatable, prognostically
# important diagnosis that requires escalation to a provocative echo / CMR.
COST_RATIO = {"Other": 1.0, "HCM_no_latent": 3.0, "Latent_pos": 5.0}


def get_probs(npy_x: str, npy_y: str, device: torch.device, cfg: dict):
    sys.path.insert(0, cfg["inference"].get("fairseq_signals_path",
                                             "/volume/DeepECG_Docker/fairseq-signals"))
    from fairseq_signals.utils import checkpoint_utils

    m = cfg["model"]
    base, _, _ = checkpoint_utils.load_model_and_task(
        os.path.join(PKG_ROOT, m["base_checkpoint"]) if not os.path.isabs(m["base_checkpoint"]) else m["base_checkpoint"],
        arg_overrides={"no_pretrained_weights": True},
        suffix="",
    )
    base.proj = nn.Sequential(nn.Dropout(0.1), nn.Linear(m["hidden_dim"], m["num_classes"]))
    state = torch.load(os.path.join(PKG_ROOT, m["finetuned_checkpoint"]) if not os.path.isabs(m["finetuned_checkpoint"]) else m["finetuned_checkpoint"],
                       map_location=device, weights_only=False)
    base.load_state_dict(state["model"])
    base = base.to(device).eval()

    x = np.load(npy_x)
    y = np.load(npy_y)
    x = torch.from_numpy(x.squeeze(-1)).permute(0, 2, 1).contiguous()

    probs = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            xb = x[i:i + 64].to(device, non_blocking=True)
            out = base(source=xb)
            logits = base.get_logits(out).float()
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0), y


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thresholds[k]), float(j[k]), float(tpr[k]), float(1 - fpr[k])


def cost_threshold(y_true: np.ndarray, y_score: np.ndarray, cost_ratio: float):
    """Minimize expected cost = cost_fn * n_FN + 1 * n_FP across a grid."""
    grid = np.unique(np.concatenate([[0.0, 1.0], np.linspace(0, 1, 1001)]))
    best = (None, np.inf, None, None, None)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    for t in grid:
        pred = y_score >= t
        fn = int(((y_true == 1) & ~pred).sum())
        fp = int(((y_true == 0) & pred).sum())
        cost = cost_ratio * fn + fp
        if cost < best[1]:
            tp = int(((y_true == 1) & pred).sum())
            tn = int(((y_true == 0) & ~pred).sum())
            sens = tp / max(n_pos, 1)
            spec = tn / max(n_neg, 1)
            best = (float(t), cost, sens, spec, (tp, fp, tn, fn))
    return best


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, t: float):
    pred = y_score >= t
    tp = int(((y_true == 1) & pred).sum())
    fp = int(((y_true == 0) & pred).sum())
    tn = int(((y_true == 0) & ~pred).sum())
    fn = int(((y_true == 1) & ~pred).sum())
    n_pos = tp + fn
    n_neg = fp + tn
    sens = tp / max(n_pos, 1)
    spec = tn / max(n_neg, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
    j = sens + spec - 1
    return {"threshold": float(t), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "f1": f1, "youden_j": j}


def main():
    cfg = yaml.safe_load(open(CFG))
    device = torch.device("cuda:3")

    print("Computing val probs ...")
    val_probs, val_y = get_probs(VAL_X, VAL_Y, device, cfg)
    print(f"  val: n={len(val_y)}, class dist={np.bincount(val_y, minlength=3).tolist()}")

    print("Computing test probs ...")
    test_probs, test_y = get_probs(TEST_X, TEST_Y, device, cfg)
    print(f"  test: n={len(test_y)}, class dist={np.bincount(test_y, minlength=3).tolist()}")

    results = {}
    rows = []
    for c, name in enumerate(CLASSES):
        val_bin = (val_y == c).astype(int)
        val_sc = val_probs[:, c]
        test_bin = (test_y == c).astype(int)
        test_sc = test_probs[:, c]

        # Youden threshold (fit on val)
        t_y, j_val, sens_val, spec_val = youden_threshold(val_bin, val_sc)
        # Cost threshold (fit on val with class-specific FN:FP cost)
        cr = COST_RATIO[name]
        t_c, cost_val, sens_c_val, spec_c_val, _ = cost_threshold(val_bin, val_sc, cr)

        # Apply both to test
        m_y_val = confusion_at_threshold(val_bin, val_sc, t_y)
        m_y_test = confusion_at_threshold(test_bin, test_sc, t_y)
        m_c_val = confusion_at_threshold(val_bin, val_sc, t_c)
        m_c_test = confusion_at_threshold(test_bin, test_sc, t_c)
        m_argmax = confusion_at_threshold(test_bin, test_sc, 0.5)

        results[name] = {
            "youden": {"val_fit": m_y_val, "test_eval": m_y_test},
            "cost_aware": {"cost_ratio_fn_to_fp": cr,
                           "val_fit": m_c_val, "test_eval": m_c_test},
            "argmax_0.5": {"test_eval": m_argmax},
        }
        for strat, payload in [
            ("youden", m_y_test), ("cost", m_c_test), ("argmax", m_argmax)
        ]:
            threshold = m_y_test["threshold"] if strat == "youden" else (
                m_c_test["threshold"] if strat == "cost" else 0.5)
            rows.append({
                "class": name, "strategy": strat, "threshold": round(threshold, 4),
                "sens": round(payload["sens"], 3),
                "spec": round(payload["spec"], 3),
                "ppv": round(payload["ppv"], 3),
                "npv": round(payload["npv"], 3),
                "f1": round(payload["f1"], 3),
                "youden_j": round(payload["youden_j"], 3),
                "tp": payload["tp"], "fp": payload["fp"],
                "fn": payload["fn"], "tn": payload["tn"],
            })

    df = pd.DataFrame(rows)
    print("\n=== Per-class operating points (test) ===")
    print(df.to_string(index=False))

    with open(OUT / "thresholds.json", "w") as f:
        json.dump(results, f, indent=2)
    df.to_csv(OUT / "threshold_metrics.csv", index=False)
    print(f"\nSaved: {OUT/'thresholds.json'}")
    print(f"Saved: {OUT/'threshold_metrics.csv'}")


if __name__ == "__main__":
    main()
