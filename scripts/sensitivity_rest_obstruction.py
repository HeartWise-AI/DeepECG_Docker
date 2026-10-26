"""Sensitivity analysis: re-evaluate the trained 3-class model on the test set
after excluding patients with a resting LVOT gradient >= 30 mmHg
(`manifest` obstruction). Reports macro AUROC + per-class OvR AUROCs +
the Latent+ Youden operating point fitted on the full validation split
(unchanged from main analysis) and re-evaluated on the reduced test set.

Output: outputs/stress_echo_fig/sensitivity_rest_obstruction.json
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
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

PKG = "/media/data1/models/DeepECG-HCM"
CFG = os.path.join(PKG, "config", "inference.yaml")
COHORT_DIR = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class"
LINKED = "/volume/DeepECG_Docker/finetune_data/stress_echo/linked_all.parquet"
OUT_JSON = Path("/volume/DeepECG_Docker/outputs/stress_echo_fig/sensitivity_rest_obstruction.json")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

CLASSES = ["Other", "HCM_no_latent", "Latent_pos"]


def get_probs(npy_x, npy_y, device, cfg):
    sys.path.insert(0, cfg["inference"].get(
        "fairseq_signals_path", "/volume/DeepECG_Docker/fairseq-signals"))
    from fairseq_signals.utils import checkpoint_utils

    m = cfg["model"]
    base_ckpt = m["base_checkpoint"]
    base_ckpt = base_ckpt if os.path.isabs(base_ckpt) else os.path.join(PKG, base_ckpt)
    ft_ckpt = m["finetuned_checkpoint"]
    ft_ckpt = ft_ckpt if os.path.isabs(ft_ckpt) else os.path.join(PKG, ft_ckpt)

    base, _, _ = checkpoint_utils.load_model_and_task(
        base_ckpt, arg_overrides={"no_pretrained_weights": True}, suffix="")
    base.proj = nn.Sequential(nn.Dropout(0.1),
                              nn.Linear(m["hidden_dim"], m["num_classes"]))
    state = torch.load(ft_ckpt, map_location=device, weights_only=False)
    base.load_state_dict(state["model"])
    base = base.to(device).eval()

    x = np.load(npy_x); y = np.load(npy_y)
    x = torch.from_numpy(x.squeeze(-1)).permute(0, 2, 1).contiguous()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            xb = x[i:i + 64].to(device, non_blocking=True)
            logits = base.get_logits(base(source=xb)).float()
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0), y


def per_class_metrics(y, p, label):
    out = {}
    for c, name in enumerate(CLASSES):
        bin_y = (y == c).astype(int)
        if bin_y.sum() == 0 or bin_y.sum() == len(bin_y):
            out[name] = {"auroc": float("nan"), "auprc": float("nan"),
                         "n_pos": int(bin_y.sum()), "n": int(len(bin_y))}
            continue
        out[name] = {
            "auroc": float(roc_auc_score(bin_y, p[:, c])),
            "auprc": float(average_precision_score(bin_y, p[:, c])),
            "n_pos": int(bin_y.sum()),
            "n": int(len(bin_y)),
            "prevalence": float(bin_y.mean()),
        }
    valid = [v["auroc"] for v in out.values() if not np.isnan(v["auroc"])]
    out["macro_auroc"] = float(np.mean(valid)) if valid else float("nan")
    return out


def fit_youden(y_bin, score):
    fpr, tpr, t = roc_curve(y_bin, score)
    j = tpr - fpr
    return float(t[int(np.argmax(j))])


def at_threshold(y_bin, score, t):
    pred = score >= t
    tp = int(((y_bin == 1) & pred).sum()); fp = int(((y_bin == 0) & pred).sum())
    tn = int(((y_bin == 0) & ~pred).sum()); fn = int(((y_bin == 1) & ~pred).sum())
    return {
        "threshold": float(t),
        "sens": tp / max(tp + fn, 1),
        "spec": tn / max(tn + fp, 1),
        "ppv":  tp / max(tp + fp, 1),
        "npv":  tn / max(tn + fn, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    cfg = yaml.safe_load(open(CFG))
    device = torch.device("cuda:3")

    print("== Computing probabilities ==")
    val_p, val_y = get_probs(os.path.join(COHORT_DIR, "val_x.npy"),
                              os.path.join(COHORT_DIR, "val_y.npy"), device, cfg)
    test_p, test_y = get_probs(os.path.join(COHORT_DIR, "test_x.npy"),
                                os.path.join(COHORT_DIR, "test_y.npy"), device, cfg)
    print(f"  val n={len(val_y)}  test n={len(test_y)}")

    # Identify rest-obstructed in the test split (ordering of cohort.parquet
    # matches the NPY arrays — same selection used in build_stress_echo_3class.py).
    cohort = pd.read_parquet(os.path.join(COHORT_DIR, "cohort.parquet"))
    linked = pd.read_parquet(LINKED)
    rest = pd.to_numeric(linked["hcm_variables.lvot_gradient_rest_mmhg"], errors="coerce")
    linked = linked.assign(rest_grad=rest)
    # Join rest_grad onto cohort by patient_id_int + exam_date
    cohort = cohort.merge(
        linked[["patient_id_int", "exam_date", "rest_grad"]],
        on=["patient_id_int", "exam_date"], how="left")
    test_mask = (cohort["split"] == "test").values
    test_cohort = cohort[test_mask].reset_index(drop=True)
    assert len(test_cohort) == len(test_y), \
        f"row mismatch: cohort test {len(test_cohort)} vs npy y {len(test_y)}"
    # Label-order sanity
    assert (test_cohort["label3"].values == test_y).all(), \
        "label ordering between cohort and npy y disagrees"

    rest_obs = (test_cohort["rest_grad"] >= 30).fillna(False).values
    keep = ~rest_obs
    print(f"\nTest rows: {len(test_y)}, rest-obstructed: {int(rest_obs.sum())}, "
          f"retained for sensitivity: {int(keep.sum())}")
    print(f"  Class distribution among excluded: "
          f"{np.bincount(test_y[rest_obs], minlength=3).tolist()}")
    print(f"  Class distribution after exclusion: "
          f"{np.bincount(test_y[keep], minlength=3).tolist()}")

    print("\n== Main test metrics (full set) ==")
    main_m = per_class_metrics(test_y, test_p, "full")
    print(json.dumps(main_m, indent=2))

    print("\n== Sensitivity (rest-obstructed excluded) ==")
    sens_m = per_class_metrics(test_y[keep], test_p[keep], "rest_obs_excl")
    print(json.dumps(sens_m, indent=2))

    # Youden for Latent+ fit on FULL validation, evaluated on both test sets
    val_bin = (val_y == 2).astype(int)
    val_sc = val_p[:, 2]
    t_lat = fit_youden(val_bin, val_sc)

    # Recompute for the main (already done in main report) and sensitivity test
    test_lat_full = at_threshold((test_y == 2).astype(int), test_p[:, 2], t_lat)
    test_lat_sens = at_threshold((test_y[keep] == 2).astype(int),
                                  test_p[keep][:, 2], t_lat)
    print("\nLatent+ Youden threshold (val-fit):", t_lat)
    print("  Full test:        ", test_lat_full)
    print("  Sensitivity test: ", test_lat_sens)

    out = {
        "test_full": {
            "n": int(len(test_y)),
            "class_n": np.bincount(test_y, minlength=3).tolist(),
            "metrics": main_m,
            "latent_at_youden": test_lat_full,
        },
        "test_rest_obs_excluded": {
            "n": int(keep.sum()),
            "n_excluded": int(rest_obs.sum()),
            "class_n": np.bincount(test_y[keep], minlength=3).tolist(),
            "metrics": sens_m,
            "latent_at_youden": test_lat_sens,
        },
        "youden_threshold_latent": t_lat,
        "exclusion_rule": "rest LVOT gradient >= 30 mmHg",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
