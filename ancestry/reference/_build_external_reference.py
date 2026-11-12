"""Generate the baked-in REFERENCE ROC/PR/calibration curves for the overlays (seed-2000).

Run ONCE, outside the container, in the ANCESTRY env (GPU recommended). Each SHIPPED
seed-2000 fold model (the exact checkpoints build.sh stages) scores its own held-out test
fold, i.e. 5-fold pooled out-of-sample predictions, and only aggregates are written:

  reference_roc_pr.json   pooled ROC / PR / calibration curve arrays + per-cohort and ALL
                          AUROC / AUPRC / ACC (mean +/- 95% CI across folds, and pooled)

Test arrays are raw ADC units; they are multiplied by 0.00488 (ADC -> mV) exactly as the
training manifest does, because the container's model expects millivolts.

reference_admixed.json is NOT regenerated here: its source (outputs/admixed_ood/stats.json)
no longer exists on the training box, so the committed seed-42-era file is kept and marked.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

HERE = Path(__file__).resolve().parent
# PROVENANCE ONLY. This ran on the analysis box, against the discovery cohorts and the
# fold checkpoints; neither is in this repository, so it cannot run here. It is shipped
# so that how the frozen tables were made is readable, not re-runnable.
from ancestry import infer as infer_lib  # noqa: E402

# Not public; see _build_reference.py. Set ANCESTRY_ANALYSIS_ROOT to rebuild.
TASK = Path(os.environ.get("ANCESTRY_ANALYSIS_ROOT", "/nonexistent/analysis-tree"))
MODELS = Path(os.environ.get("REF_MODELS", TASK / "output_uniform_split_70_10_20_seed2000_rerun/ancestry/ft_models"))  # MUST = build.sh MODELS_SRC
ARRAYS = Path(os.environ.get("REF_ARRAYS", TASK / "output_uniform_split_70_10_20_seed2000_rerun/ancestry/arrays"))  # the split THOSE models were trained on
SSL = Path(os.environ.get("SSL_CKPT", "/app/weights/checkpoint_last.pt"))
SCALE = 0.00488
COHORTS = ["UKB", "MHI", "CLSA"]
DECISION = 0.5
N_BINS = 10


def mean_ci(a):
    a = np.asarray(a, float)
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def _down(x, y, n=400):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) <= n:
        return x.tolist(), y.tolist()
    i = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return x[i].tolist(), y[i].tolist()


def curves(y, s):
    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    cy, cx = calibration_curve(y, s, n_bins=N_BINS)
    r, p = _down(rec, prec); f, t = _down(fpr, tpr)
    return {"pr": {"recall": r, "precision": p}, "roc": {"fpr": f, "tpr": t},
            "cal": {"pred": cx.tolist(), "frac_pos": cy.tolist()}}


def main():
    dev = infer_lib.pick_device("auto")
    Y = {c: [] for c in COHORTS}; P = {c: [] for c in COHORTS}
    for k in range(1, 6):
        m = infer_lib.WCREcgTransformer(str(MODELS / f"fold_{k}/checkpoint_best.pt"), str(SSL))
        m.to(dev).eval()
        for c in COHORTS:
            X = np.load(ARRAYS / f"X_test_{c}_{k}.npy", mmap_mode="r")
            y = np.load(ARRAYS / f"y_test_{c}_{k}.npy").astype(int).ravel()
            Xmv = np.asarray(X, np.float32) * SCALE
            p = infer_lib._predict_one(m, np.swapaxes(Xmv, -1, -2), dev, 64)
            Y[c].append(y); P[c].append(p)
            print(f"fold {k} {c}: n={len(y)} AUROC={roc_auc_score(y, p):.3f}", flush=True)
        del m

    out = {"decision_threshold": DECISION, "per_cohort": {}, "curves": {}}
    fold_all = {"auroc": [], "auprc": [], "acc": []}
    for c in COHORTS:
        fa = {"auroc": [], "auprc": [], "acc": []}
        for y, p in zip(Y[c], P[c]):
            fa["auroc"].append(roc_auc_score(y, p)); fa["auprc"].append(average_precision_score(y, p))
            fa["acc"].append(accuracy_score(y, (p >= DECISION).astype(int)))
        y = np.concatenate(Y[c]); p = np.concatenate(P[c])
        out["curves"][c] = curves(y, p)
        out["per_cohort"][c] = {"n": int(len(y)), "pos_rate": float(y.mean()),
                                **{k: mean_ci(v) for k, v in fa.items()},
                                "auroc_pooled": float(roc_auc_score(y, p)),
                                "auprc_pooled": float(average_precision_score(y, p))}
    for k in range(5):                                   # per-fold pooled over cohorts
        y = np.concatenate([Y[c][k] for c in COHORTS]); p = np.concatenate([P[c][k] for c in COHORTS])
        fold_all["auroc"].append(roc_auc_score(y, p)); fold_all["auprc"].append(average_precision_score(y, p))
        fold_all["acc"].append(accuracy_score(y, (p >= DECISION).astype(int)))
    y = np.concatenate([np.concatenate(Y[c]) for c in COHORTS])
    p = np.concatenate([np.concatenate(P[c]) for c in COHORTS])
    out["curves"]["ALL"] = curves(y, p)
    out["per_cohort"]["ALL"] = {"n": int(len(y)), "pos_rate": float(y.mean()),
                                **{k: mean_ci(v) for k, v in fold_all.items()},
                                "auroc_pooled": float(roc_auc_score(y, p)),
                                "auprc_pooled": float(average_precision_score(y, p))}
    out["source"] = (f"seed-2000 uniform 70/10/20 fold checkpoints {MODELS}; each fold scores its own "
                     f"held-out test fold from {ARRAYS}; 5-fold pooled; inputs ADC*0.00488 -> mV")
    out["label_convention"] = "y=1 -> non-European (positive class); score = P(non-EUR)"
    (HERE / "reference_roc_pr.json").write_text(json.dumps(out, indent=2))
    for c in [*COHORTS, "ALL"]:
        m = out["per_cohort"][c]
        print(f"  {c:4s} N={m['n']:5d}  per-fold AUROC={m['auroc'][0]:.3f}±{m['auroc'][1]:.3f}  "
              f"pooled AUROC={m['auroc_pooled']:.3f}  AUPRC={m['auprc_pooled']:.3f}  ACC={m['acc'][0]:.3f}")


if __name__ == "__main__":
    main()
