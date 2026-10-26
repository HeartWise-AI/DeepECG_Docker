"""Publication-grade 4-panel figure for the DeepECG-HCM 3-class stress-echo model.

Panels:
    A. ROC curves (per-class one-vs-rest) with AUROC + 95% CI in legend.
    B. PR curves (per-class one-vs-rest) with AUPRC + prevalence reference lines.
    C. Confusion matrix (row-normalized, predictions at argmax).
    D. Decision-curve analysis (net benefit vs threshold) for Latent+ vs rest.

Outputs (SVG + PDF + TIFF + PNG) into outputs/stress_echo_fig/.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix,
)

CSV = "/volume/DeepECG_Docker/outputs/stress_echo_test_predictions.csv"
OUT = Path("/volume/DeepECG_Docker/outputs/stress_echo_fig")
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Other", "HCM_no_latent", "Latent_pos"]
CLASS_LABELS = ["Other", "HCM w/o latent", "Latent+"]
# Low-saturation palette (NMI pastel inspired); Latent+ in a warm accent.
COLORS = {"Other": "#6E8AB8", "HCM_no_latent": "#9CB089", "Latent_pos": "#C26B6B"}

# Matplotlib publication defaults
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y_true[idx], y_score[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        stats.append(metric_fn(yb, sb))
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def plot_roc(ax, df):
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.5, zorder=0)
    for c, name, label in zip(range(3), CLASSES, CLASS_LABELS):
        y_true = (df["true_label"].values == c).astype(int)
        y_score = df[f"prob_{name}"].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        lo, hi = bootstrap_ci(y_true, y_score, roc_auc_score, seed=c)
        ax.plot(fpr, tpr, color=COLORS[name], lw=1.4,
                label=f"{label}  AUROC {auc:.3f} ({lo:.3f}–{hi:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("a  ROC (one-vs-rest)", loc="left", fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", handlelength=1.6, borderaxespad=0.2)


def plot_pr(ax, df):
    n = len(df)
    for c, name, label in zip(range(3), CLASSES, CLASS_LABELS):
        y_true = (df["true_label"].values == c).astype(int)
        y_score = df[f"prob_{name}"].values
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        lo, hi = bootstrap_ci(y_true, y_score, average_precision_score, seed=10 + c)
        prev = y_true.mean()
        ax.plot(rec, prec, color=COLORS[name], lw=1.4,
                label=f"{label}  AUPRC {ap:.3f} ({lo:.3f}–{hi:.3f}), prev {prev*100:.2f}%")
        # prevalence reference line (random classifier)
        ax.hlines(prev, 0, 1, color=COLORS[name], lw=0.5, ls=":", alpha=0.6, zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("b  Precision–Recall", loc="left", fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)


def plot_confusion(ax, df):
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=[0, 1, 2])
    cm_n = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_n[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if cm_n[i, j] > 0.55 else "black")
    ax.set_xticks(range(3), CLASS_LABELS, rotation=20, ha="right")
    ax.set_yticks(range(3), CLASS_LABELS)
    ax.set_xlabel("Predicted (argmax)")
    ax.set_ylabel("True")
    ax.set_title("c  Confusion matrix (row-normalized)", loc="left", fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)
    cb.set_label("Row-normalized rate")


def decision_curve(y_true, y_score, thresholds):
    """Net benefit at each decision threshold. Vickers & Elkin (2006)."""
    n = len(y_true)
    nb = []
    for t in thresholds:
        pred_pos = y_score >= t
        tp = np.sum((pred_pos == 1) & (y_true == 1))
        fp = np.sum((pred_pos == 1) & (y_true == 0))
        w = t / (1 - t) if t < 1 else np.inf
        nb.append((tp / n) - (fp / n) * w)
    return np.array(nb)


def plot_dca(ax, df):
    thresholds = np.linspace(0.01, 0.5, 99)
    for c, name, label in zip(range(3), CLASSES, CLASS_LABELS):
        y_true = (df["true_label"].values == c).astype(int)
        y_score = df[f"prob_{name}"].values
        nb = decision_curve(y_true, y_score, thresholds)
        ax.plot(thresholds, nb, color=COLORS[name], lw=1.4, label=f"{label} (model)")
        # Treat-all baseline per class
        prev = y_true.mean()
        treat_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
        ax.plot(thresholds, treat_all, color=COLORS[name], lw=0.6, ls="--", alpha=0.7,
                label=f"{label} (treat all)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlim(0, 0.5)
    ymin = -0.02
    ymax = 0.05
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Decision threshold (P̂)")
    ax.set_ylabel("Net benefit")
    ax.set_title("d  Decision curve analysis", loc="left", fontweight="bold")
    ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2, ncol=1)


def main():
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} test rows. class dist={np.bincount(df['true_label'], minlength=3).tolist()}")

    # 180 mm wide (full page) × 150 mm tall — 2×2 panel grid
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 150 / 25.4))
    plot_roc(axes[0, 0], df)
    plot_pr(axes[0, 1], df)
    plot_confusion(axes[1, 0], df)
    plot_dca(axes[1, 1], df)
    fig.suptitle("DeepECG-HCM — 3-class test performance (n=1,897)", fontsize=8.5,
                 fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    for ext in ["svg", "pdf", "png", "tiff"]:
        path = OUT / f"deepecg_hcm_figure.{ext}"
        dpi = 600 if ext in ("png", "tiff") else None
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
