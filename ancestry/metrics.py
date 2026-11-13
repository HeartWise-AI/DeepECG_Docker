"""Mode A -- evaluation + overlay.

Input  (mounted read-only):  X.npy  (N, 2500, 12) float32 @ 250 Hz
                             y.npy  (N,) in {0,1}, 1 = non-European (positive)
                             cohort.npy (N,) str   [optional, for per-group AUROC]

Runs the 5-fold ensemble and computes, with bootstrap 95% CIs, discrimination
(AUROC, AUPRC), calibration (Brier, ECE), and operating-point metrics at the
threshold (sensitivity, specificity, PPV, NPV, F1, accuracy) plus the confusion
matrix -- overall and per cohort. Curves are drawn against OUR baked-in reference
so results compare without any per-patient data leaving the enclave.

Outputs (aggregate only):
  metrics.json                 all metrics + 95% CI + confusion, overall & per-cohort
  metrics.csv                  same, one row per group
  roc_pr_overlay.png/.pdf      PR | ROC | Calibration, this run vs reference (ALL)
  score_distribution.png/.pdf  P(non-EUR) histogram by true class
  per_cohort_curves.png/.pdf   PR | ROC | Calibration, one line per cohort (if cohort.npy
                               with >=2 evaluable cohorts is provided)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve, brier_score_loss,
)

from ancestry import infer as infer_lib
from ancestry import lib as L
from ancestry import weights as W
from ancestry import privacy as PV
from ancestry.privacy import cell, mask


def _ece(y, s, n_bins=10):
    """Expected calibration error (uniform binning, matches calibration_curve)."""
    y = np.asarray(y, float).ravel(); s = np.asarray(s, float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            e += m.mean() * abs(s[m].mean() - y[m].mean())
    return float(e)


def _confusion(y, s, thr):
    yhat = (s >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def _scalar_metrics(y, s, thr):
    """All point metrics from (y, score, threshold) as a flat dict.
    Threshold metrics: sens (recall/TPR), spec (TNR), ppv (precision), npv, f1, acc."""
    tp, fp, tn, fn = _confusion(y, s, thr)
    div = lambda a, b: (a / b) if b else float("nan")
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "brier": float(brier_score_loss(y, s)),
        "ece": _ece(y, s),
        "sens": div(tp, tp + fn), "spec": div(tn, tn + fp),
        "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "acc": div(tp + tn, tp + fp + tn + fn),
    }


_METRIC_KEYS = ["auroc", "auprc", "brier", "ece",
                "sens", "spec", "ppv", "npv", "f1", "acc"]


def _boot_all(y, s, thr, n_boot=1000, seed=42):
    """One resample loop -> 95% CI for every scalar metric (consistent draws)."""
    rng = np.random.default_rng(seed)
    n = len(y); acc = {k: [] for k in _METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, ss = y[idx], s[idx]
        if len(np.unique(yy)) < 2:
            continue
        m = _scalar_metrics(yy, ss, thr)
        for k in _METRIC_KEYS:
            if m[k] == m[k]:  # skip NaN
                acc[k].append(m[k])
    return {k: ((float(np.percentile(acc[k], 2.5)), float(np.percentile(acc[k], 97.5)))
                if acc[k] else (float("nan"), float("nan"))) for k in _METRIC_KEYS}


def evaluate(y, s, thr=0.5, n_boot=1000):
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(s).astype(float).ravel()
    point = _scalar_metrics(y, s, thr)
    ci = _boot_all(y, s, thr, n_boot)
    tp, fp, tn, fn = _confusion(y, s, thr)
    out = {"n": int(len(y)), "pos_rate": float(y.mean()), "threshold": float(thr),
           "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}
    for k in _METRIC_KEYS:
        out[k] = float(point[k]); out[k + "_ci95"] = ci[k]
    return out


def _calibration_bins(y, s, n_bins=10, min_bin=0):
    """Uniform-bin reliability curve, dropping bins with < min_bin people (a bin of one or two
    people would display those people's labels)."""
    y = np.asarray(y, float); s = np.asarray(s, float)
    ids = np.clip(np.digitize(s, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    mp, fp = [], []
    for b in range(n_bins):
        m = ids == b
        if m.sum() and m.sum() >= min_bin:
            mp.append(float(s[m].mean())); fp.append(float(y[m].mean()))
    return np.asarray(fp), np.asarray(mp)


def overlay_figure(y, s, ref, out_png, thr=0.5, run_name="This run", title=None, min_bin=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42

    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    rall = ref["per_cohort"]["ALL"]
    rc = ref["curves"]["ALL"]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))
    NEW, OUR = "#0072B2", "#888888"

    ax = axes[0]
    ax.plot(rc["pr"]["recall"], rc["pr"]["precision"], color=OUR, lw=2, ls="--",
            label=f'Ours / reference (AUPRC={rall["auprc_pooled"]:.3f})')
    ax.step(rec, prec, where="post", color=NEW, lw=2,
            label=f'{run_name} (AUPRC={average_precision_score(y, s):.3f})')
    ax.axhline(float(y.mean()), color="gray", ls=":", label=f"prevalence={y.mean():.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax.legend(loc="lower left", fontsize=9)

    ax = axes[1]
    ax.plot(rc["roc"]["fpr"], rc["roc"]["tpr"], color=OUR, lw=2, ls="--",
            label=f'Ours / reference (AUROC={rall["auroc_pooled"]:.3f})')
    ax.plot(fpr, tpr, color=NEW, lw=2,
            label=f'{run_name} (AUROC={roc_auc_score(y, s):.3f})')
    ax.plot([0, 1], [0, 1], color="gray", ls=":")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC")
    ax.legend(loc="lower right", fontsize=9)

    # -- Calibration / reliability curve (predicted P(non-EUR) vs observed fraction) --
    ax = axes[2]
    frac_pos, mean_pred = _calibration_bins(y, s, n_bins=10, min_bin=min_bin)
    brier = brier_score_loss(y, s); ece = _ece(y, s)
    if isinstance(rc, dict) and "cal" in rc:
        ax.plot(rc["cal"]["pred"], rc["cal"]["frac_pos"], color=OUR, lw=2, ls="--",
                marker="o", ms=4, label="Ours / reference")
    ax.plot(mean_pred, frac_pos, color=NEW, lw=2, marker="o", ms=4,
            label=f"{run_name} (Brier={brier:.3f}, ECE={ece:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls=":", label="perfect calibration")
    ax.set(xlabel="Mean predicted P(non-EUR)", ylabel="Observed fraction non-EUR",
           title=f"Calibration (bins with < {min_bin} people not shown)" if min_bin else "Calibration",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(title or f"DeepECG-SSL ancestry: {run_name} vs reference (5-fold ensemble)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(out_png).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


_COHORT_COL = {"UKB": "#0072B2", "MHI": "#E69F00", "CLSA": "#009E73"}
_FALLBACK = ["#CC79A7", "#56B4E9", "#D55E00", "#000000", "#882255"]


def per_cohort_figure(y, s, cohort, ref, out_png, run_name="This run",
                      min_bin=PV.MIN_PER_CLASS):
    """1x3 (PR | ROC | Calibration) with one line per cohort. Solid = this run,
    dotted = our baked-in reference for that cohort (if present). Aggregate-only."""
    groups = [g for g in sorted(set(cohort))
              if min(y[cohort == g].sum(), (1 - y[cohort == g]).sum()) >= 20]
    if len(groups) < 2:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))
    for i, g in enumerate(groups):
        m = cohort == g; yg, sg = y[m], s[m]
        col = _COHORT_COL.get(g, _FALLBACK[i % len(_FALLBACK)])
        prec, rec, _ = precision_recall_curve(yg, sg)
        fpr, tpr, _ = roc_curve(yg, sg)
        # A reliability point exposes the mean prediction and binary-label fraction of its bin.
        # Apply the same disclosure floor as the overall calibration plot; cohort-level class
        # size alone does not stop a tail bin from containing a single participant.
        fpos, mpred = _calibration_bins(yg, sg, n_bins=10,
                                        min_bin=max(min_bin, PV.MIN_PER_CLASS))
        axes[0].step(rec, prec, where="post", color=col, lw=1.8,
                     label=f"{g} (AUPRC={average_precision_score(yg, sg):.3f})")
        axes[1].plot(fpr, tpr, color=col, lw=1.8,
                     label=f"{g} (AUROC={roc_auc_score(yg, sg):.3f})")
        axes[2].plot(mpred, fpos, color=col, lw=1.8, marker="o", ms=3,
                     label=f"{g} (Brier={brier_score_loss(yg, sg):.3f})")
        rcg = ref.get("curves", {}).get(g)
        if rcg:
            axes[0].plot(rcg["pr"]["recall"], rcg["pr"]["precision"], color=col, lw=1.1, ls=":", alpha=0.7)
            axes[1].plot(rcg["roc"]["fpr"], rcg["roc"]["tpr"], color=col, lw=1.1, ls=":", alpha=0.7)
            if "cal" in rcg:
                axes[2].plot(rcg["cal"]["pred"], rcg["cal"]["frac_pos"], color=col, lw=1.1, ls=":", alpha=0.7)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (per cohort)")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].plot([0, 1], [0, 1], color="gray", ls=":")
    axes[1].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC (per cohort)")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[2].plot([0, 1], [0, 1], color="gray", ls=":")
    axes[2].set(xlabel="Mean predicted P(non-EUR)", ylabel="Observed fraction non-EUR",
                title=f"Calibration (per cohort; bins n >= {max(min_bin, PV.MIN_PER_CLASS)})",
                xlim=(0, 1), ylim=(0, 1))
    axes[2].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{run_name}: per-cohort curves (solid = this run, dotted = reference)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(out_png).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def score_hist_figure(y, s, out_png, thr=0.5, run_name="This run"):
    """Class-conditional distribution of P(non-EUR): EUR (y=0) vs non-EUR (y=1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    bins = np.linspace(0, 1, 41)
    ax.hist(s[y == 0], bins=bins, color="#0072B2", alpha=0.6, density=True,
            label=f"European (y=0, n={cell(int((y == 0).sum()))})")
    ax.hist(s[y == 1], bins=bins, color="#D55E00", alpha=0.6, density=True,
            label=f"non-European (y=1, n={cell(int((y == 1).sum()))})")
    ax.axvline(thr, color="black", ls="--", lw=1.0, label=f"threshold={thr:.2f}")
    ax.set(xlabel="Predicted P(non-EUR)", ylabel="Density",
           title=f"{run_name}: score distribution by true class")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(out_png).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dir with X.npy, y.npy [, cohort.npy]")
    ap.add_argument("--output", required=True,
                    help="where to write this analysis's tables and plates (aggregate only)")
    ap.add_argument("--models-root", default=None,
                    help="the five ancestry fold checkpoints (default: $MODELS_ROOT, else the pinned revision in the model manifests)")
    ap.add_argument("--ssl-ckpt", default=None,
                    help="the shared SSL backbone (default: $SSL_CKPT, else the pinned revision in the model manifests)")
    ap.add_argument("--reference", default=None, help="frozen discovery metrics (default: the packaged reference_roc_pr.json)")
    ap.add_argument("--device", default="auto",
                    help="cpu, cuda, cuda:1 ...; 'auto' takes a GPU when there is one")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="ECGs per forward pass; lower it if the GPU runs out of memory")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="operating point for the reported TPR and FPR")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="bootstrap resamples behind every confidence interval")
    ap.add_argument("--run-name", default="This run", help="label for this cohort in figures")
    ap.add_argument("--min-n", type=int, default=PV.MIN_PER_CLASS,
                    help="refuse overall / skip per-cohort metrics with fewer than this per class")
    args = ap.parse_args()
    # Resolved here rather than required on the command line: one package, one
    # way to find the weights and the frozen discovery tables.
    args.models_root = W.resolve("ancestry_folds", args.models_root)
    args.ssl_ckpt = W.resolve("ssl_backbone", args.ssl_ckpt)
    args.reference = args.reference or str(L.ref_dir() / "reference_roc_pr.json")

    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    X = np.load(inp / "X.npy", mmap_mode="r")
    y = np.load(inp / "y.npy").astype(int).ravel()
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    cohort = None
    if (inp / "cohort.npy").exists():
        cohort = np.load(inp / "cohort.npy", allow_pickle=True).astype(str).ravel()

    if min(y.sum(), (1 - y).sum()) < args.min_n:
        raise SystemExit(f"[metrics] REFUSED: fewer than {args.min_n} participants in a class; "
                         "nothing was written.")
    device = infer_lib.pick_device(args.device)
    print(f"[info] device={device}  X={X.shape}  pos_rate={y.mean():.3f}", flush=True)
    models, ckpts = infer_lib.load_fold_models(args.models_root, args.ssl_ckpt, device)
    p_ens, p_folds = infer_lib.ensemble_predict(models, X, device, args.batch_size)

    ref = json.loads(Path(args.reference).read_text())
    result = {
        "run_name": args.run_name,
        "device": str(device), "n_models": len(models), "checkpoints": ckpts,
        "threshold": args.threshold,
        "label_convention": "y=1 -> non-EUR (positive); score = P(non-EUR)",
        "overall": evaluate(y, p_ens, args.threshold, args.n_boot),
        "prevalence_note": ("AUROC, Brier-skill and the curves' shape are comparable to the reference "
                            "at any non-EUR fraction. AUPRC, PPV/NPV, F1 and accuracy depend on the "
                            "non-EUR fraction (pos_rate); the reference set is ~balanced, so compare "
                            "those only to runs with a similar pos_rate (e.g. the matched first_only/)."),
        "reference_overall": ref["per_cohort"]["ALL"],
    }
    if cohort is not None:
        result["per_cohort"] = {}
        for g in sorted(set(cohort)):
            m = cohort == g
            if min(y[m].sum(), (1 - y[m]).sum()) >= args.min_n:
                result["per_cohort"][g] = evaluate(y[m], p_ens[m], args.threshold, args.n_boot)

    (out / "metrics.json").write_text(json.dumps(mask(result), indent=2))

    rows = ["group,n,pos_rate,auroc,auroc_lo,auroc_hi,auprc,auprc_lo,auprc_hi,"
            "brier,brier_lo,brier_hi,ece,ece_lo,ece_hi,"
            "sens,sens_lo,sens_hi,spec,spec_lo,spec_hi,ppv,npv,f1,acc,"
            "tp,fp,tn,fn,ref_auroc"]
    def _row(g, m, ref_auroc):
        c = m["confusion"]
        def ci(k): return f'{m[k+"_ci95"][0]:.4f},{m[k+"_ci95"][1]:.4f}'
        return (f'{g},{cell(m["n"])},{m["pos_rate"]:.4f},'
                f'{m["auroc"]:.4f},{ci("auroc")},{m["auprc"]:.4f},{ci("auprc")},'
                f'{m["brier"]:.4f},{ci("brier")},{m["ece"]:.4f},{ci("ece")},'
                f'{m["sens"]:.4f},{ci("sens")},{m["spec"]:.4f},{ci("spec")},'
                f'{m["ppv"]:.4f},{m["npv"]:.4f},{m["f1"]:.4f},{m["acc"]:.4f},'
                f'{cell(c["tp"])},{cell(c["fp"])},{cell(c["tn"])},{cell(c["fn"])},{ref_auroc:.4f}')
    rows.append(_row("OVERALL", result["overall"], ref["per_cohort"]["ALL"]["auroc_pooled"]))
    for g, m in result.get("per_cohort", {}).items():
        ra = ref["per_cohort"].get(g, {}).get("auroc_pooled", float("nan"))
        rows.append(_row(g, m, ra))
    (out / "metrics.csv").write_text("\n".join(rows) + "\n")

    figs = ["roc_pr_overlay.png/.pdf (PR | ROC | Calibration)"]
    overlay_figure(y, p_ens, ref, str(out / "roc_pr_overlay.png"),
                   thr=args.threshold, run_name=args.run_name)
    score_hist_figure(y, p_ens, str(out / "score_distribution.png"),
                      thr=args.threshold, run_name=args.run_name)
    figs.append("score_distribution.png/.pdf")
    if cohort is not None and per_cohort_figure(
            y, p_ens, cohort, ref, str(out / "per_cohort_curves.png"),
            run_name=args.run_name, min_bin=max(args.min_n, PV.MIN_PER_CLASS)):
        figs.append("per_cohort_curves.png/.pdf (>=2 evaluable cohorts)")

    o = result["overall"]
    print(f"\n[done] OVERALL  AUROC={o['auroc']:.3f} "
          f"[{o['auroc_ci95'][0]:.3f},{o['auroc_ci95'][1]:.3f}]  "
          f"AUPRC={o['auprc']:.3f}  Brier={o['brier']:.3f}  ECE={o['ece']:.3f}")
    print(f"[done]          sens={o['sens']:.3f}  spec={o['spec']:.3f}  "
          f"ppv={o['ppv']:.3f}  npv={o['npv']:.3f}  f1={o['f1']:.3f}  acc={o['acc']:.3f}  "
          f"@thr={args.threshold:.2f}   (reference AUROC={ref['per_cohort']['ALL']['auroc_pooled']:.3f})")
    if abs(y.mean() - 0.5) > 0.15:
        print(f"[note] pos_rate={y.mean():.3f}: AUROC is directly comparable to the reference; "
              "AUPRC/PPV/accuracy are not (they scale with the non-EUR fraction).")
    print(f"[done] wrote {out}/metrics.json, metrics.csv, " + ", ".join(figs))


if __name__ == "__main__":
    main()
