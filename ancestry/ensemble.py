"""Mode `ensemble` -- per-fold agreement + ensemble-size value on the partner cohort.

Because the partner cohort is out-of-sample for ALL 5 fold-models (none trained on it), this
is the clean setting the internal cross-validation cannot give: fold agreement and the 1->5
ensemble-size sweep are leakage-free and on a single common test set.

Input  (mounted read-only):  X.npy (N, 2500, 12) float32 @ 250 Hz     [required]
                             y.npy (N,) {0,1}, 1 = non-European        [optional; enables the
                                                AUROC / AUPRC / Brier ensemble-size sweep]

SECURE-ENCLAVE SAFE: only aggregates are written (vote-count distribution, pairwise agreement
matrix, Fleiss kappa, ensemble-size metric curves). No per-patient predictions are emitted.

Outputs:  ensemble_stats.json, ensemble_agreement.png/.pdf,
          ensemble_size_sweep.png/.pdf (only if y.npy provided)
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from ancestry import infer as infer_lib
from ancestry import lib as L
from ancestry import weights as W
from ancestry import privacy as PV
from ancestry.privacy import mask


def fleiss_binary(votes, R=5):
    """Fleiss' kappa for R binary raters; votes = # raters choosing class 1 per item."""
    n1 = np.asarray(votes, float); n0 = R - n1
    Pi = (n1 ** 2 + n0 ** 2 - R) / (R * (R - 1))
    Pbar = Pi.mean(); p1 = n1.mean() / R; Pe = p1 ** 2 + (1 - p1) ** 2
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def metrics3(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score
    return (float(roc_auc_score(y, p)), float(average_precision_score(y, p)),
            float(np.mean((p - y) ** 2)))


def ci(vals):
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ensemble_analysis(P, y, n_boot, run_name, ckpts=None, device=""):
    """Fold agreement (+ ensemble-size sweep when y has both classes) from the (N, R) matrix
    of per-fold P(non-EUR). Returns (stats, votes, A). Shared by `ensemble` and `validate`."""
    R = P.shape[1]; p_ens = P.mean(1)
    # ---- agreement (no labels needed) ----
    calls = (P >= 0.5).astype(int); votes = calls.sum(1)
    N = len(votes)
    A = np.array([[float((calls[:, i] == calls[:, j]).mean()) for j in range(R)] for i in range(R)])
    pair_mean = float((A.sum() - R) / (R * (R - 1)))
    fleiss = fleiss_binary(votes, R)
    sd = P.std(1)
    stats = {
        "run_name": run_name, "n": int(N), "n_models": R, "checkpoints": ckpts,
        "device": str(device),
        "note": ("partner cohort is out-of-sample for all 5 models -> agreement and the "
                 "ensemble-size sweep are leakage-free and on a common test set."),
        "agreement": {
            "vote_count_dist": {int(v): int((votes == v).sum()) for v in range(R + 1)},
            "unanimous_frac": float(np.mean((votes == 0) | (votes == R))),
            "mean_pairwise_agreement": pair_mean, "fleiss_kappa": fleiss,
            "pairwise_agreement_matrix": A.tolist(),
            "mean_within_patient_sd": float(sd.mean()), "median_within_patient_sd": float(np.median(sd)),
        },
    }

    # ---- ensemble-size sweep + single-vs-ensemble (needs labels) ----
    if y is not None and len(set(y)) == 2:
        KS = list(range(1, R + 1)); COMBOS = {k: list(itertools.combinations(range(R), k)) for k in KS}
        names = ["AUROC", "AUPRC", "Brier"]

        def sweep(idx):
            yy = y[idx]; Pi = P[idx]
            return {k: np.mean([metrics3(yy, Pi[:, list(c)].mean(1)) for c in COMBOS[k]], axis=0) for k in KS}

        pt = sweep(np.arange(N))
        rng = np.random.default_rng(0); boot = {k: [] for k in KS}
        for _ in range(n_boot):
            idx = rng.integers(0, N, N)
            if len(set(y[idx])) < 2:
                continue
            s = sweep(idx)
            for k in KS:
                boot[k].append(s[k])
        boot = {k: np.array(v) for k, v in boot.items()}
        sweep_res = {nm: {k: {"point": float(pt[k][m]),
                              "lo": float(np.percentile(boot[k][:, m], 2.5)),
                              "hi": float(np.percentile(boot[k][:, m], 97.5))} for k in KS}
                     for m, nm in enumerate(names)}
        single = np.mean([metrics3(y, P[:, k]) for k in range(R)], axis=0)
        stats["performance"] = {
            "ensemble_size_sweep": sweep_res,
            "single_fold_mean": dict(zip(names, [float(x) for x in single])),
            f"ensemble_{R}_models": dict(zip(names, [float(x) for x in metrics3(y, p_ens)])),
        }

    return stats, votes, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dir with X.npy [, y.npy]")
    ap.add_argument("--output", required=True,
                    help="where to write this analysis's tables and plates (aggregate only)")
    ap.add_argument("--models-root", default=None,
                    help="the five ancestry fold checkpoints (default: $MODELS_ROOT, else the pinned revision in the model manifests)")
    ap.add_argument("--ssl-ckpt", default=None,
                    help="the shared SSL backbone (default: $SSL_CKPT, else the pinned revision in the model manifests)")
    ap.add_argument("--device", default="auto",
                    help="cpu, cuda, cuda:1 ...; 'auto' takes a GPU when there is one")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="ECGs per forward pass; lower it if the GPU runs out of memory")
    ap.add_argument("--n-boot", type=int, default=500,
                    help="bootstrap resamples behind every confidence interval")
    ap.add_argument("--run-name", default="This cohort",
                    help="your cohort name, printed on the plates and in the tables")
    ap.add_argument("--min-n", type=int, default=PV.MIN_PER_CLASS,
                    help="skip the labelled size sweep if a class has fewer participants")
    args = ap.parse_args()
    # Resolved here rather than required on the command line: one package, one
    # way to find the weights and the frozen discovery tables.
    args.models_root = W.resolve("ancestry_folds", args.models_root)
    args.ssl_ckpt = W.resolve("ssl_backbone", args.ssl_ckpt)

    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    X = np.load(inp / "X.npy", mmap_mode="r")
    y = None
    if (inp / "y.npy").exists():
        y = np.load(inp / "y.npy").astype(int).ravel()
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")

    device = infer_lib.pick_device(args.device)
    print(f"[info] device={device}  X={X.shape}  y={'yes' if y is not None else 'no'}", flush=True)
    models, ckpts = infer_lib.load_fold_models(args.models_root, args.ssl_ckpt, device)
    R = len(models)
    p_ens, P = infer_lib.ensemble_predict(models, X, device, args.batch_size)   # P=(N,R)

    if y is not None and min(y.sum(), (1 - y).sum()) < args.min_n:
        print(f"[ensemble] fewer than {args.min_n} per class: size sweep skipped (agreement only)")
        y = None
    stats, votes, A = ensemble_analysis(P, y, args.n_boot, args.run_name, ckpts, device)
    (out / "ensemble_stats.json").write_text(json.dumps(mask(stats), indent=2))
    _figs(stats, P, votes, A, str(out), R)

    ag = stats["agreement"]
    print(f"\n[done] agreement: {ag['unanimous_frac']*100:.0f}% unanimous | "
          f"mean pairwise {ag['mean_pairwise_agreement']*100:.0f}% | Fleiss kappa {ag['fleiss_kappa']:.2f}")
    if "performance" in stats:
        s = stats["performance"]["single_fold_mean"]; e = stats["performance"][f"ensemble_{R}_models"]
        print(f"[done] AUROC single {s['AUROC']:.3f} -> {R}-model ensemble {e['AUROC']:.3f} | "
              f"Brier {s['Brier']:.3f} -> {e['Brier']:.3f}")
    print(f"[done] wrote {out}/ensemble_stats.json + figures")


def _figs(stats, P, votes, A, out, R, min_cell=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    name = stats["run_name"]; N = len(votes)

    # agreement figure: vote-count distribution + pairwise agreement matrix
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    vc = [int((votes == v).sum()) for v in range(R + 1)]
    small = [0 < c < min_cell for c in vc]                  # small cells: no bar, no exact count
    ax[0].bar(range(R + 1), [0 if sm else c for c, sm in zip(vc, small)],
              color=["#117733" if v in (0, R) else "#999999" for v in range(R + 1)],
              edgecolor="k", lw=.5)
    for v, (c, sm) in enumerate(zip(vc, small)):
        ax[0].text(v, 0 if sm else c, f"<{min_cell}" if sm else f"{c}\n{c/N*100:.0f}%",
                   ha="center", va="bottom", fontsize=8.5)
    ax[0].set(xlabel=f"# of {R} models calling non-EUR", ylabel="participants", xticks=range(R + 1),
              title=f"A  Vote-count distribution\n(unanimous = {stats['agreement']['unanimous_frac']*100:.0f}%)")
    ax[0].grid(axis="y", alpha=.3)
    im = ax[1].imshow(A, vmin=0.5, vmax=1.0, cmap="Greens")
    for i in range(R):
        for j in range(R):
            ax[1].text(j, i, f"{A[i,j]*100:.0f}", ha="center", va="center",
                       color="white" if A[i, j] > 0.85 else "black", fontsize=9)
    ax[1].set(xticks=range(R), yticks=range(R), xticklabels=[f"f{k}" for k in range(1, R + 1)],
              yticklabels=[f"f{k}" for k in range(1, R + 1)],
              title=f"B  Pairwise agreement (%)\nmean {stats['agreement']['mean_pairwise_agreement']*100:.1f}% | "
                    f"Fleiss kappa {stats['agreement']['fleiss_kappa']:.2f}")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="% same call")
    fig.suptitle(f"Fold agreement on {name} (out-of-sample, n={N})", fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{out}/ensemble_agreement.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}/ensemble_agreement.pdf", bbox_inches="tight"); plt.close(fig)

    # ensemble-size sweep figure
    if "performance" in stats:
        sw = stats["performance"]["ensemble_size_sweep"]; KS = sorted(sw["AUROC"])
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.8))
        for m, nm in enumerate(["AUROC", "AUPRC", "Brier"]):
            pts = [sw[nm][k]["point"] for k in KS]
            ax[m].plot(KS, pts, "o-", color="#0072B2", lw=2.6, ms=10)
            for k, v in zip(KS, pts):
                ax[m].annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(0, 11),
                               ha="center", fontsize=9.5)
            better = "lower is better" if nm == "Brier" else "higher is better"
            ax[m].set(xlabel="Number of models in the ensemble", xticks=KS, title=f"{nm}\n({better})")
            ax[m].grid(alpha=.3)
        fig.suptitle(f"Ensemble size 1-{R} on {name}  (out-of-sample, leakage-free, common test set; n={N})",
                     fontsize=12.5, fontweight="bold", y=1.03)
        fig.tight_layout(); fig.savefig(f"{out}/ensemble_size_sweep.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{out}/ensemble_size_sweep.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
