"""Mode B -- admixed dose-response: predicted p(EUR) vs RFMix EUR fraction.

Input (mounted read-only):  X_admixed.npy (M, 2500, 12) float32 @ 250 Hz
                            rfmix_eur.npy (M,) RFMix European ancestry fraction in [0,1]

Runs the 5-fold ensemble, takes p(EUR) = 1 - P(non-EUR), regresses it on the RFMix
EUR fraction, and overlays the dose-response on OUR baked-in reference curve. Binning
uses FIXED edges (--bin-edges) so the curve covers a consistent x-range, with a
bootstrap 95% CI band on the run curve. --run-name relabels the cohort in figures.

SECURE-ENCLAVE SAFE: only binned aggregates and summary statistics are written/plotted.
The raw per-patient scatter is never emitted.

Outputs (aggregate only):   admixed_stats.json, admixed_binned.csv,
                            admixed_regression_overlay.png / .pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ancestry import infer as infer_lib
from ancestry import lib as L
from ancestry import weights as W
from ancestry import privacy as PV
from ancestry.privacy import cell, mask


def boot_ci(arr, n_boot=2000, seed=42):
    arr = np.asarray(arr, float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(arr)
    bs = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(arr.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fixed_bins(x, y, edges, min_n=20, n_boot=2000):
    """Group points into FIXED bins (given edges), so the curve covers a consistent
    x-range regardless of how the admixed EUR fractions happen to be distributed.
    Empty bins are dropped; bins with n < min_n are kept but flagged low_n=True."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    edges = np.asarray(sorted(set(float(e) for e in edges)), float)
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    bins = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        mean, lo, hi = boot_ci(y[m], n_boot=n_boot)
        bins.append({"eur_frac": float(x[m].mean()),
                     "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                     "bin_mid": float((edges[b] + edges[b + 1]) / 2),
                     "n": int(m.sum()), "mean_pEUR": mean, "lo": lo, "hi": hi,
                     "low_n": bool(m.sum() < min_n)})
    return bins


def overlay_figure(bins, ref, stats, out_png, edges, run_name="This run", title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    NEW, OUR = "#0072B2", "#888888"

    # reference: stored as P(non-EUR); convert to p(EUR) = 1 - P
    rb = ref["admixed_only"]["bins"]
    rx = [b["eur_frac"] for b in rb]
    ry = [1.0 - b["mean_P_nonEUR"] for b in rb]
    rlo = [1.0 - b["hi"] for b in rb]; rhi = [1.0 - b["lo"] for b in rb]
    ax.fill_between(rx, rlo, rhi, color=OUR, alpha=0.18)
    # reference rho stored in P(non-EUR) space; sign flips in p(EUR) space
    ref_rho_eur = -ref["admixed_only"]["spearman_rho"]
    ax.plot(rx, ry, "s--", color=OUR, lw=2, markersize=7,
            label=f'Ours / reference (rho={ref_rho_eur:+.3f}, '
                  f'n={ref["admixed_only"]["n"]})')

    # this run: fixed-bin means with a bootstrap 95% CI band
    bx = [b["eur_frac"] for b in bins]
    by = [b["mean_pEUR"] for b in bins]
    blo = [b["lo"] for b in bins]; bhi = [b["hi"] for b in bins]
    ax.fill_between(bx, blo, bhi, color=NEW, alpha=0.22,
                    label=f"{run_name} bootstrap 95% CI")
    ax.plot(bx, by, "o-", color=NEW, lw=2.4, markersize=9, markerfacecolor="white",
            markeredgewidth=1.7,
            label=f'{run_name} (rho={stats["spearman_rho"]:+.3f}, n={stats["n"]})')
    for b in bins:
        ax.text(b["eur_frac"], b["mean_pEUR"] + 0.035, f'n={cell(b["n"])}'
                + ("*" if b.get("low_n") else ""),
                ha="center", fontsize=8, color="dimgray")

    # show the fixed bin edges as light vertical guides
    for e in edges:
        ax.axvline(e, color="lightgray", lw=0.6, ls=":", zorder=0)

    # identity = perfect calibration p(EUR) == EUR fraction
    ax.plot([0, 1], [0, 1], color="green", ls=":", lw=1.2, label="perfect calibration")
    ax.set_xlabel("RFMix European ancestry fraction", fontsize=11)
    ax.set_ylabel("Predicted p(EUR) = 1 - P(non-EUR)", fontsize=11)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    edge_str = ", ".join(f"{e:g}" for e in edges)
    ax.set_title(title or "Admixed dose-response: predicted p(EUR) vs RFMix EUR fraction\n"
                 f"(fixed bins [{edge_str}], mean +/- bootstrap 95% CI; *=low n; "
                 "per-patient points not shown)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(Path(out_png).with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    from scipy.stats import spearmanr, pearsonr

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dir with X_admixed.npy, rfmix_eur.npy")
    ap.add_argument("--output", required=True,
                    help="where to write this analysis's tables and plates (aggregate only)")
    ap.add_argument("--models-root", default=None,
                    help="the five ancestry fold checkpoints (default: $MODELS_ROOT, else the pinned revision in the model manifests)")
    ap.add_argument("--ssl-ckpt", default=None,
                    help="the shared SSL backbone (default: $SSL_CKPT, else the pinned revision in the model manifests)")
    ap.add_argument("--reference", default=None, help="frozen discovery dose-response (default: the packaged reference_admixed.json)")
    ap.add_argument("--device", default="auto",
                    help="cpu, cuda, cuda:1 ...; 'auto' takes a GPU when there is one")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="ECGs per forward pass; lower it if the GPU runs out of memory")
    ap.add_argument("--min-n", type=int, default=PV.MIN_PER_CLASS,
                    help="bins below this n keep only a masked count; their mean/CI are never written or plotted")
    ap.add_argument("--bin-edges", default="0.0,0.2,0.4,0.6,0.8,1.0",
                    help="comma-separated FIXED bin edges covering the EUR-fraction range")
    ap.add_argument("--run-name", default="This run", help="label for this cohort in figures")
    args = ap.parse_args()
    # Resolved here rather than required on the command line: one package, one
    # way to find the weights and the frozen discovery tables.
    args.models_root = W.resolve("ancestry_folds", args.models_root)
    args.ssl_ckpt = W.resolve("ssl_backbone", args.ssl_ckpt)
    args.reference = args.reference or str(L.ref_dir() / "reference_admixed.json")
    edges = [float(e) for e in args.bin_edges.split(",")]

    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    X = np.load(inp / "X_admixed.npy", mmap_mode="r")
    eur = np.load(inp / "rfmix_eur.npy").astype(float).ravel()
    if len(X) != len(eur):
        raise ValueError(f"X_admixed has {len(X)} rows but rfmix_eur has {len(eur)}")

    device = infer_lib.pick_device(args.device)
    print(f"[info] device={device}  X={X.shape}  EUR-frac range "
          f"[{eur.min():.2f},{eur.max():.2f}]", flush=True)
    models, ckpts = infer_lib.load_fold_models(args.models_root, args.ssl_ckpt, device)
    p_non, p_folds = infer_lib.ensemble_predict(models, X, device, args.batch_size)
    p_eur = 1.0 - p_non

    rho, rp = spearmanr(eur, p_eur)
    r, pp = pearsonr(eur, p_eur)
    mad = float(np.mean(np.abs(p_eur - eur)))
    bins = fixed_bins(eur, p_eur, edges, min_n=args.min_n)
    n_ok = sum(1 for b in bins if not b["low_n"])
    for b in bins:                     # low-n bins: keep only the (masked) count, never the values
        if b["low_n"]:
            for k in ("eur_frac", "mean_pEUR", "lo", "hi"):
                b[k] = None

    stats = {
        "run_name": args.run_name,
        "device": str(device), "n": int(len(eur)), "n_models": len(models),
        "checkpoints": ckpts,
        "spearman_rho": float(rho), "spearman_p": float(rp),
        "pearson_r": float(r), "pearson_p": float(pp),
        "MAD_pEUR_vs_eurfrac": mad,
        "mean_signed_dev": float(np.mean(p_eur - eur)),
        "bin_edges": edges, "n_bins": len(bins), "n_bins_ge_min": n_ok, "min_n": args.min_n,
        "bins": bins,
        "note": ("score = mean P(non-EUR) over 5 folds; p(EUR)=1-score. Fixed bins. "
                 "Only binned aggregates exported (no per-patient rows)."),
    }
    if len(bins) < 2:
        stats["WARNING"] = ("fewer than 2 non-empty bins; widen --bin-edges or check the "
                            "EUR-fraction range.")

    ref = json.loads(Path(args.reference).read_text())
    (out / "admixed_stats.json").write_text(json.dumps(mask(stats), indent=2))

    rows = ["bin_lo,bin_hi,eur_frac,n,low_n,mean_pEUR,lo,hi"]
    f4 = lambda v: "" if v is None else f"{v:.4f}"
    for b in bins:
        rows.append(f'{b["bin_lo"]:.3f},{b["bin_hi"]:.3f},{f4(b["eur_frac"])},{cell(b["n"])},'
                    f'{int(b["low_n"])},{f4(b["mean_pEUR"])},{f4(b["lo"])},{f4(b["hi"])}')
    (out / "admixed_binned.csv").write_text("\n".join(rows) + "\n")

    overlay_figure([b for b in bins if not b["low_n"]], ref, stats,
                   str(out / "admixed_regression_overlay.png"),
                   edges, run_name=args.run_name)

    print(f"\n[done] admixed dose-response: Spearman rho={rho:+.3f} (p={rp:.1e}), "
          f"Pearson r={r:+.3f}, MAD={mad:.3f}, n={len(eur)}, "
          f"{len(bins)} bins ({n_ok} with n>={args.min_n})")
    print(f"[done] reference rho={ref['admixed_only']['spearman_rho']:+.3f}")
    print(f"[done] wrote {out}/admixed_stats.json, admixed_binned.csv, "
          f"admixed_regression_overlay.png/.pdf")


if __name__ == "__main__":
    main()
