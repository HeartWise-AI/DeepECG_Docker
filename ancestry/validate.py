"""Mode `validate` -- every pre-specified external-validation analysis from ONE inference pass.

Input (mounted read-only): the `all_genotyped/` directory written by `prepare`
  X.npy           (N, 2500, 12) float32 @ 250 Hz -- everyone with an ECG, admixed included
  covariates.csv  ancestry proportions (AFR, EAS, SAS, EUR, ...), sex, age, batch, round,
                  in_matched (no ids)
  info.json       ancestries, eur_label, training gate

Analyses (labels are derived here from the RFMix proportions, so one pass serves all):
  1 primary        AUROC etc. on the training definition: dominant ancestry >= --gate,
                   admixed excluded
  2 continuous     model p(EUR) vs RFMix EUR fraction on EVERYONE (Spearman/Pearson +
                   fixed-bin calibration curve)
  3 gate sweep     AUROC with the gate at each of --gates
  4 threshold-free EUR fraction < 0.5 (y=1) vs >= 0.5 (y=0), every participant labelled
  5 matched        the 1:1 age/sex-matched subset (in_matched) -- confounding check
  6 fairness       on the primary set: per sex / age band / genotyping batch / ECG round,
                   AUROC, TPR, FPR, ECE with 95% CI and the gap to the largest group
                   (bootstrap CI); per non-EUR group (AFR/EAS/SAS ...) vs all EUR
  7 ensemble       on the primary set: fold agreement (vote counts, pairwise agreement,
                   Fleiss kappa) + 1->5 ensemble-size sweep. Leakage-free here: no partner
                   participant was in any fold's training data.
  8 stability     intrapatient reproducibility on all_genotyped_repeats/ (two recordings per
                   person, written by `prepare` when a person has more than one ECG): ICC(2,1)
                   with a bootstrap CI, within/between-person SD, mean score change, how often
                   the call at the operating threshold flips, by pairing (first/last and the
                   closest consecutive pair), by EUR / non-EUR, by interval and by ECG count.
                   Skipped, with a message, when there are no repeats.

SECURE-ENCLAVE SAFE: only aggregates are written, and two disclosure rules apply to every
output (see privacy.py): any group, gate or bin with fewer than --min-n participants in either
class is reported as suppressed, without metrics or curve points; and any count of 1 to
--min-cell - 1 people is written as "<min-cell" (JSON, CSV and figure labels).

Outputs: validate.json, plot_data.json (every curve behind every figure), validate_summary.csv,
         fairness.csv, dose_response.csv,
         validation_summary.png/.pdf, primary_roc_pr_overlay.png/.pdf,
         ensemble_stats.json, ensemble_agreement.png/.pdf, ensemble_size_sweep.png/.pdf,
         stability.json, stability.csv, stability.png/.pdf (analysis 8, when repeats exist)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from ancestry import infer as infer_lib
from ancestry import lib as L
from ancestry import weights as W
from ancestry import privacy as PV
from ancestry.admixed import fixed_bins
from ancestry.ensemble import _figs as ensemble_figs
from ancestry.ensemble import ensemble_analysis
from ancestry import stability as stab
from ancestry.privacy import cell, mask
from ancestry.metrics import _ece, evaluate, overlay_figure


# --------------------------------------------------------------------------- helpers
def auc(y, s):
    """Mann-Whitney AUROC (fast, tie-aware); NaN if a class is missing."""
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rates(y, s, thr):
    y = np.asarray(y, bool); yhat = np.asarray(s) >= thr
    tpr = float(yhat[y].mean()) if y.any() else float("nan")
    fpr = float(yhat[~y].mean()) if (~y).any() else float("nan")
    return tpr, fpr


def ci(v):
    v = np.asarray([x for x in v if x == x], float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def boot_auc(y, s, n_boot, rng):
    n = len(y)
    return ci([auc(y[i], s[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])


def nan2none(o):
    if isinstance(o, dict):
        return {k: nan2none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [nan2none(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    return o


# --------------------------------------------------------------------------- fairness
def subgroup_table(y, s, attrs, thr, n_boot, min_n, rng):
    """attrs: {attribute: array of group labels (None = not assigned)} over the primary set.
    Per group: n, AUROC/TPR/FPR/ECE + CI. Gap to the attribute's largest group with a CI
    from the SAME bootstrap resamples, so the gap CI accounts for their correlation."""
    y = np.asarray(y, bool); n = len(y)
    groups = {}
    for a, lab in attrs.items():
        lab = np.asarray(lab, object)
        keep = [g for g in pd.unique(lab[pd.notna(lab)])]
        groups[a] = sorted(keep, key=lambda g: str(g))
    rows, boots = [], {}
    for a, gs in groups.items():
        lab = np.asarray(attrs[a], object)
        sizes = {g: int((lab == g).sum()) for g in gs}
        ok = [g for g in gs if (y[lab == g]).sum() >= min_n and (~y[lab == g]).sum() >= min_n]
        ref = max(ok, key=lambda g: sizes[g]) if ok else None
        for g in gs:
            m = lab == g
            row = {"attribute": a, "group": str(g), "n": int(m.sum()),
                   "n_pos": int(y[m].sum()), "n_neg": int((~y[m]).sum()),
                   "reference_group": str(ref) if ref is not None else None}
            if g not in ok:
                row["suppressed"] = True
                rows.append(row); continue
            tpr, fpr = rates(y[m], s[m], thr)
            row.update({"suppressed": False, "auroc": auc(y[m], s[m]), "tpr": tpr, "fpr": fpr,
                        "ece": _ece(y[m].astype(int), s[m])})
            rows.append(row); boots[(a, str(g))] = {"auroc": [], "tpr": [], "fpr": [], "ece": []}
    # one shared resample loop over the whole primary set
    for _ in range(n_boot):
        i = rng.integers(0, n, n); yb, sb = y[i], s[i]
        for a in groups:
            lb = np.asarray(attrs[a], object)[i]
            for g in groups[a]:
                k = (a, str(g))
                if k not in boots:
                    continue
                m = lb == g
                tpr, fpr = rates(yb[m], sb[m], thr)
                boots[k]["auroc"].append(auc(yb[m], sb[m])); boots[k]["tpr"].append(tpr)
                boots[k]["fpr"].append(fpr)
                boots[k]["ece"].append(_ece(yb[m].astype(int), sb[m]) if m.any() else float("nan"))
    for r in rows:
        k = (r["attribute"], r["group"])
        if r.get("suppressed"):
            continue
        for met in ("auroc", "tpr", "fpr", "ece"):
            r[f"{met}_ci95"] = ci(boots[k][met])
        refk = (r["attribute"], r["reference_group"])
        for met in ("auroc", "tpr", "fpr"):
            d = np.asarray(boots[k][met]) - np.asarray(boots[refk][met])
            r[f"d_{met}"] = r[met] - next(x[met] for x in rows if (x["attribute"], x["group"]) == refk)
            r[f"d_{met}_ci95"] = ci(d)
    return rows


def ancestry_subgroups(dom, y, s, noneur, thr, n_boot, min_n, rng):
    """Each non-EUR group vs ALL EUR in the primary set: does the model find every group?"""
    eur = ~y
    rows, boots = [], {}
    for g in noneur:
        pos = dom == g
        row = {"attribute": "ancestry_group", "group": g, "n": int(pos.sum() + eur.sum()),
               "n_pos": int(pos.sum()), "n_neg": int(eur.sum())}
        if pos.sum() < min_n or eur.sum() < min_n:
            row["suppressed"] = True; rows.append(row); continue
        m = pos | eur
        tpr, _ = rates(y[m], s[m], thr)
        row.update({"suppressed": False, "auroc": auc(y[m], s[m]), "tpr": tpr})
        rows.append(row); boots[g] = {"auroc": [], "tpr": []}
    ok = [r for r in rows if not r["suppressed"]]
    ref = max(ok, key=lambda r: r["n_pos"])["group"] if ok else None
    idx_pos = {g: np.flatnonzero(dom == g) for g in boots}
    idx_neg = np.flatnonzero(eur)
    for _ in range(n_boot):
        nb = idx_neg[rng.integers(0, len(idx_neg), len(idx_neg))]
        for g in boots:
            pb = idx_pos[g][rng.integers(0, len(idx_pos[g]), len(idx_pos[g]))]
            ii = np.concatenate([pb, nb])
            boots[g]["auroc"].append(auc(y[ii], s[ii]))
            boots[g]["tpr"].append(float((s[pb] >= thr).mean()))
    for r in rows:
        r["reference_group"] = ref
        if r["suppressed"]:
            continue
        rr = next(x for x in rows if x["group"] == ref)
        for met in ("auroc", "tpr"):
            r[f"{met}_ci95"] = ci(boots[r["group"]][met])
            r[f"d_{met}"] = r[met] - rr[met]
            r[f"d_{met}_ci95"] = ci(np.asarray(boots[r["group"]][met]) - np.asarray(boots[ref][met]))
    return rows


# --------------------------------------------------------------------------- plot data
def _thin(x, y, n=200):
    x = np.asarray(x, float); y = np.asarray(y, float)
    k = np.unique(np.linspace(0, len(x) - 1, min(len(x), n)).astype(int))
    return [round(float(v), 5) for v in x[k]], [round(float(v), 5) for v in y[k]]


def plot_block(y, s, min_n, min_cell, n_bins=10):
    """Everything needed to redraw ROC / PR / calibration / score distribution for one set.
    Aggregate only: <= 200 curve points, NO decision thresholds (those would be individual
    scores), calibration bins with >= min_n people, histogram counts masked below min_cell."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    y = np.asarray(y, int); s = np.asarray(s, float)
    f, t, _ = roc_curve(y, s); pr, rc, _ = precision_recall_curve(y, s)
    fx, ty = _thin(f, t); rx, py = _thin(rc[::-1], pr[::-1])
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    cal = []
    for b in range(n_bins):
        m = ids == b
        if m.sum() >= max(min_n, 1):
            cal.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                        "mean_pred": round(float(s[m].mean()), 5),
                        "frac_pos": round(float(y[m].mean()), 5), "n": int(m.sum())})
    hedges = np.linspace(0, 1, 21)
    hist = {"bin_edges": hedges.tolist(),
            "count_y0_EUR": [cell(int(c), min_cell) for c in np.histogram(s[y == 0], hedges)[0]],
            "count_y1_nonEUR": [cell(int(c), min_cell) for c in np.histogram(s[y == 1], hedges)[0]]}
    return {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "auroc": auc(y, s),
            "roc": {"fpr": fx, "tpr": ty}, "pr": {"recall": rx, "precision": py},
            "calibration_bins": cal, "score_histogram": hist}


# --------------------------------------------------------------------------- figure
PAL = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "ink": "#0b0b0b", "ink2": "#52514e",
       "muted": "#8a8984", "grid": "#e4e3df", "surf": "#fcfcfb"}


def _group_label(r):
    a, g = r["attribute"], r["group"]
    if a == "sex":
        return g.capitalize()
    if a == "age_band":
        return f"{g} y"
    if a == "batch":
        return f"Batch {g}"
    if a == "round":
        return f"Round {g}"
    return f"{g} vs EUR"


_ATTR_TITLE = {"sex": "Sex", "age_band": "Age", "batch": "Genotyping batch",
               "round": "ECG round", "ancestry_group": "Non-EUR group"}


def summary_figure(res, curves, dose, sweep, fair, out_png, run_name, min_cell=10):
    C = lambda v: f"{cell(v, min_cell):,}" if isinstance(cell(v, min_cell), int) else cell(v, min_cell)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9.5, "font.family": "DejaVu Sans",
                         "axes.edgecolor": PAL["muted"], "axes.labelcolor": PAL["ink2"],
                         "xtick.color": PAL["ink2"], "ytick.color": PAL["ink2"],
                         "axes.linewidth": 0.8, "figure.facecolor": PAL["surf"],
                         "axes.facecolor": PAL["surf"],
                         "savefig.facecolor": PAL["surf"]})

    # fairness layout: one row per group, a gap + header row per attribute
    rows, ypos, heads, y = [], [], [], 0.0
    last = None
    for r in fair:
        if r["attribute"] != last:
            if last is not None:
                y += 0.6
            heads.append((y, _ATTR_TITLE.get(r["attribute"], r["attribute"]))); y += 1
            last = r["attribute"]
        rows.append(r); ypos.append(y); y += 1
    n_rows = max(y, 1)

    top_h, fair_h = 3.6, 0.27 * n_rows + 0.9
    fig = plt.figure(figsize=(14, 1.25 + top_h + fair_h + 0.9))
    H = fig.get_figheight()
    gs = fig.add_gridspec(2, 1, height_ratios=[top_h, fair_h], hspace=0.95 / ((top_h + fair_h) / 2),
                          left=0.05, right=0.985, top=1 - 1.25 / H, bottom=0.75 / H)
    gtop = gs[0].subgridspec(1, 3, wspace=0.3)
    gfair = gs[1].subgridspec(1, 4, width_ratios=[1.05, 1, 1, 1], wspace=0.08)

    def clean(a, title):
        a.set_title(title, loc="left", fontweight="bold", fontsize=11, pad=8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    # 1/4/5 ROC
    a = fig.add_subplot(gtop[0]); clean(a, "1, 4, 5  ROC")
    a.plot([0, 1], [0, 1], color=PAL["grid"], lw=1)
    ref = curves.get("reference")
    if ref:
        a.plot(ref[0], ref[1], color=PAL["muted"], lw=1.4, ls=":", label=f"Our reference  {ref[2]:.3f}")
    for key, col, ls, lab in [("primary", PAL["s1"], "-", "1 Primary"),
                              ("threshold_free", PAL["s2"], "-", "4 Threshold-free"),
                              ("matched", PAL["s3"], "--", "5 Age/sex matched")]:
        c = curves.get(key)
        if c:
            a.plot(c[0], c[1], color=col, lw=2, ls=ls, label=f"{lab}  {c[2]:.3f}")
    a.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.01))
    a.legend(frameon=False, fontsize=8.5, loc="lower right", title="AUROC", title_fontsize=8.5,
             alignment="left")

    # 2 dose-response
    a = fig.add_subplot(gtop[1]); clean(a, "2  Dose-response, everyone")
    a.plot([0, 1], [0, 1], color=PAL["grid"], lw=1)
    shown = [b for b in dose["bins"] if not b["low_n"]]
    if shown:
        bx = [b["eur_frac"] for b in shown]
        a.fill_between(bx, [b["lo"] for b in shown], [b["hi"] for b in shown],
                       color=PAL["s1"], alpha=0.18, lw=0)
        a.plot(bx, [b["mean_pEUR"] for b in shown], color=PAL["s1"], lw=2, marker="o", ms=6,
               mec=PAL["surf"], mew=1.5)
    a.set(xlabel="RFMix EUR ancestry fraction", ylabel="Model p(EUR), bin mean (95% CI)",
          xlim=(0, 1), ylim=(0, 1.01))
    adm = dose["admixed_spearman_rho"]
    a.text(0.03, 0.97, f"Spearman ρ  {dose['spearman_rho']:+.3f}  (n = {C(dose['n'])})\n"
           + (f"admixed only  {adm:+.3f}  (n = {C(dose['admixed_n'])})" if adm is not None
              else "admixed only: suppressed"),
           transform=a.transAxes, va="top", fontsize=8.5, color=PAL["ink2"])

    # 3 gate sweep
    a = fig.add_subplot(gtop[2]); clean(a, "3  AUROC vs ancestry gate")
    ok = [r for r in sweep if not r.get("suppressed")]
    a.axvline(res["gate"], color=PAL["grid"], lw=6, zorder=0)
    if ok:
        gx = [r["gate"] for r in ok]
        a.fill_between(gx, [r["auroc_ci95"][0] for r in ok], [r["auroc_ci95"][1] for r in ok],
                       color=PAL["s1"], alpha=0.18, lw=0)
        a.plot(gx, [r["auroc"] for r in ok], color=PAL["s1"], lw=2, marker="o", ms=6,
               mec=PAL["surf"], mew=1.5)
    a.set_xticks([r["gate"] for r in sweep])
    a.set_xticklabels([f"{r['gate']:g}\n{C(r['n_pos'])}/{C(r['n_neg'])}" for r in sweep], fontsize=8)
    a.set_xlim(max(r["gate"] for r in sweep) + 0.05, min(r["gate"] for r in sweep) - 0.05)
    a.set(xlabel="Gate (looser →)  ·  n non-EUR / n EUR", ylabel="AUROC (95% CI)")
    a.text(res["gate"], a.get_ylim()[1], " primary", fontsize=8, color=PAL["ink2"], va="top")

    # 6 fairness: label column + AUROC | TPR | FPR
    lab = fig.add_subplot(gfair[0]); lab.axis("off")
    lab.set_ylim(n_rows - 0.5, -1.6); lab.set_xlim(0, 1)
    lab.text(0, -1.4, "6  Fairness by subgroup (primary set)", fontweight="bold", fontsize=11,
             va="center")
    for yy, t in heads:
        lab.text(0, yy, t, fontweight="bold", fontsize=9.5, va="center")
    for r, yy in zip(rows, ypos):
        lab.text(0.04, yy, _group_label(r), fontsize=9, va="center", color=PAL["ink"])
        lab.text(0.98, yy, f"{C(r['n_pos'])} / {C(r['n_neg'])}", fontsize=8.5, va="center",
                 ha="right", color=PAL["muted"])
    if heads:
        lab.text(0.98, heads[0][0], "n non-EUR / EUR", fontsize=8, ha="right", va="center",
                 color=PAL["muted"])

    p = res["primary"]
    for j, (met, title, overall) in enumerate([("auroc", "AUROC", p["auroc"]),
                                               ("tpr", f"TPR (sensitivity) at {res['operating_threshold']:g}", p["sens"]),
                                               ("fpr", f"FPR (1 − specificity) at {res['operating_threshold']:g}", 1 - p["spec"])]):
        a = fig.add_subplot(gfair[j + 1])
        a.set_ylim(n_rows - 0.5, -1.6)
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)
        a.set_yticks([])
        a.text(0.0, -1.4, title, transform=a.get_yaxis_transform(), fontsize=9.5,
               fontweight="bold", va="center", color=PAL["ink2"])
        vals = []
        for r, yy in zip(rows, ypos):
            if yy in [h[0] for h in heads]:
                continue
            a.axhline(yy, color=PAL["grid"], lw=0.6, zorder=0)
            if r.get("suppressed"):
                if j == 0:
                    a.text(0.02, yy, "suppressed (n < min)", transform=a.get_yaxis_transform(),
                           fontsize=8, color=PAL["muted"], style="italic", va="center")
                continue
            if met not in r or r[met] is None or r[met] != r[met]:
                if j == 2:
                    a.text(0.02, yy, "same EUR group", transform=a.get_yaxis_transform(),
                           fontsize=8, color=PAL["muted"], style="italic", va="center")
                continue
            lo, hi = r[f"{met}_ci95"]
            col = PAL["s2"] if r["attribute"] == "ancestry_group" else PAL["s1"]
            a.plot([lo, hi], [yy, yy], color=col, lw=2, solid_capstyle="round")
            a.plot(r[met], yy, "o", color=col, ms=6.5, mec=PAL["surf"], mew=1.5, zorder=3)
            vals += [lo, hi]
        a.axvline(overall, color=PAL["ink2"], lw=0.9, ls=(0, (3, 3)), zorder=0)
        if vals:
            lo_, hi_ = min(vals + [overall]), max(vals + [overall])
            pad = max(0.02, 0.08 * (hi_ - lo_))
            if met == "auroc":
                a.set_xlim(min(0.5, lo_ - pad), max(min(1.0, hi_ + pad), 0.6))
        if met != "auroc":                     # rates: always the full 0-1 scale
            a.set_xlim(-0.02, 1.02); a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        a.tick_params(axis="x", labelsize=8.5)
        a.set_xlabel("dashed = overall primary", fontsize=8.5)

    fig.text(0.05, 1 - 0.38 / H, f"{run_name}: external validation summary", fontsize=15,
             fontweight="bold", va="top")
    fig.text(0.05, 1 - 0.78 / H,
             f"Primary (gate {res['gate']:.2f}): AUROC {p['auroc']:.3f} "
             f"[{p['auroc_ci95'][0]:.3f}, {p['auroc_ci95'][1]:.3f}]  ·  n = {C(p['n'])} "
             f"({C(int(round(p['pos_rate'] * p['n'])))} non-EUR)  ·  all intervals are bootstrap 95% CIs  ·  "
             f"groups with < {res['min_n']} per class suppressed  ·  counts < {min_cell} shown as <{min_cell}", fontsize=9.5,
             color=PAL["ink2"], va="top")
    fig.savefig(out_png, dpi=300)
    fig.savefig(Path(out_png).with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="the all_genotyped/ dir from `prepare`")
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
    ap.add_argument("--gate", type=float, default=None,
                    help="primary ancestry gate (default: the one used in prepare, 0.70)")
    ap.add_argument("--gates", default="0.5,0.6,0.7,0.8,0.9", help="gate-sweep values")
    ap.add_argument("--threshold", type=float, default=0.5, help="operating point for TPR/FPR")
    ap.add_argument("--bin-edges", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="score bins for the calibration table; fixed bins, so the returned histogram is an aggregate and not a distribution of people")
    ap.add_argument("--age-bands", default="40,55,70", help="age cut points (years)")
    ap.add_argument("--min-n", type=int, default=PV.MIN_PER_CLASS,
                    help="suppress any group/bin with fewer per class (privacy + stability)")
    ap.add_argument("--min-cell", type=int, default=PV.MIN_CELL,
                    help="counts of 1..min_cell-1 people are written as '<min_cell'")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="bootstrap resamples behind every confidence interval")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed, so every bootstrap and every split is reproducible")
    ap.add_argument("--ensemble-n-boot", type=int, default=200,
                    help="bootstrap replicates for the ensemble-size sweep (31 subsets each)")
    ap.add_argument("--repeats", default=None,
                    help="analysis 8 input: the all_genotyped_repeats/ dir from `prepare` "
                         "(default: the sibling of --input, when it exists)")
    ap.add_argument("--no-stability", action="store_true", help="skip analysis 8")
    ap.add_argument("--run-name", default="This cohort",
                    help="your cohort name, printed on the plates and in the tables")
    args = ap.parse_args()
    # Resolved here rather than required on the command line: one package, one
    # way to find the weights and the frozen discovery tables.
    args.models_root = W.resolve("ancestry_folds", args.models_root)
    args.ssl_ckpt = W.resolve("ssl_backbone", args.ssl_ckpt)
    args.reference = args.reference or str(L.ref_dir() / "reference_roc_pr.json")

    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((inp / "info.json").read_text())
    anc, eur_label = info["ancestries"], info["eur_label"]
    noneur = [a for a in anc if a != eur_label]
    gate = args.gate if args.gate is not None else float(info["threshold"])
    cov = pd.read_csv(inp / "covariates.csv")
    X = np.load(inp / "X.npy", mmap_mode="r")
    if len(X) != len(cov):
        raise ValueError(f"X has {len(X)} rows but covariates.csv has {len(cov)}")
    rng = np.random.default_rng(args.seed)

    props = cov[anc].to_numpy(float)
    dom = np.asarray(anc, object)[props.argmax(1)]
    dmax = props.max(1)
    eur = cov[eur_label].to_numpy(float)

    # refuse BEFORE inference if the primary set is too small to report on
    yy = dom[dmax >= gate] != eur_label
    if yy.sum() < args.min_n or (~yy).sum() < args.min_n:
        sys.exit(f"[validate] REFUSED: the primary set has fewer than {args.min_n} non-EUR or EUR "
                 f"participants (gate {gate:.2f}). Any metric would be unreliable and potentially "
                 "identifying, so nothing was written. Check participant_flow.csv.")
    device = infer_lib.pick_device(args.device)
    print(f"[info] device={device}  N={len(X):,}  primary gate={gate}", flush=True)
    models, ckpts = infer_lib.load_fold_models(args.models_root, args.ssl_ckpt, device)
    P, Pf = infer_lib.ensemble_predict(models, X, device, args.batch_size)  # P(non-EUR), per fold
    P = P.astype(float); Pf = Pf.astype(float)
    ref = json.loads(Path(args.reference).read_text())
    res = {"run_name": args.run_name, "n_models": len(models), "checkpoints": ckpts,
           "gate": gate, "operating_threshold": args.threshold, "min_n": args.min_n,
           "n_boot": args.n_boot, "label_convention": "y=1 -> non-EUR; score = P(non-EUR)",
           "n_total": int(len(P)),
           "n_by_dominant_ancestry": {str(k): int(v) for k, v in
                                      pd.Series(np.where(dmax >= gate, dom, "ADMIXED")).value_counts().items()},
           "reference_overall": ref["per_cohort"]["ALL"]}
    curves = {}
    from sklearn.metrics import roc_curve

    def curve(y, s):
        f, t, _ = roc_curve(y, s)
        k = np.unique(np.linspace(0, len(f) - 1, min(len(f), 200)).astype(int))
        return f[k].tolist(), t[k].tolist(), auc(y, s)

    def labelled(m):
        return (dom[m] != eur_label).astype(int), P[m]

    # 1 primary
    m1 = dmax >= gate
    y1, s1 = labelled(m1)
    print(f"[1] primary: n={m1.sum():,} (non-EUR {y1.sum():,})", flush=True)
    res["primary"] = evaluate(y1, s1, args.threshold, args.n_boot)
    curves["primary"] = curve(y1, s1)
    rc = ref.get("curves", {}).get("ALL", {}).get("roc")
    if rc:
        curves["reference"] = (rc["fpr"], rc["tpr"], ref["per_cohort"]["ALL"]["auroc_pooled"])
    overlay_figure(y1, s1, ref, str(out / "primary_roc_pr_overlay.png"), thr=args.threshold,
                   run_name=f"{args.run_name} (primary, gate {gate:.2f})", min_bin=args.min_n)

    # 2 continuous dose-response on everyone
    print("[2] dose-response on everyone", flush=True)
    peur = 1.0 - P
    edges = [float(e) for e in args.bin_edges.split(",")]
    bins = fixed_bins(eur, peur, edges, min_n=args.min_n, n_boot=args.n_boot)
    for b in bins:                                  # suppress small bins entirely
        if b["low_n"]:
            for k in ("eur_frac", "mean_pEUR", "lo", "hi"):
                b[k] = None
    adm = ~m1
    rho, prho = spearmanr(eur, peur)
    res["continuous"] = {
        "n": int(len(eur)), "spearman_rho": float(rho), "spearman_p": float(prho),
        "pearson_r": float(pearsonr(eur, peur)[0]),
        "admixed_n": int(adm.sum()),
        "admixed_spearman_rho": float(spearmanr(eur[adm], peur[adm])[0]) if adm.sum() >= args.min_n else None,
        "mean_abs_calibration_error": float(np.mean([abs(b["mean_pEUR"] - b["eur_frac"])
                                                     for b in bins if not b["low_n"]] or [np.nan])),
        "bins": bins}

    # 3 gate sweep
    print("[3] gate sweep", flush=True)
    sweep = []
    for g in sorted(float(x) for x in args.gates.split(",")):
        m = dmax >= g; y, s = labelled(m)
        r = {"gate": g, "n": int(m.sum()), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
             "n_excluded": int((~m).sum())}
        if r["n_pos"] < args.min_n or r["n_neg"] < args.min_n:
            r["suppressed"] = True
        else:
            r.update({"suppressed": False, "auroc": auc(y, s),
                      "auroc_ci95": boot_auc(y, s, args.n_boot, rng)})
        sweep.append(r)
    res["gate_sweep"] = sweep

    # 4 threshold-free binary
    print("[4] threshold-free binary (EUR fraction < 0.5)", flush=True)
    y4 = (eur < 0.5).astype(int)
    if min(y4.sum(), (1 - y4).sum()) >= args.min_n:
        res["threshold_free"] = evaluate(y4, P, args.threshold, args.n_boot)
        curves["threshold_free"] = curve(y4, P)
    else:
        res["threshold_free"] = {"suppressed": True}

    # 5 matched
    m5 = cov["in_matched"].to_numpy(int) == 1
    y5, s5 = labelled(m5)
    print(f"[5] matched: n={m5.sum():,}", flush=True)
    if m5.sum() and min(y5.sum(), (1 - y5).sum()) >= args.min_n:
        res["matched"] = evaluate(y5, s5, args.threshold, args.n_boot)
        curves["matched"] = curve(y5, s5)
    else:
        res["matched"] = {"suppressed": True, "n": int(m5.sum())}

    # 6 fairness on the primary set
    print("[6] fairness", flush=True)
    c1 = cov[m1].reset_index(drop=True)
    cuts = [float(a) for a in args.age_bands.split(",")]
    labels = ([f"<{cuts[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(cuts, cuts[1:])]
              + [f"≥{cuts[-1]:g}"])
    age = pd.to_numeric(c1["age"], errors="coerce")
    band = pd.cut(age, [-np.inf, *cuts, np.inf], right=False, labels=labels).astype(object)
    sex = c1["sex"].where(c1["sex"].isin(["MALE", "FEMALE"]))
    attrs = {"sex": sex.to_numpy(object), "age_band": band.where(age.notna()).to_numpy(object),
             "batch": c1["batch"].astype(str).to_numpy(object)}
    if c1["round"].nunique() > 1:
        attrs["round"] = c1["round"].astype(str).to_numpy(object)
    fair = subgroup_table(y1, s1, attrs, args.threshold, args.n_boot, args.min_n, rng)
    fair += ancestry_subgroups(dom[m1], y1.astype(bool), s1, noneur, args.threshold,
                               args.n_boot, args.min_n, rng)
    order = {"sex": 0, "age_band": 1, "batch": 2, "round": 3, "ancestry_group": 4}
    agelab = {l: i for i, l in enumerate(labels)}
    fair.sort(key=lambda r: (order[r["attribute"]], agelab.get(r["group"], 0), r["group"]))
    res["fairness"] = fair

    # 7 ensemble agreement + size sweep on the primary set (per-fold predictions already in hand)
    print("[7] ensemble agreement + size sweep", flush=True)
    est, votes, A = ensemble_analysis(Pf[m1], y1, args.ensemble_n_boot, args.run_name, ckpts, device)
    (out / "ensemble_stats.json").write_text(json.dumps(mask(nan2none(est), args.min_cell), indent=2))
    ensemble_figs(est, Pf[m1], votes, A, str(out), Pf.shape[1], min_cell=args.min_cell)
    res["ensemble"] = {"agreement": {k: v for k, v in est["agreement"].items()
                                     if k != "pairwise_agreement_matrix"},
                       "performance": est.get("performance")}

    # 8 intrapatient stability, on the repeat pairs (second inference pass)
    rep_dir = Path(args.repeats) if args.repeats else inp.parent / "all_genotyped_repeats"
    res["stability"] = {"available": False,
                        "reason": "--no-stability" if args.no_stability else
                                  f"no repeats directory at {rep_dir} (run `prepare` with "
                                  "--patient-id-col so a person's ECGs group together)"}
    if not args.no_stability and (rep_dir / "X.npy").exists() and (rep_dir / "covariates.csv").exists():
        covr = pd.read_csv(rep_dir / "covariates.csv")
        Xr = np.load(rep_dir / "X.npy", mmap_mode="r")
        if len(Xr) != len(covr):
            print(f"[8] SKIPPED: {rep_dir}/X.npy has {len(Xr)} rows but covariates.csv has "
                  f"{len(covr)}", flush=True)
            res["stability"] = {"available": False, "reason": "repeats X/covariates length mismatch"}
        elif covr["person"].nunique() < args.min_n:
            print(f"[8] SKIPPED: only {covr['person'].nunique()} persons with repeats "
                  f"(< --min-n {args.min_n})", flush=True)
            res["stability"] = {"available": False,
                                "reason": f"fewer than {args.min_n} persons with two ECGs"}
        else:
            print(f"[8] stability: {len(Xr):,} repeat ECGs from {covr['person'].nunique():,} "
                  "persons", flush=True)
            Pr, _ = infer_lib.ensemble_predict(models, Xr, device, args.batch_size)
            missing = [a for a in anc if a not in covr.columns]
            if missing:
                raise ValueError(f"{rep_dir}/covariates.csv is missing ancestry columns {missing}")
            pr = covr[anc].to_numpy(float)
            domr = np.where(pr.max(1) >= gate, np.asarray(anc, object)[pr.argmax(1)], "ADMIXED")
            st = stab.analyse(covr, np.asarray(Pr, float), domr, eur_label, args.threshold,
                              args.min_n, args.min_cell, args.n_boot, args.seed)
            st["available"] = True
            (out / "stability.json").write_text(json.dumps(mask(nan2none(st), args.min_cell), indent=2))
            stab.to_frame(st, args.min_cell).to_csv(out / "stability.csv", index=False)
            stab.figure(st, str(out / "stability.png"), args.run_name, args.min_cell)
            res["stability"] = {k: v for k, v in st.items()
                                if k not in ("score_grid", "delta_histogram")}

    # 9 plot data: every curve behind every figure, aggregate only
    pdata = {"label_convention": "y=1 -> non-EUR; score = P(non-EUR)",
             "rules": f"curves <= 200 points, no thresholds; calibration bins >= {args.min_n} people; "
                      f"counts 1..{args.min_cell - 1} written as '<{args.min_cell}'",
             "reference_ours": {"per_cohort_ALL": ref["per_cohort"]["ALL"],
                                "curves_ALL": ref.get("curves", {}).get("ALL")},
             "primary": plot_block(y1, s1, args.min_n, args.min_cell)}
    if not res["threshold_free"].get("suppressed"):
        pdata["threshold_free"] = plot_block(y4, P, args.min_n, args.min_cell)
    if not res["matched"].get("suppressed"):
        pdata["matched"] = plot_block(y5, s5, args.min_n, args.min_cell)
    pdata["dose_response_bins"] = res["continuous"]["bins"]
    pdata["gate_sweep"] = sweep
    pdata["ensemble_size_sweep"] = (est.get("performance") or {}).get("ensemble_size_sweep")
    pdata["ensemble_vote_count_dist"] = est["agreement"]["vote_count_dist"]
    pdata["ensemble_pairwise_agreement"] = est["agreement"]["pairwise_agreement_matrix"]
    froc = {}
    ya = y1.astype(bool)
    for a, lab in attrs.items():
        lab = np.asarray(lab, object)
        for r in fair:
            if r["attribute"] == a and not r.get("suppressed"):
                m = lab == r["group"]
                if a == "batch" or a == "round":
                    m = lab == str(r["group"])
                fx, ty = _thin(*roc_curve(ya[m], s1[m])[:2])
                froc[f"{a}:{r['group']}"] = {"fpr": fx, "tpr": ty, "auroc": r["auroc"]}
    dom1 = dom[m1]
    for r in fair:
        if r["attribute"] == "ancestry_group" and not r.get("suppressed"):
            m = (dom1 == r["group"]) | ~ya
            fx, ty = _thin(*roc_curve(ya[m], s1[m])[:2])
            froc[f"ancestry_group:{r['group']} vs EUR"] = {"fpr": fx, "tpr": ty, "auroc": r["auroc"]}
    pdata["fairness_roc"] = froc
    if res["stability"].get("available"):
        pdata["stability"] = {k: st[k] for k in ("overall", "by_interval", "by_n_ecgs",
                                                 "score_grid", "delta_histogram") if k in st}

    # ------------------------------------------------------------------ write
    K = args.min_cell
    (out / "plot_data.json").write_text(json.dumps(mask(nan2none(pdata), K), indent=2))
    (out / "validate.json").write_text(json.dumps(mask(nan2none(res), K), indent=2))
    pd.DataFrame([{k: (cell(v, K) if k == "n" else v) for k, v in b.items() if not isinstance(v, list)}
                  for b in bins]).to_csv(out / "dose_response.csv", index=False)

    def flat(r):
        o = {}
        for k, v in r.items():
            if isinstance(v, list) and len(v) == 2:
                o[k + "_lo"], o[k + "_hi"] = v
            elif not isinstance(v, (dict, list)):
                o[k] = v
        return o
    pd.DataFrame([flat(mask(r, K)) for r in fair]).to_csv(out / "fairness.csv", index=False)
    rows = []
    for key, lab in [("primary", f"1 primary (gate {gate:.2f})"),
                     ("threshold_free", "4 threshold-free (EUR<0.5)"), ("matched", "5 age/sex matched")]:
        r = res[key]
        if not r.get("suppressed"):
            rows.append({"analysis": lab, "n": r["n"], "pos_rate": r["pos_rate"],
                         "auroc": r["auroc"], "auroc_lo": r["auroc_ci95"][0],
                         "auroc_hi": r["auroc_ci95"][1], "auprc": r["auprc"], "brier": r["brier"],
                         "ece": r["ece"], "sens": r["sens"], "spec": r["spec"]})
    for r in sweep:
        if not r["suppressed"]:
            rows.append({"analysis": f"3 gate {r['gate']:.2f}", "n": r["n"],
                         "pos_rate": r["n_pos"] / r["n"], "auroc": r["auroc"],
                         "auroc_lo": r["auroc_ci95"][0], "auroc_hi": r["auroc_ci95"][1]})
    rows.append({"analysis": "2 continuous (Spearman rho, everyone)", "n": len(eur),
                 "auroc": None, "spearman_rho": res["continuous"]["spearman_rho"]})
    for r in (res["stability"].get("overall") or []):
        if r["scope"] == "all" and not r.get("suppressed"):
            rows.append({"analysis": f"8 stability ICC ({r['pairing']}, persons)",
                         "n": r["n_persons"], "icc": r["icc"], "icc_lo": r["icc_ci95"][0],
                         "icc_hi": r["icc_ci95"][1]})
    pd.DataFrame([{k: (cell(v, K) if k == "n" else v) for k, v in r.items()} for r in rows]).to_csv(
        out / "validate_summary.csv", index=False)
    summary_figure(res, curves, res["continuous"], sweep, fair,
                   str(out / "validation_summary.png"), args.run_name, K)

    p = res["primary"]
    print(f"\n[done] 1 primary AUROC={p['auroc']:.3f} [{p['auroc_ci95'][0]:.3f},{p['auroc_ci95'][1]:.3f}] "
          f"n={p['n']:,} | 2 Spearman rho={rho:+.3f} (n={len(eur):,})")
    for r in sweep:
        print(f"[done] 3 gate {r['gate']:.2f}: " + ("suppressed" if r["suppressed"] else
              f"AUROC={r['auroc']:.3f} (n_pos={r['n_pos']:,}, n_neg={r['n_neg']:,})"))
    sup = sum(r.get("suppressed", False) for r in fair)
    print(f"[done] 6 fairness: {len(fair) - sup} groups reported, {sup} suppressed (< {args.min_n}/class)")
    ag = est["agreement"]
    print(f"[done] 7 ensemble: {ag['unanimous_frac'] * 100:.0f}% unanimous, Fleiss kappa "
          f"{ag['fleiss_kappa']:.2f}" + (
              f"; AUROC single {est['performance']['single_fold_mean']['AUROC']:.3f} -> "
              f"{len(models)}-model {est['performance'][f'ensemble_{len(models)}_models']['AUROC']:.3f}"
              if est.get("performance") else ""))
    for r in (res["stability"].get("overall") or []):
        if r["scope"] == "all" and not r.get("suppressed"):
            print(f"[done] 8 stability ({r['pairing']}): ICC={r['icc']:.3f} "
                  f"[{r['icc_ci95'][0]:.3f},{r['icc_ci95'][1]:.3f}] on {r['n_persons']:,} persons, "
                  f"within-person SD {r['sd_within']:.3f}, call flips {r['flip_rate'] * 100:.1f}%")
    if not res["stability"].get("available"):
        print(f"[done] 8 stability: not run ({res['stability'].get('reason')})")
    print(f"[done] wrote {out}/validate.json, validate_summary.csv, fairness.csv, dose_response.csv, "
          "validation_summary.png/.pdf, primary_roc_pr_overlay.png/.pdf, ensemble_stats.json, "
          "ensemble_agreement.png/.pdf, ensemble_size_sweep.png/.pdf, plot_data.json"
          + (", stability.json, stability.csv, stability.png/.pdf" if res["stability"].get("available") else ""))


if __name__ == "__main__":
    main()
