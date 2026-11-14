"""Analysis 8 -- intrapatient stability: does the model give one person the same answer twice?

The score is meant to read an inherited property, so two recordings of the same person should
agree however far apart they were taken. This is the reproducibility arm of the external
validation and it needs repeated ECGs, which `prepare` writes to all_genotyped_repeats/ (two
recordings per person; see `prepare --help`, --patient-id-col and --date-col).

Reported per pairing (first_last, closest) and per scope (everyone, EUR, non-EUR):
  ICC(2,1)        two-way random effects, single measurement, absolute agreement
                  (Shrout & Fleiss), with a percentile bootstrap CI over persons
  sd_within       mean within-person SD of the score; sd_between SD of the person means;
                  snr = sd_between / sd_within
  mean |delta|    average absolute score change between the two recordings
  flip rate       share of persons whose call at the operating threshold changes
  strata          by the interval between the recordings, and by how many ECGs the person has

SECURE-ENCLAVE SAFE: every output is an aggregate over >= --min-n persons; the score grid and
the delta histogram are binned counts with cells of 1..--min-cell-1 masked. No person index,
no per-person score and no date ever leaves this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ancestry.privacy import cell

SCOPES = ("all", "EUR", "non-EUR")


# --------------------------------------------------------------------------- ICC
def icc21_matrix(Y):
    """ICC(2,1) from an (n persons, k repeats) matrix. NaN when it is not estimable."""
    Y = np.asarray(Y, float)
    n, k = Y.shape
    if n < 3 or k < 2:
        return float("nan")
    gm = Y.mean()
    ms_r = k * ((Y.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((Y.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    resid = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + gm
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    den = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / den) if den else float("nan")


def icc_ci(Y, n_boot, rng):
    """Percentile bootstrap CI, resampling persons -- the independent unit."""
    Y = np.asarray(Y, float)
    if len(Y) < 3:
        return [None, None]
    b = np.array([icc21_matrix(Y[rng.integers(0, len(Y), len(Y))]) for _ in range(n_boot)])
    b = b[np.isfinite(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if b.size else [None, None]


def _metrics(Y, thr, n_boot, rng, gaps=None):
    """Every stability number for one (n, 2) matrix of paired scores."""
    Y = np.asarray(Y, float)
    within = float(np.mean(Y.std(axis=1, ddof=1)))
    between = float(Y.mean(axis=1).std(ddof=1))
    out = {"n_persons": int(len(Y)), "icc": icc21_matrix(Y), "icc_ci95": icc_ci(Y, n_boot, rng),
           "sd_within": within, "sd_between": between,
           "snr": float(between / within) if within else None,
           "mean_abs_delta": float(np.abs(Y[:, 1] - Y[:, 0]).mean()),
           "mean_delta": float((Y[:, 1] - Y[:, 0]).mean()),
           "flip_rate": float(((Y[:, 0] >= thr) != (Y[:, 1] >= thr)).mean())}
    if gaps is not None:
        g = pd.to_numeric(pd.Series(gaps), errors="coerce").dropna()
        out["median_gap_days"] = float(g.median()) if len(g) else None
        out["median_gap_years"] = round(float(g.median()) / 365.25, 2) if len(g) else None
    return out


# --------------------------------------------------------------------------- shaping
def _pivot(df):
    """(persons x 2) score matrix, the per-person gap and the person index, for persons whose
    pair is complete. The index is the within-file person number, never a site id."""
    piv = df.pivot_table(index="person", columns="rep", values="score", aggfunc="first")
    if not {0, 1} <= set(piv.columns):
        return np.empty((0, 2)), pd.Series(dtype=float), pd.Index([])
    piv = piv[[0, 1]].dropna()
    one = df.drop_duplicates("person").set_index("person")
    gap = pd.to_numeric(one["gap_days"], errors="coerce").reindex(piv.index)
    return piv.to_numpy(float), gap, piv.index


def _grid(Y, min_cell, nbin=6):
    """Binned 2D histogram of (first, second) score -- the scatter plot, made disclosure-safe.
    Six bins a side, not twenty: a fine grid spreads people so thin that every cell falls under
    the disclosure limit and the whole panel comes back blank."""
    edges = np.linspace(0, 1, nbin + 1)
    H, _, _ = np.histogram2d(Y[:, 0], Y[:, 1], bins=[edges, edges])
    return {"bin_edges": [round(float(e), 3) for e in edges],
            "counts": [[cell(int(v), min_cell) for v in row] for row in H]}


def _delta_hist(Y, min_cell, nbin=20):
    edges = np.linspace(-1, 1, nbin + 1)
    H, _ = np.histogram(Y[:, 1] - Y[:, 0], bins=edges)
    return {"bin_edges": [round(float(e), 3) for e in edges],
            "counts": [cell(int(v), min_cell) for v in H]}


# --------------------------------------------------------------------------- main entry
def analyse(cov, scores, dom, eur_label, thr, min_n, min_cell, n_boot, seed):
    """cov: covariates.csv of all_genotyped_repeats/ ; scores: P(non-EUR) of the same rows;
    dom: dominant-ancestry label per row (ADMIXED allowed, counted only in scope 'all')."""
    rng = np.random.default_rng(seed)
    d = cov.copy()
    d["score"] = np.asarray(scores, float)
    d["dom"] = np.asarray(dom, object)
    res = {"n_rows": int(len(d)), "n_persons": int(d["person"].nunique()),
           "pairings": sorted(d["pairing"].unique().tolist()),
           "operating_threshold": thr, "min_n": min_n, "n_boot": n_boot,
           "icc_definition": "ICC(2,1), two-way random effects, single measurement, "
                             "absolute agreement (Shrout & Fleiss); CI = percentile bootstrap "
                             "over persons",
           "overall": [], "by_interval": [], "by_n_ecgs": []}

    for pairing, dp in d.groupby("pairing"):
        for scope in SCOPES:
            sub = dp if scope == "all" else dp[dp["dom"] == eur_label] if scope == "EUR" else \
                dp[(dp["dom"] != eur_label) & (dp["dom"] != "ADMIXED")]
            Y, gap, _ = _pivot(sub)
            row = {"pairing": pairing, "scope": scope, "n_persons": int(len(Y))}
            if len(Y) < min_n:
                row["suppressed"] = True
            else:
                row["suppressed"] = False
                row.update(_metrics(Y, thr, n_boot, rng, gaps=gap))
            res["overall"].append(row)

        # strata by the interval between the two recordings (quintiles of the gap)
        Y, gap, idx = _pivot(dp)
        if len(Y) >= min_n and gap.notna().any():
            g = gap.to_numpy(float)
            ok = np.isfinite(g)
            edges = np.unique(np.nanpercentile(g[ok], [0, 20, 40, 60, 80, 100]))
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                m = ok & (g >= lo) & ((g <= hi) if i == len(edges) - 2 else (g < hi))
                row = {"pairing": pairing, "days_lo": float(lo), "days_hi": float(hi),
                       "years_lo": round(float(lo) / 365.25, 2), "years_hi": round(float(hi) / 365.25, 2),
                       "n_persons": int(m.sum())}
                if m.sum() < min_n:
                    row["suppressed"] = True
                else:
                    row["suppressed"] = False
                    row.update(_metrics(Y[m], thr, n_boot, rng, gaps=g[m]))
                res["by_interval"].append(row)

        # strata by how many ECGs the person has in the cohort
        n_ecgs = (pd.to_numeric(dp.drop_duplicates("person").set_index("person")["n_ecgs"],
                                errors="coerce").reindex(idx))
        if len(Y) >= min_n and n_ecgs.notna().any():
            k = pd.to_numeric(n_ecgs, errors="coerce").to_numpy(float)
            for lab, lo, hi in [("2", 2, 2), ("3-4", 3, 4), ("5-9", 5, 9), ("10+", 10, np.inf)]:
                m = np.isfinite(k) & (k >= lo) & (k <= hi)
                if not m.sum():
                    continue
                row = {"pairing": pairing, "n_ecgs": lab, "n_persons": int(m.sum())}
                if m.sum() < min_n:
                    row["suppressed"] = True
                else:
                    row["suppressed"] = False
                    row.update(_metrics(Y[m], thr, n_boot, rng))
                res["by_n_ecgs"].append(row)

    counts = {r["pairing"]: r["n_persons"] for r in res["overall"] if r["scope"] == "all"}
    head = max(counts, key=counts.get) if counts else None
    res["headline_pairing"] = head
    if head is not None:
        Y, _, _ = _pivot(d[d["pairing"] == head])
        if len(Y) >= min_n:
            res["score_grid"] = {"pairing": head, **_grid(Y, min_cell)}
            res["delta_histogram"] = {"pairing": head, **_delta_hist(Y, min_cell)}
    return res


def to_frame(res, min_cell=10):
    """One flat table of every stability row, for stability.csv. Person counts are masked the
    same way as everywhere else: 1..min_cell-1 becomes '<min_cell'."""
    rows = []
    for key, extra in [("overall", "scope"), ("by_interval", "years_lo"), ("by_n_ecgs", "n_ecgs")]:
        for r in res.get(key, []):
            o = {"table": key, "pairing": r.get("pairing"), "stratum": r.get(extra)}
            if key == "by_interval" and not r.get("suppressed", False):
                o["stratum"] = f"{r['years_lo']}-{r['years_hi']} y"
            elif key == "by_interval":
                o["stratum"] = f"{r.get('years_lo')}-{r.get('years_hi')} y"
            for c in ("n_persons", "suppressed", "icc", "sd_within", "sd_between", "snr",
                      "mean_abs_delta", "flip_rate", "median_gap_years"):
                o[c] = r.get(c)
            ci = r.get("icc_ci95") or [None, None]
            o["icc_lo"], o["icc_hi"] = ci[0], ci[1]
            o["n_persons"] = cell(o.get("n_persons"), min_cell)
            rows.append(o)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figure
def figure(res, out_png, run_name, min_cell=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    PAL = {"s1": "#2a78d6", "s2": "#eb6834", "ink": "#0b0b0b", "ink2": "#52514e",
           "muted": "#8a8984", "grid": "#e4e3df", "surf": "#fcfcfb"}
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9.5, "font.family": "DejaVu Sans",
                         "axes.edgecolor": PAL["muted"], "axes.labelcolor": PAL["ink2"],
                         "xtick.color": PAL["ink2"], "ytick.color": PAL["ink2"],
                         "axes.linewidth": 0.8, "figure.facecolor": PAL["surf"],
                         "axes.facecolor": PAL["surf"], "savefig.facecolor": PAL["surf"]})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))

    def clean(a, t):
        a.set_title(t, loc="left", fontweight="bold", fontsize=11, pad=8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    # a: ICC by pairing and scope
    a = axes[0, 0]; clean(a, "a  ICC(2,1) by pairing and group")
    rows = [r for r in res["overall"]]
    y = np.arange(len(rows))[::-1]
    vals = [v for r in rows if not r.get("suppressed")
            for v in ([r["icc"]] + list(r.get("icc_ci95") or []))if v is not None]
    xlo = min(-0.05, min(vals) - 0.05) if vals else -0.05
    for r, yy in zip(rows, y):
        lab = f"{r['pairing']}  ·  {r['scope']}"
        if r.get("suppressed"):
            a.text(0.015, yy, f"{lab}: suppressed (n < {res['min_n']})", fontsize=8.5,
                   color=PAL["muted"], style="italic", va="center",
                   transform=a.get_yaxis_transform())
            continue
        lo, hi = r.get("icc_ci95") or [None, None]
        col = PAL["s1"] if r["scope"] == "all" else PAL["s2"]
        if lo is not None:
            a.plot([lo, hi], [yy, yy], color=col, lw=2, solid_capstyle="round")
        a.plot(r["icc"], yy, "o", color=col, ms=7, mec=PAL["surf"], mew=1.5, zorder=3)
        a.text(0.015, yy + 0.32, f"{lab}   n = {cell(r['n_persons'], min_cell)}", fontsize=8.5,
               color=PAL["ink2"], va="bottom", transform=a.get_yaxis_transform())
    a.set_yticks([]); a.set_xlim(xlo, 1); a.set_xlabel("ICC(2,1) with 95% CI")
    a.axvline(0.75, color=PAL["grid"], lw=1.2, ls=(0, (3, 3)))
    a.text(0.75, -0.9, " 0.75 = good agreement", fontsize=8, color=PAL["muted"], va="center")
    a.set_ylim(-1.2, len(rows) - 0.3)

    # b: ICC against the interval
    a = axes[0, 1]; clean(a, "b  ICC against the interval between the recordings")
    any_b = False
    for i, pairing in enumerate(res["pairings"]):
        ok = [r for r in res["by_interval"] if r["pairing"] == pairing and not r.get("suppressed")]
        if not ok:
            continue
        any_b = True
        x = [(r["years_lo"] + r["years_hi"]) / 2 for r in ok]
        col = [PAL["s1"], PAL["s2"]][i % 2]
        lo = [(r.get("icc_ci95") or [np.nan, np.nan])[0] for r in ok]
        hi = [(r.get("icc_ci95") or [np.nan, np.nan])[1] for r in ok]
        a.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
        a.plot(x, [r["icc"] for r in ok], color=col, lw=2, marker="o", ms=6, mec=PAL["surf"],
               mew=1.5, label=pairing)
        for xx, r in zip(x, ok):
            a.annotate(f"{cell(r['n_persons'], min_cell)}", (xx, r["icc"]), textcoords="offset points",
                       xytext=(0, -14), ha="center", fontsize=7.5, color=PAL["muted"])
    if any_b:
        a.legend(frameon=False, fontsize=9, title="pairing", title_fontsize=9, alignment="left")
        blo = min([(r.get("icc_ci95") or [0])[0] or 0 for r in res["by_interval"]
                   if not r.get("suppressed")] + [0])
        a.set(xlabel="Years between the two recordings (stratum midpoint; n below each point)",
              ylabel="ICC(2,1)", ylim=(min(-0.05, blo - 0.05), 1))
    else:
        a.text(0.5, 0.5, "no dates supplied:\nrun `prepare --date-col ...` for this panel",
               ha="center", va="center", fontsize=10, color=PAL["muted"], transform=a.transAxes)
        a.set_xticks([]); a.set_yticks([])

    # c: binned agreement grid
    a = axes[1, 0]
    g = res.get("score_grid")
    clean(a, f"c  First vs second recording ({g['pairing'] if g else 'n/a'})")
    if g and any(not isinstance(v, str) and v > 0 for row in g["counts"] for v in row):
        H = np.array([[0 if isinstance(v, str) else v for v in row] for row in g["counts"]], float)
        e = g["bin_edges"]
        a.pcolormesh(e, e, H.T, cmap="Blues", norm=LogNorm(vmin=max(H.min(), 1), vmax=max(H.max(), 2)),
                     shading="flat")
        a.plot([0, 1], [0, 1], color=PAL["muted"], lw=1, ls=(0, (3, 3)))
        a.set(xlabel="P(non-EUR), first recording", ylabel="P(non-EUR), second recording",
              xlim=(0, 1), ylim=(0, 1))
        a.text(0.03, 0.95, f"cells with < {min_cell} people are blank", transform=a.transAxes,
               fontsize=8, color=PAL["muted"], va="top")
    else:
        a.text(0.5, 0.5, f"every cell holds fewer than {min_cell} people:\nnothing can be shown",
               ha="center", va="center", color=PAL["muted"], transform=a.transAxes)
        a.set_xticks([]); a.set_yticks([])

    # d: change distribution
    a = axes[1, 1]; clean(a, "d  Change in score between the two recordings")
    h = res.get("delta_histogram")
    if h and any(not isinstance(v, str) and v > 0 for v in h["counts"]):
        e = np.array(h["bin_edges"], float)
        c = np.array([0 if isinstance(v, str) else v for v in h["counts"]], float)
        a.bar((e[:-1] + e[1:]) / 2, c, width=np.diff(e), color=PAL["s1"], alpha=0.85)
        a.axvline(0, color=PAL["muted"], lw=1)
        a.set(xlabel="P(non-EUR) second − first", ylabel=f"Persons (bars < {min_cell} shown as 0)")
        o = next((r for r in res["overall"] if r["pairing"] == h["pairing"] and r["scope"] == "all"
                  and not r.get("suppressed")), None)
        lo = min([i for i, v in enumerate(c) if v > 0] or [0])
        hi = max([i for i, v in enumerate(c) if v > 0] or [len(c) - 1])
        a.set_xlim(max(-1.0, e[lo] - 0.15), min(1.0, e[hi + 1] + 0.15))
        if o:
            a.set_ylim(0, max(c) * 1.42)
            a.text(0.03, 0.97, f"mean |change| {o['mean_abs_delta']:.3f}\n"
                               f"call changes in {o['flip_rate'] * 100:.1f}% of people\n"
                               f"within-person SD {o['sd_within']:.3f}  ·  between {o['sd_between']:.3f}",
                   transform=a.transAxes, va="top", fontsize=9, color=PAL["ink2"],
                   bbox=dict(facecolor=PAL["surf"], edgecolor="none", alpha=0.9, pad=3))
    else:
        a.text(0.5, 0.5, f"every bin holds fewer than {min_cell} people:\nnothing can be shown",
               ha="center", va="center", color=PAL["muted"], transform=a.transAxes)
        a.set_xticks([]); a.set_yticks([])

    head = res.get("headline_pairing")
    o = next((r for r in res["overall"] if r["scope"] == "all" and not r.get("suppressed")
              and (head is None or r["pairing"] == head)), None)
    head = (f"ICC {o['icc']:.3f} [{(o.get('icc_ci95') or [float('nan')])[0]:.3f}, "
            f"{(o.get('icc_ci95') or [0, float('nan')])[1]:.3f}] on {cell(o['n_persons'], min_cell)} "
            f"persons ({o['pairing']})" if o else "all strata suppressed")
    fig.suptitle(f"{run_name}: analysis 8, intrapatient stability — {head}", fontsize=13,
                 fontweight="bold", x=0.015, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_png, dpi=300)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    plt.close(fig)
