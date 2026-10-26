"""Table 1 for the DeepECG-HCM abstract.

Reads cohort_3class_with_tte.csv and produces, for each variable:
  - overall summary
  - by 3-class group (Other / HCM_no_latent / Latent_pos) + p-value
  - by split (train / val / test) + p-value

Tests
  - continuous (age, MLVWT, LVEDD, LVEF, LAVI, peak subaortic gradient, MR grade ordinal):
        Kruskal-Wallis H-test; report median [IQR]
  - categorical / binary (sex, LVEF<50, SAM, LAVI>34):
        chi-square (Fisher's exact if any expected cell < 5)

Source for the TTE-style variables: stress-echo own rest measurements
(`rest_*`), which have the best coverage and zero temporal mismatch.
Nearest-external-TTE coverage and median |delta| are reported as a footer.

Output
  outputs/table1/table1.csv
  outputs/table1/table1.md  (markdown for copy/paste into the abstract draft)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

CSV = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class_with_tte.csv"
OUT = Path("/volume/DeepECG_Docker/outputs/table1")
OUT.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ["Other", "HCM_no_latent", "Latent_pos"]
SPLIT_ORDER = ["train", "val", "test"]

CONTINUOUS = [
    ("age_years", "Age, years"),
    ("rest_mlvwt_mm", "MLVWT, mm"),
    ("rest_lvedd_mm", "LVEDD, mm"),
    ("rest_lvef_pct", "LVEF, %"),
    ("rest_lavi_ml_m2", "LAVI, mL/m²"),
    ("rest_peak_subaortic_gradient_mmhg", "Peak subaortic gradient at rest, mmHg"),
    ("rest_mr_grade", "MR grade (0–4 ordinal)"),
]

BINARY = [
    ("sex_male", "Male sex"),
    ("rest_lvef_lt50", "LVEF < 50%"),
    ("rest_sam_present", "SAM present"),
    ("rest_lavi_gt34", "LAVI > 34 mL/m²"),
]


def fmt_cont(vals):
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if len(vals) == 0:
        return "—", 0
    med = np.median(vals)
    q1, q3 = np.percentile(vals, [25, 75])
    return f"{med:.1f} [{q1:.1f}–{q3:.1f}]", int(len(vals))


def fmt_bin(vals):
    vals = pd.to_numeric(vals, errors="coerce")
    n_obs = int(vals.notna().sum())
    n_pos = int((vals == 1).sum())
    if n_obs == 0:
        return "—", 0
    pct = 100.0 * n_pos / n_obs
    return f"{n_pos}/{n_obs} ({pct:.1f}%)", n_obs


def kw_test(df, var, group):
    arrays = []
    for g in df[group].dropna().unique():
        v = pd.to_numeric(df.loc[df[group] == g, var], errors="coerce").dropna().values
        if len(v) >= 2:
            arrays.append(v)
    if len(arrays) < 2:
        return np.nan
    try:
        return float(stats.kruskal(*arrays).pvalue)
    except Exception:
        return np.nan


def chi2_test(df, var, group):
    sub = df[[var, group]].copy()
    sub[var] = pd.to_numeric(sub[var], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return np.nan
    ct = pd.crosstab(sub[group], sub[var])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    try:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        if (expected < 5).any():
            # Fisher fallback only meaningful for 2x2
            if ct.shape == (2, 2):
                _, p = stats.fisher_exact(ct.values)
        return float(p)
    except Exception:
        return np.nan


def main():
    df = pd.read_csv(CSV)
    df["label3_name"] = pd.Categorical(df["label3_name"], categories=CLASS_ORDER, ordered=True)
    df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)

    n_total = len(df)
    n_class = {g: int((df["label3_name"] == g).sum()) for g in CLASS_ORDER}
    n_split = {s: int((df["split"] == s).sum()) for s in SPLIT_ORDER}

    rows = []
    rows.append({
        "Variable": "N",
        "Overall": f"{n_total}",
        **{g: str(n_class[g]) for g in CLASS_ORDER},
        "P (class)": "",
        **{s: str(n_split[s]) for s in SPLIT_ORDER},
        "P (split)": "",
    })

    # Continuous
    for var, label in CONTINUOUS:
        ov, _ = fmt_cont(df[var])
        per_class = {g: fmt_cont(df.loc[df["label3_name"] == g, var])[0] for g in CLASS_ORDER}
        per_split = {s: fmt_cont(df.loc[df["split"] == s, var])[0] for s in SPLIT_ORDER}
        p_class = kw_test(df, var, "label3_name")
        p_split = kw_test(df, var, "split")
        rows.append({
            "Variable": label, "Overall": ov,
            **per_class, "P (class)": fmt_p(p_class),
            **per_split, "P (split)": fmt_p(p_split),
        })

    # Binary
    for var, label in BINARY:
        ov, _ = fmt_bin(df[var])
        per_class = {g: fmt_bin(df.loc[df["label3_name"] == g, var])[0] for g in CLASS_ORDER}
        per_split = {s: fmt_bin(df.loc[df["split"] == s, var])[0] for s in SPLIT_ORDER}
        p_class = chi2_test(df, var, "label3_name")
        p_split = chi2_test(df, var, "split")
        rows.append({
            "Variable": label, "Overall": ov,
            **per_class, "P (class)": fmt_p(p_class),
            **per_split, "P (split)": fmt_p(p_split),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table1.csv", index=False)
    print(f"Saved {OUT/'table1.csv'}")

    # Markdown rendering
    cols = ["Variable", "Overall"] + CLASS_ORDER + ["P (class)"] + SPLIT_ORDER + ["P (split)"]
    md_lines = ["| " + " | ".join(cols) + " |",
                "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in out.iterrows():
        md_lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")

    # Footer: TTE coverage notes
    nearest_n = int(df["tte_to_stress_days"].notna().sum())
    if nearest_n:
        med_delta = float(df["tte_to_stress_days"].abs().median())
    else:
        med_delta = float("nan")
    md_lines.append("")
    md_lines.append(f"**Notes.** Continuous variables shown as median [IQR]; "
                    f"binary as n/N (%). P-values by Kruskal–Wallis (continuous and "
                    f"ordinal MR grade) or χ² with Fisher's exact fallback (binary). "
                    f"TTE-derived metrics are taken from the stress echocardiogram's "
                    f"own rest evaluation (delta = 0 days). An external regular TTE "
                    f"could be matched for {nearest_n}/{n_total} "
                    f"({100*nearest_n/n_total:.1f}%) studies; median absolute time "
                    f"from stress echo {med_delta:.0f} days.")
    (OUT / "table1.md").write_text("\n".join(md_lines))
    print(f"Saved {OUT/'table1.md'}")

    # Print to stdout
    print("\n" + "\n".join(md_lines))


def fmt_p(p):
    if p is None or (isinstance(p, float) and (np.isnan(p))):
        return "—"
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


if __name__ == "__main__":
    main()
