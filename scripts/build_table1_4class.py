"""Table 1 for the 4-class DeepECG-HCM cohort.

Same source CSV as the 3-class table, but joined with the 4-class label so
HCM_no_obs and Manifest_obs are reported separately.

Output: outputs/table1_4class/table1.{csv,md}
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

CSV3 = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class_with_tte.csv"
COHORT4 = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_4class/cohort.parquet"
OUT = Path("/volume/DeepECG_Docker/outputs/table1_4class")
OUT.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ["Other", "HCM_no_obs", "Manifest_pos", "Latent_pos"]
DISPLAY = ["Other", "Manifest−/Latent−", "Manifest+", "Latent+"]
LABEL_NAMES = {0: "Other", 1: "HCM_no_obs", 2: "Manifest_pos", 3: "Latent_pos"}
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
        return "—"
    med = np.median(vals); q1, q3 = np.percentile(vals, [25, 75])
    return f"{med:.1f} [{q1:.1f}–{q3:.1f}]"


def fmt_bin(vals):
    vals = pd.to_numeric(vals, errors="coerce")
    n_obs = int(vals.notna().sum()); n_pos = int((vals == 1).sum())
    if n_obs == 0: return "—"
    return f"{n_pos} / {n_obs} ({100.0 * n_pos / n_obs:.1f}%)"


def kw(df, var, group):
    arrays = []
    for g in df[group].dropna().unique():
        v = pd.to_numeric(df.loc[df[group] == g, var], errors="coerce").dropna().values
        if len(v) >= 2: arrays.append(v)
    if len(arrays) < 2: return np.nan
    try: return float(stats.kruskal(*arrays).pvalue)
    except Exception: return np.nan


def chi2(df, var, group):
    sub = df[[var, group]].copy()
    sub[var] = pd.to_numeric(sub[var], errors="coerce")
    sub = sub.dropna()
    if sub.empty: return np.nan
    ct = pd.crosstab(sub[group], sub[var])
    if ct.shape[0] < 2 or ct.shape[1] < 2: return np.nan
    try:
        chi2v, p, _, expected = stats.chi2_contingency(ct)
        if (expected < 5).any() and ct.shape == (2, 2):
            _, p = stats.fisher_exact(ct.values)
        return float(p)
    except Exception: return np.nan


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return "—"
    if p < 0.001: return "<0.001"
    if p < 0.01: return f"{p:.3f}"
    return f"{p:.2f}"


def main():
    df3 = pd.read_csv(CSV3)
    coh4 = pd.read_parquet(COHORT4)[["patient_id_int", "exam_date", "label4"]]
    df3["exam_date"] = pd.to_datetime(df3["stress_echo_date"])
    coh4["exam_date"] = pd.to_datetime(coh4["exam_date"])
    df = df3.merge(coh4.rename(columns={"patient_id_int": "patient_id"}),
                   on=["patient_id", "exam_date"], how="left")
    df["label4_name"] = df["label4"].map(LABEL_NAMES)
    df["label4_name"] = pd.Categorical(df["label4_name"], categories=CLASS_ORDER, ordered=True)
    df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)

    n_total = len(df)
    n_class = {g: int((df["label4_name"] == g).sum()) for g in CLASS_ORDER}
    n_split = {s: int((df["split"] == s).sum()) for s in SPLIT_ORDER}

    rows = [{
        "Variable": "N",
        "Overall": f"{n_total:,}",
        **{disp: f"{n_class[g]:,}" for g, disp in zip(CLASS_ORDER, DISPLAY)},
        "P (class)": "",
        **{s.title(): f"{n_split[s]:,}" for s in SPLIT_ORDER},
        "P (split)": "",
    }]

    for var, label in CONTINUOUS:
        rows.append({
            "Variable": label,
            "Overall": fmt_cont(df[var]),
            **{disp: fmt_cont(df.loc[df["label4_name"] == g, var])
               for g, disp in zip(CLASS_ORDER, DISPLAY)},
            "P (class)": fmt_p(kw(df, var, "label4_name")),
            **{s.title(): fmt_cont(df.loc[df["split"] == s, var]) for s in SPLIT_ORDER},
            "P (split)": fmt_p(kw(df, var, "split")),
        })
    for var, label in BINARY:
        rows.append({
            "Variable": label,
            "Overall": fmt_bin(df[var]),
            **{disp: fmt_bin(df.loc[df["label4_name"] == g, var])
               for g, disp in zip(CLASS_ORDER, DISPLAY)},
            "P (class)": fmt_p(chi2(df, var, "label4_name")),
            **{s.title(): fmt_bin(df.loc[df["split"] == s, var]) for s in SPLIT_ORDER},
            "P (split)": fmt_p(chi2(df, var, "split")),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table1.csv", index=False)

    cols = ["Variable", "Overall"] + DISPLAY + ["P (class)"] + [s.title() for s in SPLIT_ORDER] + ["P (split)"]
    md = ["| " + " | ".join(cols) + " |",
          "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in out.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    (OUT / "table1.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved {OUT/'table1.csv'} and {OUT/'table1.md'}")


if __name__ == "__main__":
    main()
