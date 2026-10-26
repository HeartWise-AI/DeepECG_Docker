"""Build cohort_3class_with_tte.csv: stress-echo cohort joined with the nearest
regular TTE per patient.

Sources
  - finetune_data/stress_echo/linked_all.parquet      (cohort: 9550 stress echoes)
  - finetune_data/stress_echo/cohort_3class/cohort_3class_demo.parquet
                                                       (split-aligned demo arrays)
  - /media/data1/ravram/DeepECHO/TTE_reports_concat/df_report_2006_2024.csv
                                                       (221k regular TTE reports)
  - /media/data1/muse_ge/ECG_ad20241231_metadata.v1.6._with_translation_ROXs42Bb.cleaned.parquet
                                                       (age/sex from ECG metadata)

Output
  finetune_data/stress_echo/cohort_3class_with_tte.csv

Columns (the `rest_*` block is taken from the stress-echo's own rest
echocardiography; the `tte_*` block is from the nearest external TTE report).

  patient_id, stress_echo_date, ecg_date, split, label3, label3_name,
  age_years, sex_male,
  # stress-echo own rest measurements (same study, delta=0)
  rest_mlvwt_mm, rest_lvedd_mm, rest_lvef_pct, rest_lvef_lt50,
  rest_lavi_ml_m2, rest_lavi_gt34, rest_mr_grade,
  rest_sam_present, rest_peak_subaortic_gradient_mmhg,
  # nearest external TTE
  tte_date, tte_to_stress_days,
  tte_mlvwt_mm, tte_lvedd_mm, tte_lvef_pct, tte_lvef_lt50,
  tte_lavi_ml_m2, tte_lavi_gt34, tte_mr_grade, tte_sam_present,
  tte_peak_subaortic_gradient_mmhg
"""

import os
import re
import numpy as np
import pandas as pd

LINKED = "/volume/DeepECG_Docker/finetune_data/stress_echo/linked_all.parquet"
COHORT_DIR = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class"
TTE_CSV = "/media/data1/ravram/DeepECHO/TTE_reports_concat/df_report_2006_2024.csv"
ECG_META = "/media/data1/muse_ge/ECG_ad20241231_metadata.v1.6._with_translation_ROXs42Bb.cleaned.parquet"
OUT_CSV = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class_with_tte.csv"

LABEL_NAMES = {0: "Other", 1: "HCM_no_latent", 2: "Latent_pos"}

TTE_USECOLS = [
    "Dossier", "StudyDate",
    "LVID Diastole (2D)",
    "IVS Diastole Thickness (2D)",
    "LVIW Diastolic Thickness (2D)",
    "Visually Estimated EF",
    "LA Volume Index (BP MOD) DuBois",
    "Referring MR Grade",
    "Mitral valve findings",
    "Left ventricule findings",
    " [MeasurementsDoppler]_[LVOTPeakVelocity]",
]


def parse_mr_grade(v):
    """Map heterogeneous MR grade strings → ordinal 0..4 or NaN."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in {"", "nan", "none"}:
        return np.nan
    # Common text labels
    table = {
        "trace": 0.5, "trivial": 0.5,
        "mild": 1, "leger": 1, "léger": 1, "légère": 1, "legere": 1,
        "mild-moderate": 1.5, "mild to moderate": 1.5,
        "moderate": 2, "modere": 2, "modéré": 2, "moderee": 2, "modérée": 2,
        "moderate-severe": 2.5, "moderate to severe": 2.5,
        "severe": 3, "severe+": 4, "very severe": 4,
    }
    if s in table:
        return table[s]
    # Try numeric
    try:
        f = float(s)
        if 0 <= f <= 4:
            return f
    except Exception:
        pass
    # Some rows look like "2022-01-04 00:00:00" → corrupted excel date for 4/1: treat as 4.0?
    m = re.match(r"\d{4}-(\d{2})-(\d{2})", s)
    if m:
        # Excel date corruption: month/day digits likely encode the original "1/4" or "0.5"
        # too unreliable — drop
        return np.nan
    return np.nan


def parse_sam(text):
    """1 if 'SAM' / 'systolic anterior motion' is present and not negated; else 0."""
    if pd.isna(text):
        return np.nan
    s = str(text).lower()
    if not s.strip():
        return np.nan
    positive = ("systolic anterior motion" in s) or re.search(r"(^|[^a-z])sam([^a-z]|$)", s)
    if not positive:
        return 0
    # crude negation guard
    neg_terms = ["no systolic anterior motion", "absence of systolic anterior motion",
                 "pas de mouvement systolique antérieur", "no sam"]
    if any(n in s for n in neg_terms):
        return 0
    return 1


def parse_mr_grade_text(v):
    """Stress-echo extraction stores MR grade as text like 'mild', 'moderate', 'severe'."""
    return parse_mr_grade(v)


def main():
    print("Loading linked cohort ...")
    linked = pd.read_parquet(LINKED)
    linked["label3"] = linked.apply(
        lambda r: 2 if r["latent_pos"] else (1 if r["hcm_any"] else 0), axis=1
    )
    print(f"  cohort: {len(linked)} rows")

    # ---- stress-echo own rest measurements (same study, delta=0) ----
    linked["rest_mlvwt_mm"] = linked["hcm_variables.max_wall_thickness_mm"]
    linked["rest_lvedd_mm"] = linked["rest_echocardiography.lv_diastolic_diameter_mm"]
    linked["rest_lvef_pct"] = linked["hcm_variables.lvef_rest_percent"]
    linked["rest_lavi_ml_m2"] = linked["hcm_variables.la_volume_index_ml_m2"]
    linked["rest_mr_grade"] = linked["rest_echocardiography.mitral_valve.regurgitation_grade"].apply(parse_mr_grade_text)
    linked["rest_sam_present"] = linked["hcm_variables.sam_present"].astype("float")
    # Clinical convention: if no rest LVOT gradient is reported on the echo,
    # treat it as 0 mmHg (normal). Reports only document a gradient when
    # an outflow obstruction is suspected; absence is informative.
    linked["rest_peak_subaortic_gradient_mmhg"] = linked["hcm_variables.lvot_gradient_rest_mmhg"].fillna(0.0)
    linked["rest_lvef_lt50"] = (linked["rest_lvef_pct"] < 50).astype("float")
    linked.loc[linked["rest_lvef_pct"].isna(), "rest_lvef_lt50"] = np.nan
    linked["rest_lavi_gt34"] = (linked["rest_lavi_ml_m2"] > 34).astype("float")
    linked.loc[linked["rest_lavi_ml_m2"].isna(), "rest_lavi_gt34"] = np.nan

    print("Loading ECG metadata for age/sex ...")
    meta = pd.read_parquet(
        ECG_META,
        columns=[
            "npy_path",
            "RestingECG_PatientDemographics_PatientAge",
            "RestingECG_PatientDemographics_AgeUnits",
            "RestingECG_PatientDemographics_Gender",
        ],
    )
    meta.columns = ["npy_path", "age_raw", "age_units", "gender"]

    def age_to_years(a, u):
        if pd.isna(a):
            return np.nan
        try:
            a = float(a)
        except Exception:
            return np.nan
        if u == "MONTHS":
            return a / 12
        if u == "DAYS":
            return a / 365.25
        if u == "WEEKS":
            return a / 52.18
        return a

    meta["age_years"] = meta.apply(lambda r: age_to_years(r["age_raw"], r["age_units"]), axis=1)
    meta["sex_male"] = (meta["gender"].astype(str).str.upper().str.startswith("M")).astype(int)
    meta["sex_female"] = (meta["gender"].astype(str).str.upper().str.startswith("F")).astype(int)
    linked = linked.merge(meta[["npy_path", "age_years", "sex_male", "sex_female"]],
                          on="npy_path", how="left")
    print(f"  age non-null: {linked['age_years'].notna().sum()}/{len(linked)}")

    print("Loading TTE database ...")
    tte = pd.read_csv(TTE_CSV, usecols=TTE_USECOLS, low_memory=False)
    print(f"  TTE rows: {len(tte)}")

    # Normalize
    tte["StudyDate"] = pd.to_datetime(tte["StudyDate"], errors="coerce")
    tte = tte.dropna(subset=["Dossier", "StudyDate"]).reset_index(drop=True)
    print(f"  TTE rows after date parse: {len(tte)}")

    # MLVWT proxy = max(IVS, LVIW) (mm)
    tte["mlvwt_mm"] = tte[["IVS Diastole Thickness (2D)",
                            "LVIW Diastolic Thickness (2D)"]].max(axis=1)
    tte["lvedd_mm"] = tte["LVID Diastole (2D)"]
    tte["lvef_pct"] = tte["Visually Estimated EF"]
    tte["lavi_ml_m2"] = tte["LA Volume Index (BP MOD) DuBois"]
    tte["mr_grade"] = tte["Referring MR Grade"].apply(parse_mr_grade)
    sam_mv = tte["Mitral valve findings"].apply(parse_sam)
    sam_lv = tte["Left ventricule findings"].apply(parse_sam)
    tte["sam_present"] = np.where((sam_mv == 1) | (sam_lv == 1), 1,
                          np.where((sam_mv == 0) | (sam_lv == 0), 0, np.nan))
    # Peak subaortic gradient = 4 * v^2 (v in m/s); impute 0 when v missing
    # (clinical convention: absence of reported LVOT velocity = no obstruction).
    v = pd.to_numeric(tte[" [MeasurementsDoppler]_[LVOTPeakVelocity]"], errors="coerce")
    tte["peak_subaortic_gradient_mmhg"] = (4.0 * (v ** 2)).fillna(0.0)

    tte_keep = tte[["Dossier", "StudyDate", "mlvwt_mm", "lvedd_mm", "lvef_pct",
                    "lavi_ml_m2", "mr_grade", "sam_present",
                    "peak_subaortic_gradient_mmhg"]].copy()
    tte_keep = tte_keep.sort_values(["Dossier", "StudyDate"]).reset_index(drop=True)

    print("Matching nearest TTE per stress-echo ...")
    # group tte by patient → for each cohort row, get nearest
    tte_by_pat = {pid: g for pid, g in tte_keep.groupby("Dossier")}

    rows = []
    n_match = 0
    for _, r in linked.iterrows():
        pid = int(r["patient_id_int"])
        stress_date = pd.Timestamp(r["exam_date"])
        nearest = None; nearest_delta = None
        if pid in tte_by_pat:
            g = tte_by_pat[pid]
            deltas = (g["StudyDate"] - stress_date).dt.days
            j = int(deltas.abs().values.argmin())
            nearest = g.iloc[j]
            nearest_delta = int(deltas.iloc[j])
            n_match += 1

        row = {
            "patient_id": pid,
            "stress_echo_date": stress_date.date().isoformat(),
            "ecg_date": pd.Timestamp(r["ecg_date"]).date().isoformat() if pd.notna(r["ecg_date"]) else "",
            "npy_path": r["npy_path"],
            "split": r["split"],
            "label3": int(r["label3"]),
            "label3_name": LABEL_NAMES[int(r["label3"])],
            "age_years": r.get("age_years", np.nan),
            "sex_male": r.get("sex_male", np.nan),
            # stress-echo own rest measurements
            "rest_mlvwt_mm": r["rest_mlvwt_mm"],
            "rest_lvedd_mm": r["rest_lvedd_mm"],
            "rest_lvef_pct": r["rest_lvef_pct"],
            "rest_lvef_lt50": r["rest_lvef_lt50"],
            "rest_lavi_ml_m2": r["rest_lavi_ml_m2"],
            "rest_lavi_gt34": r["rest_lavi_gt34"],
            "rest_mr_grade": r["rest_mr_grade"],
            "rest_sam_present": r["rest_sam_present"],
            "rest_peak_subaortic_gradient_mmhg": r["rest_peak_subaortic_gradient_mmhg"],
        }
        if nearest is not None:
            row.update({
                "tte_date": nearest["StudyDate"].date().isoformat(),
                "tte_to_stress_days": nearest_delta,
                "tte_mlvwt_mm": nearest["mlvwt_mm"],
                "tte_lvedd_mm": nearest["lvedd_mm"],
                "tte_lvef_pct": nearest["lvef_pct"],
                "tte_lavi_ml_m2": nearest["lavi_ml_m2"],
                "tte_mr_grade": nearest["mr_grade"],
                "tte_sam_present": nearest["sam_present"],
                "tte_peak_subaortic_gradient_mmhg": nearest["peak_subaortic_gradient_mmhg"],
            })
        else:
            row.update({"tte_date": "", "tte_to_stress_days": np.nan,
                        "tte_mlvwt_mm": np.nan, "tte_lvedd_mm": np.nan,
                        "tte_lvef_pct": np.nan, "tte_lavi_ml_m2": np.nan,
                        "tte_mr_grade": np.nan, "tte_sam_present": np.nan,
                        "tte_peak_subaortic_gradient_mmhg": np.nan})
        rows.append(row)
    print(f"  matched a TTE for {n_match}/{len(linked)} ({n_match/len(linked):.1%})")

    out = pd.DataFrame(rows)
    out["tte_lvef_lt50"] = (out["tte_lvef_pct"] < 50).astype("float")
    out.loc[out["tte_lvef_pct"].isna(), "tte_lvef_lt50"] = np.nan
    out["tte_lavi_gt34"] = (out["tte_lavi_ml_m2"] > 34).astype("float")
    out.loc[out["tte_lavi_ml_m2"].isna(), "tte_lavi_gt34"] = np.nan

    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}  ({len(out)} rows, {out.shape[1]} cols)")

    # quick summary
    print("\nVariable coverage (rest = stress-echo own; tte = nearest external TTE):")
    for c in ["age_years", "sex_male",
              "rest_mlvwt_mm", "rest_lvedd_mm", "rest_lvef_pct",
              "rest_lavi_ml_m2", "rest_mr_grade", "rest_sam_present",
              "rest_peak_subaortic_gradient_mmhg",
              "tte_mlvwt_mm", "tte_lvedd_mm", "tte_lvef_pct",
              "tte_lavi_ml_m2", "tte_mr_grade", "tte_sam_present",
              "tte_peak_subaortic_gradient_mmhg"]:
        n = out[c].notna().sum()
        print(f"  {c:42s}: {n}/{len(out)} ({n/len(out):.1%})")
    print("\ntte_to_stress_days |delta| summary (days):")
    s = out["tte_to_stress_days"].abs().describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    print(s)


if __name__ == "__main__":
    main()
