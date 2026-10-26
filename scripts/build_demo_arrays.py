"""Add age + sex covariates aligned to each split's NPY ordering.

Reads `linked_all.parquet`, joins ECG metadata for age/sex, standardizes age
using TRAIN-set stats only, and writes `<split>_demo.npy` arrays of shape (N, 3)
= [age_z, sex_male, sex_female].

Output: finetune_data/stress_echo/cohort_3class/{train,val,test}_demo.npy
        + cohort_3class_demo.parquet (for audit).
"""
import os
import numpy as np
import pandas as pd

LINKED = "/volume/DeepECG_Docker/finetune_data/stress_echo/linked_all.parquet"
COHORT_DIR = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class"
META_PATH = "/media/data1/muse_ge/ECG_ad20241231_metadata.v1.6._with_translation_ROXs42Bb.cleaned.parquet"


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
    return a  # YEARS / NaN -> years


def main():
    linked = pd.read_parquet(LINKED)
    linked["label3"] = linked.apply(
        lambda r: 2 if r["latent_pos"] else (1 if r["hcm_any"] else 0), axis=1
    )

    meta = pd.read_parquet(
        META_PATH,
        columns=[
            "npy_path",
            "RestingECG_PatientDemographics_PatientAge",
            "RestingECG_PatientDemographics_AgeUnits",
            "RestingECG_PatientDemographics_Gender",
        ],
    )
    meta.columns = ["npy_path", "age_raw", "age_units", "gender"]
    merged = linked.merge(meta, on="npy_path", how="left")
    merged["age_years"] = merged.apply(
        lambda r: age_to_years(r["age_raw"], r["age_units"]), axis=1
    )
    merged["sex_male"] = (
        merged["gender"].astype(str).str.upper().str.startswith("M").astype(int)
    )
    merged["sex_female"] = (
        merged["gender"].astype(str).str.upper().str.startswith("F").astype(int)
    )

    # Sanity vs the existing cohort_3class which was built with the same
    # patient split: same row order is achieved by sorting on (split, npy_path).
    # The build_stress_echo_3class.py script iterates rows in the merged
    # dataframe's natural order, which matches the linked_all parquet order.
    # We replicate that ordering exactly.
    splits = {}
    for split in ["train", "val", "test"]:
        sub = merged[merged["split"] == split].reset_index(drop=True)
        splits[split] = sub

    # z-score age using train stats only
    train_age = splits["train"]["age_years"]
    mu, sigma = float(train_age.mean()), float(train_age.std())
    print(f"Age z-score from train: mu={mu:.2f}, sigma={sigma:.2f}")

    # Save demo arrays
    for split, sub in splits.items():
        age_z = ((sub["age_years"].fillna(mu) - mu) / max(sigma, 1e-6)).values
        sex_m = sub["sex_male"].values
        sex_f = sub["sex_female"].values
        demo = np.stack([age_z, sex_m, sex_f], axis=1).astype(np.float32)

        # Cross-check: existing _y.npy must align with this ordering
        y_existing = np.load(os.path.join(COHORT_DIR, f"{split}_y.npy"))
        y_here = sub["label3"].astype(np.int64).values
        if len(y_existing) != len(y_here) or not (y_existing == y_here).all():
            raise RuntimeError(f"label ordering mismatch in {split}: existing has "
                               f"{len(y_existing)} rows, here {len(y_here)}; "
                               f"agreement={(y_existing == y_here).sum()}/{len(y_here)}")
        np.save(os.path.join(COHORT_DIR, f"{split}_demo.npy"), demo)
        print(f"  {split}: n={len(demo)}  demo shape={demo.shape}  age_z mean/std={demo[:,0].mean():.3f}/{demo[:,0].std():.3f}  "
              f"male={demo[:,1].sum():.0f}  female={demo[:,2].sum():.0f}")

    # Save audit parquet
    audit = merged[[
        "patient_id_int", "exam_date", "ecg_date", "npy_path", "split",
        "age_years", "sex_male", "sex_female", "label3",
    ]].copy()
    audit.to_parquet(os.path.join(COHORT_DIR, "cohort_3class_demo.parquet"), index=False)
    print(f"Saved audit: {COHORT_DIR}/cohort_3class_demo.parquet")
    # Save stats so inference can apply same z-score
    pd.DataFrame([{"age_mu": mu, "age_sigma": sigma}]).to_csv(
        os.path.join(COHORT_DIR, "demo_stats.csv"), index=False
    )
    print(f"Saved demo_stats.csv (age_mu={mu:.2f}, age_sigma={sigma:.2f})")


if __name__ == "__main__":
    main()
