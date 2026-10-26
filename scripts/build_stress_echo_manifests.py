"""Build fairseq-signals manifests for stress-echo latent-obstruction fine-tuning.

Cohorts:
  A: latent+ vs everyone else (all linked stress-echo rows).
  B: latent+ vs HCM-any-but-not-latent (controls restricted to HCM patients).

Shared 70/10/20 patient split so a patient's ECG stays in the same split across
both cohorts.

Latent+ rule:   rest<30 AND max(provoked, post_ex) >= 50
HCM-any rule:   max_wall >= 15 OR IVS >= 15 OR SAM+ OR known_cardiomyopathy~'hcm'
ECG link:       closest ECG (by |exam_date - ecg_date|) within +/- 30 days, 1 per echo.
Text label:     `diag` = translated_diagnosis, fallback to diagnosis.

Usage:
    python scripts/build_stress_echo_manifests.py
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm


ECHO_PARQUET = "finetune_data/stress_echo/stress_echo_full.parquet"
META_PARQUET = "/media/data1/muse_ge/ECG_ad20241231_metadata.v1.6._with_translation_ROXs42Bb.cleaned.parquet"
LABELS_PARQUET = "/media/data1/muse_ge/ECG_ad20241231_gt_labels_v1.6.parquet"

OUT_ROOT = "finetune_data/stress_echo"

SPLIT_SEED = 42
MATCH_WINDOW_DAYS = 30


def build_linked_dataset() -> pd.DataFrame:
    echo = pd.read_parquet(ECHO_PARQUET)
    echo["patient_id_int"] = echo["study_identifiers.patient_id"].astype(int)
    echo["exam_date"] = pd.to_datetime(echo["study_identifiers.exam_date"], errors="coerce")

    stress_grad = echo[[
        "hcm_variables.lvot_gradient_provoked_mmhg",
        "post_exercise_imaging.lvot_gradient_post_exercise_mmhg",
    ]].max(axis=1)
    rest_grad = echo["hcm_variables.lvot_gradient_rest_mmhg"]
    echo["latent_pos"] = ((rest_grad < 30) & (stress_grad >= 50)).fillna(False)

    max_wall = echo["hcm_variables.max_wall_thickness_mm"].fillna(0)
    ivs = echo["hcm_variables.ivs_thickness_basal_mm"].fillna(0)
    sam = echo["hcm_variables.sam_present"].astype(str).str.lower().isin(["true", "1", "yes"])
    known = echo["demographics.known_cardiomyopathy"].fillna("").astype(str).str.lower()
    echo["hcm_any"] = (
        (max_wall >= 15)
        | (ivs >= 15)
        | sam
        | known.str.contains(r"hcm|hypertr|cmh", regex=True, na=False)
    )
    echo = echo.dropna(subset=["exam_date"])
    print(f"Echo rows with valid exam date: {len(echo)}")
    print(f"  latent_pos exams: {int(echo['latent_pos'].sum())}")
    print(f"  hcm_any exams:    {int(echo['hcm_any'].sum())}")

    meta_cols = ["PatientID", "ecg_date", "npy_path", "diagnosis", "translated_diagnosis"]
    meta = pd.read_parquet(META_PARQUET, columns=meta_cols)
    meta["ecg_date"] = pd.to_datetime(meta["ecg_date"], errors="coerce")
    meta = meta.dropna(subset=["ecg_date", "npy_path"])
    meta = meta[meta["PatientID"].isin(echo["patient_id_int"].unique())]
    print(f"ECG metadata rows for echo patients: {len(meta)}")

    merged = echo.merge(
        meta, left_on="patient_id_int", right_on="PatientID", how="inner"
    )
    merged["date_diff_days"] = (merged["ecg_date"] - merged["exam_date"]).dt.days.abs()
    merged = merged[merged["date_diff_days"] <= MATCH_WINDOW_DAYS]
    print(f"Candidate ECG-echo pairs within +/-{MATCH_WINDOW_DAYS}d: {len(merged)}")

    merged = merged.sort_values(["patient_id_int", "exam_date", "date_diff_days"])
    one_per_echo = merged.drop_duplicates(
        subset=["patient_id_int", "exam_date"], keep="first"
    ).reset_index(drop=True)
    print(f"Unique echoes with 1 matched ECG: {len(one_per_echo)}")
    print(f"  latent_pos linked: {int(one_per_echo['latent_pos'].sum())}")
    print(f"  hcm_any linked:    {int(one_per_echo['hcm_any'].sum())}")

    one_per_echo["diag"] = one_per_echo["translated_diagnosis"].fillna(
        one_per_echo["diagnosis"]
    )
    return one_per_echo


def assign_patient_split(df: pd.DataFrame) -> pd.Series:
    patients = np.array(sorted(df["patient_id_int"].unique()))
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(0.70 * n)
    n_val = int(0.10 * n)
    train_p = set(patients[:n_train].tolist())
    val_p = set(patients[n_train : n_train + n_val].tolist())
    test_p = set(patients[n_train + n_val :].tolist())

    def which(pid):
        if pid in train_p:
            return "train"
        if pid in val_p:
            return "val"
        return "test"

    return df["patient_id_int"].map(which)


def load_and_stack(paths, labels, desc):
    sigs, keep_labels, keep_paths, failed = [], [], [], 0
    for p, y in tqdm(list(zip(paths, labels)), desc=desc):
        try:
            arr = np.load(p).astype(np.float32)
            if arr.shape == (12, 2500):
                arr = arr.T
            if arr.ndim == 2 and arr.shape == (2500, 12):
                arr = arr[:, :, np.newaxis]
            elif arr.ndim == 3 and arr.shape == (2500, 12, 1):
                pass
            else:
                failed += 1
                continue
            sigs.append(arr)
            keep_labels.append(y)
            keep_paths.append(p)
        except Exception:
            failed += 1
    X = np.stack(sigs, axis=0) if sigs else np.zeros((0, 2500, 12, 1), dtype=np.float32)
    Y = np.array(keep_labels, dtype=np.float32)[:, None] if keep_labels else np.zeros((0, 1), np.float32)
    print(f"  {desc}: kept {len(sigs)}, failed {failed}, pos={int(Y.sum())}")
    return X, Y, keep_paths


def write_manifest(out_dir, split, x_path, y_path, x_shape):
    with open(os.path.join(out_dir, f"{split}.tsv"), "w") as f:
        f.write(
            f"x_path:{x_path}\n"
            f"x_shape:{x_shape}\n"
            f"y_path:{y_path}\n"
            f"label_indexes:[0]\n"
        )


def build_cohort(linked: pd.DataFrame, cohort_name: str, mask: pd.Series) -> str:
    cohort = linked[mask].copy()
    print(f"\n=== Cohort {cohort_name}: {len(cohort)} rows (pos={int(cohort['latent_pos'].sum())}) ===")
    out_dir = os.path.join(OUT_ROOT, cohort_name)
    manifest_dir = os.path.join(out_dir, "manifest")
    os.makedirs(manifest_dir, exist_ok=True)

    # Save the cohort table (for audit / later eval)
    cohort[[
        "patient_id_int", "exam_date", "ecg_date", "date_diff_days",
        "npy_path", "diag", "latent_pos", "hcm_any", "split",
    ]].to_parquet(os.path.join(out_dir, "cohort.parquet"), index=False)

    fairseq_names = {"train": "train", "val": "valid", "test": "test"}
    for split in ["train", "val", "test"]:
        sub = cohort[cohort["split"] == split]
        X, Y, _ = load_and_stack(sub["npy_path"].tolist(),
                                 sub["latent_pos"].astype(np.float32).tolist(),
                                 f"{cohort_name}/{split}")
        x_path = os.path.abspath(os.path.join(out_dir, f"{split}_x.npy"))
        y_path = os.path.abspath(os.path.join(out_dir, f"{split}_y.npy"))
        np.save(x_path, X)
        np.save(y_path, Y)
        write_manifest(manifest_dir, fairseq_names[split], x_path, y_path, X.shape)
    print(f"  manifests -> {manifest_dir}")
    return manifest_dir


def main():
    os.chdir("/volume/DeepECG_Docker")
    linked = build_linked_dataset()
    linked["split"] = assign_patient_split(linked)
    print("\nSplit counts (linked):")
    print(linked["split"].value_counts())
    print("Positives per split:")
    print(linked.groupby("split")["latent_pos"].sum())

    # Save the global linked manifest for reproducibility
    os.makedirs(OUT_ROOT, exist_ok=True)
    linked.to_parquet(os.path.join(OUT_ROOT, "linked_all.parquet"), index=False)

    # Cohort A: latent+ vs everyone else
    mask_a = pd.Series(True, index=linked.index)
    build_cohort(linked, "cohort_latent_all", mask_a)

    # Cohort B: latent+ vs HCM-any-but-not-latent
    mask_b = linked["latent_pos"] | (linked["hcm_any"] & ~linked["latent_pos"])
    build_cohort(linked, "cohort_latent_vs_hcm", mask_b)

    print("\nAll done.")


if __name__ == "__main__":
    main()
