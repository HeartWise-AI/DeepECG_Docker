"""Build 4-class stress-echo cohort that splits the prior HCM_no_latent class
into HCM-no-obstruction vs HCM-with-rest (manifest) obstruction.

Labels (mutually exclusive, ordered by severity):
    0 = Other            : not HCM, no obstruction
    1 = HCM_no_obs       : HCM-any features but no rest, no provoked >=50
    2 = Manifest_obs     : resting LVOT gradient >= 30 mmHg
    3 = Latent+          : rest < 30 AND max(provoked, post_exercise) >= 50

Rows and the patient-level 70/10/20 split match `cohort_3class/`, so we can
reuse the existing X NPYs and only write new Y NPYs + a cohort parquet.

Output:
    finetune_data/stress_echo/cohort_4class/
        cohort.parquet
        train_x.npy, val_x.npy, test_x.npy   (symlinks -> cohort_3class)
        train_y.npy, val_y.npy, test_y.npy
        train_demo.npy, val_demo.npy, test_demo.npy   (symlinks)
"""

import os
import numpy as np
import pandas as pd

LINKED = "/volume/DeepECG_Docker/finetune_data/stress_echo/linked_all.parquet"
SRC_DIR = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class"
OUT_DIR = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_4class"

LABEL_NAMES = {0: "Other", 1: "HCM_no_obs", 2: "Manifest_obs", 3: "Latent_pos"}


def label_row(r):
    rest_v = r["_rest"]
    stress_v = r["_max_stress"]
    rest_obs = pd.notna(rest_v) and rest_v >= 30
    latent = (pd.notna(rest_v) and rest_v < 30
              and pd.notna(stress_v) and stress_v >= 50)
    if latent:
        return 3
    if rest_obs:
        return 2
    if r["hcm_any"]:
        return 1
    return 0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    linked = pd.read_parquet(LINKED).copy()
    rest = pd.to_numeric(linked["hcm_variables.lvot_gradient_rest_mmhg"], errors="coerce")
    prov = pd.to_numeric(linked["hcm_variables.lvot_gradient_provoked_mmhg"], errors="coerce")
    post = pd.to_numeric(linked["post_exercise_imaging.lvot_gradient_post_exercise_mmhg"], errors="coerce")
    linked["_rest"] = rest
    linked["_max_stress"] = pd.concat([prov, post], axis=1).max(axis=1)
    linked["label4"] = linked.apply(label_row, axis=1)

    counts = linked["label4"].value_counts().sort_index().rename(LABEL_NAMES)
    print("Overall:", counts.to_dict())

    by_split = (linked.groupby("split")["label4"]
                .value_counts().unstack(fill_value=0).rename(columns=LABEL_NAMES))
    print("By split:\n", by_split)

    linked[[
        "patient_id_int", "exam_date", "ecg_date", "date_diff_days",
        "npy_path", "diag", "latent_pos", "hcm_any", "label4", "split",
        "_rest", "_max_stress",
    ]].rename(columns={"_rest": "rest_grad", "_max_stress": "max_stress_grad"}
              ).to_parquet(os.path.join(OUT_DIR, "cohort.parquet"), index=False)

    # Cross-check ordering against existing cohort_3class (same upstream filter+order)
    src_cohort = pd.read_parquet(os.path.join(SRC_DIR, "cohort.parquet"))
    assert len(src_cohort) == len(linked), \
        f"row count mismatch: src {len(src_cohort)} vs new {len(linked)}"
    for col in ["patient_id_int", "exam_date", "npy_path", "split"]:
        if not (src_cohort[col].values == linked[col].values).all():
            raise RuntimeError(f"ordering disagreement on {col}")
    print("Row ordering matches cohort_3class -> reusing X NPYs.")

    for split in ["train", "val", "test"]:
        sub = linked[linked["split"] == split].reset_index(drop=True)
        y = sub["label4"].astype(np.int64).values

        # Verify against the source 3-class y by checking that the
        # rest-obstructed subset maps cleanly.
        src_y3 = np.load(os.path.join(SRC_DIR, f"{split}_y.npy"))
        n = len(y)
        assert n == len(src_y3), f"{split}: len mismatch {n} vs {len(src_y3)}"

        np.save(os.path.join(OUT_DIR, f"{split}_y.npy"), y)
        # Symlink X + demo (no re-stacking)
        for s in [f"{split}_x.npy", f"{split}_demo.npy"]:
            target = os.path.join(SRC_DIR, s)
            dst = os.path.join(OUT_DIR, s)
            if os.path.lexists(dst):
                os.remove(dst)
            if os.path.exists(target):
                os.symlink(target, dst)
        print(f"  {split}: n={n}  class dist={np.bincount(y, minlength=4).tolist()}")


if __name__ == "__main__":
    main()
