"""Build 3-class stress-echo cohort from the already-linked dataset.

Labels:
    0 = "other"              (neither HCM nor latent)
    1 = "hcm_no_latent"      (hcm_any & not latent_pos)
    2 = "latent_pos"

Reuses linked_all.parquet produced by build_stress_echo_manifests.py, which locks
the 70/10/20 patient split.

Usage:
    python scripts/build_stress_echo_3class.py
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

LINKED = "/volume/DeepECG_Docker/finetune_data/stress_echo/linked_all.parquet"
OUT_DIR = "/volume/DeepECG_Docker/finetune_data/stress_echo/cohort_3class"


def label_row(r):
    if r["latent_pos"]:
        return 2
    if r["hcm_any"]:
        return 1
    return 0


def load_and_stack(paths, labels, desc):
    sigs, keep_labels, failed = [], [], 0
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
        except Exception:
            failed += 1
    X = np.stack(sigs, axis=0) if sigs else np.zeros((0, 2500, 12, 1), dtype=np.float32)
    Y = np.array(keep_labels, dtype=np.int64)
    print(f"  {desc}: kept {len(sigs)}, failed {failed}, class dist={np.bincount(Y) if len(Y) else 'empty'}")
    return X, Y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    linked = pd.read_parquet(LINKED)
    linked["label3"] = linked.apply(label_row, axis=1)
    cohort = linked.copy()
    print(f"Total: {len(cohort)}  class dist: {np.bincount(cohort['label3'])}")

    cohort[[
        "patient_id_int", "exam_date", "ecg_date", "date_diff_days",
        "npy_path", "diag", "latent_pos", "hcm_any", "label3", "split",
    ]].to_parquet(os.path.join(OUT_DIR, "cohort.parquet"), index=False)

    for split in ["train", "val", "test"]:
        sub = cohort[cohort["split"] == split]
        X, Y = load_and_stack(sub["npy_path"].tolist(), sub["label3"].tolist(), f"3class/{split}")
        np.save(os.path.join(OUT_DIR, f"{split}_x.npy"), X)
        np.save(os.path.join(OUT_DIR, f"{split}_y.npy"), Y)
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
