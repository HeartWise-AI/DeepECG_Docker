"""
Compute AUC for Acute Coronary Obstruction prediction from WCR 77-class output.

Usage:
    python scripts/compute_acs_auc.py --probabilities_csv outputs/acs_npy_baseline/wcr_77_classes_*_probabilities.csv
    python scripts/compute_acs_auc.py --probabilities_csv outputs/acs_dig_baseline/wcr_77_classes_*_probabilities.csv
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve

# Acute MI is index 69 in WCR 77-class output
# The column name in the probabilities CSV is "Acute MI_sig_model"
ACUTE_MI_COL = "Acute MI_sig_model"


def compute_auc(probabilities_csv: str, linked_parquet: str = "data/acs_linked_dataset.parquet"):
    """Compute AUC for Acute Obstruction prediction."""

    # Load probabilities
    print(f"Loading probabilities from: {probabilities_csv}")
    df_prob = pd.read_csv(probabilities_csv)
    print(f"Probabilities shape: {df_prob.shape}")

    # Load linked dataset with ground truth
    print(f"Loading ground truth from: {linked_parquet}")
    linked = pd.read_parquet(linked_parquet)
    print(f"Linked dataset shape: {linked.shape}")

    # Extract ecg_key from file_name in probabilities
    # file_name looks like "0000311_05-07-2021_09-50-46.npy" or similar
    df_prob['ecg_key'] = df_prob['file_name'].apply(
        lambda x: os.path.splitext(os.path.basename(str(x)))[0]
    )

    # Merge on ecg_key
    merged = df_prob.merge(linked[['ecg_key', 'Acute_Obstruction', 'Split']], on='ecg_key', how='inner')
    print(f"Merged: {len(merged)} rows (of {len(df_prob)} probabilities)")

    if ACUTE_MI_COL not in merged.columns:
        print(f"WARNING: '{ACUTE_MI_COL}' not found. Available sig_model columns:")
        sig_cols = [c for c in merged.columns if '_sig_model' in c]
        for c in sig_cols:
            print(f"  {c}")
        return

    # Compute overall AUC
    y_true = merged['Acute_Obstruction'].values
    y_pred = merged[ACUTE_MI_COL].values

    print(f"\n=== OVERALL RESULTS ===")
    print(f"Total: {len(y_true)}, Positive: {y_true.sum()}, Negative: {(1-y_true).sum()}")

    auc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    # Optimal threshold (Youden)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_threshold = thresholds[best_idx]
    y_binary = (y_pred >= best_threshold).astype(int)
    f1 = f1_score(y_true, y_binary)

    print(f"AUC: {auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1: {f1:.4f} (threshold={best_threshold:.4f})")
    print(f"Sensitivity: {tpr[best_idx]:.4f}")
    print(f"Specificity: {1-fpr[best_idx]:.4f}")

    # Per-split AUC
    print(f"\n=== PER-SPLIT RESULTS ===")
    for split in ['train', 'test', 'val']:
        mask = merged['Split'] == split
        if mask.sum() == 0:
            continue
        split_true = merged.loc[mask, 'Acute_Obstruction'].values
        split_pred = merged.loc[mask, ACUTE_MI_COL].values
        if split_true.sum() == 0:
            print(f"{split}: N={mask.sum()}, no positives")
            continue
        split_auc = roc_auc_score(split_true, split_pred)
        split_auprc = average_precision_score(split_true, split_pred)
        print(f"{split}: N={mask.sum()}, Pos={split_true.sum()}, AUC={split_auc:.4f}, AUPRC={split_auprc:.4f}")

    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities_csv", type=str, required=True,
                        help="Path to WCR probabilities CSV (supports glob)")
    parser.add_argument("--linked_parquet", type=str, default="data/acs_linked_dataset.parquet")
    args = parser.parse_args()

    # Support glob patterns
    csv_files = glob.glob(args.probabilities_csv)
    if not csv_files:
        print(f"No files found matching: {args.probabilities_csv}")
    else:
        for csv_file in sorted(csv_files):
            print(f"\n{'='*60}")
            print(f"Processing: {csv_file}")
            print(f"{'='*60}")
            compute_auc(csv_file, args.linked_parquet)
