"""
Evaluate a fine-tuned WCR checkpoint on digitized ECG signals.

This script:
1. Loads the linked dataset
2. Finds matching digitized NPY files
3. Runs inference with the fine-tuned WCR model
4. Computes AUC vs Acute_Obstruction

Usage:
    python scripts/eval_digitized_acs.py \
        --checkpoint checkpoints_acs_npy_ceiling/checkpoint_best.pt \
        --dig_npy_dir data/acs_digitized_npy \
        --split test
"""

import argparse
import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from fairseq_signals.utils import checkpoint_utils


def load_model(checkpoint_path, device):
    """Load fine-tuned model from checkpoint."""
    model, cfg, task = checkpoint_utils.load_model_and_task(
        checkpoint_path,
        arg_overrides={"model_path": "/volume/DeepECG_Docker/weights/wcr_77_classes/base_ssl.pt"},
        suffix=""
    )
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dig_npy_dir", type=str, required=True)
    parser.add_argument("--linked_parquet", type=str, default="data/acs_linked_dataset.parquet")
    parser.add_argument("--split", type=str, default=None, help="If set, only evaluate this split")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)

    linked = pd.read_parquet(args.linked_parquet)
    print(f"Linked dataset: {len(linked)} rows")

    # Find matching digitized NPYs
    linked['dig_npy_path'] = linked['ecg_key'].apply(
        lambda ek: os.path.join(args.dig_npy_dir, f"{ek}.npy")
    )
    linked['has_dig'] = linked['dig_npy_path'].apply(os.path.exists)
    print(f"Found {linked.has_dig.sum()} digitized NPY files")

    splits = [args.split] if args.split else ['train', 'val', 'test']

    for split in splits:
        split_df = linked[(linked.Split == split) & linked.has_dig].copy()
        n = len(split_df)
        n_pos = int(split_df.Acute_Obstruction.sum())
        print(f"\n{'='*50}")
        print(f"{split}: n={n}, pos={n_pos}")
        print(f"{'='*50}")

        if n < 10:
            print("  Too few samples, skipping")
            continue

        # Load NPYs
        signals = []
        labels = []
        for _, row in split_df.iterrows():
            arr = np.load(row['dig_npy_path']).astype(np.float32)
            if arr.shape == (12, 2500):
                arr = arr.T  # (2500, 12)
            if arr.ndim == 2 and arr.shape == (2500, 12):
                signals.append(arr)
                labels.append(row['Acute_Obstruction'])

        X = np.stack(signals)  # (N, 2500, 12)
        Y = np.array(labels, dtype=np.float32)

        # Run inference
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X), args.batch_size):
                batch_x = X[i:i+args.batch_size]
                batch_x = np.transpose(batch_x, (0, 2, 1))  # (B, 12, 2500)
                source = torch.from_numpy(batch_x).float().to(device)
                padding_mask = torch.zeros(source.shape[0], source.shape[2], dtype=torch.bool, device=device)
                net_input = {"source": source, "padding_mask": padding_mask}
                net_output = model(**net_input)
                logits = model.get_logits(net_output).float()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        all_probs = np.concatenate(all_probs, axis=0)
        if all_probs.ndim == 2:
            all_probs = all_probs[:, 0]

        auroc = roc_auc_score(Y, all_probs)
        auprc = average_precision_score(Y, all_probs)

        # Best F1
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs > thresh).astype(int)
            f1 = f1_score(Y, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        preds_05 = (all_probs > 0.5).astype(int)

        print(f"  AUROC:  {auroc:.4f}")
        print(f"  AUPRC:  {auprc:.4f}")
        print(f"  F1 (t=0.5):    {f1_score(Y, preds_05, zero_division=0):.4f}")
        print(f"  Prec (t=0.5):  {precision_score(Y, preds_05, zero_division=0):.4f}")
        print(f"  Rec (t=0.5):   {recall_score(Y, preds_05):.4f}")
        print(f"  Best F1:       {best_f1:.4f} (threshold={best_thresh:.2f})")


if __name__ == "__main__":
    main()
