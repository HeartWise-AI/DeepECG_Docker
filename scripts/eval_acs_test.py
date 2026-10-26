"""
Evaluate a fine-tuned WCR checkpoint on the ACS test set.

Usage:
    python scripts/eval_acs_test.py \
        --checkpoint checkpoints_acs_npy_ceiling/checkpoint_best.pt \
        --manifest_dir finetune_data/acs_npy/manifest_npy \
        --split test
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from fairseq_signals.utils import checkpoint_utils


def load_model(checkpoint_path, device):
    """Load fine-tuned model from checkpoint."""
    # Use checkpoint_utils to load model + task from the checkpoint
    model, cfg, task = checkpoint_utils.load_model_and_task(
        checkpoint_path,
        arg_overrides={"model_path": "/volume/DeepECG_Docker/weights/wcr_77_classes/base_ssl.pt"},
        suffix=""
    )
    model = model.to(device)
    model.eval()
    return model


def load_data(manifest_dir, split):
    """Load X and Y from manifest."""
    manifest_path = os.path.join(manifest_dir, f"{split}.tsv")
    with open(manifest_path) as f:
        lines = f.read().strip().split("\n")

    info = {}
    for line in lines:
        key, val = line.split(":", 1)
        info[key] = val

    X = np.load(info["x_path"])  # (N, 2500, 12, 1)
    Y = np.load(info["y_path"])  # (N, 1)

    return X, Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)

    split_name = "valid" if args.split == "val" else args.split
    print(f"Loading {args.split} data from {args.manifest_dir}...")
    X, Y = load_data(args.manifest_dir, split_name)
    print(f"  X: {X.shape}, Y: {Y.shape}")

    if Y.ndim == 2:
        Y = Y[:, 0]

    # Run inference in batches
    all_probs = []
    n = len(X)

    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            batch_x = X[i:i+args.batch_size]
            # Shape: (B, 2500, 12, 1) -> need (B, 12, 2500) for WCR
            batch_x = batch_x[:, :, :, 0]  # (B, 2500, 12)
            batch_x = np.transpose(batch_x, (0, 2, 1))  # (B, 12, 2500)

            source = torch.from_numpy(batch_x).float().to(device)
            padding_mask = torch.zeros(source.shape[0], source.shape[2], dtype=torch.bool, device=device)

            net_input = {"source": source, "padding_mask": padding_mask}
            net_output = model(**net_input)
            logits = model.get_logits(net_output).float()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

            if (i // args.batch_size) % 10 == 0:
                print(f"  Batch {i//args.batch_size + 1}/{(n + args.batch_size - 1)//args.batch_size}")

    all_probs = np.concatenate(all_probs, axis=0)
    if all_probs.ndim == 2:
        all_probs = all_probs[:, 0]

    # Compute metrics
    auroc = roc_auc_score(Y, all_probs)
    auprc = average_precision_score(Y, all_probs)

    # Find best threshold for F1
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs > thresh).astype(int)
        f1 = f1_score(Y, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    preds_best = (all_probs > best_thresh).astype(int)
    preds_05 = (all_probs > 0.5).astype(int)

    print(f"\n{'='*50}")
    print(f"Results on {args.split} set (n={n}, pos={int(Y.sum())})")
    print(f"{'='*50}")
    print(f"  AUROC:  {auroc:.4f}")
    print(f"  AUPRC:  {auprc:.4f}")
    print(f"  F1 (t=0.5):    {f1_score(Y, preds_05):.4f}")
    print(f"  Prec (t=0.5):  {precision_score(Y, preds_05, zero_division=0):.4f}")
    print(f"  Rec (t=0.5):   {recall_score(Y, preds_05):.4f}")
    print(f"  Best F1:       {best_f1:.4f} (threshold={best_thresh:.2f})")
    print(f"  Prec (best t): {precision_score(Y, preds_best, zero_division=0):.4f}")
    print(f"  Rec (best t):  {recall_score(Y, preds_best):.4f}")


if __name__ == "__main__":
    main()
