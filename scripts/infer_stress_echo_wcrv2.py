"""Inference on the held-out stress-echo test set with the WCRv2 3-class model.

Reproduces the reported test metrics (macro AUROC 0.8713) from the sweep winner.

Usage:
    python scripts/infer_stress_echo_wcrv2.py
    python scripts/infer_stress_echo_wcrv2.py --config scripts/stress_echo_inference_config.yaml --device cuda:3
    python scripts/infer_stress_echo_wcrv2.py --out_csv outputs/stress_echo_preds.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")

DEFAULT_CFG = "/volume/DeepECG_Docker/scripts/stress_echo_inference_config.yaml"


def load_model(cfg, device):
    from fairseq_signals.utils import checkpoint_utils
    m = cfg["model"]
    base, _, _ = checkpoint_utils.load_model_and_task(
        m["base_checkpoint"],
        arg_overrides={"model_path": m["base_ssl_placeholder"]},
        suffix="",
    )
    base.proj = nn.Sequential(nn.Dropout(0.1), nn.Linear(m["hidden_dim"], m["num_classes"]))
    state = torch.load(m["finetuned_checkpoint"], map_location=device, weights_only=False)
    base.load_state_dict(state["model"])
    return base.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device or cfg["inference"]["device"])
    bs = cfg["inference"]["batch_size"]
    class_names = cfg["model"]["class_names"]

    x = np.load(cfg["data"]["x_npy"])        # (N, 2500, 12, 1)
    y = np.load(cfg["data"]["y_npy"])        # (N,)
    x = torch.from_numpy(x.squeeze(-1)).permute(0, 2, 1).contiguous()  # (N, 12, 2500)
    print(f"Loaded test set: x={tuple(x.shape)}  y dist={np.bincount(y, minlength=3).tolist()}")

    model = load_model(cfg, device)
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = x[i:i + bs].to(device, non_blocking=True)
            out = model(source=xb)
            logits = model.get_logits(out).float()
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(probs, axis=0)

    # Metrics
    print(f"\n{'Class':<22}{'n':>6}{'prev %':>10}{'AUROC':>10}{'AUPRC':>10}")
    aurocs, auprcs = [], []
    for c, name in enumerate(class_names):
        tb = (y == c).astype(int)
        auroc = roc_auc_score(tb, probs[:, c]) if 0 < tb.sum() < len(tb) else float("nan")
        auprc = average_precision_score(tb, probs[:, c]) if tb.sum() else float("nan")
        aurocs.append(auroc); auprcs.append(auprc)
        print(f"{name:<22}{tb.sum():>6}{100 * tb.mean():>10.2f}{auroc:>10.4f}{auprc:>10.4f}")
    print(f"{'Macro':<22}{len(y):>6}{100.0:>10.2f}{np.nanmean(aurocs):>10.4f}{np.nanmean(auprcs):>10.4f}")

    if args.out_csv:
        cohort = pd.read_parquet(cfg["data"]["cohort_parquet"])
        test = cohort[cohort["split"] == "test"].reset_index(drop=True)
        assert len(test) == len(probs), f"{len(test)} vs {len(probs)}"
        for c, name in enumerate(class_names):
            test[f"prob_{name}"] = probs[:, c]
        test["pred_label"] = probs.argmax(axis=1)
        test["true_label"] = y
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        test.to_csv(args.out_csv, index=False)
        print(f"\nPredictions CSV -> {args.out_csv}")


if __name__ == "__main__":
    main()
