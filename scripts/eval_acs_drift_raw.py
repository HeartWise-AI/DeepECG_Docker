"""Quantify ACS-head AUROC on the RAW test set (pos+neg) for the digitized v2
checkpoint vs the raw-signal location fine-tune, to measure ACS drift."""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
_o = torch.load
torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})
sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")
V = ["LAD", "RCA", "LCX", "Left_Main"]; ADC = 0.00488


class DualHeadWCR(nn.Module):
    def __init__(self, b, h=768):
        super().__init__(); self.base = b
        self.vessel_head = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 4))

    def forward(self, s):
        cap = {}
        hk = self.base.proj.register_forward_hook(lambda m, i, o: cap.__setitem__("p", i[0]))
        try:
            out = self.base(source=s); a = self.base.get_logits(out)
        finally:
            hk.remove()
        return a, self.vessel_head(cap["p"])


def build(ckpt, device):
    from fairseq_signals.utils import checkpoint_utils
    src = "checkpoints_acs_npy_ceiling/checkpoint_best.pt"
    bm, _, _ = checkpoint_utils.load_model_and_task(src, arg_overrides={"model_path": "weights/wcr_77_classes/base_ssl.pt"}, suffix="")
    h = bm.proj.in_features; bm.proj = nn.Linear(h, 1)
    m = DualHeadWCR(bm, h); m.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
    return m.to(device).eval()


@torch.no_grad()
def acs_probs(m, X, device, bs=64):
    out = []
    for i in range(0, len(X), bs):
        b = np.transpose(X[i:i+bs], (0, 2, 1))
        a, _ = m(torch.from_numpy(b).float().to(device))
        out.append(torch.sigmoid(a).cpu().numpy())
    a = np.concatenate(out, 0)
    return a[:, 0] if a.ndim == 2 else a


def main():
    device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:3")
    df = pd.read_parquet("data/acs_multilayout_v2_dual_linked.parquet")
    df["key"] = df.npy_path.apply(lambda p: os.path.basename(p).replace(".npy", ""))
    df = df[df.Split == "test"].drop_duplicates("key")
    df = df[df.npy_path.apply(os.path.exists)]
    # cap negatives for speed, keep all positives
    pos = df[df.Acute_Obstruction == 1]; neg = df[df.Acute_Obstruction == 0]
    neg = neg.sample(n=min(3000, len(neg)), random_state=42)
    use = pd.concat([pos, neg])
    X, Y = [], []
    for _, r in use.iterrows():
        a = np.load(r.npy_path).squeeze().astype(np.float32)
        if a.shape == (12, 2500): a = a.T
        if a.shape != (2500, 12): continue
        X.append(np.nan_to_num(a) * ADC); Y.append(r.Acute_Obstruction)
    X = np.stack(X).astype(np.float32); Y = np.array(Y, dtype=np.float32)
    print(f"raw test for ACS: n={len(Y)} pos={int(Y.sum())} (capped negatives)")
    for tag, ck in [("digitized v2 (orig)", "checkpoints_acs_dual_head_v2_digitized/best_model.pt"),
                    ("raw location finetune", "checkpoints_acs_location_v2_raw/best_model.pt")]:
        m = build(ck, device); p = acs_probs(m, X, device)
        print(f"  {tag:24s}: ACS AUROC {roc_auc_score(Y, p):.4f}  AUPRC {average_precision_score(Y, p):.4f}")


if __name__ == "__main__":
    main()
