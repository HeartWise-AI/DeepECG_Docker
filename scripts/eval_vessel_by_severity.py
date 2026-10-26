"""Evaluate culprit-vessel AUROC of a checkpoint stratified by occlusion severity:
ACCO (complete, sev=3) vs AICO (incomplete, sev=2) vs all ACS+.
Tests the hypothesis that vessel localization is reliable only for complete occlusions."""
import os, sys, numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
_o = torch.load; torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})
sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")
sys.path.insert(0, "/volume/DeepECG_Docker/scripts")
from train_acs_dual_head_raw import build_model, predict, boot_ci, VESSEL_CLASSES, ADC_TO_MV

device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
ck = sys.argv[2] if len(sys.argv) > 2 else "checkpoints_acs_dual_head_v2_raw/best_model.pt"

# severity map from old linked
old = pd.read_parquet("data/acs_linked_dataset.parquet")
old["key"] = old["ecg_key"].astype(str)
sev = old.drop_duplicates("key").set_index("key")["condition_severity_modified"].to_dict()

dual = pd.read_parquet("data/acs_multilayout_v2_dual_linked.parquet")
dual["key"] = dual.npy_path.apply(lambda p: os.path.basename(p).replace(".npy", ""))
df = dual[dual.Split == "test"].drop_duplicates("key")
df = df[df.npy_path.apply(os.path.exists)]
df = df[df.Acute_Obstruction == 1]                       # ACS+ only
df["sev"] = df["key"].map(sev)
df = df[df["sev"].notna()]
print(f"test ACS+ with severity label: {len(df)}  | ACCO(3)={int((df.sev==3).sum())} AICO(2)={int((df.sev==2).sum())}")

sigs, ves, sevs = [], [], []
for _, r in df.iterrows():
    a = np.load(r.npy_path).squeeze().astype(np.float32)
    if a.shape == (12, 2500): a = a.T
    if a.shape != (2500, 12): continue
    sigs.append(np.nan_to_num(a) * ADC_TO_MV)
    ves.append([r[v] for v in VESSEL_CLASSES]); sevs.append(int(r.sev))
X = np.stack(sigs).astype(np.float32); V = np.array(ves, np.float32); S = np.array(sevs)

model, _ = build_model(ck, "weights/wcr_77_classes", device)
model.load_state_dict(torch.load(ck, map_location="cpu")["model_state_dict"]); model.eval()
_, pv = predict(model, X, device)

def report(mask, tag):
    print(f"\n=== {tag} (n={int(mask.sum())}) ===")
    for j, v in enumerate(VESSEL_CLASSES):
        yt = V[mask, j]; pp = pv[mask, j]
        if yt.sum() >= 5 and (1 - yt).sum() >= 5:
            au = roc_auc_score(yt, pp); ci = boot_ci(yt, pp, roc_auc_score)
            print(f"  {v}: AUROC {au:.3f} CI({ci[0]:.3f},{ci[1]:.3f}) AUPRC {average_precision_score(yt,pp):.3f} (n_pos={int(yt.sum())})")
        else:
            print(f"  {v}: n_pos={int(yt.sum())} insufficient")
    # macro over evaluable
    aus = [roc_auc_score(V[mask, j], pv[mask, j]) for j in range(4)
           if V[mask, j].sum() >= 5 and (1 - V[mask, j]).sum() >= 5]
    print(f"  macro AUROC (evaluable): {np.mean(aus):.3f}" if aus else "  macro: n/a")

report(np.ones(len(S), bool), "ALL ACS+ (ACCO+AICO)")
report(S == 3, "ACCO only (complete occlusion)")
report(S == 2, "AICO only (incomplete occlusion)")
