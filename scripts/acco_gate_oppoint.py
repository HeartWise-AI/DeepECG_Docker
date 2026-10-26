"""Deployment operating points for the ACCO gate + top-1 culprit-vessel accuracy
when the gate fires. Uses cached embeddings + saved 3-head checkpoint."""
import os, sys, json, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_curve
_o = torch.load; torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})
sys.path.insert(0, "/volume/DeepECG_Docker/scripts")
from train_acs_acco_vessel import build_labels, VESSEL_CLASSES
EMB = "outputs/acs_acco_vessel"; CK = "checkpoints_acs_acco_vessel/best_model.pt"
device = torch.device("cpu")
ck = torch.load(CK, map_location="cpu")
hidden = 768
acco_head = nn.Sequential(nn.Linear(hidden,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,1))
acco_head.load_state_dict(ck["acco_head_state_dict"]); acco_head.eval()
vessel_head = nn.Sequential(nn.Linear(hidden,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,4))
vessel_head.load_state_dict({k.replace("vessel_head.",""):v for k,v in ck["model_state_dict"].items() if k.startswith("vessel_head.")}); vessel_head.eval()
L = build_labels()
def probs(sp):
    E = torch.from_numpy(np.asarray(np.load(f"{EMB}/emb_{sp}.npy", mmap_mode="r"))).float()
    with torch.no_grad():
        return torch.sigmoid(acco_head(E)[:,0]).numpy(), torch.sigmoid(vessel_head(E)).numpy()
pa_va,_ = probs("val"); pa_te, pv_te = probs("test")
acco_va, known_va = L["val"]["acco"], L["val"]["sev_known"]
acco_te, known_te, ves_te = L["test"]["acco"], L["test"]["sev_known"], L["test"]["vessel"]
yv, sv = acco_va[known_va==1], pa_va[known_va==1]
yt, st = acco_te[known_te==1], pa_te[known_te==1]
def thr_at_spec(spec):
    neg = np.sort(sv[yv==0]); return float(neg[int(np.ceil(spec*len(neg)))-1]) if len(neg) else 0.5
print("ACCO gate operating points (threshold fit on val, applied to test):")
for spec in [0.80, 0.90, 0.95]:
    t = thr_at_spec(spec); pred = st >= t
    tp=int(((pred)&(yt==1)).sum()); fp=int(((pred)&(yt==0)).sum()); fn=int(((~pred)&(yt==1)).sum()); tn=int(((~pred)&(yt==0)).sum())
    sens=tp/(tp+fn) if tp+fn else 0; spc=tn/(tn+fp) if tn+fp else 0; ppv=tp/(tp+fp) if tp+fp else 0
    # top-1 vessel accuracy among gated-positive cases that are truly ACCO with a vessel label
    gate = pa_te >= t
    m = gate & (acco_te==1) & (ves_te.sum(1)>0)
    if m.sum():
        pred_v = pv_te[m].argmax(1); true_v = ves_te[m].argmax(1)
        top1 = float((pred_v==true_v).mean())
    else: top1 = float('nan')
    shown = int(gate.sum())
    print(f"  target spec {spec:.2f}: thr={t:.4f} | test ACCO sens={sens:.3f} spec={spc:.3f} ppv={ppv:.3f} "
          f"| vessel shown for {shown}/{len(pa_te)} ECGs | top-1 vessel acc (true-ACCO,labeled)={top1:.3f} (n={int(m.sum())})")
