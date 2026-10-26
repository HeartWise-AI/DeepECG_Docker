"""Evaluate the unified raw dual-head best_model.pt on cached raw val/test.
ACS (full test, Youden fit on val) + per-vessel AUROC among ACS+ test. Bootstrap CIs."""
import json, os, sys, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
_o = torch.load; torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})
sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")
sys.path.insert(0, "/volume/DeepECG_Docker/scripts")
from train_acs_dual_head_raw import DualHeadWCR, build_model, predict, boot_ci, VESSEL_CLASSES

CACHE = "outputs/acs_dual_head_raw/cache"
device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
ck = sys.argv[2] if len(sys.argv) > 2 else "checkpoints_acs_dual_head_v2_raw/best_model.pt"

model, hidden = build_model(ck, "weights/wcr_77_classes", device)
state = torch.load(ck, map_location="cpu")
model.load_state_dict(state["model_state_dict"]); model.eval()
print(f"loaded {ck} (epoch {state.get('epoch')}, val_acs {state.get('val_acs_auroc'):.4f}, "
      f"val_vessel {state.get('val_vessel_macro'):.4f})")

def load(split):
    X = np.load(f"{CACHE}/raw_{split}_X.npy", mmap_mode="r")
    z = np.load(f"{CACHE}/raw_{split}_meta.npz")
    return X, z["acs"], z["ves"]

Xva, Ava, _ = load("val"); Xte, Ate, Vte = load("test")
pav, _ = predict(model, Xva, device)
pte, vte = predict(model, Xte, device)

auroc = roc_auc_score(Ate, pte); auprc = average_precision_score(Ate, pte)
fpr, tpr, thr = roc_curve(Ava, pav); thy = float(thr[np.argmax(tpr - fpr)])
pred = (pte >= thy).astype(int)
tp = int(((pred==1)&(Ate==1)).sum()); fp = int(((pred==1)&(Ate==0)).sum())
fn = int(((pred==0)&(Ate==1)).sum()); tn = int(((pred==0)&(Ate==0)).sum())
acs = {"auroc": auroc, "auroc_ci": boot_ci(Ate, pte, roc_auc_score),
       "auprc": auprc, "auprc_ci": boot_ci(Ate, pte, average_precision_score),
       "youden_thr_val": thy, "sens": tp/(tp+fn), "spec": tn/(tn+fp),
       "ppv": tp/(tp+fp) if tp+fp else 0, "npv": tn/(tn+fn) if tn+fn else 0,
       "n_test": int(len(Ate)), "n_pos": int(Ate.sum())}
m = Ate == 1; vessel = {}
for j, v in enumerate(VESSEL_CLASSES):
    yt = Vte[m, j]; pp = vte[m, j]
    if yt.sum() >= 5 and (1-yt).sum() >= 5:
        vessel[v] = {"auroc": float(roc_auc_score(yt, pp)), "auroc_ci": boot_ci(yt, pp, roc_auc_score),
                     "auprc": float(average_precision_score(yt, pp)), "n_pos": int(yt.sum())}
    else:
        vessel[v] = {"auroc": None, "n_pos": int(yt.sum())}
print("\n=== TEST (unified raw dual-head, ep%s) ===" % state.get("epoch"))
print(f"  ACS: AUROC {acs['auroc']:.4f} CI{tuple(round(x,3) for x in acs['auroc_ci'])} AUPRC {acs['auprc']:.4f}"
      f" | Youden Sens {acs['sens']:.3f} Spec {acs['spec']:.3f} NPV {acs['npv']:.3f} (n={acs['n_test']}, pos={acs['n_pos']})")
for v in VESSEL_CLASSES:
    d = vessel[v]
    print(f"  {v}: AUROC {d['auroc']:.3f} (n_pos={d['n_pos']})" if d["auroc"] else f"  {v}: n_pos={d['n_pos']} insufficient")
json.dump({"checkpoint": ck, "epoch": state.get("epoch"),
           "val_acs_auroc": state.get("val_acs_auroc"), "val_vessel_macro": state.get("val_vessel_macro"),
           "acs": acs, "vessel": vessel}, open("outputs/acs_dual_head_raw/results.json", "w"), indent=2)
print("\nsaved -> outputs/acs_dual_head_raw/results.json")
