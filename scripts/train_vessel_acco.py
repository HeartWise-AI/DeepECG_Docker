"""Train a culprit-vessel head on ACCO (complete occlusion) cases ONLY, on top of
the frozen unified raw backbone (so ACS is untouched). Deployment intent: only emit
a culprit vessel for complete occlusions; AICO / non-ACS report no vessel.

Frozen backbone -> precompute pooled embeddings (input to base.proj) for ACCO
vessel-labeled cases, then train the 4-vessel MLP. Eval on ACCO test."""
import json, os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
_o = torch.load; torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})
sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")
sys.path.insert(0, "/volume/DeepECG_Docker/scripts")
from train_acs_dual_head_raw import build_model, boot_ci, VESSEL_CLASSES, ADC_TO_MV

device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
UNIFIED = "checkpoints_acs_dual_head_v2_raw/best_model.pt"
SAVE = "checkpoints_acs_dual_head_v2_raw_acco_vessel"
os.makedirs(SAVE, exist_ok=True)
torch.manual_seed(42); np.random.seed(42)

old = pd.read_parquet("data/acs_linked_dataset.parquet"); old["key"] = old.ecg_key.astype(str)
sev = old.drop_duplicates("key").set_index("key")["condition_severity_modified"].to_dict()
dual = pd.read_parquet("data/acs_multilayout_v2_dual_linked.parquet")
dual["key"] = dual.npy_path.apply(lambda p: os.path.basename(p).replace(".npy", ""))

model, hidden = build_model(UNIFIED, "weights/wcr_77_classes", device)
model.load_state_dict(torch.load(UNIFIED, map_location="cpu")["model_state_dict"]); model.eval()

@torch.no_grad()
def embed(split, acco_only):
    d = dual[dual.Split == split].drop_duplicates("key")
    d = d[d.npy_path.apply(os.path.exists) & (d.Acute_Obstruction == 1)]
    d["sev"] = d.key.map(sev)
    d = d[d.sev.notna()]
    if acco_only:
        d = d[(d.sev == 3) & (d[VESSEL_CLASSES].sum(axis=1) > 0)]
    else:
        d = d[d.sev == 3]  # ACCO for eval
    embs, ves = [], []
    cap = {}
    h = model.base.proj.register_forward_hook(lambda m, i, o: cap.__setitem__("p", i[0]))
    try:
        for _, r in d.iterrows():
            a = np.load(r.npy_path).squeeze().astype(np.float32)
            if a.shape == (12, 2500): a = a.T
            if a.shape != (2500, 12): continue
            x = (np.nan_to_num(a) * ADC_TO_MV).T[None]  # (1,12,2500)
            model.base(source=torch.from_numpy(np.ascontiguousarray(x)).float().to(device))
            embs.append(cap["p"].squeeze(0).cpu().numpy())
            ves.append([r[v] for v in VESSEL_CLASSES])
    finally:
        h.remove()
    return np.stack(embs).astype(np.float32), np.array(ves, np.float32)

Etr, Vtr = embed("train", True)
Eva, Vva = embed("val", False)
Ete, Vte = embed("test", False)
print(f"ACCO train={len(Etr)} val={len(Eva)} test={len(Ete)}")

# warm-start head from the unified model's vessel head
head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4)).to(device)
sd = {k.replace("vessel_head.", ""): v for k, v in
      torch.load(UNIFIED, map_location="cpu")["model_state_dict"].items() if k.startswith("vessel_head.")}
head.load_state_dict(sd)
pos = Vtr.sum(0); pw = torch.tensor((len(Vtr) - pos) / np.clip(pos, 1, None), dtype=torch.float32, device=device)
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
Xt = torch.from_numpy(Etr).to(device); Tt = torch.from_numpy(Vtr).to(device)

def macro(E, V, idxs):
    head.eval()
    with torch.no_grad():
        p = torch.sigmoid(head(torch.from_numpy(E).to(device))).cpu().numpy()
    aus = [roc_auc_score(V[:, j], p[:, j]) for j in idxs if V[:, j].sum() >= 3 and (1 - V[:, j]).sum() >= 3]
    return float(np.mean(aus)) if aus else 0.0, p

best, best_state, wait = -1, None, 0
for ep in range(400):
    head.train(); opt.zero_grad()
    loss = F.binary_cross_entropy_with_logits(head(Xt), Tt, pos_weight=pw)
    loss.backward(); opt.step()
    m, _ = macro(Eva, Vva, [0, 1, 2])
    if m > best:
        best, wait, best_state = m, 0, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    else:
        wait += 1
        if wait >= 60: break
head.load_state_dict(best_state)
print(f"best val ACCO macro (LAD/RCA/LCX) = {best:.4f}")

_, pte = macro(Ete, Vte, [0, 1, 2])
print("\n=== TEST: ACCO-only vessel head, on ACCO test cases ===")
res = {}
old_acco = {"LAD": 0.822, "RCA": 0.768, "LCX": 0.737, "Left_Main": 0.674}  # unified model on ACCO
for j, v in enumerate(VESSEL_CLASSES):
    yt = Vte[:, j]; pp = pte[:, j]
    if yt.sum() >= 5 and (1 - yt).sum() >= 5:
        au = roc_auc_score(yt, pp); ci = boot_ci(yt, pp, roc_auc_score)
        res[v] = {"auroc": float(au), "auroc_ci": ci, "auprc": float(average_precision_score(yt, pp)), "n_pos": int(yt.sum())}
        print(f"  {v}: AUROC {au:.3f} CI({ci[0]:.3f},{ci[1]:.3f}) (was {old_acco[v]:.3f} unified)  n_pos={int(yt.sum())}")
    else:
        res[v] = {"auroc": None, "n_pos": int(yt.sum())}
        print(f"  {v}: n_pos={int(yt.sum())} insufficient")
aus = [res[v]["auroc"] for v in VESSEL_CLASSES if res[v]["auroc"]]
print(f"  macro AUROC: {np.mean(aus):.3f}")

# splice ACCO vessel head into a full checkpoint
base_sd = dict(torch.load(UNIFIED, map_location="cpu")["model_state_dict"])
for k, v in best_state.items():
    base_sd[f"vessel_head.{k}"] = v
torch.save({"model_state_dict": base_sd, "vessel_classes": VESSEL_CLASSES,
            "base_model": "unified raw + ACCO-only vessel head (frozen backbone)",
            "vessel_trained_on": "ACCO (complete occlusion) only; report vessel only for ACCO",
            "init_from": UNIFIED, "test_acco_vessel": res, "val_acco_macro": best},
           os.path.join(SAVE, "best_model.pt"))
json.dump({"checkpoint": os.path.join(SAVE, "best_model.pt"), "val_acco_macro": best,
           "test_acco_vessel": res}, open("outputs/acs_dual_head_raw/results_acco_vessel.json", "w"), indent=2)
print(f"\nsaved -> {SAVE}/best_model.pt")
