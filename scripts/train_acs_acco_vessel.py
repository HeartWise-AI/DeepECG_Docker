"""
3-head deployment model on the frozen unified-raw backbone:
  - ACS head (existing base.proj, UNCHANGED): ACS = ACCO u AICO.
  - ACCO-vs-rest head (NEW): P(complete occlusion). Gates vessel display.
  - Vessel location head: trained ONLY on ACCO cases; location for AICO / no-ACS
    is "unknown" (masked out of training).

Deployment rule: show a culprit vessel ONLY when P(ACCO) > its val-fit Youden
threshold; otherwise vessel = unknown.

Frozen backbone -> pooled embeddings are precomputed/cached once, so the two heads
train as fast MLPs and we can iterate. Severity from condition_severity_modified
(data/acs_linked_dataset.parquet); 3=ACCO, 2=AICO.

Usage: python scripts/train_acs_acco_vessel.py --device cuda:1
"""
import argparse, json, os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
_o = torch.load; torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})
sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals"); sys.path.insert(0, "/volume/DeepECG_Docker/scripts")
from train_acs_dual_head_raw import build_model, VESSEL_CLASSES

CACHE = "outputs/acs_dual_head_raw/cache"
UNIFIED = "checkpoints_acs_dual_head_v2_raw/best_model.pt"
SAVE = "checkpoints_acs_acco_vessel"
EMB = "outputs/acs_acco_vessel"


def boot_ci(y, p, fn, n=1000, seed=42):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < 2: continue
        out.append(fn(y[s], p[s]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else [None, None]


def build_labels():
    old = pd.read_parquet("data/acs_linked_dataset.parquet"); old["key"] = old.ecg_key.astype(str)
    sev = old.drop_duplicates("key").set_index("key")["condition_severity_modified"].to_dict()
    dual = pd.read_parquet("data/acs_multilayout_v2_dual_linked.parquet")
    dual["key"] = dual.npy_path.apply(lambda p: os.path.basename(p).replace(".npy", ""))
    out = {}
    for sp in ["train", "val", "test"]:
        df = dual[dual.Split == sp].drop_duplicates("key")
        df = df[df.npy_path.apply(os.path.exists)]
        s = df["key"].map(sev)
        out[sp] = {
            "acs": df["Acute_Obstruction"].values.astype(np.float32),
            "acco": (s == 3).values.astype(np.float32),
            "sev_known": s.notna().values.astype(np.float32),
            "vessel": df[VESSEL_CLASSES].values.astype(np.float32),
        }
    return out


@torch.no_grad()
def extract_emb(model, split, device, bs=256):
    f = f"{EMB}/emb_{split}.npy"
    if os.path.exists(f):
        return np.load(f, mmap_mode="r")
    X = np.load(f"{CACHE}/raw_{split}_X.npy", mmap_mode="r")
    cap = {}; h = model.base.proj.register_forward_hook(lambda m, i, o: cap.__setitem__("p", i[0]))
    embs = []
    try:
        for i in range(0, len(X), bs):
            b = np.ascontiguousarray(np.transpose(X[i:i+bs], (0, 2, 1)))
            model.base(source=torch.from_numpy(b).float().to(device))
            embs.append(cap["p"].cpu().numpy())
    finally:
        h.remove()
    E = np.concatenate(embs).astype(np.float32)
    os.makedirs(EMB, exist_ok=True); np.save(f, E)
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=2e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--acco_pos_frac", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--vessel_weight", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(SAVE, exist_ok=True)
    device = torch.device(args.device)

    model, hidden = build_model(UNIFIED, "weights/wcr_77_classes", device)
    model.load_state_dict(torch.load(UNIFIED, map_location="cpu")["model_state_dict"]); model.eval()
    proj_w = model.base.proj.weight.detach().cpu().numpy(); proj_b = model.base.proj.bias.detach().cpu().numpy()

    L = build_labels()
    E = {sp: np.asarray(extract_emb(model, sp, device)) for sp in ["train", "val", "test"]}
    for sp in ["train", "val", "test"]:
        assert len(E[sp]) == len(L[sp]["acs"]), (sp, len(E[sp]), len(L[sp]["acs"]))
    print(f"emb train{E['train'].shape} val{E['val'].shape} test{E['test'].shape}")

    def acs_prob(sp):  # unchanged ACS head
        return 1 / (1 + np.exp(-(E[sp] @ proj_w.T + proj_b)[:, 0]))

    acco_head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(128, 1)).to(device)
    vessel_head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(256, 4)).to(device)
    sd = {k.replace("vessel_head.", ""): v for k, v in torch.load(UNIFIED, map_location="cpu")["model_state_dict"].items() if k.startswith("vessel_head.")}
    vessel_head.load_state_dict(sd)

    Etr = torch.from_numpy(E["train"]).to(device)
    acco_tr = torch.from_numpy(L["train"]["acco"]).to(device)
    known_tr = torch.from_numpy(L["train"]["sev_known"]).to(device)
    ves_tr = torch.from_numpy(L["train"]["vessel"]).to(device)
    acco_mask_train = (L["train"]["acco"] == 1)
    ves_train_mask = acco_mask_train & (L["train"]["vessel"].sum(1) > 0)   # ACCO + vessel-labeled
    print(f"train ACCO={int(acco_mask_train.sum())} ACCO+vessel-labeled={int(ves_train_mask.sum())} sev_known={int(known_tr.sum())}")

    # ACCO pos_weight (within sev_known) and vessel pos_weight (within ACCO-labeled)
    nk = known_tr.sum().item(); npos = (acco_tr * known_tr).sum().item()
    acco_pw = torch.tensor([(nk - npos) / max(npos, 1)], device=device)
    vp = L["train"]["vessel"][ves_train_mask].sum(0); vn = ves_train_mask.sum() - vp
    ves_pw = torch.tensor(vn / np.clip(vp, 1, None), dtype=torch.float32, device=device)
    print(f"acco_pos_weight={acco_pw.item():.1f}  vessel_pos_weight={ves_pw.tolist()}")

    # oversample ACCO for batches
    w = np.ones(len(E["train"]))
    pos_idx = np.where(acco_mask_train)[0]
    if len(pos_idx):
        w[pos_idx] = (args.acco_pos_frac/(1-args.acco_pos_frac)) * ((len(w)-len(pos_idx))/len(pos_idx))
    w = w/w.sum()

    params = list(acco_head.parameters()) + list(vessel_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    try:
        import wandb; os.environ.setdefault("WANDB_MODE", "offline")
        wandb.init(project="deepecg-acs", name="acs-acco-vessel-3head", dir=".", config=vars(args)); W = True
    except Exception as e:
        print("wandb off", e); W = False

    Eva = torch.from_numpy(E["val"]).to(device)
    n = len(E["train"]); bs = args.batch
    best, best_state, wait, best_ep = -1, None, 0, 0
    acco_va, known_va, ves_va = L["val"]["acco"], L["val"]["sev_known"], L["val"]["vessel"]
    acco_mask_va = acco_va == 1

    for ep in range(args.epochs):
        acco_head.train(); vessel_head.train()
        idx = np.random.choice(n, size=n, replace=True, p=w)
        for s in range(0, n, bs):
            bi = idx[s:s+bs]
            e = Etr[bi]
            opt.zero_grad()
            al = acco_head(e)[:, 0]; vl = vessel_head(e)
            km = known_tr[bi] > 0
            la = F.binary_cross_entropy_with_logits(al[km], acco_tr[bi][km], pos_weight=acco_pw) if km.any() else torch.tensor(0., device=device)
            vm = (acco_tr[bi] == 1) & (ves_tr[bi].sum(1) > 0)
            lv = F.binary_cross_entropy_with_logits(vl[vm], ves_tr[bi][vm], pos_weight=ves_pw) if vm.any() else torch.tensor(0., device=device)
            (la + args.vessel_weight*lv).backward(); opt.step()

        acco_head.eval(); vessel_head.eval()
        with torch.no_grad():
            pa = torch.sigmoid(acco_head(Eva)[:, 0]).cpu().numpy()
            pv = torch.sigmoid(vessel_head(Eva)).cpu().numpy()
        acco_auc = roc_auc_score(acco_va[known_va == 1], pa[known_va == 1])
        vmac = np.mean([roc_auc_score(ves_va[acco_mask_va, j], pv[acco_mask_va, j]) for j in range(3)
                        if ves_va[acco_mask_va, j].sum() >= 3])
        combined = 0.5*acco_auc + 0.5*vmac
        if combined > best:
            best, best_ep, wait = combined, ep, 0
            best_state = ({k: v.detach().cpu().clone() for k, v in acco_head.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in vessel_head.state_dict().items()})
        else:
            wait += 1
            if wait >= args.patience: break
        if W: wandb.log({"epoch": ep+1, "val_acco_auroc": acco_auc, "val_vessel_acco_macro": vmac, "combined": combined})
        if ep % 25 == 0 or wait == 0:
            print(f"  ep{ep+1:03d} val_ACCO_auroc={acco_auc:.4f} val_vessel_ACCO_macro={vmac:.4f} combined={combined:.4f}", flush=True)

    acco_head.load_state_dict(best_state[0]); vessel_head.load_state_dict(best_state[1])
    print(f"\nBest combined val={best:.4f} @ ep{best_ep+1}")

    # ---- TEST ----
    Ete = torch.from_numpy(E["test"]).to(device)
    acco_te, known_te, ves_te = L["test"]["acco"], L["test"]["sev_known"], L["test"]["vessel"]
    acco_head.eval(); vessel_head.eval()
    with torch.no_grad():
        pacco = torch.sigmoid(acco_head(Ete)[:, 0]).cpu().numpy()
        pves = torch.sigmoid(vessel_head(Ete)).cpu().numpy()
    pacs = acs_prob("test")
    # ACCO gate quality
    acco_auc = roc_auc_score(acco_te[known_te == 1], pacco[known_te == 1])
    acco_ap = average_precision_score(acco_te[known_te == 1], pacco[known_te == 1])
    # Youden threshold for ACCO on val
    pacco_va = None
    with torch.no_grad():
        pacco_va = torch.sigmoid(acco_head(Eva)[:, 0]).cpu().numpy()
    fpr, tpr, thr = roc_curve(acco_va[known_va == 1], pacco_va[known_va == 1]); thy = float(thr[np.argmax(tpr-fpr)])
    # vessel among true-ACCO test
    m = acco_te == 1
    vessel = {}
    for j, v in enumerate(VESSEL_CLASSES):
        yt = ves_te[m, j]; pp = pves[m, j]
        vessel[v] = ({"auroc": float(roc_auc_score(yt, pp)), "auroc_ci": boot_ci(yt, pp, roc_auc_score),
                      "n_pos": int(yt.sum())} if yt.sum() >= 5 and (1-yt).sum() >= 5 else {"auroc": None, "n_pos": int(yt.sum())})
    res = {
        "acs_auroc": float(roc_auc_score(L["test"]["acs"], pacs)),
        "acco_vs_rest": {"auroc": float(acco_auc), "auroc_ci": boot_ci(acco_te[known_te==1], pacco[known_te==1], roc_auc_score),
                          "auprc": float(acco_ap), "youden_thr_val": thy,
                          "n_pos": int(acco_te.sum()), "n": int(known_te.sum())},
        "vessel_on_ACCO": vessel,
    }
    print("\n=== TEST (3-head: ACS / ACCO-vs-rest / vessel-on-ACCO) ===")
    print(f"  ACS AUROC: {res['acs_auroc']:.4f} (unchanged head)")
    print(f"  ACCO-vs-rest AUROC: {acco_auc:.4f} CI{tuple(round(x,3) for x in res['acco_vs_rest']['auroc_ci'])} "
          f"AUPRC {acco_ap:.3f} (gate thr={thy:.3f})")
    for v in VESSEL_CLASSES:
        d = vessel[v]
        print(f"  vessel/{v}: AUROC {d['auroc']:.3f} (n_pos={d['n_pos']})" if d["auroc"] else f"  vessel/{v}: n_pos={d['n_pos']} insufficient")
    aus = [vessel[v]["auroc"] for v in VESSEL_CLASSES if vessel[v]["auroc"]]
    res["vessel_on_ACCO_macro"] = float(np.mean(aus))
    print(f"  vessel-on-ACCO macro: {np.mean(aus):.3f}")

    base_sd = dict(torch.load(UNIFIED, map_location="cpu")["model_state_dict"])
    for k, v in best_state[1].items(): base_sd[f"vessel_head.{k}"] = v
    torch.save({"model_state_dict": base_sd,
                "acco_head_state_dict": best_state[0],
                "vessel_classes": VESSEL_CLASSES, "init_from": UNIFIED,
                "deploy": "show vessel only if P(ACCO) > youden_thr_val; vessel head trained on ACCO only",
                "youden_thr_acco_val": thy, "test": res},
               os.path.join(SAVE, "best_model.pt"))
    json.dump(res, open(f"{EMB}/results.json", "w"), indent=2)
    if W:
        wandb.summary.update({"test_acs_auroc": res["acs_auroc"], "test_acco_auroc": acco_auc,
                              "test_vessel_acco_macro": res["vessel_on_ACCO_macro"]}); wandb.finish()
    print(f"\nsaved -> {SAVE}/best_model.pt and {EMB}/results.json")


if __name__ == "__main__":
    main()
