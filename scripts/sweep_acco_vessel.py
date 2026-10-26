"""
Auto research sweep: 5 experiments over the 3-head deployment model's head
hyperparameters (frozen unified-raw backbone; ACCO-vs-rest gate + ACCO-only vessel
head trained as MLPs on cached pooled embeddings). Select on val combined =
0.5*ACCO_AUROC + 0.5*vessel-on-ACCO macro. Save the BEST run.

Usage: python scripts/sweep_acco_vessel.py --device cuda:3
"""
import argparse, json, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
_o = torch.load; torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})
sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals"); sys.path.insert(0, "/volume/DeepECG_Docker/scripts")
from train_acs_acco_vessel import build_labels, extract_emb, boot_ci, VESSEL_CLASSES
from train_acs_dual_head_raw import build_model

UNIFIED = "checkpoints_acs_dual_head_v2_raw/best_model.pt"
SAVE = "checkpoints_acs_acco_vessel"   # best run overwrites the deployment checkpoint

# 5 experiments
CONFIGS = [
    {"name": "e1_baseline",   "dropout": 0.3, "lr": 1e-3, "wd": 2e-3, "acco_frac": 0.30, "vw": 1.0, "ah": 128, "vh": 256},
    {"name": "e2_highreg",    "dropout": 0.5, "lr": 1e-3, "wd": 5e-3, "acco_frac": 0.30, "vw": 1.0, "ah": 128, "vh": 256},
    {"name": "e3_moreacco",   "dropout": 0.3, "lr": 1e-3, "wd": 2e-3, "acco_frac": 0.40, "vw": 2.0, "ah": 128, "vh": 256},
    {"name": "e4_bigheads",   "dropout": 0.4, "lr": 1e-3, "wd": 3e-3, "acco_frac": 0.35, "vw": 1.5, "ah": 256, "vh": 512},
    {"name": "e5_lowlr",      "dropout": 0.3, "lr": 5e-4, "wd": 1e-3, "acco_frac": 0.35, "vw": 1.0, "ah": 128, "vh": 256},
]


def mlp(din, dh, dout, p):
    return nn.Sequential(nn.Linear(din, dh), nn.ReLU(), nn.Dropout(p), nn.Linear(dh, dout))


def run_config(cfg, E, L, hidden, device, epochs=600, patience=80, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    acco = mlp(hidden, cfg["ah"], 1, cfg["dropout"]).to(device)
    vessel = mlp(hidden, cfg["vh"], 4, cfg["dropout"]).to(device)
    Etr = torch.from_numpy(E["train"]).to(device); Eva = torch.from_numpy(E["val"]).to(device)
    acco_tr = torch.from_numpy(L["train"]["acco"]).to(device)
    known_tr = torch.from_numpy(L["train"]["sev_known"]).to(device)
    ves_tr = torch.from_numpy(L["train"]["vessel"]).to(device)
    acco_m = L["train"]["acco"] == 1
    nk = known_tr.sum().item(); npos = (acco_tr * known_tr).sum().item()
    acco_pw = torch.tensor([(nk - npos) / max(npos, 1)], device=device)
    vmask = acco_m & (L["train"]["vessel"].sum(1) > 0)
    vp = L["train"]["vessel"][vmask].sum(0); vn = vmask.sum() - vp
    ves_pw = torch.tensor(vn / np.clip(vp, 1, None), dtype=torch.float32, device=device)
    w = np.ones(len(E["train"])); pi = np.where(acco_m)[0]
    w[pi] = (cfg["acco_frac"]/(1-cfg["acco_frac"]))*((len(w)-len(pi))/len(pi)); w /= w.sum()
    opt = torch.optim.AdamW(list(acco.parameters()) + list(vessel.parameters()), lr=cfg["lr"], weight_decay=cfg["wd"])
    n = len(E["train"]); bs = 2048
    acco_va, known_va, ves_va = L["val"]["acco"], L["val"]["sev_known"], L["val"]["vessel"]
    am_va = acco_va == 1
    best, best_state, wait = -1, None, 0
    for ep in range(epochs):
        acco.train(); vessel.train()
        idx = np.random.choice(n, size=n, replace=True, p=w)
        for s in range(0, n, bs):
            bi = idx[s:s+bs]; e = Etr[bi]
            opt.zero_grad()
            km = known_tr[bi] > 0
            la = F.binary_cross_entropy_with_logits(acco(e)[:, 0][km], acco_tr[bi][km], pos_weight=acco_pw) if km.any() else torch.tensor(0., device=device)
            vm = (acco_tr[bi] == 1) & (ves_tr[bi].sum(1) > 0)
            lv = F.binary_cross_entropy_with_logits(vessel(e)[vm], ves_tr[bi][vm], pos_weight=ves_pw) if vm.any() else torch.tensor(0., device=device)
            (la + cfg["vw"]*lv).backward(); opt.step()
        acco.eval(); vessel.eval()
        with torch.no_grad():
            pa = torch.sigmoid(acco(Eva)[:, 0]).cpu().numpy(); pv = torch.sigmoid(vessel(Eva)).cpu().numpy()
        aa = roc_auc_score(acco_va[known_va == 1], pa[known_va == 1])
        vmac = np.mean([roc_auc_score(ves_va[am_va, j], pv[am_va, j]) for j in range(3) if ves_va[am_va, j].sum() >= 3])
        comb = 0.5*aa + 0.5*vmac
        if comb > best:
            best, wait = comb, 0
            best_state = ({k: v.detach().cpu().clone() for k, v in acco.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in vessel.state_dict().items()})
        else:
            wait += 1
            if wait >= patience: break
    return best, best_state, (acco, vessel)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:3"); args = ap.parse_args()
    device = torch.device(args.device)
    model, hidden = build_model(UNIFIED, "weights/wcr_77_classes", device)
    model.load_state_dict(torch.load(UNIFIED, map_location="cpu")["model_state_dict"]); model.eval()
    proj_w = model.base.proj.weight.detach().cpu().numpy(); proj_b = model.base.proj.bias.detach().cpu().numpy()
    L = build_labels()
    E = {sp: np.asarray(extract_emb(model, sp, device)) for sp in ["train", "val", "test"]}
    try:
        import wandb; os.environ.setdefault("WANDB_MODE", "offline"); WB = True
    except Exception:
        WB = False

    results = []
    for cfg in CONFIGS:
        if WB:
            run = wandb.init(project="deepecg-acs", name=f"sweep-{cfg['name']}", group="acco-vessel-sweep", config=cfg, reinit=True)
        val_comb, state, _ = run_config(cfg, E, L, hidden, device)
        results.append({"cfg": cfg, "val_combined": val_comb, "state": state})
        print(f"  [{cfg['name']}] val_combined={val_comb:.4f}", flush=True)
        if WB:
            wandb.summary["val_combined"] = val_comb; wandb.finish()

    results.sort(key=lambda r: r["val_combined"], reverse=True)
    best = results[0]
    print("\n=== SWEEP RANKING (val combined) ===")
    for r in results:
        print(f"  {r['cfg']['name']:14s} {r['val_combined']:.4f}")
    print(f"\nBEST: {best['cfg']['name']} (val_combined={best['val_combined']:.4f})")

    # ---- eval best on test + save deployment checkpoint ----
    acco = mlp(hidden, best["cfg"]["ah"], 1, best["cfg"]["dropout"]).to(device); acco.load_state_dict(best["state"][0]); acco.eval()
    vessel = mlp(hidden, best["cfg"]["vh"], 4, best["cfg"]["dropout"]).to(device); vessel.load_state_dict(best["state"][1]); vessel.eval()
    Ete = torch.from_numpy(E["test"]).to(device); Eva = torch.from_numpy(E["val"]).to(device)
    with torch.no_grad():
        pacco_te = torch.sigmoid(acco(Ete)[:, 0]).cpu().numpy(); pves_te = torch.sigmoid(vessel(Ete)).cpu().numpy()
        pacco_va = torch.sigmoid(acco(Eva)[:, 0]).cpu().numpy()
    acco_te, known_te, ves_te = L["test"]["acco"], L["test"]["sev_known"], L["test"]["vessel"]
    acco_va, known_va = L["val"]["acco"], L["val"]["sev_known"]
    acs_te = 1/(1+np.exp(-(E["test"] @ proj_w.T + proj_b)[:, 0]))
    gate_auc = roc_auc_score(acco_te[known_te == 1], pacco_te[known_te == 1])
    m = acco_te == 1
    vessel_res = {}
    for j, v in enumerate(VESSEL_CLASSES):
        yt = ves_te[m, j]; pp = pves_te[m, j]
        vessel_res[v] = ({"auroc": float(roc_auc_score(yt, pp)), "n_pos": int(yt.sum())}
                         if yt.sum() >= 5 and (1-yt).sum() >= 5 else {"auroc": None, "n_pos": int(yt.sum())})
    # top-1 at spec 0.90 gate
    yv, sv = acco_va[known_va == 1], pacco_va[known_va == 1]
    neg = np.sort(sv[yv == 0]); thr = float(neg[int(np.ceil(0.90*len(neg)))-1])
    gate = pacco_te >= thr; gm = gate & (acco_te == 1) & (ves_te.sum(1) > 0)
    top1 = float((pves_te[gm].argmax(1) == ves_te[gm].argmax(1)).mean()) if gm.sum() else None
    macro = float(np.mean([vessel_res[v]["auroc"] for v in VESSEL_CLASSES if vessel_res[v]["auroc"]]))
    test = {"acs_auroc": float(roc_auc_score(L["test"]["acs"], acs_te)), "acco_gate_auroc": float(gate_auc),
            "vessel_on_ACCO": vessel_res, "vessel_macro": macro, "top1_gate_spec90": top1, "gate_thr_spec90": thr}
    print("\n=== BEST on TEST ===")
    print(f"  ACS {test['acs_auroc']:.4f} | ACCO gate {gate_auc:.4f} | vessel macro {macro:.3f} | top-1@spec90 {top1}")

    base_sd = dict(torch.load(UNIFIED, map_location="cpu")["model_state_dict"])
    for k, v in best["state"][1].items(): base_sd[f"vessel_head.{k}"] = v
    torch.save({"model_state_dict": base_sd, "acco_head_state_dict": best["state"][0],
                "acco_head_arch": {"hidden": best["cfg"]["ah"], "dropout": best["cfg"]["dropout"]},
                "vessel_head_arch": {"hidden": best["cfg"]["vh"], "dropout": best["cfg"]["dropout"]},
                "vessel_classes": VESSEL_CLASSES, "init_from": UNIFIED, "sweep_best_cfg": best["cfg"],
                "gate_thr_spec90_val": thr, "test": test,
                "deploy": "show vessel only if P(ACCO) > gate_thr; vessel head trained on ACCO only"},
               os.path.join(SAVE, "best_model.pt"))
    json.dump({"ranking": [{"name": r["cfg"]["name"], "val_combined": r["val_combined"], "cfg": r["cfg"]} for r in results],
               "best": best["cfg"], "best_test": test},
              open("outputs/acs_acco_vessel/sweep_results.json", "w"), indent=2)
    print(f"\nsaved best -> {SAVE}/best_model.pt and outputs/acs_acco_vessel/sweep_results.json")


if __name__ == "__main__":
    main()
