"""
Partial fine-tune of the unified RAW dual-head model to push culprit-vessel
localization without wrecking ACS.

Recipe (per request: "unfreeze ~20% of layers, not only the head"):
  - Continue from checkpoints_acs_dual_head_v2_raw/best_model.pt (unified raw model).
  - Freeze the bottom 9 of 12 encoder layers + conv feature extractor (these hold the
    ACS-aware features) and unfreeze ONLY the top 3 transformer layers (~25%) + final
    layer norms + both heads (ACS proj + vessel head).
  - Joint loss with ACCO/AICO oversampling; vessel weight raised so localization moves
    while the frozen base anchors ACS.
  - Reuses the cached raw signals (outputs/acs_dual_head_raw/cache).

Usage:
    python scripts/train_acs_raw_partialft.py --device cuda:3
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

_o = torch.load
torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})
sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")
sys.path.insert(0, "/volume/DeepECG_Docker/scripts")
from train_acs_dual_head_raw import DualHeadWCR, build_model, predict, boot_ci, VESSEL_CLASSES  # noqa

CACHE = "outputs/acs_dual_head_raw/cache"


def load(split):
    X = np.load(f"{CACHE}/raw_{split}_X.npy", mmap_mode="r")
    z = np.load(f"{CACHE}/raw_{split}_meta.npz")
    return X, z["acs"], z["ves"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints_acs_dual_head_v2_raw/best_model.pt")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--unfreeze_top", type=int, default=3, help="top N of 12 encoder layers to train")
    ap.add_argument("--lr_enc", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--vessel_weight", type=float, default=3.0)
    ap.add_argument("--pos_frac", type=float, default=0.4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", default="checkpoints_acs_dual_head_v2_raw_partialft")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    model, hidden = build_model(args.init, "weights/wcr_77_classes", device)
    st = torch.load(args.init, map_location="cpu")
    model.load_state_dict(st["model_state_dict"])
    print(f"continued from {args.init} (ep{st.get('epoch')}, val_acs {st.get('val_acs_auroc'):.4f})", flush=True)

    # ---- freeze all base params except top-N layers + final LNs + ACS proj; vessel head trainable ----
    top_ids = set(range(12 - args.unfreeze_top, 12))
    enc_train, frozen = [], 0
    for name, p in model.named_parameters():
        keep = (
            name.startswith("vessel_head")
            or name.startswith("base.proj")
            or any(f"layers.{i}." in name for i in top_ids)
            or ("encoder.encoder.layer_norm" in name)
            or name.startswith("base.encoder.layer_norm")
        )
        p.requires_grad = keep
        if keep and not (name.startswith("vessel_head") or name.startswith("base.proj")):
            enc_train.append(p)
        if not keep:
            frozen += 1
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"unfrozen top layers {sorted(top_ids)} | trainable params {n_train/1e6:.1f}M / {n_total/1e6:.1f}M "
          f"({100*n_train/n_total:.0f}%); frozen tensors={frozen}", flush=True)

    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and (n.startswith("vessel_head") or n.startswith("base.proj"))]
    opt = torch.optim.AdamW([
        {"params": [p for p in enc_train if p.requires_grad], "lr": args.lr_enc},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler()

    Xtr, Atr, Vtr = load("train"); Xva, Ava, Vva = load("val"); Xte, Ate, Vte = load("test")
    n = len(Atr)
    acs_pos_weight = torch.tensor([(1 - args.pos_frac) / args.pos_frac], device=device)
    lab = Vtr.sum(1) > 0
    vp = Vtr[lab].sum(0); vn = lab.sum() - vp
    vessel_pos_weight = torch.tensor(vn / np.clip(vp, 1, None), dtype=torch.float32, device=device)
    pos_idx = np.where(np.asarray(Atr) == 1)[0]; neg_idx = np.where(np.asarray(Atr) == 0)[0]
    w = np.ones(n); w[pos_idx] = (args.pos_frac / (1 - args.pos_frac)) * (len(neg_idx) / len(pos_idx)); w /= w.sum()
    Atr_t = torch.from_numpy(np.asarray(Atr)).float(); Vtr_t = torch.from_numpy(np.asarray(Vtr)).float()

    try:
        import wandb
        os.environ.setdefault("WANDB_MODE", "offline")
        wandb.init(project="deepecg-acs", name="acs-raw-partialft-top%d" % args.unfreeze_top,
                   dir=".", config=vars(args)); use_wandb = True
    except Exception as e:
        print("wandb off", e); use_wandb = False

    best, best_state, wait, best_ep = -1, None, 0, 0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.choice(n, size=n, replace=True, p=w)
        t0 = time.time(); run = 0.0
        for s in range(0, n, args.batch_size):
            idx = np.sort(perm[s:s + args.batch_size])
            b = np.ascontiguousarray(np.transpose(Xtr[idx], (0, 2, 1)))
            src = torch.from_numpy(b).float().to(device)
            at = Atr_t[idx].unsqueeze(1).to(device); vt = Vtr_t[idx].to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                al, vl = model(src)
                la = F.binary_cross_entropy_with_logits(al, at, pos_weight=acs_pos_weight)
                m = (at.squeeze(1) == 1) & (vt.sum(1) > 0)
                lv = (F.binary_cross_entropy_with_logits(vl[m], vt[m], pos_weight=vessel_pos_weight)
                      if m.any() else torch.tensor(0.0, device=device))
                loss = la + args.vessel_weight * lv
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update()
            run += loss.item() * len(idx)

        pa, pv = predict(model, Xva, device)
        acs_auc = roc_auc_score(Ava, pa)
        vmac = np.mean([roc_auc_score(Vva[:, j], pv[:, j]) for j in range(3) if Vva[:, j].sum() >= 3])
        combined = 0.4 * acs_auc + 0.6 * vmac  # weight vessel more (that's what we're improving)
        print(f"  ep{ep+1:02d} loss={run/n:.4f} val_ACS={acs_auc:.4f} val_vessel={vmac:.4f} "
              f"combined={combined:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if use_wandb:
            wandb.log({"epoch": ep+1, "val_acs_auroc": acs_auc, "val_vessel_macro": vmac, "combined": combined})
        if combined > best:
            best, best_ep, wait = combined, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "vessel_classes": VESSEL_CLASSES,
                        "base_model": "WCRv2 raw unified + top%d partial FT" % args.unfreeze_top,
                        "init_from": args.init, "epoch": ep+1,
                        "val_acs_auroc": acs_auc, "val_vessel_macro": vmac},
                       os.path.join(args.save_dir, "best_model.pt"))
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  early stop @ ep{ep+1}", flush=True); break

    print(f"\nBest combined val={best:.4f} @ ep{best_ep+1}", flush=True)
    model.load_state_dict(best_state)

    pav, _ = predict(model, Xva, device); pa, pv = predict(model, Xte, device)
    auroc = roc_auc_score(Ate, pa); auprc = average_precision_score(Ate, pa)
    fpr, tpr, thr = roc_curve(Ava, pav); thy = float(thr[np.argmax(tpr - fpr)])
    pred = (pa >= thy).astype(int)
    tp = int(((pred==1)&(Ate==1)).sum()); fp=int(((pred==1)&(Ate==0)).sum())
    fn = int(((pred==0)&(Ate==1)).sum()); tn=int(((pred==0)&(Ate==0)).sum())
    acs = {"auroc": auroc, "auroc_ci": boot_ci(Ate, pa, roc_auc_score), "auprc": auprc,
           "sens": tp/(tp+fn), "spec": tn/(tn+fp), "npv": tn/(tn+fn) if tn+fn else 0,
           "n_test": int(len(Ate)), "n_pos": int(Ate.sum())}
    m = Ate == 1; vessel = {}
    for j, v in enumerate(VESSEL_CLASSES):
        yt = Vte[m, j]; pp = pv[m, j]
        vessel[v] = ({"auroc": float(roc_auc_score(yt, pp)), "auroc_ci": boot_ci(yt, pp, roc_auc_score),
                      "auprc": float(average_precision_score(yt, pp)), "n_pos": int(yt.sum())}
                     if yt.sum() >= 5 and (1-yt).sum() >= 5 else {"auroc": None, "n_pos": int(yt.sum())})
    print("\n=== TEST (raw unified + top%d partial FT) ===" % args.unfreeze_top)
    print(f"  ACS: AUROC {acs['auroc']:.4f} AUPRC {acs['auprc']:.4f} | Sens {acs['sens']:.3f} Spec {acs['spec']:.3f} NPV {acs['npv']:.3f}")
    for v in VESSEL_CLASSES:
        d = vessel[v]
        print(f"  {v}: AUROC {d['auroc']:.3f} (n_pos={d['n_pos']})" if d["auroc"] else f"  {v}: n_pos={d['n_pos']} insufficient")
    json.dump({"checkpoint": os.path.join(args.save_dir, "best_model.pt"), "best_combined_val": best,
               "acs": acs, "vessel": vessel}, open("outputs/acs_dual_head_raw/results_partialft.json", "w"), indent=2)
    if use_wandb:
        wandb.summary.update({"test_acs_auroc": acs["auroc"]})
        for v in VESSEL_CLASSES:
            if vessel[v]["auroc"] is not None:
                wandb.summary[f"test_{v}_auroc"] = vessel[v]["auroc"]
        wandb.finish()
    print("\nsaved -> outputs/acs_dual_head_raw/results_partialft.json", flush=True)


if __name__ == "__main__":
    main()
