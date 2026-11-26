"""DINOv3 probe heads of DeepECG_Docker against the cached features and saved scores of their fits.

Run where the zo59vdxk run folder is (lightning_shreya; its DeepECG-PNG venv has torch,
transformers, safetensors and numpy), from a checkout of this branch:

    /volume/DeepECG-PNG-dinov3-siglip2/.venv/bin/python scripts/dinov3_probe_parity.py \\
        --snapshot /media/data1/models/DeepECG-PNG/acs_ft20_dinov3_siglip2_epoch030 \\
        --images 16 --device cuda:0 --out dinov3_parity.json

--probes-dir picks the probe folder under --snapshot. Default: probes_all18 when it exists (the
NAS layout once probes/ holds only the six heads served by PACS-AI), otherwise probes (the
Hugging Face archive layout, all 18 heads). The script compares every head of that folder.

1. Feature -> score, every head: the val and test features cached for the fit (provenance.json
   split_files) go through models/dinov3_bank.head_probabilities (float32, as served) and are
   compared with the saved scores of the fit (provenance.json predictions). Pass: max |diff| <= 2e-5.
2. Image -> score: the first --images test pages of the ACS and SHD/LVEF caches are read from
   their files and run end to end (utils/image_preprocessing.load_ecg_page with the bank's
   preprocessing, the DINOv3 encoder under bfloat16 autocast on --device, the heads). Reported:
   max |feature diff| and max |score diff| against the cached features. bf16 noise is expected.

Nothing is written but --out (aggregate numbers only: no file names or patient data).
Pure reading: needs no Hugging Face access and no fairseq-signals (models/__init__ is not run).
models/dinov3_bank.py is the code served; it imports only torch, transformers, safetensors and utils.
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
# Import models/dinov3_bank.py without models/__init__.py (it loads the WCR / fairseq models).
_models = types.ModuleType("models")
_models.__path__ = [os.path.join(REPO, "models")]
sys.modules["models"] = _models
# The DeepECG-PNG venv this runs in ships its own top-level `utils` package, which would shadow
# this repo's (no utils/__init__.py: a namespace package). Bind `utils` to this repo explicitly.
_utils = types.ModuleType("utils")
_utils.__path__ = [os.path.join(REPO, "utils")]
sys.modules["utils"] = _utils
from models import dinov3_bank as dw  # noqa: E402

TOLERANCE = 2e-5


def safe(name: str) -> str:
    """DeepECG-PNG scripts/probe_report.py _safe."""
    return name.replace("<=", "le").replace("<", "lt").replace("/", "_")


def cache(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def saved_scores(predictions: dict, provenance: dict, head: dict, split: str, n_rows: int, labels=None):
    """[(output, rows or None, saved scores)] for one split, or a reason string."""
    names = list((provenance.get("verification") or {}).get("scores_vs_saved_report_predictions") or {})
    if "cls77" == provenance.get("cohort"):
        labels = [str(x) for x in predictions["labels"]]
        scores = predictions[f"{split}_score"]
        if scores.shape[0] != n_rows:
            return f"{split}_score has {scores.shape[0]} rows, the cache {n_rows}"
        return [(label, None, scores[:, j]) for j, label in enumerate(labels)]
    key = f"{safe(provenance['task'])}__{split}__score"
    if key in predictions:                                   # scripts/probe_report.py fit_binary
        return [(head["outputs"][0], predictions[f"{safe(provenance['task'])}__{split}__rows"], predictions[key])]
    # scripts/probe_extra_cohorts.py report: "<view>__<table>__sv|st", view names unmodified, table "w_o aug"
    # for the clean arm and "w_ aug" for the aug arm; ACCO vs AICO rows are the occlusion ECGs only.
    suffix = "sv" if split == "val" else "st"
    table = {"clean": "w_o aug", "aug": "w_ aug"}.get(provenance.get("arm"))
    if table is None:
        return f"unknown probe arm {provenance.get('arm')!r}"
    keep = None
    if "in_B == 1" in (provenance.get("row_filter") or ""):
        keep = np.asarray(labels)[:, 2] == 1                 # acs label columns: ACO, ACCO, in_B
    n = n_rows if keep is None else int(keep.sum())
    found = []
    for name in names:
        key = f"{name}__{table}__{suffix}"
        if key not in predictions:
            return f"no saved key {key!r}"
        output = name.split(": ", 1)[1] if ": " in name else head["outputs"][0]
        if output not in head["outputs"]:
            return f"view {name!r} names no output of the head"
        if len(predictions[key]) != n:
            return f"{key} has {len(predictions[key])} rows, the cache {n}"
        found.append((output, keep, predictions[key]))
    return found


def default_probes_dir(snapshot: str) -> str:
    """probes_all18 (NAS, after the 6-head probes/ swap) when present, else probes."""
    return "probes_all18" if os.path.isdir(os.path.join(snapshot, "probes_all18")) else "probes"


def compare_head(key: str, snapshot: str, bank) -> dict:
    folder = os.path.join(snapshot, bank.probes_dir, key)
    provenance = json.load(open(os.path.join(folder, "provenance.json")))
    task = key.rsplit("_", 1)[0]                              # <task>_clean / <task>_aug
    head = bank.load_head(key, dw.PROBE_OUTPUTS[task])
    predictions = dict(np.load(provenance["predictions"]["path"], allow_pickle=False))
    result = {"head": key, "splits": {}}
    for split in ("val", "test"):
        data = cache(provenance["split_files"][split]["path"])
        features = data["features"].float()
        ours = dw.head_probabilities(features, head).numpy()
        saved = saved_scores(predictions, provenance, head, split, len(features), data.get("labels"))
        if isinstance(saved, str):
            result["splits"][split] = {"status": "not compared", "reason": saved}
            continue
        diffs = {}
        for output, rows, scores in saved:
            j = head["outputs"].index(output)
            mine = ours[:, j] if rows is None else ours[np.asarray(rows), j]
            diffs[output] = float(np.abs(mine - np.asarray(scores, dtype=np.float64)).max())
        numeric = [v for v in diffs.values() if isinstance(v, float)]
        result["splits"][split] = {
            "n": int(len(features)),
            "max_abs_diff": max(numeric) if numeric else None,
            "pass": bool(numeric) and max(numeric) <= TOLERANCE and len(numeric) == len(diffs),
            "unmatched": {k: v for k, v in diffs.items() if not isinstance(v, float)},
        }
    return result


def image_parity(snapshot: str, bank, n_images: int) -> dict:
    from utils.image_preprocessing import load_ecg_page
    out = {}
    for key in ("aco_clean", "shd_clean"):
        provenance = json.load(open(os.path.join(snapshot, bank.probes_dir, key, "provenance.json")))
        data = cache(provenance["split_files"]["test"]["path"])
        files = list(data["files"])[:n_images]
        cached = data["features"][: len(files)].float()
        pages = torch.stack([load_ecg_page(f, bank.preprocessing) for f in files])
        bank._features.clear()
        mine = bank.features(pages)
        heads = {t: bank.load_head(f"{t}_clean", o) for t, o in dw.PROBE_OUTPUTS.items()}
        score_diff = {name: float((dw.head_probabilities(mine, h) - dw.head_probabilities(cached, h)).abs().max())
                      for name, h in heads.items()}
        out[provenance["cohort"]] = {
            "n_images": len(files),
            "feature_max_abs_diff": float((mine - cached).abs().max()),
            "feature_min_cosine": float(torch.nn.functional.cosine_similarity(mine, cached).min()),
            "score_max_abs_diff": score_diff,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--snapshot", required=True, help="local copy of the repository (base/ and probes/)")
    parser.add_argument("--probes-dir", default=None,
                        help="probe folder under --snapshot (default: probes_all18 if present, else probes)")
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="dinov3_parity.json")
    args = parser.parse_args()
    probes_dir = args.probes_dir or default_probes_dir(args.snapshot)
    bank = dw._DeepECG_DINOv3_Bank(args.snapshot, torch.device(args.device), probes_dir=probes_dir)
    root = os.path.join(args.snapshot, probes_dir)
    keys = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    print(f"probes folder: {root} ({len(keys)} heads)")
    report = {"tolerance": TOLERANCE, "probes_dir": probes_dir, "n_heads": len(keys), "environment": {"torch": torch.__version__,
              "transformers": __import__("transformers").__version__, "device": args.device},
              "feature_to_score": [compare_head(k, args.snapshot, bank) for k in keys]}
    if args.images > 0:
        report["image_to_score"] = image_parity(args.snapshot, bank, args.images)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    for r in report["feature_to_score"]:
        for split, s in r["splits"].items():
            print(f"{r['head']:22s} {split:4s} " + (f"max|diff| {s['max_abs_diff']:.2e} {'PASS' if s['pass'] else 'CHECK'}"
                  + (f" unmatched {s['unmatched']}" if s.get("unmatched") else "") if "max_abs_diff" in s
                  else f"not compared: {s['reason']}"))
    for cohort, s in (report.get("image_to_score") or {}).items():
        print(f"image->score {cohort}: n {s['n_images']}, feature max|diff| {s['feature_max_abs_diff']:.3e}, "
              f"min cosine {s['feature_min_cosine']:.6f}, worst score diff {max(s['score_max_abs_diff'].values()):.3e}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
