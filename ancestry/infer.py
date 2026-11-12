"""Core model loading + 5-fold ensemble inference for the DeepECG-SSL ancestry models.

Self-contained re-implementation of `scripts/wcr.py::WCREcgTransformer` with the
fairseq-signals location taken from $FAIRSEQ_ROOT (the original wcr.py hard-codes a
path that only existed on the training box). Loads the 5 fine-tuned `FT_ANCESTRY_*`
fold heads on top of the shared SSL backbone and averages their sigmoid outputs.

Device: uses CUDA when available and not disabled, else CPU. The same image runs
both -- torch is the cu121 wheel, which falls back to CPU transparently.

Output convention (matches 01_ancestry_genetic_perf.ipynb):
    logit -> sigmoid = P(non-European).  p(EUR) = 1 - P(non-EUR).
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@contextlib.contextmanager
def _quiet():
    """Silence the chatty debug prints in the fairseq-signals checkpoint loader."""
    if os.environ.get("VERBOSE_LOAD", "") == "1":
        yield
        return
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield

# The dockerfile clones fairseq-signals here and sets FAIRSEQ_ROOT to match. Outside the
# image, set FAIRSEQ_ROOT to a checkout of the same pinned commit.
FAIRSEQ_ROOT = os.environ.get("FAIRSEQ_ROOT", "/app/fairseq-signals")


def _load_checkpoint_utils():
    if FAIRSEQ_ROOT not in sys.path:
        sys.path.insert(0, FAIRSEQ_ROOT)
    cu_path = Path(FAIRSEQ_ROOT) / "fairseq_signals/utils/checkpoint_utils.py"
    if not cu_path.exists():
        raise FileNotFoundError(
            f"fairseq-signals not found at FAIRSEQ_ROOT={FAIRSEQ_ROOT} "
            f"(missing {cu_path}). Set $FAIRSEQ_ROOT to the source tree.")
    spec = importlib.util.spec_from_file_location("checkpoint_utils", str(cu_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class WCREcgTransformer(nn.Module):
    """Fine-tuned ECG transformer classifier (single-logit ancestry head)."""

    def __init__(self, model_path, pretrained_path=None, suffix=""):
        super().__init__()
        cu = _load_checkpoint_utils()
        overrides = {}
        if pretrained_path is not None:
            overrides["model_path"] = pretrained_path
        with _quiet():
            model, _saved_cfg, _task = cu.load_model_and_task(
                model_path, arg_overrides=overrides, suffix=suffix)
        self.model = model

    def forward(self, x, padding_mask=None, return_embedding=False):
        net_input = {"source": x, "padding_mask": padding_mask}
        if not return_embedding:
            return self.model.get_logits(self.model(**net_input))
        cap = {}

        def pre_hook(_module, inputs):
            cap["embedding"] = inputs[0].detach()

        h = self.model.proj.register_forward_pre_hook(pre_hook)
        try:
            logits = self.model.get_logits(self.model(**net_input))
        finally:
            h.remove()
        return logits, cap.get("embedding")


def pick_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if os.environ.get("FORCE_CPU", "") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_fold_checkpoints(models_root, n_folds=5, variant="raw", bs=128):
    """roberto layout: <root>/<variant>/fold_{k}/bs_{bs}/checkpoint_best.pt.

    Falls back to <root>/fold_{k}/checkpoint_best.pt (flat layout) if the
    roberto layout is absent.
    """
    root = Path(models_root)
    out = []
    for k in range(1, n_folds + 1):
        p = root / variant / f"fold_{k}" / f"bs_{bs}" / "checkpoint_best.pt"
        if not p.exists():
            p = root / f"fold_{k}" / "checkpoint_best.pt"
        if not p.exists():
            raise FileNotFoundError(f"no checkpoint for fold {k} under {root}")
        out.append(str(p))
    return out


def load_fold_models(models_root, ssl_ckpt, device, n_folds=5):
    ckpts = discover_fold_checkpoints(models_root, n_folds=n_folds)
    models = []
    for k, c in enumerate(ckpts, 1):
        print(f"[load] fold {k}: {c}", flush=True)
        m = WCREcgTransformer(c, str(ssl_ckpt))
        m.to(device).eval()
        models.append(m)
    return models, ckpts


@torch.no_grad()
def _predict_one(model, X_nct, device, batch_size):
    """X_nct: (N, C, T) float32 already lead-major. Returns P(non-EUR) (N,)."""
    N = X_nct.shape[0]
    out = np.empty(N, dtype=np.float32)
    step = max(batch_size, (N // 10) // batch_size * batch_size)   # ~10 progress lines
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        xb = torch.from_numpy(np.ascontiguousarray(X_nct[s:e], dtype=np.float32)).to(device)
        logits = model(xb)
        out[s:e] = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        if N >= 2000 and e % step < batch_size and e < N:
            print(f"    {e:,}/{N:,} ECGs", flush=True)
    return out


# Training inputs are millivolts (raw ADC x 0.00488): the median 12-lead ECG peaks at ~1.5 mV.
MV_PEAK_RANGE = (0.2, 20.0)


def check_units(X_ntc, n_sample=256, seed=0):
    """Refuse inputs that are clearly not millivolts or not finite. Wrong units (uV, raw ADC
    counts, z-scored signals) do not raise anywhere else -- they silently give meaningless
    scores -- so this is the one guard. Set ANCESTRY_SKIP_UNIT_CHECK=1 to bypass (not advised)."""
    n = len(X_ntc)
    if n == 0:
        raise ValueError("no ECGs in the input")
    idx = np.sort(np.random.default_rng(seed).choice(n, min(n, n_sample), replace=False))
    xs = np.asarray(X_ntc[idx], dtype=np.float32)
    if not np.isfinite(xs).all():
        raise ValueError("input ECGs contain NaN or inf values; clean them before scoring")
    peak = float(np.median(np.abs(xs).max(axis=(1, 2))))
    lo, hi = MV_PEAK_RANGE
    msg = (f"[units] median per-ECG peak |amplitude| = {peak:.3g} "
           f"(expected millivolts, ~0.5-5; accepted {lo}-{hi})")
    print(msg, flush=True)
    if os.environ.get("ANCESTRY_SKIP_UNIT_CHECK", "") == "1":
        return peak
    if peak > hi:
        raise ValueError(msg + ". This looks like microvolts or raw ADC counts: convert to mV "
                         "(uV / 1000, or ADC x your lead gain, e.g. x 0.00488) and re-run prepare.")
    if peak < lo:
        raise ValueError(msg + ". This looks like volts or a normalised signal: the model needs "
                         "unnormalised millivolts.")
    return peak


def ensemble_predict(models, X_ntc, device, batch_size=64, check=True):
    """X_ntc: (N, T=2500, C=12) @ 250 Hz. Returns (P_ens (N,), P_folds (N, 5)).

    Model expects lead-major (N, C, T); we swap the last two axes here.
    P_ens is the mean of the 5 folds' P(non-EUR).
    """
    X = np.asarray(X_ntc)
    if X.ndim != 3:
        raise ValueError(f"expected (N, 2500, 12), got {X.shape}")
    if check:
        check_units(X)
    if X.shape[1] == 12 and X.shape[2] != 12:
        # already (N, C, T) -- leave as is
        X_nct = X
    else:
        X_nct = np.swapaxes(X, -1, -2)
    preds = []
    for k, m in enumerate(models, 1):
        print(f"[infer] model {k}/{len(models)} on {X_nct.shape[0]:,} ECGs", flush=True)
        preds.append(_predict_one(m, X_nct, device, batch_size))
    P = np.stack(preds, axis=1)
    return P.mean(axis=1), P
