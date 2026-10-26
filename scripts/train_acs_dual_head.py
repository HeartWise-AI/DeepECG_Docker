"""
Dual-head WCR training: ACS prediction + Culprit vessel localization.

Two output heads:
    Head 1 (proj):    Binary ACS prediction (1 output)
    Head 2 (vessel):  Multi-label vessel localization (4 outputs: LAD, RCA, LCX, Left_Main)

Vessel loss is only computed for ACS-positive samples with vessel labels.
Combined loss = loss_acs + vessel_weight * loss_vessel

Pipeline per sample:
    Multi-layout PNG → smartphone augmentation → digitize → resample 250Hz → PSA → WCR → predict

Uses 2 GPUs:
    - GPU for digitizer (default: cuda:0)
    - GPU for WCR training (default: cuda:1)

Usage:
    cd /volume/DeepECG_Docker
    python scripts/train_acs_dual_head.py --no_wandb
"""

import argparse
import os
import sys
import time
import threading
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")
sys.path.insert(0, "/volume/Open-ECG-Digitizer")

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_LENGTH = 2500
PTBXL_POWER_RATIO = 3.003154
VESSEL_CLASSES = ["LAD", "RCA", "LCX", "Left_Main"]
NUM_VESSELS = len(VESSEL_CLASSES)


# ---------------------------------------------------------------------------
# Smartphone augmentations (same as v3)
# ---------------------------------------------------------------------------

class SmartphoneAugmentation:
    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return image
        img = image.copy()
        if np.random.random() < 0.5:
            img = self._perspective_warp(img)
        if np.random.random() < 0.7:
            img = self._brightness_contrast(img)
        if np.random.random() < 0.5:
            img = self._color_jitter(img)
        if np.random.random() < 0.3:
            img = self._motion_blur(img)
        if np.random.random() < 0.4:
            img = self._gaussian_noise(img)
        if np.random.random() < 0.5:
            img = self._jpeg_compress(img)
        if np.random.random() < 0.3:
            img = self._downsample_upsample(img)
        if np.random.random() < 0.4:
            img = self._random_shadow(img)
        if np.random.random() < 0.4:
            img = self._gradient_lighting(img)
        if np.random.random() < 0.3:
            img = self._color_temperature(img)
        if np.random.random() < 0.2:
            img = self._paper_curl(img)
        if np.random.random() < 0.3:
            img = self._random_crop_margins(img)
        if np.random.random() < 0.3:
            img = self._clahe_augment(img)
        return img

    def _perspective_warp(self, img):
        h, w = img.shape[:2]
        offset = int(min(h, w) * np.random.uniform(0.02, 0.08))
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src.copy()
        for i in range(4):
            dst[i, 0] += np.random.randint(-offset, offset + 1)
            dst[i, 1] += np.random.randint(-offset, offset + 1)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def _brightness_contrast(self, img):
        alpha = np.random.uniform(0.7, 1.3)
        beta = np.random.randint(-30, 31)
        return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    def _color_jitter(self, img):
        for c in range(3):
            shift = np.random.randint(-15, 16)
            img[:, :, c] = np.clip(img[:, :, c].astype(np.int16) + shift, 0, 255).astype(np.uint8)
        return img

    def _motion_blur(self, img):
        size = np.random.choice([3, 5, 7])
        angle = np.random.randint(0, 180)
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        kernel /= kernel.sum() + 1e-9
        return cv2.filter2D(img, -1, kernel)

    def _gaussian_noise(self, img):
        sigma = np.random.uniform(5, 20)
        noise = np.random.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _jpeg_compress(self, img):
        quality = np.random.randint(30, 80)
        _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def _downsample_upsample(self, img):
        h, w = img.shape[:2]
        factor = np.random.uniform(0.3, 0.7)
        small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))))
        return cv2.resize(small, (w, h))

    def _random_shadow(self, img):
        h, w = img.shape[:2]
        overlay = img.astype(np.float32)
        opacity = np.random.uniform(0.3, 0.7)
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        ax = np.random.randint(w // 6, w // 2)
        ay = np.random.randint(h // 6, h // 2)
        angle = np.random.randint(0, 180)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 1.0, -1)
        ksize = max(31, min(h, w) // 4) | 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        shadow = overlay * (1.0 - mask[:, :, None] * opacity)
        return np.clip(shadow, 0, 255).astype(np.uint8)

    def _gradient_lighting(self, img):
        h, w = img.shape[:2]
        strength = np.random.uniform(0.15, 0.30)
        if np.random.random() < 0.5:
            grad = np.linspace(1.0 - strength, 1.0 + strength, w, dtype=np.float32)
            grad = np.tile(grad[None, :, None], (h, 1, 3))
        else:
            grad = np.linspace(1.0 - strength, 1.0 + strength, h, dtype=np.float32)
            grad = np.tile(grad[:, None, None], (1, w, 3))
        return np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)

    def _color_temperature(self, img):
        warm = np.random.random() < 0.5
        shift = np.random.randint(10, 30)
        result = img.copy()
        if warm:
            result[:, :, 2] = np.clip(result[:, :, 2].astype(np.int16) + shift, 0, 255).astype(np.uint8)
            result[:, :, 1] = np.clip(result[:, :, 1].astype(np.int16) + shift // 3, 0, 255).astype(np.uint8)
            result[:, :, 0] = np.clip(result[:, :, 0].astype(np.int16) - shift // 2, 0, 255).astype(np.uint8)
        else:
            result[:, :, 0] = np.clip(result[:, :, 0].astype(np.int16) + shift, 0, 255).astype(np.uint8)
            result[:, :, 2] = np.clip(result[:, :, 2].astype(np.int16) - shift // 2, 0, 255).astype(np.uint8)
        return result

    def _paper_curl(self, img):
        h, w = img.shape[:2]
        amplitude = np.random.uniform(2, 8)
        frequency = np.random.uniform(1, 3)
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                map_x[y, x] = x
                map_y[y, x] = y + amplitude * np.sin(2 * np.pi * frequency * x / w)
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _random_crop_margins(self, img):
        h, w = img.shape[:2]
        top = int(h * np.random.uniform(0, 0.15)) if np.random.random() < 0.5 else 0
        bottom = int(h * np.random.uniform(0, 0.15)) if np.random.random() < 0.5 else 0
        left = int(w * np.random.uniform(0, 0.15)) if np.random.random() < 0.5 else 0
        right = int(w * np.random.uniform(0, 0.15)) if np.random.random() < 0.5 else 0
        y1, y2 = top, max(h - bottom, top + 1)
        x1, x2 = left, max(w - right, left + 1)
        return cv2.resize(img[y1:y2, x1:x2], (w, h))

    def _clahe_augment(self, img):
        clip_limit = np.random.uniform(1.0, 4.0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def apply_psa(ecg_array: np.ndarray) -> np.ndarray:
    mag = np.abs(np.fft.fft(ecg_array[:, 0]))
    current_power = float(np.mean(mag))
    if current_power < 1e-10:
        return ecg_array
    factor = PTBXL_POWER_RATIO / current_power
    scaled = np.real(np.fft.ifft(np.fft.fft(ecg_array, axis=0) * factor, axis=0))
    return scaled.astype(np.float32)


def canonical_to_array(canonical: torch.Tensor) -> np.ndarray:
    data = canonical.squeeze().cpu().numpy()
    if data.ndim == 1:
        data = data[None, :]
    data = np.nan_to_num(data, nan=0.0)
    data = data.T
    if data.shape[0] != TARGET_LENGTH:
        data = resample(data, TARGET_LENGTH, axis=0)
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Digitizer wrapper
# ---------------------------------------------------------------------------

class DigitizerWrapper:
    def __init__(self, device: str = "cuda:0", config_name: str = "inference_wrapper_george-moody-2024.yml"):
        from src.config.default import get_cfg
        from src.utils import import_class_from_path

        config_path = os.path.join("/volume/Open-ECG-Digitizer/src/config", config_name)
        cfg = get_cfg(config_path)
        cfg.MODEL.KWARGS.device = device
        cfg.MODEL.KWARGS.config.LAYOUT_IDENTIFIER.KWARGS.device = device

        orig_dir = os.getcwd()
        os.chdir("/volume/Open-ECG-Digitizer")
        wrapper_class = import_class_from_path(cfg.MODEL.class_path)
        self.model = wrapper_class(**cfg.MODEL.KWARGS)
        self.device = device
        os.chdir(orig_dir)
        print(f"Digitizer loaded on {device}")

    @torch.no_grad()
    def digitize(self, img_bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        got_values = self.model(img_tensor, layout_should_include_substring=None)
        canonical = None
        if "signal" in got_values and isinstance(got_values["signal"], dict):
            canonical = got_values["signal"].get("canonical_lines")
        if canonical is None:
            return np.zeros((TARGET_LENGTH, 12), dtype=np.float32)
        ecg = canonical_to_array(canonical)
        ecg = apply_psa(ecg)
        return ecg


# ---------------------------------------------------------------------------
# Dual-head WCR model
# ---------------------------------------------------------------------------

class DualHeadWCR(nn.Module):
    """Wrapper around WCR model that adds a vessel classification head."""

    def __init__(self, base_model, hidden_dim: int = 768):
        super().__init__()
        self.base = base_model
        # Head 1: ACS prediction (the existing proj layer in base_model)
        # Head 2: Vessel localization (new)
        self.vessel_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, NUM_VESSELS),
        )
        nn.init.xavier_uniform_(self.vessel_head[0].weight)
        nn.init.xavier_uniform_(self.vessel_head[3].weight)

    def forward(self, source):
        """Forward pass returning both ACS logits and vessel logits.

        Uses a forward hook on base.proj to capture the pooled features
        (with gradients) before the ACS projection head.
        """
        captured = {}

        def hook_fn(module, input, output):
            # input[0] is the mean-pooled encoder features going into proj
            captured['pooled'] = input[0]

        handle = self.base.proj.register_forward_hook(hook_fn)
        try:
            output = self.base(source=source)
            acs_logits = self.base.get_logits(output)  # (B, 1)
        finally:
            handle.remove()

        pooled = captured['pooled']  # (B, hidden_dim) — has gradients
        vessel_logits = self.vessel_head(pooled)  # (B, NUM_VESSELS)

        return acs_logits, vessel_logits


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

DIGITIZE_TIMEOUT = 30


def _digitize_worker(digitizer, img, result_holder):
    try:
        result_holder['ecg'] = digitizer.digitize(img)
    except Exception as e:
        result_holder['error'] = str(e)


def digitize_with_timeout(digitizer, img, timeout=DIGITIZE_TIMEOUT):
    result = {}
    t = threading.Thread(target=_digitize_worker, args=(digitizer, img, result), daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None
    if 'error' in result:
        raise Exception(result['error'])
    return result.get('ecg')


def process_batch_dual(png_paths, acs_labels, vessel_labels_list, digitizer, augment_fn, train_device):
    """Load, augment, digitize a batch → (B, 12, 2500) tensor + ACS labels + vessel labels."""
    ecg_list = []
    valid_acs = []
    valid_vessels = []

    for png_path, acs_label, vessel_labels in zip(png_paths, acs_labels, vessel_labels_list):
        try:
            img = cv2.imread(png_path)
            if img is None:
                continue
            if augment_fn is not None:
                img = augment_fn(img)
            ecg = digitize_with_timeout(digitizer, img)
            if ecg is None:
                print(f"  TIMEOUT digitizing {os.path.basename(png_path)}, skipping")
                continue
            ecg_list.append(torch.from_numpy(ecg.T))  # (12, 2500)
            valid_acs.append(acs_label)
            valid_vessels.append(vessel_labels)
        except Exception as e:
            print(f"  Error processing {os.path.basename(png_path)}: {e}")
            continue

    if not ecg_list:
        return None, None, None

    sources = torch.stack(ecg_list).float().to(train_device)
    acs_targets = torch.tensor(valid_acs, dtype=torch.float32).unsqueeze(1).to(train_device)
    vessel_targets = torch.tensor(valid_vessels, dtype=torch.float32).to(train_device)  # (B, 4)
    return sources, acs_targets, vessel_targets


def evaluate_npy_dual(model, npy_dir, linked_df, split, device, batch_size=256):
    """Evaluate dual-head model on pre-digitized NPY files."""
    model.eval()
    split_df = linked_df[linked_df.Split == split].copy()

    # Build NPY paths
    split_df['dig_npy_path'] = split_df['npy_path'].apply(
        lambda p: os.path.join(npy_dir, os.path.basename(p).replace('.npy', '') + '.npy')
    )
    split_df = split_df[split_df['dig_npy_path'].apply(os.path.exists)]

    signals, acs_labels, vessel_labels = [], [], []
    for _, row in split_df.iterrows():
        arr = np.load(row['dig_npy_path']).astype(np.float32)
        if arr.shape == (12, 2500):
            arr = arr.T
        if arr.ndim == 2 and arr.shape == (2500, 12):
            signals.append(arr)
            acs_labels.append(row['Acute_Obstruction'])
            vessel_labels.append([row['LAD'], row['RCA'], row['LCX'], row['Left_Main']])

    if len(signals) < 10 or sum(acs_labels) == 0:
        return {"auc": 0.0, "auprc": 0.0, "n": len(signals), "pos": sum(acs_labels),
                "vessel_aucs": {}}

    X = np.stack(signals)
    Y_acs = np.array(acs_labels, dtype=np.float32)
    Y_vessels = np.array(vessel_labels, dtype=np.float32)

    all_acs_probs, all_vessel_probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = np.transpose(X[i:i+batch_size], (0, 2, 1))
            source = torch.from_numpy(batch_x).float().to(device)
            acs_logits, vessel_logits = model(source)
            all_acs_probs.append(torch.sigmoid(acs_logits).cpu().numpy())
            all_vessel_probs.append(torch.sigmoid(vessel_logits).cpu().numpy())

    all_acs_probs = np.concatenate(all_acs_probs, axis=0)
    all_vessel_probs = np.concatenate(all_vessel_probs, axis=0)
    if all_acs_probs.ndim == 2:
        all_acs_probs = all_acs_probs[:, 0]

    auc = roc_auc_score(Y_acs, all_acs_probs)
    auprc = average_precision_score(Y_acs, all_acs_probs)

    # Vessel AUCs (only for ACS-positive samples with vessel labels)
    acs_mask = Y_acs == 1
    vessel_aucs = {}
    for j, name in enumerate(VESSEL_CLASSES):
        y_true = Y_vessels[acs_mask, j]
        y_pred = all_vessel_probs[acs_mask, j]
        if y_true.sum() >= 5 and (1 - y_true).sum() >= 5:
            vessel_aucs[name] = roc_auc_score(y_true, y_pred)
        else:
            vessel_aucs[name] = 0.0

    return {
        "auc": auc, "auprc": auprc,
        "n": len(Y_acs), "pos": int(Y_acs.sum()),
        "vessel_aucs": vessel_aucs,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dual-head WCR: ACS + Culprit vessel")
    parser.add_argument("--linked_parquet", type=str,
                        default="data/acs_multilayout_3x4_dual_linked.parquet")
    parser.add_argument("--model_dir", type=str, default="weights/wcr_77_classes")
    parser.add_argument("--digitizer_device", type=str, default="cuda:0")
    parser.add_argument("--train_device", type=str, default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--save_dir", type=str, default="checkpoints_acs_dual_head")
    parser.add_argument("--val_every_n_steps", type=int, default=200)
    parser.add_argument("--augment_p", type=float, default=0.8)
    parser.add_argument("--vessel_weight", type=float, default=0.5,
                        help="Weight for vessel loss relative to ACS loss")
    parser.add_argument("--no_wandb", action="store_true", default=False)
    parser.add_argument("--init_checkpoint", type=str,
                        default="checkpoints_acs_npy_ceiling/checkpoint_best.pt")
    parser.add_argument("--val_npy_dir", type=str, default="data/acs_digitized_npy")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Load linked dataset ----
    print("Loading linked dataset...")
    linked = pd.read_parquet(args.linked_parquet)
    print(f"  Total: {len(linked)} rows")
    print(f"  ACS+: {linked['Acute_Obstruction'].sum()}")
    for v in VESSEL_CLASSES:
        acs_df = linked[linked['Acute_Obstruction'] == 1]
        print(f"    {v}: {acs_df[v].sum()}")

    # Build data per split
    def get_split_data(split_name):
        df = linked[linked["Split"] == split_name]
        paths = df["png_path"].tolist()
        acs = df["Acute_Obstruction"].tolist()
        vessels = df[VESSEL_CLASSES].values.tolist()
        return paths, acs, vessels

    train_paths, train_acs, train_vessels = get_split_data("train")
    val_paths, val_acs, val_vessels = get_split_data("val")
    test_paths, test_acs, test_vessels = get_split_data("test")

    print(f"  Train: {len(train_paths)} (acs+={sum(train_acs)})")
    print(f"  Val:   {len(val_paths)} (acs+={sum(val_acs)})")
    print(f"  Test:  {len(test_paths)} (acs+={sum(test_acs)})")

    # ---- Load digitizer ----
    print("\nLoading digitizer...")
    digitizer = DigitizerWrapper(device=args.digitizer_device)

    # ---- Load WCR model and wrap with dual head ----
    print("\nLoading WCR model...")
    from fairseq_signals.utils import checkpoint_utils

    if args.init_checkpoint:
        print(f"  Initializing from: {args.init_checkpoint}")
        base_model, cfg, task = checkpoint_utils.load_model_and_task(
            args.init_checkpoint,
            arg_overrides={"model_path": os.path.join(args.model_dir, "base_ssl.pt")},
            suffix=""
        )
        hidden_dim = base_model.proj.in_features
        print(f"  Base model proj: {hidden_dim} → {base_model.proj.out_features}")
    else:
        base_ssl_path = os.path.join(args.model_dir, "base_ssl.pt")
        wcr_path = os.path.join(args.model_dir, "wcr_77_classes.pt")
        base_model, cfg, task = checkpoint_utils.load_model_and_task(
            wcr_path, arg_overrides={"model_path": base_ssl_path}, suffix=""
        )
        hidden_dim = base_model.proj.in_features
        # Reset ACS head
        base_model.proj = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(base_model.proj.weight)
        nn.init.constant_(base_model.proj.bias, 0.0)

    model = DualHeadWCR(base_model, hidden_dim=hidden_dim)
    model = model.to(args.train_device)
    print(f"  Dual-head model: ACS (1) + Vessel ({NUM_VESSELS})")
    print(f"  Vessel classes: {VESSEL_CLASSES}")

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    augment_fn = SmartphoneAugmentation(p=args.augment_p)

    # ---- Class weights ----
    n_pos = sum(train_acs)
    n_neg = len(train_acs) - n_pos
    acs_pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(args.train_device)
    print(f"\nACS pos weight: {acs_pos_weight.item():.2f}")

    # Vessel pos weights (within ACS-positive samples)
    train_vessels_arr = np.array(train_vessels)
    train_acs_arr = np.array(train_acs)
    acs_mask = train_acs_arr == 1
    vessel_pos_weights = []
    for j, name in enumerate(VESSEL_CLASSES):
        v_pos = train_vessels_arr[acs_mask, j].sum()
        v_neg = acs_mask.sum() - v_pos
        w = v_neg / max(v_pos, 1)
        vessel_pos_weights.append(w)
        print(f"  {name} pos weight: {w:.2f} (pos={int(v_pos)}, neg={int(v_neg)})")
    vessel_pos_weight = torch.tensor(vessel_pos_weights, dtype=torch.float32).to(args.train_device)

    # ---- Training loop ----
    best_val_auc = 0.0
    patience_counter = 0
    global_step = 0

    print(f"\nStarting dual-head training:")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Digitizer: {args.digitizer_device}, Training: {args.train_device}")
    print(f"  Vessel weight: {args.vessel_weight}")
    print(f"  Val every {args.val_every_n_steps} steps, patience={args.patience}")
    print(f"  ~{len(train_paths) // args.batch_size} steps/epoch")
    print()

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()

        indices = np.random.permutation(len(train_paths))
        epoch_acs_loss = 0.0
        epoch_vessel_loss = 0.0
        epoch_samples = 0

        for step_idx in range(0, len(indices), args.batch_size):
            batch_indices = indices[step_idx:step_idx + args.batch_size]
            batch_paths = [train_paths[i] for i in batch_indices]
            batch_acs = [train_acs[i] for i in batch_indices]
            batch_vessels = [train_vessels[i] for i in batch_indices]

            step_start = time.time()

            sources, acs_targets, vessel_targets = process_batch_dual(
                batch_paths, batch_acs, batch_vessels,
                digitizer, augment_fn, args.train_device
            )
            if sources is None:
                continue

            optimizer.zero_grad()

            acs_logits, vessel_logits = model(sources)

            # ACS loss (all samples)
            loss_acs = F.binary_cross_entropy_with_logits(
                acs_logits, acs_targets, pos_weight=acs_pos_weight
            )

            # Vessel loss (only ACS-positive samples)
            acs_positive_mask = acs_targets.squeeze(1) == 1
            # Also require at least one vessel label
            has_vessel = vessel_targets.sum(dim=1) > 0
            vessel_mask = acs_positive_mask & has_vessel

            if vessel_mask.any():
                vessel_logits_pos = vessel_logits[vessel_mask]
                vessel_targets_pos = vessel_targets[vessel_mask]
                loss_vessel = F.binary_cross_entropy_with_logits(
                    vessel_logits_pos, vessel_targets_pos,
                    pos_weight=vessel_pos_weight
                )
            else:
                loss_vessel = torch.tensor(0.0, device=args.train_device)

            loss = loss_acs + args.vessel_weight * loss_vessel
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size_actual = sources.shape[0]
            epoch_acs_loss += loss_acs.item() * batch_size_actual
            epoch_vessel_loss += loss_vessel.item() * batch_size_actual
            epoch_samples += batch_size_actual
            global_step += 1

            step_time = time.time() - step_start
            steps_done = step_idx // args.batch_size + 1
            total_steps = len(indices) // args.batch_size

            if steps_done % 10 == 0 or steps_done <= 3:
                n_acs_pos = int(acs_positive_mask.sum())
                n_vessel = int(vessel_mask.sum()) if vessel_mask.any() else 0
                print(
                    f"  Epoch {epoch+1} | Step {steps_done}/{total_steps} | "
                    f"ACS: {loss_acs.item():.4f} | Vessel: {loss_vessel.item():.4f} | "
                    f"Time: {step_time:.1f}s | B={batch_size_actual} (acs+={n_acs_pos}, v={n_vessel})"
                )

            # ---- Periodic validation ----
            if global_step % args.val_every_n_steps == 0:
                print(f"\n  === Validation at step {global_step} ===")
                val_metrics = evaluate_npy_dual(
                    model, args.val_npy_dir, linked, "val",
                    args.train_device, batch_size=256
                )
                print(
                    f"  Val ACS AUC: {val_metrics['auc']:.4f} | "
                    f"Val AUPRC: {val_metrics['auprc']:.4f} | "
                    f"N={val_metrics['n']}, Pos={val_metrics['pos']}"
                )
                if val_metrics['vessel_aucs']:
                    for name, vauc in val_metrics['vessel_aucs'].items():
                        print(f"    Vessel {name} AUC: {vauc:.4f}")

                if val_metrics["auc"] > best_val_auc:
                    best_val_auc = val_metrics["auc"]
                    patience_counter = 0
                    ckpt_path = os.path.join(args.save_dir, "best_model.pt")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_auc": best_val_auc,
                        "vessel_aucs": val_metrics['vessel_aucs'],
                        "args": vars(args),
                        "vessel_classes": VESSEL_CLASSES,
                    }, ckpt_path)
                    print(f"  New best! Saved to {ckpt_path}")
                else:
                    patience_counter += 1
                    print(f"  No improvement. Patience: {patience_counter}/{args.patience}")

                model.train()

        # ---- End of epoch ----
        epoch_time = time.time() - epoch_start
        avg_acs = epoch_acs_loss / max(epoch_samples, 1)
        avg_vessel = epoch_vessel_loss / max(epoch_samples, 1)

        print(f"\n{'='*70}")
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"ACS Loss: {avg_acs:.4f} | Vessel Loss: {avg_vessel:.4f} | "
            f"Time: {epoch_time/3600:.1f}h | Best AUC: {best_val_auc:.4f}"
        )
        print(f"{'='*70}\n")

        # Save epoch checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_auc": best_val_auc,
        }, os.path.join(args.save_dir, f"checkpoint_epoch{epoch+1}.pt"))

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ---- Final test evaluation ----
    print("\n=== Final Test Evaluation ===")
    best_ckpt = os.path.join(args.save_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=args.train_device)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded best model from step {state['global_step']}")

    test_metrics = evaluate_npy_dual(
        model, args.val_npy_dir, linked, "test",
        args.train_device, batch_size=256
    )
    print(f"  Test ACS AUC: {test_metrics['auc']:.4f}")
    print(f"  Test AUPRC: {test_metrics['auprc']:.4f}")
    print(f"  Test N={test_metrics['n']}, Pos={test_metrics['pos']}")
    if test_metrics['vessel_aucs']:
        print("  Vessel AUCs:")
        for name, vauc in test_metrics['vessel_aucs'].items():
            print(f"    {name}: {vauc:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
