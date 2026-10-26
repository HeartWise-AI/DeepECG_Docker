"""
Online augmentation training v2 for ACS (Acute Coronary Obstruction) prediction.

Enhanced smartphone augmentations: shadows, gradient lighting, color temperature,
paper curl, random crop margins, CLAHE — in addition to all v1 augmentations.

Pipeline per sample:
    PNG → smartphone augmentation → digitize → resample to 250Hz → PSA → WCR → predict

Uses 2 GPUs:
    - GPU for digitizer (default: cuda:2)
    - GPU for WCR training (default: cuda:3)

Usage:
    cd /volume/DeepECG_Docker
    python scripts/train_acs_online_augment_v2.py \
        --init_checkpoint checkpoints_acs_npy_ceiling/checkpoint_best.pt \
        --val_npy_dir data/acs_digitized_npy \
        --no_wandb
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

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add fairseq-signals and digitizer to path
sys.path.insert(0, "/volume/DeepECG_Docker/fairseq-signals")
sys.path.insert(0, "/volume/Open-ECG-Digitizer")

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_LENGTH = 2500  # 10s at 250 Hz
DIGITIZER_OUTPUT_LENGTH = 10000  # 10s at 1000 Hz (digitizer default)
PTBXL_POWER_RATIO = 3.003154


# ---------------------------------------------------------------------------
# Enhanced smartphone augmentations (v2)
# ---------------------------------------------------------------------------

class SmartphoneAugmentation:
    """Augmentations mimicking smartphone photos of ECG paper prints.

    v2 additions: shadow, gradient lighting, color temperature, paper curl,
    random crop margins, CLAHE augmentation.
    """

    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return image
        img = image.copy()

        # --- v1 augmentations ---
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

        # --- v2 augmentations ---
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

    # --- v1 augmentations (unchanged) ---

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

    # --- v2 augmentations (new) ---

    def _random_shadow(self, img):
        """Dark ellipse overlay simulating finger/hand shadow."""
        h, w = img.shape[:2]
        overlay = img.astype(np.float32)
        opacity = np.random.uniform(0.3, 0.7)

        # Random ellipse shadow
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        ax = np.random.randint(w // 6, w // 2)
        ay = np.random.randint(h // 6, h // 2)
        angle = np.random.randint(0, 180)

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 1.0, -1)
        # Blur for soft edges
        ksize = max(31, min(h, w) // 4) | 1  # ensure odd
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        shadow = overlay * (1.0 - mask[:, :, None] * opacity)
        return np.clip(shadow, 0, 255).astype(np.uint8)

    def _gradient_lighting(self, img):
        """Linear gradient across image simulating uneven lighting."""
        h, w = img.shape[:2]
        strength = np.random.uniform(0.15, 0.30)

        # Random direction: 0=left-to-right, 1=top-to-bottom
        if np.random.random() < 0.5:
            grad = np.linspace(1.0 - strength, 1.0 + strength, w, dtype=np.float32)
            grad = np.tile(grad[None, :, None], (h, 1, 3))
        else:
            grad = np.linspace(1.0 - strength, 1.0 + strength, h, dtype=np.float32)
            grad = np.tile(grad[:, None, None], (1, w, 3))

        result = img.astype(np.float32) * grad
        return np.clip(result, 0, 255).astype(np.uint8)

    def _color_temperature(self, img):
        """Warm (yellow) or cool (blue) color shift."""
        warm = np.random.random() < 0.5
        shift = np.random.randint(10, 30)
        result = img.copy()
        if warm:
            # Warm: boost R, slightly boost G, reduce B
            result[:, :, 2] = np.clip(result[:, :, 2].astype(np.int16) + shift, 0, 255).astype(np.uint8)      # R (BGR)
            result[:, :, 1] = np.clip(result[:, :, 1].astype(np.int16) + shift // 3, 0, 255).astype(np.uint8)  # G
            result[:, :, 0] = np.clip(result[:, :, 0].astype(np.int16) - shift // 2, 0, 255).astype(np.uint8)  # B
        else:
            # Cool: boost B, reduce R
            result[:, :, 0] = np.clip(result[:, :, 0].astype(np.int16) + shift, 0, 255).astype(np.uint8)       # B
            result[:, :, 2] = np.clip(result[:, :, 2].astype(np.int16) - shift // 2, 0, 255).astype(np.uint8)  # R
        return result

    def _paper_curl(self, img):
        """Sinusoidal vertical distortion simulating paper curl."""
        h, w = img.shape[:2]
        amplitude = np.random.uniform(2, 8)
        frequency = np.random.uniform(1, 3)

        # Build remap arrays
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                map_x[y, x] = x
                map_y[y, x] = y + amplitude * np.sin(2 * np.pi * frequency * x / w)

        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _random_crop_margins(self, img):
        """Crop 5-15% off random edges simulating imperfect framing."""
        h, w = img.shape[:2]
        top = int(h * np.random.uniform(0, 0.15)) if np.random.random() < 0.5 else 0
        bottom = int(h * np.random.uniform(0, 0.15)) if np.random.random() < 0.5 else 0
        left = int(w * np.random.uniform(0, 0.15)) if np.random.random() < 0.5 else 0
        right = int(w * np.random.uniform(0, 0.15)) if np.random.random() < 0.5 else 0

        y1, y2 = top, max(h - bottom, top + 1)
        x1, x2 = left, max(w - right, left + 1)
        cropped = img[y1:y2, x1:x2]
        # Resize back to original dimensions so digitizer gets consistent input
        return cv2.resize(cropped, (w, h))

    def _clahe_augment(self, img):
        """Apply CLAHE with random clipLimit as augmentation."""
        clip_limit = np.random.uniform(1.0, 4.0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def apply_psa(ecg_array: np.ndarray) -> np.ndarray:
    """Power Spectral Alignment on (2500, 12) array."""
    mag = np.abs(np.fft.fft(ecg_array[:, 0]))
    current_power = float(np.mean(mag))
    if current_power < 1e-10:
        return ecg_array
    factor = PTBXL_POWER_RATIO / current_power
    scaled = np.real(np.fft.ifft(np.fft.fft(ecg_array, axis=0) * factor, axis=0))
    return scaled.astype(np.float32)


def canonical_to_array(canonical: torch.Tensor) -> np.ndarray:
    """Convert digitizer canonical_lines (12, N) to (2500, 12) numpy array."""
    data = canonical.squeeze().cpu().numpy()
    if data.ndim == 1:
        data = data[None, :]
    data = np.nan_to_num(data, nan=0.0)
    data = data.T  # (N, 12)
    if data.shape[0] != TARGET_LENGTH:
        data = resample(data, TARGET_LENGTH, axis=0)
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Digitizer wrapper
# ---------------------------------------------------------------------------

class DigitizerWrapper:
    """Wraps the Open-ECG-Digitizer for programmatic use."""

    def __init__(self, device: str = "cuda:2", config_name: str = "inference_wrapper_george-moody-2024.yml"):
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
        """Digitize a BGR image to (2500, 12) ECG array with PSA."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        img_tensor = img_tensor.to(self.device)

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
# WCR model loader
# ---------------------------------------------------------------------------

def load_wcr_model(model_dir: str, device: str, num_labels: int = 1):
    """Load WCR model from fairseq-signals checkpoint, replace classification head."""
    from fairseq_signals.utils import checkpoint_utils

    base_ssl_path = os.path.join(model_dir, "base_ssl.pt")
    wcr_path = os.path.join(model_dir, "wcr_77_classes.pt")

    overrides = {"model_path": base_ssl_path}
    model, cfg, task = checkpoint_utils.load_model_and_task(
        wcr_path, arg_overrides=overrides, suffix=""
    )

    old_proj = model.proj
    in_features = old_proj.in_features
    model.proj = nn.Linear(in_features, num_labels)
    nn.init.xavier_uniform_(model.proj.weight)
    nn.init.constant_(model.proj.bias, 0.0)

    model = model.to(device)
    print(f"WCR model loaded on {device}, head: {in_features} → {num_labels}")
    return model


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

DIGITIZE_TIMEOUT = 30  # seconds per image


def _digitize_worker(digitizer, img, result_holder):
    """Run digitization in a daemon thread."""
    try:
        result_holder['ecg'] = digitizer.digitize(img)
    except Exception as e:
        result_holder['error'] = str(e)


def digitize_with_timeout(digitizer, img, timeout=DIGITIZE_TIMEOUT):
    """Run digitization with a thread-based timeout. Returns None if timeout."""
    result = {}
    t = threading.Thread(target=_digitize_worker, args=(digitizer, img, result), daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None
    if 'error' in result:
        raise Exception(result['error'])
    return result.get('ecg')


def process_batch(png_paths, labels, digitizer, augment_fn, train_device):
    """Load, augment, digitize a batch of PNGs → (B, 12, 2500) tensor."""
    ecg_list = []
    valid_labels = []

    for png_path, label in zip(png_paths, labels):
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

            ecg_tensor = torch.from_numpy(ecg.T)  # (12, 2500)
            ecg_list.append(ecg_tensor)
            valid_labels.append(label)
        except Exception as e:
            print(f"  Error processing {os.path.basename(png_path)}: {e}")
            continue

    if not ecg_list:
        return None, None

    sources = torch.stack(ecg_list).float().to(train_device)  # (B, 12, 2500)
    targets = torch.tensor(valid_labels, dtype=torch.float32).unsqueeze(1).to(train_device)  # (B, 1)
    return sources, targets


def evaluate(model, png_paths, labels, digitizer, train_device, batch_size=32):
    """Evaluate model on a set of PNGs (no augmentation)."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(png_paths), batch_size):
            batch_paths = png_paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            sources, targets = process_batch(
                batch_paths, batch_labels, digitizer, augment_fn=None, train_device=train_device
            )
            if sources is None:
                continue

            output = model(source=sources)
            logits = model.get_logits(output)  # (B, 1)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs.tolist())
            all_labels.extend(targets.cpu().numpy().flatten().tolist())

    if len(all_preds) == 0 or sum(all_labels) == 0:
        return {"auc": 0.0, "auprc": 0.0, "n": len(all_preds), "pos": sum(all_labels)}

    auc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    return {"auc": auc, "auprc": auprc, "n": len(all_preds), "pos": sum(all_labels)}


def evaluate_npy(model, npy_dir, linked_df, split, device, batch_size=256):
    """Evaluate model on pre-digitized NPY files. Much faster than online digitization."""
    model.eval()
    split_df = linked_df[linked_df.Split == split].copy()
    split_df['dig_npy_path'] = split_df['ecg_key'].apply(
        lambda ek: os.path.join(npy_dir, f"{ek}.npy")
    )
    split_df = split_df[split_df['dig_npy_path'].apply(os.path.exists)]

    signals, labels = [], []
    for _, row in split_df.iterrows():
        arr = np.load(row['dig_npy_path']).astype(np.float32)
        if arr.shape == (12, 2500):
            arr = arr.T
        if arr.ndim == 2 and arr.shape == (2500, 12):
            signals.append(arr)
            labels.append(row['Acute_Obstruction'])

    if len(signals) < 10 or sum(labels) == 0:
        return {"auc": 0.0, "auprc": 0.0, "n": len(signals), "pos": sum(labels)}

    X = np.stack(signals)
    Y = np.array(labels, dtype=np.float32)

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = np.transpose(X[i:i+batch_size], (0, 2, 1))  # (B, 12, 2500)
            source = torch.from_numpy(batch_x).float().to(device)
            output = model(source=source)
            logits = model.get_logits(output).float()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    if all_probs.ndim == 2:
        all_probs = all_probs[:, 0]

    auc = roc_auc_score(Y, all_probs)
    auprc = average_precision_score(Y, all_probs)
    return {"auc": auc, "auprc": auprc, "n": len(Y), "pos": int(Y.sum())}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Online augmentation training v2 for ACS prediction")
    parser.add_argument("--linked_parquet", type=str, default="data/acs_linked_dataset.parquet")
    parser.add_argument("--png_base_dir", type=str, default="/media/data1/ravram/DeepECG/ecg_png_parquet")
    parser.add_argument("--model_dir", type=str, default="weights/wcr_77_classes")
    parser.add_argument("--digitizer_device", type=str, default="cuda:2")
    parser.add_argument("--train_device", type=str, default="cuda:3")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--save_dir", type=str, default="checkpoints_acs_online_augment_v2")
    parser.add_argument("--wandb_project", type=str, default="ecg-digitization-ft")
    parser.add_argument("--wandb_run_name", type=str, default="acs_online_augment_v2")
    parser.add_argument("--val_every_n_steps", type=int, default=200,
                        help="Validate every N training steps")
    parser.add_argument("--val_max_samples", type=int, default=1000,
                        help="Max validation samples per eval (for speed)")
    parser.add_argument("--augment_p", type=float, default=0.8,
                        help="Probability of applying augmentation per sample")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use mixed precision training (default: OFF for stability)")
    parser.add_argument("--no_wandb", action="store_true", default=False)
    parser.add_argument("--init_checkpoint", type=str,
                        default="checkpoints_acs_npy_ceiling/checkpoint_best.pt",
                        help="Fairseq checkpoint to initialize from (default: ceiling model)")
    parser.add_argument("--val_npy_dir", type=str, default="data/acs_digitized_npy",
                        help="Pre-digitized NPY dir for fast validation")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Load linked dataset ----
    print("Loading linked dataset...")
    linked = pd.read_parquet(args.linked_parquet)
    print(f"  Total: {len(linked)} rows, Pos: {linked['Acute_Obstruction'].sum()}")

    # Build PNG path lists per split
    def get_split_data(split_name):
        df = linked[linked["Split"] == split_name].copy()
        paths = [os.path.join(args.png_base_dir, p) for p in df["png_path"].tolist()]
        labels = df["Acute_Obstruction"].tolist()
        return paths, labels

    train_paths, train_labels = get_split_data("train")
    val_paths, val_labels = get_split_data("val")
    test_paths, test_labels = get_split_data("test")

    print(f"  Train: {len(train_paths)} (pos={sum(train_labels)})")
    print(f"  Val:   {len(val_paths)} (pos={sum(val_labels)})")
    print(f"  Test:  {len(test_paths)} (pos={sum(test_labels)})")

    # ---- Init WandB ----
    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # ---- Load digitizer ----
    print("\nLoading digitizer...")
    digitizer = DigitizerWrapper(device=args.digitizer_device)

    # ---- Load WCR model ----
    print("\nLoading WCR model...")
    if args.init_checkpoint:
        from fairseq_signals.utils import checkpoint_utils
        print(f"  Initializing from fairseq checkpoint: {args.init_checkpoint}")
        model, cfg, task = checkpoint_utils.load_model_and_task(
            args.init_checkpoint,
            arg_overrides={"model_path": os.path.join(args.model_dir, "base_ssl.pt")},
            suffix=""
        )
        model = model.to(args.train_device)
        print(f"  Model loaded with proj: {model.proj.in_features} → {model.proj.out_features}")
    else:
        model = load_wcr_model(args.model_dir, device=args.train_device, num_labels=1)

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda") if args.fp16 else None
    augment_fn = SmartphoneAugmentation(p=args.augment_p)

    # ---- Compute class weight for imbalanced data ----
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(args.train_device)
    print(f"\nPos weight: {pos_weight.item():.2f} (neg={n_neg}, pos={n_pos})")

    # ---- Training loop ----
    best_val_auc = 0.0
    patience_counter = 0
    global_step = 0
    train_losses = []

    print(f"\nStarting training v2: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"  LR: {args.lr}, FP16: {args.fp16}")
    print(f"  Digitizer: {args.digitizer_device}, Training: {args.train_device}")
    print(f"  Val every {args.val_every_n_steps} steps, patience={args.patience}")
    print(f"  ~{len(train_paths) // args.batch_size} steps/epoch")
    print(f"  Augmentations: v2 (shadow, gradient, color temp, paper curl, crop, CLAHE)")
    print()

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()

        # Shuffle training data
        indices = np.random.permutation(len(train_paths))
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_correct = 0

        for step_idx in range(0, len(indices), args.batch_size):
            batch_indices = indices[step_idx:step_idx + args.batch_size]
            batch_paths = [train_paths[i] for i in batch_indices]
            batch_labels = [train_labels[i] for i in batch_indices]

            step_start = time.time()

            # Process batch: augment + digitize
            sources, targets = process_batch(
                batch_paths, batch_labels, digitizer, augment_fn, args.train_device
            )
            if sources is None:
                continue

            # Forward pass
            optimizer.zero_grad()

            if args.fp16:
                with torch.amp.autocast("cuda"):
                    output = model(source=sources)
                    logits = model.get_logits(output)
                    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(source=sources)
                logits = model.get_logits(output)
                loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Track metrics
            batch_loss = loss.item()
            batch_size_actual = sources.shape[0]
            epoch_loss += batch_loss * batch_size_actual
            epoch_samples += batch_size_actual

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                epoch_correct += (preds == targets).sum().item()

            train_losses.append(batch_loss)
            global_step += 1

            step_time = time.time() - step_start
            steps_done = step_idx // args.batch_size + 1
            total_steps = len(indices) // args.batch_size

            if steps_done % 10 == 0 or steps_done <= 3:
                print(
                    f"  Epoch {epoch+1}/{args.epochs} | Step {steps_done}/{total_steps} | "
                    f"Loss: {batch_loss:.4f} | Time: {step_time:.1f}s | "
                    f"Batch: {batch_size_actual}"
                )

            if not args.no_wandb:
                import wandb
                wandb.log({
                    "train/loss": batch_loss,
                    "train/step_time": step_time,
                    "train/batch_size": batch_size_actual,
                    "global_step": global_step,
                }, step=global_step)

            # ---- Periodic validation ----
            if global_step % args.val_every_n_steps == 0:
                print(f"\n  === Validation at step {global_step} ===")

                if args.val_npy_dir:
                    val_metrics = evaluate_npy(
                        model, args.val_npy_dir, linked, "val",
                        args.train_device, batch_size=256
                    )
                else:
                    val_sample_paths = val_paths[:args.val_max_samples]
                    val_sample_labels = val_labels[:args.val_max_samples]
                    val_metrics = evaluate(
                        model, val_sample_paths, val_sample_labels,
                        digitizer, args.train_device, batch_size=args.batch_size
                    )

                print(
                    f"  Val AUC: {val_metrics['auc']:.4f} | "
                    f"Val AUPRC: {val_metrics['auprc']:.4f} | "
                    f"N={val_metrics['n']}, Pos={val_metrics['pos']}"
                )

                if not args.no_wandb:
                    wandb.log({
                        "val/auc": val_metrics["auc"],
                        "val/auprc": val_metrics["auprc"],
                        "val/n_samples": val_metrics["n"],
                    }, step=global_step)

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
                        "args": vars(args),
                    }, ckpt_path)
                    print(f"  New best! Saved to {ckpt_path}")
                else:
                    patience_counter += 1
                    print(f"  No improvement. Patience: {patience_counter}/{args.patience}")

                model.train()

        # ---- End of epoch ----
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(epoch_samples, 1)
        avg_acc = epoch_correct / max(epoch_samples, 1)

        print(f"\n{'='*60}")
        print(
            f"Epoch {epoch+1}/{args.epochs} complete | "
            f"Avg Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | "
            f"Time: {epoch_time/3600:.1f}h | "
            f"Best Val AUC: {best_val_auc:.4f}"
        )
        print(f"{'='*60}\n")

        if not args.no_wandb:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/epoch_acc": avg_acc,
                "train/epoch_time_hours": epoch_time / 3600,
                "epoch": epoch + 1,
            }, step=global_step)

        # Save epoch checkpoint
        ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_auc": best_val_auc,
        }, ckpt_path)

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    # ---- Final test evaluation ----
    print("\n=== Final Test Evaluation ===")
    best_ckpt = os.path.join(args.save_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=args.train_device)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded best model from step {state['global_step']}")

    # Evaluate on pre-digitized test NPY (fast)
    if args.val_npy_dir:
        print("\nTest on pre-digitized NPY:")
        test_npy_metrics = evaluate_npy(
            model, args.val_npy_dir, linked, "test",
            args.train_device, batch_size=256
        )
        print(f"  Test AUC: {test_npy_metrics['auc']:.4f}")
        print(f"  Test AUPRC: {test_npy_metrics['auprc']:.4f}")
        print(f"  Test N={test_npy_metrics['n']}, Pos={test_npy_metrics['pos']}")

        if not args.no_wandb:
            wandb.log({
                "test/auc_npy": test_npy_metrics["auc"],
                "test/auprc_npy": test_npy_metrics["auprc"],
            })

    # Also evaluate with online digitization (slower but end-to-end)
    print("\nTest on online-digitized PNGs:")
    test_metrics = evaluate(
        model, test_paths, test_labels, digitizer, args.train_device, batch_size=args.batch_size
    )
    print(f"  Test AUC: {test_metrics['auc']:.4f}")
    print(f"  Test AUPRC: {test_metrics['auprc']:.4f}")
    print(f"  Test N={test_metrics['n']}, Pos={test_metrics['pos']}")

    if not args.no_wandb:
        wandb.log({
            "test/auc": test_metrics["auc"],
            "test/auprc": test_metrics["auprc"],
        })
        wandb.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()
