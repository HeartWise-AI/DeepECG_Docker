"""
Convert digitized ECG CSVs from Open-ECG-Digitizer to NPY format for DeepECG.

Input: *_timeseries_canonical.csv files (from digitizer)
Output: NPY files (2500, 12) at 250 Hz

Usage:
    python scripts/convert_digitized_csv_to_npy.py \
        --csv_dir data/acs_digitized_csv \
        --npy_dir data/acs_digitized_npy \
        --linked_parquet data/acs_linked_dataset.parquet
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_LENGTH = 2500  # 10s at 250 Hz
PTBXL_POWER_RATIO = 3.003154


def apply_psa(ecg_array: np.ndarray) -> np.ndarray:
    """Apply Power Spectral Alignment to a (2500, 12) array."""
    mag = np.abs(np.fft.fft(ecg_array[:, 0]))
    current_power = float(np.mean(mag))
    if current_power < 1e-10:
        return ecg_array
    factor = PTBXL_POWER_RATIO / current_power
    scaled = np.real(np.fft.ifft(np.fft.fft(ecg_array, axis=0) * factor, axis=0))
    return scaled.astype(np.float32)


def csv_to_npy(csv_path: str) -> np.ndarray:
    """Convert a digitized CSV to (2500, 12) NPY array."""
    df = pd.read_csv(csv_path)

    signals = []
    for lead in LEAD_ORDER:
        if lead in df.columns:
            sig = df[lead].values.astype(np.float32)
            sig = np.nan_to_num(sig, nan=0.0)
            if len(sig) != TARGET_LENGTH:
                sig = resample(sig, TARGET_LENGTH)
            signals.append(sig)
        else:
            # Missing lead - fill with zeros
            signals.append(np.zeros(TARGET_LENGTH, dtype=np.float32))

    ecg_array = np.stack(signals, axis=-1)  # (2500, 12)
    return ecg_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True)
    parser.add_argument("--npy_dir", type=str, required=True)
    parser.add_argument("--linked_parquet", type=str, default="data/acs_linked_dataset.parquet")
    parser.add_argument("--apply_psa", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.npy_dir, exist_ok=True)

    # Load linked dataset to map PNG names to ecg_keys
    linked = pd.read_parquet(args.linked_parquet)
    # Create mapping: png_basename (without .png) -> ecg_key
    linked['png_basename'] = linked['png_path'].apply(lambda x: os.path.basename(str(x)))
    # The digitized CSV files are named like: MUSE_20220816_151452_48000.xml_timeseries_canonical.csv
    # The PNG files are named like: MUSE_20220816_151452_48000.xml.png
    # So csv_stem = png_basename.replace('.png', '') + '_timeseries_canonical'
    # Or: png_basename = csv_stem.replace('_timeseries_canonical', '') + '.png'

    # Build mapping from csv filename stem to ecg_key
    png_to_ecg = dict(zip(linked['png_basename'], linked['ecg_key']))

    # Find all CSV files
    csv_files = glob.glob(os.path.join(args.csv_dir, "*_timeseries_canonical.csv"))
    print(f"Found {len(csv_files)} digitized CSV files")

    converted = 0
    failed = 0
    unmatched = 0

    for csv_path in tqdm(csv_files, desc="Converting CSV to NPY"):
        try:
            # Extract the PNG basename from the CSV filename
            csv_basename = os.path.basename(csv_path)
            # Remove _timeseries_canonical.csv to get the XML name
            xml_name = csv_basename.replace("_timeseries_canonical.csv", "")
            # The PNG name is xml_name + .png
            png_name = xml_name + ".png"

            if png_name not in png_to_ecg:
                unmatched += 1
                continue

            ecg_key = png_to_ecg[png_name]
            npy_path = os.path.join(args.npy_dir, f"{ecg_key}.npy")

            ecg_array = csv_to_npy(csv_path)

            if args.apply_psa:
                ecg_array = apply_psa(ecg_array)

            np.save(npy_path, ecg_array.astype(np.float32))
            converted += 1

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"Error converting {csv_path}: {e}")

    print(f"\nConversion complete:")
    print(f"  Converted: {converted}")
    print(f"  Failed: {failed}")
    print(f"  Unmatched: {unmatched}")
    print(f"  Output dir: {args.npy_dir}")


if __name__ == "__main__":
    main()
