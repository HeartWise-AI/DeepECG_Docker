"""
Create stacked NPY files and fairseq-signals manifests for ACS fine-tuning.

Reads the linked dataset and creates:
- Stacked train/test/val X arrays (N, 2500, 12, 1)
- Stacked train/test/val Y arrays (N,)
- TSV manifests for fairseq-signals NpyECGDataset

Usage:
    python scripts/create_finetune_manifests.py \
        --npy_source_dir /media/data1/anolin/temp_new_dataset/ecg_npy \
        --output_dir finetune_data/acs_npy \
        --variant npy

    python scripts/create_finetune_manifests.py \
        --npy_source_dir data/acs_digitized_npy \
        --output_dir finetune_data/acs_dig \
        --variant dig
"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_and_stack(npy_paths: list, labels: list, source_dir: str = None) -> tuple:
    """Load NPY files and stack into arrays."""
    signals = []
    valid_labels = []
    failed = 0

    for npy_path, label in tqdm(zip(npy_paths, labels), total=len(npy_paths), desc="Loading"):
        try:
            if source_dir and not os.path.isabs(npy_path):
                full_path = os.path.join(source_dir, npy_path)
            else:
                full_path = npy_path

            arr = np.load(full_path).astype(np.float32)

            # Ensure shape is (2500, 12)
            if arr.shape == (12, 2500):
                arr = arr.T
            if arr.ndim == 2 and arr.shape == (2500, 12):
                arr = arr[:, :, np.newaxis]  # -> (2500, 12, 1)
            elif arr.ndim == 3 and arr.shape == (2500, 12, 1):
                pass
            else:
                failed += 1
                if failed <= 5:
                    print(f"Unexpected shape {arr.shape} for {full_path}")
                continue

            signals.append(arr)
            valid_labels.append(label)

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"Failed to load {full_path}: {e}")

    if failed > 0:
        print(f"Failed to load {failed} files")

    X = np.stack(signals, axis=0)  # (N, 2500, 12, 1)
    Y = np.array(valid_labels, dtype=np.float32)[:, np.newaxis]  # (N, 1)
    return X, Y


def create_manifest(x_path: str, y_path: str, x_shape: tuple) -> str:
    """Create fairseq-signals manifest TSV content."""
    return (
        f"x_path:{x_path}\n"
        f"x_shape:{x_shape}\n"
        f"y_path:{y_path}\n"
        f"label_indexes:[0]\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linked_parquet", type=str, default="data/acs_linked_dataset.parquet")
    parser.add_argument("--npy_source_dir", type=str, default=None,
                        help="Directory containing NPY files (for digitized variant)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--variant", type=str, choices=["npy", "dig"], required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_dir = os.path.join(args.output_dir, f"manifest_{args.variant}")
    os.makedirs(manifest_dir, exist_ok=True)

    # Load linked dataset
    linked = pd.read_parquet(args.linked_parquet)
    print(f"Linked dataset: {len(linked)} rows")

    for split_name, fairseq_name in [("train", "train"), ("val", "valid"), ("test", "test")]:
        split_df = linked[linked['Split'] == split_name].copy()
        print(f"\n{split_name}: {len(split_df)} rows (pos={split_df['Acute_Obstruction'].sum()})")

        if args.variant == "npy":
            npy_paths = split_df['npy_path'].tolist()
            source_dir = None
        else:
            # For digitized, use ecg_key to find NPY in source dir
            npy_paths = [f"{ek}.npy" for ek in split_df['ecg_key'].tolist()]
            source_dir = args.npy_source_dir

        labels = split_df['Acute_Obstruction'].tolist()

        X, Y = load_and_stack(npy_paths, labels, source_dir)
        print(f"Stacked X: {X.shape}, Y: {Y.shape}, pos={Y.sum():.0f}")

        # Save
        x_filename = f"{args.variant}_{split_name}_x.npy"
        y_filename = f"{args.variant}_{split_name}_y.npy"
        x_path = os.path.join(args.output_dir, x_filename)
        y_path = os.path.join(args.output_dir, y_filename)

        np.save(x_path, X)
        np.save(y_path, Y)
        print(f"Saved {x_path} ({os.path.getsize(x_path)/1024/1024:.1f} MB)")

        # Create manifest
        manifest_path = os.path.join(manifest_dir, f"{fairseq_name}.tsv")
        with open(manifest_path, "w") as f:
            f.write(create_manifest(
                x_path=os.path.abspath(x_path),
                y_path=os.path.abspath(y_path),
                x_shape=X.shape
            ))
        print(f"Manifest: {manifest_path}")

    print(f"\nDone! Output in {args.output_dir}")
    print(f"Manifests in {manifest_dir}")


if __name__ == "__main__":
    main()
