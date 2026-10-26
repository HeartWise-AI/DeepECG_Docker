"""
Create mixed training manifests: clean NPY + digitized NPY for training.
Val/Test use digitized-only (to measure real-world performance).

This doubles effective training data for samples that have both versions,
helping the model learn to handle both clean and digitized domains.

Usage:
    python scripts/create_mixed_manifests.py \
        --npy_source_dir /media/data1/anolin/temp_new_dataset/ecg_npy \
        --dig_npy_dir data/acs_digitized_npy \
        --output_dir finetune_data/acs_mixed \
        --linked_parquet data/acs_linked_dataset.parquet
"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_and_stack(paths_labels, desc="Loading"):
    """Load NPY files and stack into arrays."""
    signals = []
    valid_labels = []
    failed = 0

    for path, label in tqdm(paths_labels, desc=desc):
        try:
            arr = np.load(path).astype(np.float32)
            if arr.shape == (12, 2500):
                arr = arr.T
            if arr.ndim == 2 and arr.shape == (2500, 12):
                arr = arr[:, :, np.newaxis]  # -> (2500, 12, 1)
            elif arr.ndim == 3 and arr.shape == (2500, 12, 1):
                pass
            else:
                failed += 1
                continue
            signals.append(arr)
            valid_labels.append(label)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"Failed: {path}: {e}")

    if failed > 0:
        print(f"Failed to load {failed} files")

    X = np.stack(signals, axis=0)
    Y = np.array(valid_labels, dtype=np.float32)[:, np.newaxis]  # (N, 1)
    return X, Y


def create_manifest(x_path, y_path, x_shape):
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
                        help="Directory with original NPY files (clean)")
    parser.add_argument("--dig_npy_dir", type=str, required=True,
                        help="Directory with digitized NPY files")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_dir = os.path.join(args.output_dir, "manifest_mixed")
    os.makedirs(manifest_dir, exist_ok=True)

    linked = pd.read_parquet(args.linked_parquet)
    print(f"Linked dataset: {len(linked)} rows")

    # Find which samples have digitized NPYs
    linked['dig_npy_path'] = linked['ecg_key'].apply(
        lambda ek: os.path.join(args.dig_npy_dir, f"{ek}.npy")
    )
    linked['has_dig'] = linked['dig_npy_path'].apply(os.path.exists)
    print(f"Digitized NPYs available: {linked.has_dig.sum()}")

    # TRAIN: Mix clean NPY (all) + digitized NPY (available)
    train_df = linked[linked.Split == 'train'].copy()
    train_paths_labels = []

    # Add all clean NPYs
    if args.npy_source_dir:
        for _, row in train_df.iterrows():
            npy_path = row['npy_path']
            if not os.path.isabs(npy_path) and args.npy_source_dir:
                npy_path = os.path.join(args.npy_source_dir, os.path.basename(npy_path))
            if os.path.exists(npy_path):
                train_paths_labels.append((npy_path, row['Acute_Obstruction']))

    # Add digitized NPYs
    for _, row in train_df[train_df.has_dig].iterrows():
        train_paths_labels.append((row['dig_npy_path'], row['Acute_Obstruction']))

    print(f"\nTrain: {len(train_paths_labels)} samples "
          f"(clean: {len(train_df)}, dig: {train_df.has_dig.sum()})")

    X_train, Y_train = load_and_stack(train_paths_labels, "Train")
    print(f"Train stacked: X={X_train.shape}, Y={Y_train.shape}, pos={Y_train.sum():.0f}")

    # VAL: Digitized only (for realistic evaluation)
    val_df = linked[(linked.Split == 'val') & linked.has_dig].copy()
    val_paths_labels = [(row['dig_npy_path'], row['Acute_Obstruction'])
                        for _, row in val_df.iterrows()]
    print(f"\nVal (dig only): {len(val_paths_labels)} samples")

    X_val, Y_val = load_and_stack(val_paths_labels, "Val")
    print(f"Val stacked: X={X_val.shape}, Y={Y_val.shape}, pos={Y_val.sum():.0f}")

    # TEST: Digitized only
    test_df = linked[(linked.Split == 'test') & linked.has_dig].copy()
    test_paths_labels = [(row['dig_npy_path'], row['Acute_Obstruction'])
                         for _, row in test_df.iterrows()]
    print(f"\nTest (dig only): {len(test_paths_labels)} samples")

    X_test, Y_test = load_and_stack(test_paths_labels, "Test")
    print(f"Test stacked: X={X_test.shape}, Y={Y_test.shape}, pos={Y_test.sum():.0f}")

    # Save
    for name, fairseq_name, X, Y in [
        ("train", "train", X_train, Y_train),
        ("val", "valid", X_val, Y_val),
        ("test", "test", X_test, Y_test),
    ]:
        x_path = os.path.join(args.output_dir, f"mixed_{name}_x.npy")
        y_path = os.path.join(args.output_dir, f"mixed_{name}_y.npy")
        np.save(x_path, X)
        np.save(y_path, Y)
        print(f"Saved {x_path} ({os.path.getsize(x_path)/1024/1024:.1f} MB)")

        manifest_path = os.path.join(manifest_dir, f"{fairseq_name}.tsv")
        with open(manifest_path, "w") as f:
            f.write(create_manifest(
                x_path=os.path.abspath(x_path),
                y_path=os.path.abspath(y_path),
                x_shape=X.shape
            ))
        print(f"Manifest: {manifest_path}")

    print(f"\nDone! Output in {args.output_dir}")


if __name__ == "__main__":
    main()
