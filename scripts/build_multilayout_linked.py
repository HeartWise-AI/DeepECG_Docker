#!/usr/bin/env python3
"""
Build a linked dataset parquet for multi-layout WCR training.

Merges:
- Multi-layout rendering manifest (from render_acs_multilayout.py)
- Original linked dataset splits (for consistent train/val/test)

New rows (in v6 but not in old linked dataset) get assigned to train split.

Output: data/acs_multilayout_linked.parquet with columns:
    npy_path, png_path, layout, Acute_Obstruction, Split
"""

import os
import pandas as pd

# Paths
OLD_LINKED = "/volume/DeepECG_Docker/data/acs_linked_dataset.parquet"
MANIFEST = "/volume/Open-ECG-Digitizer/data/acs_multilayout/manifest.csv"
OUTPUT = "/volume/DeepECG_Docker/data/acs_multilayout_linked.parquet"

def main():
    # Load old linked dataset for splits
    old = pd.read_parquet(OLD_LINKED)
    print(f"Old linked dataset: {len(old)} rows")
    print(f"  Splits: {old['Split'].value_counts().to_dict()}")

    # Create npy_path → Split mapping
    split_map = dict(zip(old['npy_path'], old['Split']))

    # Load multi-layout manifest
    manifest = pd.read_csv(MANIFEST)
    print(f"\nManifest: {len(manifest)} rows")
    print(f"  Layouts: {manifest['layout'].nunique()} unique")

    # Add Acute_Obstruction label
    manifest['Acute_Obstruction'] = (manifest['condition_severity_modified'] >= 2).astype(int)

    # Add Split from old dataset, default new rows to "train"
    manifest['Split'] = manifest['npy_path'].map(split_map).fillna('train')

    # Summary
    print(f"\nFinal dataset: {len(manifest)} rows")
    print(f"  Splits: {manifest['Split'].value_counts().to_dict()}")
    print(f"  Positive: {manifest['Acute_Obstruction'].sum()} ({100*manifest['Acute_Obstruction'].mean():.1f}%)")
    print(f"  Layout distribution:")
    for layout, count in manifest['layout'].value_counts().head(5).items():
        print(f"    {layout}: {count}")

    # Verify PNG paths exist (sample check)
    n_check = min(100, len(manifest))
    exists = manifest['png_path'].head(n_check).apply(os.path.exists)
    print(f"\n  PNG accessibility ({n_check} checked): {exists.sum()}/{n_check}")

    # Save
    manifest[['npy_path', 'png_path', 'layout', 'Acute_Obstruction', 'Split']].to_parquet(OUTPUT)
    print(f"\nSaved: {OUTPUT}")


if __name__ == '__main__':
    main()
