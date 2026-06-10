"""Data loading for the NYCU HAR competition.

Each file = one 5-minute window = one labeled sample (300 rows x 6 sensor columns).
We load every file once into a dense (n_files, 300, 6) array + an aligned metadata
frame, and cache it to disk so feature engineering in later steps is fast and
fully deterministic (file order is fixed by sorted globbing).

Confirmed by audit (trusted, not re-checked here): every file is exactly 300 rows,
index is contiguous 0..299, no NaN/inf.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd

BASE = "/root/dm-assignment3/nycu-data-mining-assignment-3"
TRAIN_DIR = os.path.join(BASE, "train", "train")
TEST_DIR = os.path.join(BASE, "test", "test")
SUBMISSION = os.path.join(BASE, "sample_submission.csv")
CACHE_DIR = "/root/dm-assignment3/cache"

# Sensor column order used everywhere downstream. Do not reorder.
SENSOR_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
N_STEPS = 300


def _list_files(split_dir: str) -> list[str]:
    """Deterministic, sorted list of all CSV paths under User_*/ subfolders."""
    return sorted(glob.glob(os.path.join(split_dir, "User_*", "*.csv")))


def _load_split_uncached(split: str):
    split_dir = TRAIN_DIR if split == "train" else TEST_DIR
    has_label = split == "train"
    files = _list_files(split_dir)

    n = len(files)
    X = np.empty((n, N_STEPS, len(SENSOR_COLS)), dtype=np.float64)
    file_ids = np.empty(n, dtype=np.int64)
    users = np.empty(n, dtype=object)
    labels = np.full(n, -1, dtype=np.int64)

    for i, fp in enumerate(files):
        df = pd.read_csv(fp)
        assert len(df) == N_STEPS, f"{fp} has {len(df)} rows, expected {N_STEPS}"
        X[i] = df[SENSOR_COLS].to_numpy(dtype=np.float64)
        # file_id from the column (numeric); filename is zero-padded but equal.
        file_ids[i] = int(df["file_id"].iloc[0])
        users[i] = os.path.basename(os.path.dirname(fp))
        if has_label:
            labels[i] = int(df["label"].iloc[0])

    meta = pd.DataFrame({"file_id": file_ids, "user": users})
    if has_label:
        meta["label"] = labels
    return X, meta


def load_split(split: str, use_cache: bool = True):
    """Return (X_raw, meta).

    X_raw : float64 array, shape (n_files, 300, 6), columns = SENSOR_COLS order.
    meta  : DataFrame with columns [file_id, user] (+ 'label' for train),
            row-aligned to X_raw.
    """
    assert split in ("train", "test")
    os.makedirs(CACHE_DIR, exist_ok=True)
    npz_path = os.path.join(CACHE_DIR, f"{split}_raw.npz")
    meta_path = os.path.join(CACHE_DIR, f"{split}_meta.csv")

    if use_cache and os.path.exists(npz_path) and os.path.exists(meta_path):
        X = np.load(npz_path)["X"]
        meta = pd.read_csv(meta_path)
        return X, meta

    X, meta = _load_split_uncached(split)
    np.savez_compressed(npz_path, X=X)
    meta.to_csv(meta_path, index=False)
    return X, meta


if __name__ == "__main__":
    for split in ("train", "test"):
        X, meta = load_split(split, use_cache=False)
        line = f"{split}: X={X.shape}, users={meta['user'].nunique()}"
        if "label" in meta.columns:
            line += f", label dist={meta['label'].value_counts().sort_index().to_dict()}"
        print(line)
