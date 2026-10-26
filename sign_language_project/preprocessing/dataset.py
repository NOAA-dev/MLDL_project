"""Person 1: dataset loading and batch generation.

Conventions shared with the whole team:
    X : float32, shape (N, 1, H, W), values in [0, 1]      (layout N, C, H, W)
    y : int64,   shape (N,), values in 0 .. num_classes-1  (class IDs from classes.json)
"""

import json
from pathlib import Path

import numpy as np

from preprocessing.augment import augment_batch

DEFAULT_PROCESSED_DIR = "data/processed"


# ---------- loading ----------

def load_class_info(processed_dir=DEFAULT_PROCESSED_DIR):
    """Dict with class_names (list), class_to_id (dict), num_classes, img_size, channels."""
    with open(Path(processed_dir) / "classes.json") as f:
        return json.load(f)


def load_split(split, processed_dir=DEFAULT_PROCESSED_DIR):
    """Load 'train', 'val' or 'test'. Returns X float32 (N, 1, H, W) in [0, 1] and y int64 (N,)."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got '{split}'")
    data = np.load(Path(processed_dir) / f"{split}.npz")
    X = data["X"].astype(np.float32) / 255.0
    y = data["y"].astype(np.int64)
    return X, y


def one_hot(y, num_classes):
    """int labels (N,) -> float32 one-hot matrix (N, num_classes)."""
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out

def to_channels_last(X):
    """(N, 1, H, W) -> (N, H, W, 1): the layout used by the CNN layers."""
    return np.ascontiguousarray(X.transpose(0, 2, 3, 1))

# ---------- batches ----------

def batch_generator(X, y, batch_size=32, shuffle=True, augment=False, seed=0, drop_last=False,
                    channels_last=False):
    """Yield (X_batch, y_batch) once over the whole dataset (one epoch).

    X_batch: float32 (B, 1, H, W), or (B, H, W, 1) if channels_last=True. y_batch: int64 (B,).
    Same seed -> same order and same augmentation. Use seed=base_seed + epoch for a new order each epoch.
    Use augment=True for training only.
    """
    if len(X) != len(y):
        raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X)) if shuffle else np.arange(len(X))

    for start in range(0, len(X), batch_size):
        idx = order[start:start + batch_size]
        if drop_last and len(idx) < batch_size:
            break
        X_batch = X[idx]
        if augment:
            X_batch = augment_batch(X_batch, rng)
        if channels_last:
            X_batch = to_channels_last(X_batch)
        yield X_batch, y[idx]


def num_batches(n_samples, batch_size, drop_last=False):
    if drop_last:
        return n_samples // batch_size
    return (n_samples + batch_size - 1) // batch_size


# ---------- checks and statistics ----------

def check_batch(X_batch, y_batch, img_size, num_classes, channels_last=False):
    """Raise AssertionError if a batch breaks the team's tensor contract."""
    assert X_batch.ndim == 4, f"X must be 4D, got {X_batch.shape}"
    expected = (img_size, img_size, 1) if channels_last else (1, img_size, img_size)
    assert X_batch.shape[1:] == expected, f"bad X shape {X_batch.shape}"
    assert X_batch.dtype == np.float32, f"X dtype must be float32, got {X_batch.dtype}"
    assert X_batch.min() >= 0.0 and X_batch.max() <= 1.0, "X values must be in [0, 1]"
    assert y_batch.shape == (X_batch.shape[0],), f"bad y shape {y_batch.shape}"
    assert y_batch.dtype == np.int64, f"y dtype must be int64, got {y_batch.dtype}"
    assert y_batch.min() >= 0 and y_batch.max() < num_classes, "y contains an invalid class ID"
    assert np.isfinite(X_batch).all(), "X contains NaN or inf"


def print_statistics(processed_dir=DEFAULT_PROCESSED_DIR):
    info = load_class_info(processed_dir)
    print(f"Classes: {info['num_classes']}  image size: {info['img_size']}x{info['img_size']}  channels: {info['channels']}")
    for split in ("train", "val", "test"):
        X, y = load_split(split, processed_dir)
        counts = np.bincount(y, minlength=info["num_classes"])
        print(f"{split:5s} X{X.shape} y{y.shape}  per-class min/max: {counts.min()}/{counts.max()}  "
              f"pixel mean/std: {X.mean():.3f}/{X.std():.3f}")


if __name__ == "__main__":
    print_statistics()
