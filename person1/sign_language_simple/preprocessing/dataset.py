"""
Person 1: load the saved dataset and make batches for training.
"""

import json
from pathlib import Path

import numpy as np

from preprocessing.augment import augment_batch

DATA_DIR = "data/processed"


def load_class_names(data_dir=DATA_DIR):
    with open(Path(data_dir) / "classes.json") as f:
        return json.load(f)["class_names"]


def load_split(split, data_dir=DATA_DIR):
    """split = 'train', 'val' or 'test'."""
    data = np.load(Path(data_dir) / f"{split}.npz")
    X = data["X"].astype(np.float32) / 255.0   # scale pixels to [0,1]
    y = data["y"].astype(np.int64)
    return X, y


def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    for i, label in enumerate(y):
        out[i, label] = 1.0
    return out


def get_batches(X, y, batch_size=32, augment=False, seed=0):
    """Simple batch generator. Shuffles once, then yields batches."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))

    for start in range(0, len(X), batch_size):
        idx = order[start:start + batch_size]
        X_batch = X[idx]
        y_batch = y[idx]
        if augment:
            X_batch = augment_batch(X_batch, rng)
        yield X_batch, y_batch


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        X, y = load_split(split)
        print(split, "X shape:", X.shape, "y shape:", y.shape)
