"""
Person 1: raw ASL images -> cleaned, resized, split dataset.

Run:
    python -m preprocessing.preprocess_images --raw_dir data/raw/asl_alphabet_train
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 32
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# test = whatever is left (0.15)
SEED = 42


def preprocess_image(image, img_size=IMG_SIZE):
    """One PIL image -> uint8 array (img_size, img_size). Grayscale + square crop + resize."""
    gray = image.convert("L")
    w, h = gray.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    gray = gray.crop((left, top, left + side, top + side))
    gray = gray.resize((img_size, img_size))
    return np.array(gray, dtype=np.uint8)


def load_clean_images(raw_dir, max_per_class=500):
    """Go through every class folder, clean each image, skip bad/duplicate ones."""
    raw_dir = Path(raw_dir)
    class_names = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())

    images = []
    labels = []
    seen_hashes = set()

    for class_id, class_name in enumerate(class_names):
        folder = raw_dir / class_name
        count = 0

        for file_path in folder.iterdir():
            if count >= max_per_class:
                break

            try:
                img = Image.open(file_path)
                img.load()
            except Exception:
                continue  # skip broken file

            pixels = preprocess_image(img)

            fingerprint = hashlib.md5(pixels.tobytes()).hexdigest()
            if fingerprint in seen_hashes:
                continue  # skip duplicate image
            seen_hashes.add(fingerprint)

            images.append(pixels)
            labels.append(class_id)
            count += 1

    print("Classes:", class_names)
    print("Total images kept:", len(images))

    X = np.array(images, dtype=np.uint8)
    y = np.array(labels, dtype=np.int64)
    return X, y, class_names


def split_dataset(X, y, seed=SEED):
    """Shuffle and split into train/val/test."""
    rng = np.random.default_rng(seed)
    n = len(X)
    order = rng.permutation(n)

    n_train = int(n * TRAIN_FRACTION)
    n_val = int(n * VAL_FRACTION)

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    return {
        "train": (X[train_idx], y[train_idx]),
        "val": (X[val_idx], y[val_idx]),
        "test": (X[test_idx], y[test_idx]),
    }


def save_dataset(out_dir, splits, class_names):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (X, y) in splits.items():
        # add channel dimension -> (N,1,H,W), the shape the whole team uses
        X_reshaped = X.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
        np.savez(out_dir / f"{name}.npz", X=X_reshaped, y=y)
        print(f"Saved {name}: {len(y)} images")

    with open(out_dir / "classes.json", "w") as f:
        json.dump({"class_names": class_names}, f)


def run_preprocessing(raw_dir, out_dir, max_per_class=500):
    X, y, class_names = load_clean_images(raw_dir, max_per_class)
    splits = split_dataset(X, y)
    save_dataset(out_dir, splits, class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--max_per_class", type=int, default=500)
    args = parser.parse_args()

    run_preprocessing(args.raw_dir, args.out_dir, args.max_per_class)
