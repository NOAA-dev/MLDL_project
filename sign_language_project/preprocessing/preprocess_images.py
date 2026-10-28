import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 64
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}




def preprocess_image(image, img_size=IMG_SIZE):
    gray = image.convert("L")
    w, h = gray.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    gray = gray.crop((left, top, left + side, top + side))
    gray = gray.resize((img_size, img_size), Image.BILINEAR)
    return np.array(gray, dtype=np.uint8)


def preprocess_for_inference(image, img_size=IMG_SIZE):
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    pixels = preprocess_image(image, img_size)
    return (pixels.astype(np.float32) / 255.0).reshape(1, 1, img_size, img_size)



def find_class_names(raw_dir, class_names=None):
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset folder not found: {raw_dir}")
    found = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
    if class_names is None:
        if not found:
            raise ValueError(f"No class folders inside {raw_dir}")
        return found
    missing = [c for c in class_names if c not in found]
    if missing:
        raise ValueError(f"Class folders missing in {raw_dir}: {missing}")
    return list(class_names)


def load_clean_images(raw_dir, class_names, img_size=IMG_SIZE, max_per_class=None, seed=SEED):
    rng = np.random.default_rng(seed)
    raw_dir = Path(raw_dir)
    seen_hashes = set()
    images, labels = [], []
    report = {"corrupted": [], "duplicates": [], "not_an_image": [], "ignored_folders": []}

    all_folders = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
    report["ignored_folders"] = [f for f in all_folders if f not in class_names]

    for class_id, class_name in enumerate(class_names):
        files = sorted((raw_dir / class_name).iterdir())
        order = rng.permutation(len(files))
        kept = 0
        for i in order:
            if max_per_class is not None and kept >= max_per_class:
                break
            path = files[i]
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                report["not_an_image"].append(str(path))
                continue
            try:
                with Image.open(path) as img:
                    img.load()
                    pixels = preprocess_image(img, img_size)
            except (OSError, ValueError, Image.DecompressionBombError):
                report["corrupted"].append(str(path))
                continue

            digest = hashlib.md5(pixels.tobytes()).hexdigest()
            if digest in seen_hashes:
                report["duplicates"].append(str(path))
                continue
            seen_hashes.add(digest)
            images.append(pixels)
            labels.append(class_id)
            kept += 1

        if kept == 0:
            raise ValueError(f"Class '{class_name}' has no valid images")

    return np.stack(images), np.array(labels, dtype=np.int64), report



def stratified_split(y, fractions=SPLIT_FRACTIONS, seed=SEED):
    train_frac, val_frac, test_frac = fractions
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1")

    rng = np.random.default_rng(seed)
    split_idx = {"train": [], "val": [], "test": []}

    for class_id in np.unique(y):
        idx = rng.permutation(np.where(y == class_id)[0])
        n_test = int(round(len(idx) * test_frac))
        n_val = int(round(len(idx) * val_frac))
        n_train = len(idx) - n_val - n_test
        if min(n_train, n_val, n_test) < 1:
            raise ValueError(f"Class {class_id} has only {len(idx)} images; too few to split")
        split_idx["test"].append(idx[:n_test])
        split_idx["val"].append(idx[n_test:n_test + n_val])
        split_idx["train"].append(idx[n_test + n_val:])

    return {name: rng.permutation(np.concatenate(parts)) for name, parts in split_idx.items()}




def save_processed(out_dir, X, y, split_idx, class_names, img_size, report):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = X.reshape(-1, 1, img_size, img_size)
    for name, idx in split_idx.items():
        np.savez_compressed(out_dir / f"{name}.npz", X=X[idx], y=y[idx])

    class_info = {
        "class_names": class_names,
        "class_to_id": {name: i for i, name in enumerate(class_names)},
        "num_classes": len(class_names),
        "img_size": img_size,
        "channels": 1,
    }
    with open(out_dir / "classes.json", "w") as f:
        json.dump(class_info, f, indent=2)

    train_pixels = X[split_idx["train"]].astype(np.float32) / 255.0
    stats = {
        "total_images": int(len(y)),
        "split_sizes": {name: int(len(idx)) for name, idx in split_idx.items()},
        "per_class_counts": {
            name: {class_names[c]: int((y[idx] == c).sum()) for c in range(len(class_names))}
            for name, idx in split_idx.items()
        },
        "train_pixel_mean": float(train_pixels.mean()),
        "train_pixel_std": float(train_pixels.std()),
        "removed": {
            "corrupted": len(report["corrupted"]),
            "duplicates": len(report["duplicates"]),
            "not_an_image": len(report["not_an_image"]),
            "ignored_folders": report["ignored_folders"],
        },
        "removed_files": {
            "corrupted": report["corrupted"],
            "duplicates": report["duplicates"],
            "not_an_image": report["not_an_image"],
        },
    }
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats




def run_preprocessing(raw_dir, out_dir, img_size=IMG_SIZE, max_per_class=None,
                      class_names=None, fractions=SPLIT_FRACTIONS, seed=SEED):
    class_names = find_class_names(raw_dir, class_names)
    X, y, report = load_clean_images(raw_dir, class_names, img_size, max_per_class, seed)
    split_idx = stratified_split(y, fractions, seed)
    return save_processed(out_dir, X, y, split_idx, class_names, img_size, report)


def main():
    parser = argparse.ArgumentParser(description="Preprocess the raw ASL image dataset")
    parser.add_argument("--raw_dir", required=True, help="folder with one sub-folder per class")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--max_per_class", type=int, default=500)
    parser.add_argument("--classes", nargs="+", default=None,
                        help="class folders to use (default: every sub-folder, sorted by name)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    stats = run_preprocessing(args.raw_dir, args.out_dir, args.img_size, args.max_per_class,
                              class_names=args.classes, seed=args.seed)
    print("Total images kept:", stats["total_images"])
    print("Split sizes:", stats["split_sizes"])
    print("Removed:", stats["removed"])
    print(f"Train pixel mean/std: {stats['train_pixel_mean']:.4f} / {stats['train_pixel_std']:.4f}")
    print("Saved to", args.out_dir)


if __name__ == "__main__":
    main()
