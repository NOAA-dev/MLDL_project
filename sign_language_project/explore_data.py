import argparse
from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from preprocessing.augment import augment_batch
from preprocessing.dataset import load_class_info, load_split
from preprocessing.preprocess_images import IMAGE_EXTENSIONS, preprocess_image

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/processed")
parser.add_argument("--raw_dir", default=None, help="folder containing A, B, ..., space (optional, for before-preprocessing plots)")
parser.add_argument("--classes", nargs="+", default=["V", "W", "M", "N", "P", "Q"], help="letters to inspect closely")
parser.add_argument("--out_dir", default="data_visualization")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_show", action="store_true")
args = parser.parse_args()

if args.no_show:
    matplotlib.use("Agg")

out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True)
rng = np.random.default_rng(args.seed)

class_info = load_class_info(args.data_dir)
class_names = class_info["class_names"]
num_classes = class_info["num_classes"]
for name in args.classes:
    if name not in class_names:
        raise SystemExit(f"Unknown class '{name}'. Choose from: {class_names}")

splits = {name: load_split(name, args.data_dir) for name in ("train", "val", "test")}
X_train, y_train = splits["train"]
x = np.arange(num_classes)


# BEFORE preprocessing 
if args.raw_dir:
    raw_dir = Path(args.raw_dir)
    raw_files = {}
    for name in class_names:
        files = [p for p in sorted((raw_dir / name).iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
        raw_files[name] = files

    counts = [len(raw_files[name]) for name in class_names]
    fig, ax = plt.subplots(figsize=(15, 5.5), constrained_layout=True)
    ax.bar(x, counts)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_title("Raw dataset: images per class (before preprocessing)")
    ax.set_ylabel("Number of images")
    fig.savefig(out_dir / "raw_class_counts.png", dpi=130)
    print(f"Raw images: {sum(counts)} in total, {min(counts)} to {max(counts)} per class")

    sizes = []
    for name in class_names:
        for path in rng.choice(raw_files[name], size=min(20, len(raw_files[name])), replace=False):
            with Image.open(path) as img:
                sizes.append(img.size)
    unique_sizes, size_counts = np.unique(np.array(sizes), axis=0, return_counts=True)
    print("Raw image sizes (width x height) found in a sample:")
    for size, count in zip(unique_sizes, size_counts):
        print(f"  {int(size[0])} x {int(size[1])}: {int(count)} images")

    n = 8
    chosen = [(name, rng.choice(raw_files[name])) for name in rng.choice(class_names, size=n, replace=False)]
    fig, axes = plt.subplots(2, n, figsize=(16, 5.5), constrained_layout=True)
    for i, (name, path) in enumerate(chosen):
        with Image.open(path) as img:
            raw = img.convert("RGB")
            processed = preprocess_image(img, class_info["img_size"])
        axes[0, i].imshow(raw)
        axes[0, i].set_title(f"Raw: {name}\n{raw.size[0]}x{raw.size[1]} colour", fontsize=10)
        axes[1, i].imshow(processed, cmap="gray", vmin=0, vmax=255)
        axes[1, i].set_title(f"Processed: {processed.shape[1]}x{processed.shape[0]} gray", fontsize=10)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    fig.suptitle("Before vs after preprocessing (grayscale, centre crop, resize)", fontsize=14)
    fig.savefig(out_dir / "raw_vs_processed.png", dpi=130)


# AFTER preprocessing 

# 1. Class balance
bottom = np.zeros(num_classes)
fig, ax = plt.subplots(figsize=(15, 5.5), constrained_layout=True)
for split in ("train", "val", "test"):
    counts = np.bincount(splits[split][1], minlength=num_classes)
    ax.bar(x, counts, bottom=bottom, label=f"{split} ({counts.sum()} images)")
    bottom += counts
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_ylim(0, bottom.max() * 1.2)
ax.set_title("Images per class, split into train / val / test (after preprocessing)")
ax.set_ylabel("Number of images")
ax.legend(loc="upper center", ncol=3)
fig.savefig(out_dir / "class_balance.png", dpi=130)

# 2. Variety inside the chosen classes
n_examples = 8
fig, axes = plt.subplots(len(args.classes), n_examples, figsize=(16, 9), constrained_layout=True, squeeze=False)
for row, name in enumerate(args.classes):
    class_id = class_names.index(name)
    picks = rng.choice(np.where(y_train == class_id)[0], size=n_examples, replace=False)
    for col, index in enumerate(picks):
        ax = axes[row, col]
        ax.imshow(X_train[index, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[row, 0].set_ylabel(name, rotation=0, fontsize=20, labelpad=25, va="center")
fig.suptitle(f"{n_examples} different training images for each of: {', '.join(args.classes)}", fontsize=14)
fig.savefig(out_dir / "variety_within_classes.png", dpi=130)

# 3. Pixel values and lighting
image_means = X_train.mean(axis=(1, 2, 3))
white_share = (X_train > 0.98).mean() * 100
fig, (ax_pixels, ax_images) = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
ax_pixels.hist(X_train.ravel(), bins=50)
ax_pixels.axvline(X_train.mean(), color="tab:red", linestyle="--", label=f"mean {X_train.mean():.3f}, std {X_train.std():.3f}")
ax_pixels.set_title(f"All pixels in the training set ({white_share:.1f}% are pure white)")
ax_pixels.set_xlabel("Pixel value (0 = black, 1 = white)")
ax_pixels.set_ylabel("Number of pixels")
ax_pixels.legend(loc="upper left")
ax_images.hist(image_means, bins=40, color="tab:green")
ax_images.set_title("Average brightness of each image (lighting differences)")
ax_images.set_xlabel("Mean pixel value of one image")
ax_images.set_ylabel("Number of images")
fig.savefig(out_dir / "pixel_values.png", dpi=130)

# 4. Augmentation
n_versions = 4
aug_classes = (args.classes * 4)[:4]
fig, axes = plt.subplots(len(aug_classes), n_versions + 1, figsize=(16, 9), constrained_layout=True)
for row, name in enumerate(aug_classes):
    class_id = class_names.index(name)
    index = rng.choice(np.where(y_train == class_id)[0])
    original = X_train[index:index + 1]
    axes[row, 0].imshow(original[0, 0], cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"Original: {name}", fontsize=11)
    for col in range(1, n_versions + 1):
        augmented = augment_batch(original, rng)
        axes[row, col].imshow(augmented[0, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, col].set_title("Augmented", fontsize=11)
for ax in axes.flat:
    ax.axis("off")
fig.suptitle("Training augmentation: random rotation, shift, zoom, brightness, contrast (grey corners are filler)", fontsize=14)
fig.savefig(out_dir / "augmentation.png", dpi=130)

# 5. One example per class
cols = 8
rows = int(np.ceil(num_classes / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, 2.2 * rows), constrained_layout=True)
for ax in axes.flat:
    ax.axis("off")
for class_id, ax in enumerate(axes.flat[:num_classes]):
    index = rng.choice(np.where(y_train == class_id)[0])
    ax.imshow(X_train[index, 0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(class_names[class_id], fontsize=12)
fig.suptitle("One training example per class", fontsize=14)
fig.savefig(out_dir / "examples_per_class.png", dpi=130)

print(f"Saved figures to {out_dir}/")
if not args.no_show:
    plt.show()

print(f"Saved figures to {out_dir}/")
if not args.no_show:
    plt.show()