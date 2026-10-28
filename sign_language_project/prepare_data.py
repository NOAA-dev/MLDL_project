import argparse
from pathlib import Path

from preprocessing.preprocess_images import run_preprocessing, IMG_SIZE

parser = argparse.ArgumentParser()
parser.add_argument("--raw_dir", default=None, help="folder that directly contains A, B, ..., space")
parser.add_argument("--img_size", type=int, default=IMG_SIZE)
parser.add_argument("--max_per_class", type=int, default=500)
args = parser.parse_args()

# 1. Locate the raw images
if args.raw_dir is None:
    import kagglehub
    download_path = Path(kagglehub.dataset_download("grassknoted/asl-alphabet"))
    candidates = [p.parent for p in download_path.rglob("A") if p.is_dir() and (p.parent / "B").is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No folder with class sub-folders A, B, ... found inside {download_path}")
    train_dir = candidates[0]
else:
    train_dir = Path(args.raw_dir)
print("Training images folder:", train_dir)

# 2. Preprocess
stats = run_preprocessing(train_dir, "data/processed", img_size=args.img_size, max_per_class=args.max_per_class)

# 3. Report
print("Total images kept:", stats["total_images"])
print("Split sizes:", stats["split_sizes"])
print("Removed:", stats["removed"])
print(f"Train pixel mean/std: {stats['train_pixel_mean']:.4f} / {stats['train_pixel_std']:.4f}")
