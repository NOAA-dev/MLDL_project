"""
Download the ASL Alphabet dataset and preprocess it.

Run:
    python prepare_data.py
"""

from pathlib import Path
from preprocessing.preprocess_images import run_preprocessing

import kagglehub

# Step 1: download the dataset
download_path = Path(kagglehub.dataset_download("grassknoted/asl-alphabet"))

# Step 2: find the folder that has class subfolders (A, B, C, ...)
raw_dir = None
for folder in download_path.rglob("A"):
    if folder.is_dir():
        raw_dir = folder.parent
        break

print("Using raw images from:", raw_dir)

# Step 3: preprocess
run_preprocessing(raw_dir, out_dir="data/processed", max_per_class=500)
