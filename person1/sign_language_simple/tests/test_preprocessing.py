"""
Simple test for Person 1's preprocessing code.
Builds a small fake image dataset so we don't need the real one.

Run:
    python tests/test_preprocessing.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.preprocess_images import preprocess_image, run_preprocessing
from preprocessing.dataset import load_class_names, load_split

IMG_SIZE = 32
CLASSES = ["A", "B", "C"]


def make_fake_images(folder):
    rng = np.random.default_rng(0)
    for name in CLASSES:
        class_folder = folder / name
        class_folder.mkdir(parents=True)
        for i in range(20):
            pixels = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(class_folder / f"{name}{i}.png")


def main():
    tmp = Path(tempfile.mkdtemp())
    raw_dir = tmp / "raw"
    out_dir = tmp / "processed"

    print("Creating fake dataset...")
    make_fake_images(raw_dir)

    print("\nTest 1: preprocess_image()")
    img = Image.fromarray(np.zeros((50, 80, 3), dtype=np.uint8))
    result = preprocess_image(img)
    assert result.shape == (IMG_SIZE, IMG_SIZE), "wrong shape!"
    print("PASS - shape is correct:", result.shape)

    print("\nTest 2: run_preprocessing()")
    run_preprocessing(raw_dir, out_dir, max_per_class=100)

    class_names = load_class_names(out_dir)
    assert class_names == CLASSES, "class names don't match!"
    print("PASS - classes match:", class_names)

    print("\nTest 3: check train/val/test files")
    for split in ("train", "val", "test"):
        X, y = load_split(split, out_dir)
        assert X.shape[1:] == (1, IMG_SIZE, IMG_SIZE), f"wrong X shape for {split}"
        assert X.min() >= 0.0 and X.max() <= 1.0, f"pixel values out of range for {split}"
        print(f"PASS - {split}: X{X.shape} y{y.shape}")

    shutil.rmtree(tmp)
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
