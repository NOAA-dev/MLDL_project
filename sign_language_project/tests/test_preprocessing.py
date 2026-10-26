"""Tests for Person 1 (preprocessing). Run from the project root:

    python tests/test_preprocessing.py

Uses a small synthetic image dataset created in a temp folder, so the real ASL data is not needed.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.augment import augment_batch, random_affine
from preprocessing.dataset import (batch_generator, check_batch, load_class_info, load_split,
                                   num_batches, one_hot)
from preprocessing.preprocess_images import (preprocess_for_inference, preprocess_image,
                                             run_preprocessing, stratified_split)

IMG_SIZE = 64
CLASSES = ["A", "B", "C", "nothing"]
IMAGES_PER_CLASS = 40


# ---------- synthetic data ----------

def make_raw_dataset(root):
    rng = np.random.default_rng(0)
    for k, name in enumerate(CLASSES):
        folder = root / name
        folder.mkdir(parents=True)
        for i in range(IMAGES_PER_CLASS):
            pixels = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
            pixels[20 * k:20 * k + 60, 40:100] = 255
            Image.fromarray(pixels).save(folder / f"{name}{i}.png")

    shutil.copy(root / "A" / "A0.png", root / "A" / "A0_copy.png")          # exact duplicate
    shutil.copy(root / "B" / "B1.png", root / "B" / "B1_copy.png")          # exact duplicate
    shutil.copy(root / "A" / "A2.png", root / "C" / "A2_in_C.png")          # duplicate across classes
    (root / "B" / "broken.png").write_bytes(b"this is not an image")        # corrupted
    (root / "C" / "truncated.png").write_bytes((root / "C" / "C0.png").read_bytes()[:300])  # truncated
    (root / "A" / "notes.txt").write_text("irrelevant file")                # not an image
    (root / "junk_folder").mkdir()
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(root / "junk_folder" / "x.png")


class FixedRng:
    """Returns scripted values from uniform(); used to test the affine maths exactly."""
    def __init__(self, values):
        self.values = list(values)

    def uniform(self, low, high):
        return self.values.pop(0)


# ---------- tests ----------

def test_preprocess_image_shape_and_crop():
    wide = Image.fromarray(np.random.default_rng(1).integers(0, 256, (100, 300, 3), dtype=np.uint8))
    out = preprocess_image(wide, IMG_SIZE)
    assert out.shape == (IMG_SIZE, IMG_SIZE) and out.dtype == np.uint8
    gray = Image.fromarray(np.full((50, 50), 7, dtype=np.uint8))
    assert (preprocess_image(gray, IMG_SIZE) == 7).all()


def test_pipeline_cleaning_and_split(tmp):
    raw, out = tmp / "raw", tmp / "processed"
    make_raw_dataset(raw)
    stats = run_preprocessing(raw, out, img_size=IMG_SIZE, max_per_class=None, class_names=sorted(CLASSES))

    assert stats["removed"]["corrupted"] == 2, stats["removed"]
    assert stats["removed"]["duplicates"] == 3, stats["removed"]
    assert stats["removed"]["not_an_image"] == 1, stats["removed"]
    assert stats["removed"]["ignored_folders"] == ["junk_folder"]
    assert stats["total_images"] == 4 * IMAGES_PER_CLASS, stats["total_images"]  # only the original images remain
    assert sum(stats["split_sizes"].values()) == stats["total_images"]
    return out


def test_class_mapping(out):
    info = load_class_info(out)
    assert info["class_names"] == sorted(CLASSES)
    assert [info["class_to_id"][c] for c in info["class_names"]] == list(range(len(CLASSES)))
    assert info["num_classes"] == len(CLASSES) and info["img_size"] == IMG_SIZE and info["channels"] == 1


def test_loaded_tensors(out):
    num_classes = len(CLASSES)
    for split in ("train", "val", "test"):
        X, y = load_split(split, out)
        assert X.shape[1:] == (1, IMG_SIZE, IMG_SIZE) and X.dtype == np.float32
        assert y.shape == (X.shape[0],) and y.dtype == np.int64
        check_batch(X, y, IMG_SIZE, num_classes)
        assert set(np.unique(y)) == set(range(num_classes)), f"{split} is missing a class"


def test_splits_do_not_overlap(out):
    seen = {}
    for split in ("train", "val", "test"):
        X, _ = load_split(split, out)
        for image in X:
            key = image.tobytes()
            assert key not in seen, f"image in both {seen[key]} and {split}"
            seen[key] = split


def test_split_is_stratified_and_reproducible():
    y = np.repeat(np.arange(4), 40)
    a = stratified_split(y, seed=1)
    b = stratified_split(y, seed=1)
    c = stratified_split(y, seed=2)
    for name in a:
        assert (a[name] == b[name]).all()
    assert not (a["train"] == c["train"]).all()
    for name, expected in (("train", 28), ("val", 6), ("test", 6)):
        assert (np.bincount(y[a[name]]) == expected).all(), name


def test_split_rejects_tiny_class():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    try:
        stratified_split(y)
    except ValueError as err:
        assert "too few" in str(err)
    else:
        raise AssertionError("expected ValueError for a class with 2 images")


def test_one_hot():
    result = one_hot(np.array([0, 2, 1]), 3)
    assert result.shape == (3, 3) and result.dtype == np.float32
    assert (result == np.eye(3, dtype=np.float32)[[0, 2, 1]]).all()


def test_batch_generator(out):
    X, y = load_split("train", out)
    batches = list(batch_generator(X, y, batch_size=16, shuffle=True, seed=3))
    assert len(batches) == num_batches(len(X), 16)
    assert sum(len(b[1]) for b in batches) == len(X)
    assert batches[-1][0].shape[0] == len(X) % 16 or len(X) % 16 == 0
    for xb, yb in batches:
        check_batch(xb, yb, IMG_SIZE, len(CLASSES))
    seen = np.concatenate([b[0].reshape(len(b[0]), -1) for b in batches])
    assert len({row.tobytes() for row in seen}) == len(X), "an epoch must visit each sample exactly once"

    dropped = list(batch_generator(X, y, batch_size=16, drop_last=True))
    assert all(b[0].shape[0] == 16 for b in dropped) and len(dropped) == len(X) // 16

    again = list(batch_generator(X, y, batch_size=16, shuffle=True, seed=3))
    other = list(batch_generator(X, y, batch_size=16, shuffle=True, seed=4))
    assert (again[0][1] == batches[0][1]).all() and (again[0][0] == batches[0][0]).all()
    assert not (other[0][1] == batches[0][1]).all()

    ordered = list(batch_generator(X, y, batch_size=16, shuffle=False))
    assert (np.concatenate([b[1] for b in ordered]) == y).all()


def test_batch_generator_augmentation(out):
    X, y = load_split("train", out)
    X_copy = X.copy()
    plain = next(batch_generator(X, y, batch_size=8, shuffle=False, augment=False))
    aug = next(batch_generator(X, y, batch_size=8, shuffle=False, augment=True, seed=5))
    aug_again = next(batch_generator(X, y, batch_size=8, shuffle=False, augment=True, seed=5))
    check_batch(aug[0], aug[1], IMG_SIZE, len(CLASSES))
    assert (aug[1] == plain[1]).all(), "augmentation must not change labels"
    assert not np.allclose(aug[0], plain[0]), "augmentation changed nothing"
    assert (aug[0] == aug_again[0]).all(), "augmentation must be deterministic for a fixed seed"
    assert (X == X_copy).all(), "augmentation modified its input"


def test_affine_identity():
    image = np.random.default_rng(2).random((IMG_SIZE, IMG_SIZE)).astype(np.float32)
    out = random_affine(image, FixedRng([0.0, 0.0, 0.0, 1.0]), 10, 0.1, (0.9, 1.1))
    assert np.allclose(out, image, atol=1e-5)


def test_affine_shift_and_fill():
    image = np.random.default_rng(3).random((IMG_SIZE, IMG_SIZE)).astype(np.float32)
    shift = 3 / IMG_SIZE
    out = random_affine(image, FixedRng([0.0, shift, 0.0, 1.0]), 10, 0.1, (0.9, 1.1))
    assert np.allclose(out[:, 3:], image[:, :-3], atol=1e-5), "content should move 3 px to the right"
    assert np.allclose(out[:, :3], image.mean(), atol=1e-5), "uncovered area should hold the mean value"


def test_affine_rotation_90():
    image = np.random.default_rng(4).random((IMG_SIZE, IMG_SIZE)).astype(np.float32)
    out = random_affine(image, FixedRng([90.0, 0.0, 0.0, 1.0]), 90, 0.1, (0.9, 1.1))
    assert np.allclose(out, np.rot90(image, 1), atol=1e-4) or np.allclose(out, np.rot90(image, -1), atol=1e-4)


def test_affine_zoom_out_fills_corners():
    image = np.random.default_rng(5).random((IMG_SIZE, IMG_SIZE)).astype(np.float32)
    out = random_affine(image, FixedRng([0.0, 0.0, 0.0, 0.5]), 10, 0.1, (0.5, 1.5))
    assert np.isclose(out[0, 0], image.mean(), atol=1e-5)


def test_augment_rejects_bad_shape():
    try:
        augment_batch(np.zeros((4, 32, 32), dtype=np.float32), np.random.default_rng(0))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a 3D batch")


def test_inference_preprocessing(tmp):
    path = tmp / "single.png"
    pixels = np.random.default_rng(6).integers(0, 256, (120, 90, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    from_path = preprocess_for_inference(path, IMG_SIZE)
    from_array = preprocess_for_inference(pixels, IMG_SIZE)
    assert from_path.shape == (1, 1, IMG_SIZE, IMG_SIZE) and from_path.dtype == np.float32
    assert from_path.min() >= 0.0 and from_path.max() <= 1.0
    assert np.array_equal(from_path, from_array)


def test_missing_class_folder(tmp):
    from preprocessing.preprocess_images import find_class_names
    try:
        find_class_names(tmp / "raw", ["A", "Z"])
    except ValueError as err:
        assert "Z" in str(err)
        return
    raise AssertionError("expected ValueError for a missing class folder")


# ---------- runner ----------

def main():
    tmp = Path(tempfile.mkdtemp())
    results = []

    def run(test, *args):
        try:
            value = test(*args)
            results.append((test.__name__, "PASS"))
            return value
        except AssertionError as err:
            results.append((test.__name__, f"FAIL: {err}"))

    try:
        run(test_preprocess_image_shape_and_crop)
        out = run(test_pipeline_cleaning_and_split, tmp)
        if out is not None:
            run(test_class_mapping, out)
            run(test_loaded_tensors, out)
            run(test_splits_do_not_overlap, out)
            run(test_batch_generator, out)
            run(test_batch_generator_augmentation, out)
            run(test_missing_class_folder, tmp)
        run(test_split_is_stratified_and_reproducible)
        run(test_split_rejects_tiny_class)
        run(test_one_hot)
        run(test_affine_identity)
        run(test_affine_shift_and_fill)
        run(test_affine_rotation_90)
        run(test_affine_zoom_out_fills_corners)
        run(test_augment_rejects_bad_shape)
        run(test_inference_preprocessing, tmp)
    finally:
        shutil.rmtree(tmp)

    for name, status in results:
        print(f"{status[:4]}  {name}" + ("" if status == "PASS" else f"  -> {status}"))
    failed = [r for r in results if r[1] != "PASS"]
    print(f"\n{len(results) - len(failed)}/{len(results)} tests passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
