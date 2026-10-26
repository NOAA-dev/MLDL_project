"""Tests for Person 6 (evaluation and demo). Run from the project root:

    python -m tests.test_evaluation

Needs numpy, Pillow and matplotlib.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cnn.cnn_model import CNN, CNNConfig
from demo.predict_image import predict_image
from evaluation.confusion_matrix import compute_confusion_matrix, normalize_rows, top_confusions
from evaluation.metrics import (accuracy_score, class_report, load_trained_model, measure_inference_time,
                                predict_all)
from evaluation.plots import plot_class_f1, plot_confusion_matrix, plot_history, plot_misclassified
from training.checkpoints import save_checkpoint

NAMES = ["A", "B", "C"]
CLASS_INFO = {"img_size": 16, "num_classes": 3, "class_names": NAMES}


def small_model():
    np.random.seed(0)
    return CNN(CNNConfig(input_shape=(16, 16, 1), num_classes=3, conv_channels=(2, 4)))


def expect_error(error_type, function, *args):
    try:
        function(*args)
    except error_type:
        return
    raise AssertionError(f"{function.__name__} should raise {error_type.__name__}")


def test_confusion_matrix_by_hand():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    cm = compute_confusion_matrix(y_true, y_pred, 3)
    assert np.array_equal(cm, [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    assert np.allclose(normalize_rows(cm).sum(axis=1), 1.0)
    assert accuracy_score(y_true, y_pred) == 4 / 6


def test_report_by_hand():
    cm = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    per_class, macro, weighted = class_report(cm, NAMES)
    assert np.allclose([r["precision"] for r in per_class], [0.5, 2 / 3, 1.0])
    assert np.allclose([r["recall"] for r in per_class], [0.5, 1.0, 0.5])
    assert np.allclose([r["f1"] for r in per_class], [0.5, 0.8, 2 / 3])
    assert [r["support"] for r in per_class] == [2, 2, 2]
    assert np.isclose(macro["precision"], (0.5 + 2 / 3 + 1.0) / 3)
    assert np.isclose(weighted["f1"], macro["f1"])      # equal supports


def test_report_matches_scikit_learn_if_installed():
    try:
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError:
        return
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 200)
    y_pred = np.where(rng.random(200) < 0.7, y_true, rng.integers(0, 5, 200))
    cm = compute_confusion_matrix(y_true, y_pred, 5)
    per_class, macro, weighted = class_report(cm, list("ABCDE"))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(5), zero_division=0)
    assert np.allclose([x["precision"] for x in per_class], p)
    assert np.allclose([x["recall"] for x in per_class], r)
    assert np.allclose([x["f1"] for x in per_class], f)
    assert np.isclose(macro["f1"], f.mean()) and np.isclose(weighted["f1"], (f * s / s.sum()).sum())


def test_class_never_predicted_or_missing_gives_zero_not_nan():
    cm = np.array([[3, 0, 0], [2, 0, 0], [0, 0, 0]])
    per_class, macro, weighted = class_report(cm, NAMES)
    assert all(np.isfinite([r["precision"], r["recall"], r["f1"]]).all() for r in per_class)
    assert per_class[1]["precision"] == 0 and per_class[2]["recall"] == 0
    assert np.isfinite(list(macro.values())).all() and np.isfinite(list(weighted.values())).all()


def test_top_confusions_order():
    cm = np.array([[5, 3, 0], [1, 6, 4], [0, 2, 8]])
    rows = top_confusions(cm, NAMES, top_n=3)
    assert [(r["true"], r["predicted"], r["count"]) for r in rows] == [("B", "C", 4), ("A", "B", 3), ("C", "B", 2)]
    assert np.isclose(rows[0]["share_of_true_class"], 4 / 11)
    assert len(top_confusions(np.eye(3, dtype=int) * 4, NAMES)) == 0


def test_invalid_inputs_raise():
    expect_error(ValueError, compute_confusion_matrix, np.array([0, 3]), np.array([0, 1]), 3)
    expect_error(ValueError, compute_confusion_matrix, np.array([0, 1]), np.array([0]), 3)
    expect_error(ValueError, accuracy_score, np.array([0, 1]), np.array([0]))


def test_predict_all_matches_model():
    model = small_model()
    X = np.random.default_rng(1).random((10, 1, 16, 16)).astype(np.float32)
    pred, probs = predict_all(model, X, batch_size=4)
    assert pred.shape == (10,) and probs.shape == (10, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.array_equal(pred, model.predict(X.transpose(0, 2, 3, 1)))


def test_checkpoint_round_trip_and_architecture_check():
    np.random.seed(1)
    model = CNN(CNNConfig(input_shape=(16, 16, 1), num_classes=3))      # same default layers as run_training.py
    X = np.random.default_rng(2).random((4, 1, 16, 16)).astype(np.float32)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "m.npz"
        save_checkpoint(model, path, epoch=1)
        rebuilt = load_trained_model(path, CLASS_INFO)
        assert np.allclose(predict_all(rebuilt, X)[1], predict_all(model, X)[1])
        expect_error(ValueError, load_trained_model, path, {"img_size": 16, "num_classes": 5})


def test_predict_image_top_k():
    model = small_model()
    pixels = np.random.default_rng(3).integers(0, 256, (40, 30, 3), dtype=np.uint8)
    result = predict_image(model, pixels, NAMES, top_k=3)
    assert [name for name, _ in result] and len(result) == 3
    probabilities = [p for _, p in result]
    assert probabilities == sorted(probabilities, reverse=True)
    assert abs(sum(probabilities) - 1.0) < 1e-9 and set(n for n, _ in result) == set(NAMES)
    assert len(predict_image(model, pixels, NAMES, top_k=1)) == 1
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "x.png"
        Image.fromarray(pixels).save(path)
        assert predict_image(model, path, NAMES, top_k=3) == result


def test_inference_time():
    model = small_model()
    X = np.random.default_rng(4).random((8, 1, 16, 16)).astype(np.float32)
    timing = measure_inference_time(model, X, n_single=5, batch_size=8)
    assert timing["single_image_ms"] > 0 and timing["batch_ms_per_image"] > 0 and timing["batch_size"] == 8


def test_plots_write_files():
    cm = np.array([[5, 3, 0], [1, 6, 4], [0, 2, 8]])
    per_class, _, _ = class_report(cm, NAMES)
    history = {"train_loss": [1.0, 0.8], "train_accuracy": [0.4, 0.6], "val_loss": [1.1, 0.9], "val_accuracy": [0.3, 0.5]}
    X = np.random.default_rng(5).random((6, 1, 16, 16))
    y_true, y_pred = np.array([0, 1, 2, 0, 1, 2]), np.array([0, 2, 2, 1, 1, 0])
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        plot_history(history, folder / "h.png")
        plot_history({"train_loss": [1.0], "train_accuracy": [0.5]}, folder / "h2.png")
        plot_confusion_matrix(cm, NAMES, folder / "cm.png")
        plot_class_f1(per_class, folder / "f1.png")
        assert plot_misclassified(X, y_true, y_pred, NAMES, folder / "mis.png") is True
        assert plot_misclassified(X, y_true, y_true, NAMES, folder / "none.png") is False
        for name in ("h.png", "h2.png", "cm.png", "f1.png", "mis.png"):
            assert (folder / name).stat().st_size > 1000, name
        assert not (folder / "none.png").exists()


if __name__ == "__main__":
    tests = [value for name, value in globals().copy().items() if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
        except AssertionError as err:
            failed += 1
            print("FAIL", test.__name__, "->", err)
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)