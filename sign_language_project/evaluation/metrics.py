"""Person 6: predictions, accuracy, precision / recall / F1 and inference time."""

import time

import numpy as np

from cnn.cnn_model import CNN, CNNConfig
from preprocessing.dataset import to_channels_last
from training.checkpoints import load_checkpoint


def load_trained_model(checkpoint_path, class_info):
    """Rebuild the CNN exactly as run_training.py does, then load the saved weights."""
    img_size = class_info["img_size"]
    config = CNNConfig(input_shape=(img_size, img_size, 1), num_classes=class_info["num_classes"])
    model = CNN(config)
    load_checkpoint(model, checkpoint_path)
    return model


def predict_all(model, X, batch_size=64):
    """X: (N, 1, H, W) from load_split. Returns class IDs (N,) and probabilities (N, num_classes)."""
    X = to_channels_last(X)
    probabilities = []
    for start in range(0, len(X), batch_size):
        probabilities.append(model.forward(X[start:start + batch_size]))
    probabilities = np.concatenate(probabilities)
    return probabilities.argmax(axis=1), probabilities


def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must have the same non-zero length.")
    return float(np.mean(y_true == y_pred))


def class_report(cm, class_names):
    """Per-class precision, recall, F1 and support from a confusion matrix (rows = true, columns = predicted).

    Returns (per_class list of dicts, macro average dict, weighted average dict).
    A class that is never predicted or never present gets 0 instead of a division error.
    """
    true_positive = np.diag(cm).astype(float)
    predicted_total = cm.sum(axis=0).astype(float)
    actual_total = cm.sum(axis=1).astype(float)

    precision = np.divide(true_positive, predicted_total, out=np.zeros_like(true_positive), where=predicted_total > 0)
    recall = np.divide(true_positive, actual_total, out=np.zeros_like(true_positive), where=actual_total > 0)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(true_positive), where=total > 0)

    per_class = []
    for i, name in enumerate(class_names):
        per_class.append({
            "class": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(actual_total[i]),
        })

    weights = actual_total / actual_total.sum()
    macro = {"precision": float(precision.mean()), "recall": float(recall.mean()), "f1": float(f1.mean())}
    weighted = {"precision": float((precision * weights).sum()),
                "recall": float((recall * weights).sum()),
                "f1": float((f1 * weights).sum())}
    return per_class, macro, weighted


def measure_inference_time(model, X, n_single=50, batch_size=64):
    """Average time to predict one image, and the per-image time inside a batch (milliseconds)."""
    X = to_channels_last(X)
    n_single = min(n_single, len(X))
    model.forward(X[:1])

    start = time.perf_counter()
    for i in range(n_single):
        model.forward(X[i:i + 1])
    single_ms = (time.perf_counter() - start) / n_single * 1000

    batch = X[:batch_size]
    start = time.perf_counter()
    model.forward(batch)
    batch_ms_per_image = (time.perf_counter() - start) / len(batch) * 1000

    return {"single_image_ms": single_ms, "batch_size": len(batch), "batch_ms_per_image": batch_ms_per_image}