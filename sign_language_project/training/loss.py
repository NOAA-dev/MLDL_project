import numpy as np


def _validate_probabilities_and_labels(y_pred, y_true):
    probabilities = np.asarray(y_pred, dtype=np.float64)
    labels = np.asarray(y_true)
    if probabilities.ndim != 2:
        raise ValueError("y_pred must have shape (N, C).")
    n_samples, n_classes = probabilities.shape
    if n_samples == 0 or n_classes == 0:
        raise ValueError("y_pred cannot be empty.")
    if labels.shape != (n_samples,):
        raise ValueError("y_true must have shape (N,).")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("y_true must contain integer class IDs.")
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("y_true contains an invalid class ID.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("y_pred must contain finite values.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Probabilities must be between 0 and 1.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("Each row of y_pred must sum to 1.")
    return probabilities, labels


def cross_entropy_loss(y_pred, y_true, epsilon=1e-12):
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be between 0 and 1.")
    probabilities, labels = _validate_probabilities_and_labels(y_pred, y_true)
    chosen = np.clip(probabilities[np.arange(len(labels)), labels], epsilon, 1.0)
    return float(-np.mean(np.log(chosen)))


def cross_entropy_gradient(y_pred, y_true):
    probabilities, labels = _validate_probabilities_and_labels(y_pred, y_true)
    grad = probabilities.copy()
    grad[np.arange(len(labels)), labels] -= 1.0
    grad /= len(labels)
    return grad
