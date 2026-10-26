"""Numerically stable softmax for class logits."""
import numpy as np


class Softmax:
    """Softmax layer. This layer exposes probabilities but no standalone backward.

    For cross-entropy training, the recommended output-logit gradient is
    (probabilities - one_hot_targets) / batch_size and is then sent directly
    into the preceding Dense.backward(). This combines softmax and cross-entropy
    derivatives and avoids a redundant Jacobian calculation.
    """

    def forward(self, logits):
        logits = np.asarray(logits)
        if logits.ndim != 2:
            raise ValueError(f"Softmax expects rank-2 logits (N, classes), got {logits.shape}.")
        if logits.shape[1] == 0:
            raise ValueError("Softmax requires at least one class.")
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        denom = np.sum(exp_values, axis=1, keepdims=True)
        return exp_values / denom
