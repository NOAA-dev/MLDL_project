import numpy as np
from .loss import cross_entropy_gradient


def backward(model, y_pred, y_true):
    grad = cross_entropy_gradient(y_pred, y_true)
    if not hasattr(model, "layers"):
        raise TypeError("model must expose an ordered .layers list.")
    for layer in reversed(model.layers):
        backward_fn = getattr(layer, "backward", None)
        if not callable(backward_fn):
            raise TypeError(f"Layer {type(layer).__name__} does not implement backward().")
        grad = backward_fn(grad)
        grad = np.asarray(grad)
    return grad
