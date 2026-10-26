"""Connect gradients through the existing Person 3/4 layer objects."""
import numpy as np
from .loss import cross_entropy_gradient


def backward(model, y_pred, y_true):
    """Run the model's cached layer backward methods in reverse order.

    Call model.forward(X) immediately before this function for the same batch.
    CNN.layers must be ordered input-to-output and include the final Dense layer;
    the softmax is intentionally separate and is combined with cross-entropy here.
    Returns the gradient with respect to the model input.
    """
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
