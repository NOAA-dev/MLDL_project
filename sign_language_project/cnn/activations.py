import numpy as np


class ReLU:

    def __init__(self):
        self._input = None

    def forward(self, x):
        x = np.asarray(x)
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError("ReLU input must be numeric.")
        self._input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        if self._input is None:
            raise RuntimeError("Call forward() before backward().")
        grad_output = np.asarray(grad_output)
        if grad_output.shape != self._input.shape:
            raise ValueError(
                f"grad_output shape {grad_output.shape} does not match "
                f"input shape {self._input.shape}."
            )
        return grad_output * (self._input > 0)


class Flatten:

    def __init__(self):
        self._input_shape = None

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim < 2:
            raise ValueError("Flatten expects a batch dimension and at least one feature dimension.")
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        if self._input_shape is None:
            raise RuntimeError("Call forward() before backward().")
        grad_output = np.asarray(grad_output)
        expected = (self._input_shape[0], int(np.prod(self._input_shape[1:])))
        if grad_output.shape != expected:
            raise ValueError(f"Expected gradient shape {expected}, got {grad_output.shape}.")
        return grad_output.reshape(self._input_shape)
