"""Max-pooling layer implemented with NumPy only."""
import numpy as np


class MaxPool:
    """2D max pooling for NHWC tensors.

    Args:
        pool_size: integer or (height, width), default 2.
        stride: integer or (height, width); defaults to pool_size.
    Padding is intentionally not implemented; pooling uses valid windows.
    """

    def __init__(self, pool_size=2, stride=None):
        self.pool_size = self._pair(pool_size, "pool_size")
        self.stride = self._pair(stride if stride is not None else pool_size, "stride")
        if min(*self.pool_size, *self.stride) <= 0:
            raise ValueError("pool_size and stride values must be positive.")
        self._input_shape = None
        self._argmax = None

    @staticmethod
    def _pair(value, name):
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(v, int) for v in value):
            return tuple(value)
        raise TypeError(f"{name} must be an integer or a pair of integers.")

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError(f"MaxPool expects NHWC rank-4 input, got shape {x.shape}.")
        n, h, w, c = x.shape
        ph, pw = self.pool_size
        sh, sw = self.stride
        if h < ph or w < pw:
            raise ValueError("Pooling window cannot be larger than input spatial dimensions.")
        out_h = (h - ph) // sh + 1
        out_w = (w - pw) // sw + 1
        out = np.empty((n, out_h, out_w, c), dtype=x.dtype)
        argmax = np.empty((n, out_h, out_w, c), dtype=np.int64)

        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i*sh:i*sh+ph, j*sw:j*sw+pw, :]  # N, ph, pw, C
                flat = patch.reshape(n, ph * pw, c)
                argmax[:, i, j, :] = np.argmax(flat, axis=1)
                out[:, i, j, :] = np.max(flat, axis=1)

        self._input_shape = x.shape
        self._argmax = argmax
        return out

    def backward(self, grad_output):
        if self._input_shape is None or self._argmax is None:
            raise RuntimeError("Call forward() before backward().")
        grad_output = np.asarray(grad_output)
        n, h, w, c = self._input_shape
        ph, pw = self.pool_size
        sh, sw = self.stride
        out_h = (h - ph) // sh + 1
        out_w = (w - pw) // sw + 1
        expected = (n, out_h, out_w, c)
        if grad_output.shape != expected:
            raise ValueError(f"Expected grad_output shape {expected}, got {grad_output.shape}.")
        grad_input = np.zeros(self._input_shape, dtype=grad_output.dtype)
        for i in range(out_h):
            for j in range(out_w):
                for batch in range(n):
                    for channel in range(c):
                        flat_index = self._argmax[batch, i, j, channel]
                        di, dj = divmod(int(flat_index), pw)
                        grad_input[batch, i*sh+di, j*sw+dj, channel] += grad_output[batch, i, j, channel]
        return grad_input
