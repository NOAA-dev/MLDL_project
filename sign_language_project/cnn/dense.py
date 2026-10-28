import numpy as np


class Dense:

    def __init__(self, in_features, out_features, seed=42, weight_scale=None):
        if not isinstance(in_features, int) or not isinstance(out_features, int):
            raise TypeError("in_features and out_features must be integers.")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        rng = np.random.default_rng(seed)
        if weight_scale is None:
    
            limit = np.sqrt(6.0 / (in_features + out_features))
        else:
            if weight_scale <= 0:
                raise ValueError("weight_scale must be positive.")
            limit = float(weight_scale)
        self.W = rng.uniform(-limit, limit, size=(in_features, out_features))
        self.b = np.zeros(out_features, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._input = None

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"Dense expects rank-2 input (N, features), got {x.shape}.")
        if x.shape[1] != self.W.shape[0]:
            raise ValueError(f"Expected {self.W.shape[0]} input features, got {x.shape[1]}.")
        self._input = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        if self._input is None:
            raise RuntimeError("Call forward() before backward().")
        grad_output = np.asarray(grad_output)
        expected = (self._input.shape[0], self.W.shape[1])
        if grad_output.shape != expected:
            raise ValueError(f"Expected grad_output shape {expected}, got {grad_output.shape}.")
        self.dW = self._input.T @ grad_output
        self.db = np.sum(grad_output, axis=0)
        return grad_output @ self.W.T

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]
