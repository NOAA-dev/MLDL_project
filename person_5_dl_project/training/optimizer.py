"""Plain stochastic gradient descent for live (parameter, gradient) pairs."""
import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number.")
        self.learning_rate = float(learning_rate)

    def step(self, model):
        """Update parameters in-place using model.parameters() -> [(param, grad)]."""
        if not hasattr(model, "parameters"):
            raise TypeError("model must provide parameters().")
        pairs = model.parameters()
        if not pairs:
            raise ValueError("model.parameters() returned no trainable parameters.")
        for index, pair in enumerate(pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError(f"Parameter entry {index} must be (parameter, gradient).")
            parameter, gradient = pair
            if not isinstance(parameter, np.ndarray) or not isinstance(gradient, np.ndarray):
                raise TypeError("Parameters and gradients must be NumPy arrays.")
            if parameter.shape != gradient.shape:
                raise ValueError(f"Parameter/gradient shape mismatch: {parameter.shape} vs {gradient.shape}.")
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError(f"Non-finite gradient in parameter entry {index}.")
        # Validate all pairs before changing any parameter.
        for parameter, gradient in pairs:
            parameter -= self.learning_rate * gradient
