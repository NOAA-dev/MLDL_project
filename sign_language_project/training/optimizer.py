import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number.")
        self.learning_rate = float(learning_rate)

    def step(self, model):
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

class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.velocities = None

    def step(self, model):
        pairs = model.parameters()
        if not pairs:
            raise ValueError("model.parameters() returned no trainable parameters.")
        for index, (parameter, gradient) in enumerate(pairs):
            if parameter.shape != gradient.shape:
                raise ValueError(f"Parameter/gradient shape mismatch: {parameter.shape} vs {gradient.shape}.")
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError(f"Non-finite gradient in parameter entry {index}.")
        if self.velocities is None:
            self.velocities = [np.zeros_like(parameter) for parameter, _ in pairs]
        if len(self.velocities) != len(pairs):
            raise ValueError("The number of parameters changed since the first step.")
        for velocity, (parameter, gradient) in zip(self.velocities, pairs):
            velocity *= self.momentum
            velocity -= self.learning_rate * gradient
            parameter += velocity            
