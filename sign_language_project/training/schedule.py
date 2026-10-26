"""Learning-rate schedules: each is a callable epoch (0-based) -> learning rate."""
import math


def _check(initial_lr):
    if not math.isfinite(initial_lr) or initial_lr <= 0:
        raise ValueError("initial_lr must be a finite positive number.")
    return float(initial_lr)


class StepDecay:
    """lr = initial_lr * drop ** (epoch // every)."""

    def __init__(self, initial_lr, drop=0.5, every=10):
        if not 0 < drop < 1 or every < 1:
            raise ValueError("drop must be in (0, 1) and every must be >= 1.")
        self.initial_lr, self.drop, self.every = _check(initial_lr), drop, int(every)

    def __call__(self, epoch):
        return self.initial_lr * self.drop ** (epoch // self.every)


class ExponentialDecay:
    """lr = initial_lr * gamma ** epoch."""

    def __init__(self, initial_lr, gamma=0.95):
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1].")
        self.initial_lr, self.gamma = _check(initial_lr), gamma

    def __call__(self, epoch):
        return self.initial_lr * self.gamma ** epoch


class CosineDecay:
    """Cosine curve from initial_lr down to min_lr over total_epochs (then stays at min_lr)."""

    def __init__(self, initial_lr, total_epochs, min_lr=None):
        self.initial_lr = _check(initial_lr)
        self.min_lr = self.initial_lr * 0.01 if min_lr is None else _check(min_lr)
        if total_epochs < 1 or self.min_lr > self.initial_lr:
            raise ValueError("total_epochs must be >= 1 and min_lr <= initial_lr.")
        self.total_epochs = int(total_epochs)

    def __call__(self, epoch):
        t = min(epoch, self.total_epochs) / self.total_epochs
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * t))
