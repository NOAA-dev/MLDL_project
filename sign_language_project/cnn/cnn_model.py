"""Person 4 - CNN architecture and forward pass.

Tensor convention: channels-last NHWC (set by Person 3's layers, which only accept NHWC):
    images : (N, H, W, C), cast to float64 here; values normalised by Person 1
    logits : (N, num_classes)
    probs  : (N, num_classes), rows sum to 1

Default architecture (configurable via CNNConfig):
    Input (N,64,64,1)
    -> Conv(1->8, k3, s1, p1)  -> ReLU -> MaxPool(2,2)   (N,32,32,8)
    -> Conv(8->16, k3, s1, p1) -> ReLU -> MaxPool(2,2)   (N,16,16,16)
    -> Flatten                                            (N,4096)
    -> Dense(4096->num_classes) -> Softmax                (N,num_classes)

Person 3's layers (verified from the repo): ReLU/Flatten in activations.py, MaxPool in
pooling.py, Dense in dense.py, Softmax in softmax.py (forward only).
Conv2D (Person 2) is STILL ASSUMED: Conv2D(in_channels, out_channels, kernel_size,
stride, padding) with NHWC input/output.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class CNNConfig:
    input_shape: tuple = (64, 64, 1)      # (H, W, C)
    num_classes: int = 26
    conv_channels: tuple = (8, 16)        # one conv block per entry
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    pool_size: int = 2
    pool_stride: int = 2
    seed: int = 0
    dtype: type = np.float64              # Person 3 layers default to float64


def conv_out(size, k, s, p):
    return (size + 2 * p - k) // s + 1


def pool_out(size, k, s):
    return (size - k) // s + 1


def compute_shapes(cfg):
    """Per-sample (no batch dim) shape after every layer, NHWC. Pure arithmetic."""
    h, w, c = cfg.input_shape
    shapes = [("input", (h, w, c))]
    for i, oc in enumerate(cfg.conv_channels, 1):
        h = conv_out(h, cfg.kernel_size, cfg.stride, cfg.padding)
        w = conv_out(w, cfg.kernel_size, cfg.stride, cfg.padding)
        c = oc
        shapes.append((f"conv{i}", (h, w, c)))
        shapes.append((f"relu{i}", (h, w, c)))
        h, w = pool_out(h, cfg.pool_size, cfg.pool_stride), pool_out(w, cfg.pool_size, cfg.pool_stride)
        if h < 1 or w < 1:
            raise ValueError(f"Spatial size collapsed to {h}x{w} at block {i}")
        shapes.append((f"pool{i}", (h, w, c)))
    shapes.append(("flatten", (h * w * c,)))
    shapes.append(("dense", (cfg.num_classes,)))
    shapes.append(("softmax", (cfg.num_classes,)))
    return shapes


class CNN:
    def __init__(self, cfg=None, layer_lib=None):
        """layer_lib: optional object whose attributes override layer classes
        (tests use it to inject a mock Conv2D until Person 2 delivers)."""
        from cnn.activations import ReLU, Flatten
        from cnn.pooling import MaxPool
        from cnn.dense import Dense
        from cnn.softmax import Softmax
        Conv2D = getattr(layer_lib, "Conv2D", None)
        if Conv2D is None:
            from cnn.convolution import Conv2D  # Person 2
        ReLU = getattr(layer_lib, "ReLU", ReLU)
        Flatten = getattr(layer_lib, "Flatten", Flatten)
        MaxPool = getattr(layer_lib, "MaxPool", MaxPool)
        Dense = getattr(layer_lib, "Dense", Dense)
        Softmax = getattr(layer_lib, "Softmax", Softmax)
        self.cfg = cfg or CNNConfig()
        cfg = self.cfg
        np.random.seed(cfg.seed)  # deterministic init for layers drawing from np.random
        self.shapes = compute_shapes(cfg)
        self.layers, self.names = [], []
        c = cfg.input_shape[2]
        for i, oc in enumerate(cfg.conv_channels, 1):
            self._add(f"conv{i}", Conv2D(c, oc, cfg.kernel_size, cfg.stride, cfg.padding))
            self._add(f"relu{i}", ReLU())
            self._add(f"pool{i}", MaxPool(cfg.pool_size, cfg.pool_stride))
            c = oc
        self._add("flatten", Flatten())
        self._add("dense", Dense(self.shapes[-3][1][0], cfg.num_classes, seed=cfg.seed))
        # Softmax kept outside self.layers: Person 3's Softmax has no backward.
        # Person 5 sends (probs - one_hot) / N straight into dense.backward().
        self.softmax = Softmax()

    def _add(self, name, layer):
        self.names.append(name)
        self.layers.append(layer)

    def _check_input(self, X):
        if X.ndim != 4 or tuple(X.shape[1:]) != tuple(self.cfg.input_shape):
            raise ValueError(f"Expected X of shape (N,)+{tuple(self.cfg.input_shape)} (NHWC), got {X.shape}")

    def forward_logits(self, X, verbose=False):
        """X (N,H,W,C) -> logits (N,num_classes). Verifies every intermediate shape."""
        X = np.asarray(X)
        self._check_input(X)
        out = X.astype(self.cfg.dtype, copy=False)
        n = X.shape[0]
        for name, layer, (_, sshape) in zip(self.names, self.layers, self.shapes[1:]):
            out = layer.forward(out)
            if out.shape != (n,) + tuple(sshape):
                raise RuntimeError(f"{name}: got {out.shape}, expected {(n,) + tuple(sshape)}")
            if verbose:
                print(f"{name:8s} {out.shape}")
        return out

    def forward(self, X, verbose=False):
        """X (N,H,W,C) -> class probabilities (N,num_classes)."""
        probs = self.softmax.forward(self.forward_logits(X, verbose))
        if verbose:
            print(f"{'softmax':8s} {probs.shape}")
        return probs

    __call__ = forward

    def predict(self, X):
        """Class IDs (N,). Also accepts one image (H,W,C)."""
        X = np.asarray(X)
        if X.ndim == 3:  # single image
            X = X[None]
        return np.argmax(self.forward(X), axis=1)

    def predict_single(self, img):
        """img (H,W,C) -> (class_id:int, probs:(num_classes,))."""
        p = self.forward(np.asarray(img)[None])[0]
        return int(np.argmax(p)), p

    def parameters(self):
        """All (param, grad) array pairs in layer order, from layers exposing parameters()
        (Person 3's Dense does; Person 2's Conv2D is expected to). Live references, so
        in-place updates (p -= lr * g) change the model."""
        pairs = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                pairs.extend(layer.parameters())
        return pairs

    def summary(self):
        for name, shape in self.shapes:
            print(f"{name:8s} {shape}")
