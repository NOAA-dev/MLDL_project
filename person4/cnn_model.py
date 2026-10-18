"""Person 4 - CNN architecture and forward pass.

Tensor convention (team-wide assumption, NCHW):
    images : (N, C, H, W) float32, values normalised by Person 1
    logits : (N, num_classes)
    probs  : (N, num_classes), rows sum to 1

Default architecture (configurable via CNNConfig):
    Input (N,1,64,64)
    -> Conv(1->8, k3, s1, p1)  -> ReLU -> MaxPool(2,2)   (N,8,32,32)
    -> Conv(8->16, k3, s1, p1) -> ReLU -> MaxPool(2,2)   (N,16,16,16)
    -> Flatten                                            (N,4096)
    -> Dense(4096->num_classes) -> Softmax                (N,num_classes)

Layer interfaces ASSUMED from Persons 2/3:
    Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0)
    ReLU(), MaxPool(pool_size=2, stride=2), Flatten(), Dense(in_features, out_features),
    Softmax(); each has .forward(x).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class CNNConfig:
    input_shape: tuple = (1, 64, 64)      # (C, H, W)
    num_classes: int = 26
    conv_channels: tuple = (8, 16)        # one conv block per entry
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    pool_size: int = 2
    pool_stride: int = 2
    seed: int = 0


def conv_out(size, k, s, p):
    return (size + 2 * p - k) // s + 1


def pool_out(size, k, s):
    return (size - k) // s + 1


def compute_shapes(cfg):
    """Per-sample shape after every layer: list of (layer_name, shape). Pure arithmetic."""
    c, h, w = cfg.input_shape
    shapes = [("input", (c, h, w))]
    for i, oc in enumerate(cfg.conv_channels, 1):
        h = conv_out(h, cfg.kernel_size, cfg.stride, cfg.padding)
        w = conv_out(w, cfg.kernel_size, cfg.stride, cfg.padding)
        c = oc
        shapes.append((f"conv{i}", (c, h, w)))
        shapes.append((f"relu{i}", (c, h, w)))
        h, w = pool_out(h, cfg.pool_size, cfg.pool_stride), pool_out(w, cfg.pool_size, cfg.pool_stride)
        if h < 1 or w < 1:
            raise ValueError(f"Spatial size collapsed to {h}x{w} at block {i}")
        shapes.append((f"pool{i}", (c, h, w)))
    shapes.append(("flatten", (c * h * w,)))
    shapes.append(("dense", (cfg.num_classes,)))
    shapes.append(("softmax", (cfg.num_classes,)))
    return shapes


class CNN:
    def __init__(self, cfg=None, layer_lib=None):
        """layer_lib: optional namespace with Conv2D, ReLU, MaxPool, Flatten, Dense, Softmax
        (used to inject test mocks). Default: import Person 2/3 modules."""
        self.cfg = cfg or CNNConfig()
        if layer_lib is None:
            from cnn.convolution import Conv2D
            from cnn.activations import ReLU
            from cnn.pooling import MaxPool
            from cnn.dense import Dense, Flatten  # Flatten module location: confirm with Person 3
            from cnn.softmax import Softmax
        else:
            Conv2D, ReLU, MaxPool = layer_lib.Conv2D, layer_lib.ReLU, layer_lib.MaxPool
            Dense, Flatten, Softmax = layer_lib.Dense, layer_lib.Flatten, layer_lib.Softmax
        cfg = self.cfg
        np.random.seed(cfg.seed)  # deterministic init for layers drawing from np.random
        self.shapes = compute_shapes(cfg)
        self.layers, self.names = [], []
        c = cfg.input_shape[0]
        for i, oc in enumerate(cfg.conv_channels, 1):
            self._add(f"conv{i}", Conv2D(c, oc, cfg.kernel_size, cfg.stride, cfg.padding))
            self._add(f"relu{i}", ReLU())
            self._add(f"pool{i}", MaxPool(cfg.pool_size, cfg.pool_stride))
            c = oc
        self._add("flatten", Flatten())
        self._add("dense", Dense(self.shapes[-3][1][0], cfg.num_classes))
        self.softmax = Softmax()  # separate: Person 5 may fuse softmax + cross-entropy

    def _add(self, name, layer):
        self.names.append(name)
        self.layers.append(layer)

    def _check_input(self, X):
        if X.ndim != 4 or tuple(X.shape[1:]) != tuple(self.cfg.input_shape):
            raise ValueError(f"Expected X of shape (N,{self.cfg.input_shape}), got {X.shape}")

    def forward_logits(self, X, verbose=False):
        """X (N,C,H,W) -> logits (N,num_classes). Verifies every intermediate shape."""
        self._check_input(X)
        out = X.astype(np.float32, copy=False)
        n = X.shape[0]
        for name, layer, (_, sshape) in zip(self.names, self.layers, self.shapes[1:]):
            out = layer.forward(out)
            if out.shape != (n,) + tuple(sshape):
                raise RuntimeError(f"{name}: got {out.shape}, expected {(n,) + tuple(sshape)}")
            if verbose:
                print(f"{name:8s} {out.shape}")
        return out

    def forward(self, X, verbose=False):
        """X (N,C,H,W) -> class probabilities (N,num_classes)."""
        probs = self.softmax.forward(self.forward_logits(X, verbose))
        if verbose:
            print(f"{'softmax':8s} {probs.shape}")
        return probs

    __call__ = forward

    def predict(self, X):
        """Class IDs (N,). Also accepts one image (C,H,W)."""
        if X.ndim == 3:
            X = X[None]
        return np.argmax(self.forward(X), axis=1)

    def predict_single(self, img):
        """img (C,H,W) -> (class_id:int, probs:(num_classes,))."""
        p = self.forward(img[None])[0]
        return int(np.argmax(p)), p

    def summary(self):
        for name, shape in self.shapes:
            print(f"{name:8s} {shape}")
