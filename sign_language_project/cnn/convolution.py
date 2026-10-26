"""Person 2: 2D convolution layer implemented from scratch with NumPy.

Channels-last layout (NHWC), matching Person 3's layers and Person 4's model:
    input  x   : (N, H, W, C_in)
    weights W  : (k, k, C_in, C_out)
    bias b     : (C_out,)
    output     : (N, H_out, W_out, C_out)
    H_out = (H + 2 * padding - k) // stride + 1   (same formula for the width)
"""

import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None, dtype=np.float64):
        if min(in_channels, out_channels, kernel_size, stride) < 1 or padding < 0:
            raise ValueError("channels, kernel_size and stride must be >= 1 and padding >= 0")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization: std = sqrt(2 / fan_in).
        # seed=None draws from np.random, so np.random.seed() in the model makes it reproducible.
        fan_in = kernel_size * kernel_size * in_channels
        weight_shape = (kernel_size, kernel_size, in_channels, out_channels)
        if seed is None:
            noise = np.random.standard_normal(weight_shape)
        else:
            noise = np.random.default_rng(seed).standard_normal(weight_shape)
        self.W = (noise * np.sqrt(2.0 / fan_in)).astype(dtype)
        self.b = np.zeros(out_channels, dtype=dtype)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.cache = None

    # same names as the mock Conv2D used in Person 4's tests
    @property
    def kernels(self):
        return self.W

    @property
    def bias(self):
        return self.b

    @property
    def dkernels(self):
        return self.dW

    @property
    def dbias(self):
        return self.db

    def parameters(self):
        """[(W, dW), (b, db)]: live arrays, so an in-place update (p -= lr * g) changes the layer."""
        return [(self.W, self.dW), (self.b, self.db)]

    def output_size(self, height, width):
        out_h = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        if out_h < 1 or out_w < 1:
            raise ValueError(f"Kernel {self.kernel_size} does not fit in input {height}x{width} with padding {self.padding}")
        return out_h, out_w

    def forward(self, x):
        if x.ndim != 4 or x.shape[3] != self.in_channels:
            raise ValueError(f"Expected input (N, H, W, {self.in_channels}), got {x.shape}")

        x = x.astype(self.W.dtype, copy=False)
        n, height, width, c = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        out_h, out_w = self.output_size(height, width)

        x_pad = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))

        # im2col: every k x k x C patch becomes one row
        cols = np.zeros((n, out_h, out_w, k, k, c), dtype=x.dtype)
        for i in range(k):
            for j in range(k):
                cols[:, :, :, i, j, :] = x_pad[:, i:i + s * out_h:s, j:j + s * out_w:s, :]
        cols_2d = cols.reshape(n * out_h * out_w, k * k * c)

        # convolution = one matrix multiply: (N*H_out*W_out, k*k*C) @ (k*k*C, C_out)
        w_2d = self.W.reshape(k * k * c, self.out_channels)
        out = cols_2d @ w_2d + self.b
        out = out.reshape(n, out_h, out_w, self.out_channels)

        self.cache = (x.shape, cols_2d)
        return out

    def backward(self, grad_out):
        if self.cache is None:
            raise RuntimeError("backward() called before forward()")
        x_shape, cols_2d = self.cache
        n, height, width, c = x_shape
        k, s, p = self.kernel_size, self.stride, self.padding
        out_h, out_w = self.output_size(height, width)

        if grad_out.shape != (n, out_h, out_w, self.out_channels):
            raise ValueError(f"Expected grad_out {(n, out_h, out_w, self.out_channels)}, got {grad_out.shape}")

        grad_out = grad_out.astype(self.W.dtype, copy=False)
        grad_2d = grad_out.reshape(n * out_h * out_w, self.out_channels)

        # gradients are written into the existing arrays, so references from parameters() stay valid;
        # each call overwrites the previous gradients (no accumulation)
        self.db[...] = grad_2d.sum(axis=0)
        self.dW[...] = (cols_2d.T @ grad_2d).reshape(self.W.shape)

        # gradient of the input: send each row's gradient back onto its patch
        w_2d = self.W.reshape(k * k * c, self.out_channels)
        dcols = (grad_2d @ w_2d.T).reshape(n, out_h, out_w, k, k, c)

        dx_pad = np.zeros((n, height + 2 * p, width + 2 * p, c), dtype=grad_out.dtype)
        for i in range(k):
            for j in range(k):
                dx_pad[:, i:i + s * out_h:s, j:j + s * out_w:s, :] += dcols[:, :, :, i, j, :]

        if p == 0:
            return dx_pad
        return dx_pad[:, p:-p, p:-p, :]