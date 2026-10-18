"""TEST-ONLY mock layers (NOT production). Forward-only NumPy, same signatures as assumed Person 2/3 layers."""
import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.k, self.s, self.p = kernel_size, stride, padding
        self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * 0.1
        self.bias = np.zeros(out_channels, dtype=np.float32)
    def forward(self, x):
        n, c, h, w = x.shape
        xp = np.pad(x, ((0,0),(0,0),(self.p,self.p),(self.p,self.p)))
        oh, ow = (h+2*self.p-self.k)//self.s+1, (w+2*self.p-self.k)//self.s+1
        out = np.zeros((n, self.kernels.shape[0], oh, ow), dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                patch = xp[:, :, i*self.s:i*self.s+self.k, j*self.s:j*self.s+self.k]
                out[:, :, i, j] = np.tensordot(patch, self.kernels, axes=([1,2,3],[1,2,3])) + self.bias
        return out

class ReLU:
    def forward(self, x): return np.maximum(0, x)

class MaxPool:
    def __init__(self, pool_size=2, stride=2): self.k, self.s = pool_size, stride
    def forward(self, x):
        n, c, h, w = x.shape
        oh, ow = (h-self.k)//self.s+1, (w-self.k)//self.s+1
        out = np.zeros((n, c, oh, ow), dtype=x.dtype)
        for i in range(oh):
            for j in range(ow):
                out[:, :, i, j] = x[:, :, i*self.s:i*self.s+self.k, j*self.s:j*self.s+self.k].max(axis=(2,3))
        return out

class Flatten:
    def forward(self, x): return x.reshape(x.shape[0], -1)

class Dense:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * 0.01
        self.b = np.zeros(out_features, dtype=np.float32)
    def forward(self, x): return x @ self.W + self.b

class Softmax:
    def forward(self, z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
