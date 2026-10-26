"""TEST-ONLY mock Conv2D (NOT production). NHWC, forward only, with parameters() so the
model's parameter listing can be exercised. Person 3's layers are used for real."""
import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.k, self.s, self.p = kernel_size, stride, padding
        self.kernels = np.random.randn(kernel_size, kernel_size, in_channels, out_channels) * 0.1
        self.bias = np.zeros(out_channels)
        self.dkernels, self.dbias = np.zeros_like(self.kernels), np.zeros_like(self.bias)
    def forward(self, x):
        n, h, w, c = x.shape
        xp = np.pad(x, ((0,0),(self.p,self.p),(self.p,self.p),(0,0)))
        oh, ow = (h+2*self.p-self.k)//self.s+1, (w+2*self.p-self.k)//self.s+1
        out = np.zeros((n, oh, ow, self.kernels.shape[3]))
        for i in range(oh):
            for j in range(ow):
                patch = xp[:, i*self.s:i*self.s+self.k, j*self.s:j*self.s+self.k, :]
                out[:, i, j, :] = np.tensordot(patch, self.kernels, axes=([1,2,3],[0,1,2])) + self.bias
        return out
    def parameters(self):
        return [(self.kernels, self.dkernels), (self.bias, self.dbias)]
