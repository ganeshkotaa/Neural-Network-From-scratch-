"""Core neural network layers implemented from scratch using NumPy.

All tensors use NCHW format for images: (batch, channels, height, width).
"""

from __future__ import annotations

import numpy as np


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def im2col(x, kernel_h, kernel_w, stride=1, padding=0):
    """Convert a batch of images to column format for efficient convolution.

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: Stride
        padding: Zero padding

    Returns:
        col: 2D array of shape (N*out_h*out_w, C*kernel_h*kernel_w)
        out_h: Output height
        out_w: Output width
    """
    N, C, H, W = x.shape
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1
    assert out_h > 0 and out_w > 0, "Invalid output size in im2col"

    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )

    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=x.dtype)

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x_i in range(kernel_w):
            x_max = x_i + stride * out_w
            col[:, :, y, x_i, :, :] = x_padded[:, :, y:y_max:stride, x_i:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col, out_h, out_w


def col2im(col, x_shape, kernel_h, kernel_w, stride=1, padding=0):
    """Reconstruct images from column format (inverse of im2col).

    Args:
        col: 2D array of shape (N*out_h*out_w, C*kernel_h*kernel_w)
        x_shape: Original input shape (N, C, H, W)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: Stride
        padding: Zero padding

    Returns:
        x: Reconstructed array of shape (N, C, H, W)
    """
    N, C, H, W = x_shape
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=col.dtype)

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x_i in range(kernel_w):
            x_max = x_i + stride * out_w
            x_padded[:, :, y:y_max:stride, x_i:x_max:stride] += col[:, :, y, x_i, :, :]

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Conv2D:
    """2D convolution layer with forward and backward pass.

    Uses im2col for efficient convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weight_scale=None):
        kernel_h, kernel_w = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_h * kernel_w
        scale = weight_scale if weight_scale is not None else np.sqrt(2.0 / fan_in)

        self.W = scale * np.random.randn(out_channels, in_channels, kernel_h, kernel_w)
        self.b = np.zeros(out_channels, dtype=np.float32)

        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input array of shape (N, C, H, W)

        Returns:
            out: Output array of shape (N, out_channels, out_h, out_w)
        """
        assert x.ndim == 4, "Conv2D expects input shape (N, C, H, W)"
        N, C, H, W = x.shape
        assert C == self.in_channels, "Input channels mismatch"

        col, out_h, out_w = im2col(x, self.kernel_h, self.kernel_w, self.stride, self.padding)
        col_W = self.W.reshape(self.out_channels, -1).T  # (C*KH*KW, out_channels)

        out = col @ col_W + self.b
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        """Backward pass.

        Args:
            dout: Upstream gradient of shape (N, out_channels, out_h, out_w)

        Returns:
            dx: Gradient w.r.t input x, shape (N, C, H, W)
        """
        assert dout.ndim == 4, "Conv2D expects dout shape (N, C, H, W)"
        N, _, out_h, out_w = dout.shape

        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        self.db = np.sum(dout_reshaped, axis=0)
        self.dW = (self.col.T @ dout_reshaped).transpose(1, 0).reshape(self.W.shape)

        dcol = dout_reshaped @ self.col_W.T
        dx = col2im(dcol, self.x.shape, self.kernel_h, self.kernel_w, self.stride, self.padding)
        return dx


class ReLU:
    """ReLU activation."""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2D:
    """Max pooling layer."""

    def __init__(self, pool_size=2, stride=2):
        pool_h, pool_w = _pair(pool_size)
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride

        self.x_shape = None
        self.arg_max = None
        self.col = None
        self.out_h = None
        self.out_w = None

    def forward(self, x):
        assert x.ndim == 4, "MaxPool2D expects input shape (N, C, H, W)"
        N, C, H, W = x.shape
        out_h = (H - self.pool_h) // self.stride + 1
        out_w = (W - self.pool_w) // self.stride + 1
        assert out_h > 0 and out_w > 0, "Invalid output size in MaxPool2D"

        x_reshaped = x.reshape(N * C, 1, H, W)
        col, out_h, out_w = im2col(x_reshaped, self.pool_h, self.pool_w, self.stride, 0)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, C, out_h, out_w)

        self.x_shape = x.shape
        self.arg_max = arg_max
        self.col = col
        self.out_h = out_h
        self.out_w = out_w
        return out

    def backward(self, dout):
        N, C, out_h, out_w = dout.shape
        dmax = np.zeros_like(self.col)
        dout_flat = dout.reshape(-1)
        dmax[np.arange(self.arg_max.size), self.arg_max] = dout_flat

        dcol = dmax
        dx = col2im(dcol, (N * C, 1, self.x_shape[2], self.x_shape[3]), self.pool_h, self.pool_w, self.stride, 0)
        dx = dx.reshape(self.x_shape)
        return dx


class Flatten:
    """Flatten layer."""

    def __init__(self):
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.x_shape)


class Linear:
    """Fully connected layer."""

    def __init__(self, in_features, out_features, weight_scale=None):
        fan_in = in_features
        scale = weight_scale if weight_scale is not None else np.sqrt(2.0 / fan_in)
        self.W = scale * np.random.randn(in_features, out_features)
        self.b = np.zeros(out_features, dtype=np.float32)

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        assert x.ndim == 2, "Linear expects input shape (N, D)"
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx


class SoftmaxCrossEntropy:
    """Softmax + Cross-Entropy loss for numerical stability."""

    def __init__(self):
        self.probs = None
        self.labels = None

    def forward(self, logits, labels):
        """Compute loss.

        Args:
            logits: (N, K)
            labels: (N,) integer class labels

        Returns:
            loss: scalar
        """
        assert logits.ndim == 2, "Logits should be (N, K)"
        N = logits.shape[0]

        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(logits_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_logprobs = -np.log(probs[np.arange(N), labels] + 1e-12)
        loss = np.mean(correct_logprobs)

        self.probs = probs
        self.labels = labels
        return loss

    def backward(self):
        """Gradient of loss w.r.t logits."""
        N = self.probs.shape[0]
        dlogits = self.probs.copy()
        dlogits[np.arange(N), self.labels] -= 1
        dlogits /= N
        return dlogits
