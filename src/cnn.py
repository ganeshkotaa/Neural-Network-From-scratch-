"""CNN architecture built from the custom NumPy layers."""

from __future__ import annotations

import numpy as np

from layers import Conv2D, ReLU, MaxPool2D, Flatten, Linear


class CNN:
    """Simple CNN: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> ReLU -> FC."""

    def __init__(self):
        self.conv1 = Conv2D(1, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)

        self.conv2 = Conv2D(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)

        self.flatten = Flatten()
        self.fc1 = Linear(16 * 7 * 7, 128)
        self.relu3 = ReLU()
        self.fc2 = Linear(128, 10)

        self.layers = [
            self.conv1,
            self.relu1,
            self.pool1,
            self.conv2,
            self.relu2,
            self.pool2,
            self.flatten,
            self.fc1,
            self.relu3,
            self.fc2,
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params(self):
        """Return list of (name, param, grad) tuples for optimization."""
        params = []
        params.append(("conv1.W", self.conv1.W, self.conv1.dW))
        params.append(("conv1.b", self.conv1.b, self.conv1.db))
        params.append(("conv2.W", self.conv2.W, self.conv2.dW))
        params.append(("conv2.b", self.conv2.b, self.conv2.db))
        params.append(("fc1.W", self.fc1.W, self.fc1.dW))
        params.append(("fc1.b", self.fc1.b, self.fc1.db))
        params.append(("fc2.W", self.fc2.W, self.fc2.dW))
        params.append(("fc2.b", self.fc2.b, self.fc2.db))
        return params

    def count_params(self):
        total = 0
        for _, param, _ in self.params():
            total += param.size
        return total

    def summary(self, input_shape=(1, 1, 28, 28)):
        """Print layer-by-layer output shapes for a given input shape."""
        x = np.zeros(input_shape, dtype=np.float32)
        print("Input:", x.shape)
        for layer in self.layers:
            x = layer.forward(x)
            print(f"{layer.__class__.__name__}: {x.shape}")
        print("Total params:", self.count_params())

    def get_state(self):
        """Return a serializable state dict of parameters."""
        return {
            "conv1.W": self.conv1.W,
            "conv1.b": self.conv1.b,
            "conv2.W": self.conv2.W,
            "conv2.b": self.conv2.b,
            "fc1.W": self.fc1.W,
            "fc1.b": self.fc1.b,
            "fc2.W": self.fc2.W,
            "fc2.b": self.fc2.b,
        }

    def set_state(self, state):
        """Load parameters from a state dict."""
        self.conv1.W = state["conv1.W"]
        self.conv1.b = state["conv1.b"]
        self.conv2.W = state["conv2.W"]
        self.conv2.b = state["conv2.b"]
        self.fc1.W = state["fc1.W"]
        self.fc1.b = state["fc1.b"]
        self.fc2.W = state["fc2.W"]
        self.fc2.b = state["fc2.b"]
