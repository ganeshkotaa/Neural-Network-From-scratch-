"""MNIST downloader and loader (IDX format)."""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np

MNIST_URLS = [
    "https://yann.lecun.com/exdb/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_if_missing(data_dir, filename):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        last_err = None
        for base_url in MNIST_URLS:
            url = base_url + filename
            try:
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, path)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
        if last_err is not None:
            raise last_err
    return path


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid IDX image file")
        data = f.read()
        images = np.frombuffer(data, dtype=np.uint8).reshape(num, rows, cols)
    return images


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid IDX label file")
        data = f.read()
        labels = np.frombuffer(data, dtype=np.uint8)
    return labels


def load_mnist(data_dir="data", normalize=True, one_channel=True):
    paths = {k: _download_if_missing(data_dir, v) for k, v in FILES.items()}

    x_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    x_test = _read_idx_images(paths["test_images"])
    y_test = _read_idx_labels(paths["test_labels"])

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    if one_channel:
        x_train = x_train[:, None, :, :]
        x_test = x_test[:, None, :, :]

    return x_train, y_train, x_test, y_test


def train_val_split(x, y, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    val_size = int(len(x) * val_ratio)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]
