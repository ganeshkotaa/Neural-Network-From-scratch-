"""Visualizations: learned filters and prediction samples."""

from __future__ import annotations

import argparse
import os

import numpy as np

from cnn import CNN
from mnist import load_mnist


def save_filters(conv_weights, out_path):
    # conv_weights: (out_channels, in_channels, kH, kW)
    num_filters = conv_weights.shape[0]
    kH = conv_weights.shape[2]
    kW = conv_weights.shape[3]

    cols = int(np.ceil(np.sqrt(num_filters)))
    rows = int(np.ceil(num_filters / cols))

    canvas = np.zeros((rows * kH, cols * kW), dtype=np.float32)
    for idx in range(num_filters):
        r = idx // cols
        c = idx % cols
        filt = conv_weights[idx, 0]
        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
        canvas[r * kH : (r + 1) * kH, c * kW : (c + 1) * kW] = filt

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(cols, rows))
        plt.imshow(canvas, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception:
        np.save(out_path.replace(".png", ".npy"), canvas)


def save_predictions(model, x, y, out_path, count=16):
    idx = np.random.choice(len(x), size=count, replace=False)
    xb = x[idx]
    yb = y[idx]
    logits = model.forward(xb)
    preds = np.argmax(logits, axis=1)

    try:
        import matplotlib.pyplot as plt

        cols = int(np.ceil(np.sqrt(count)))
        rows = int(np.ceil(count / cols))
        plt.figure(figsize=(cols * 2, rows * 2))
        for i in range(count):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(xb[i, 0], cmap="gray")
            plt.title(f"y={yb[i]} p={preds[i]}")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception:
        np.save(out_path.replace(".png", ".npy"), np.stack([xb[:, 0], yb, preds], axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.join("outputs", "model.npz"))
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = CNN()
    state = dict(np.load(args.model, allow_pickle=True))
    model.set_state(state)

    x_train, y_train, x_test, y_test = load_mnist(data_dir=args.data_dir)

    save_filters(model.conv1.W, os.path.join(args.out_dir, "conv1_filters.png"))
    save_predictions(model, x_test, y_test, os.path.join(args.out_dir, "predictions.png"))


if __name__ == "__main__":
    main()
