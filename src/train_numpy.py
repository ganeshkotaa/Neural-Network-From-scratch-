"""Train the NumPy CNN on MNIST."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from cnn import CNN
from layers import SoftmaxCrossEntropy
from mnist import load_mnist, train_val_split


def try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


def iterate_minibatches(x, y, batch_size, shuffle=True, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(x), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]


def accuracy(logits, labels):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == labels)


def clip_gradients(params, max_norm):
    if max_norm is None:
        return 1.0
    total_norm = 0.0
    for _, _, grad in params:
        if grad is None:
            continue
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for _, _, grad in params:
            if grad is None:
                continue
            grad *= scale
        return scale
    return 1.0


def sgd_momentum_update(params, velocities, lr, momentum):
    for name, param, grad in params:
        if grad is None:
            continue
        if name not in velocities:
            velocities[name] = np.zeros_like(param)
        velocities[name] = momentum * velocities[name] - lr * grad
        param += velocities[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    np.random.seed(args.seed)

    x_train, y_train, x_test, y_test = load_mnist(data_dir=args.data_dir)
    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train, val_ratio=args.val_ratio, seed=args.seed)

    model = CNN()
    loss_fn = SoftmaxCrossEntropy()

    velocities = {}
    best_val_acc = 0.0
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    tqdm = try_import("tqdm")
    matplotlib = try_import("matplotlib")

    os.makedirs("outputs", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_losses = []
        train_accs = []

        batch_iter = iterate_minibatches(x_train, y_train, args.batch_size, shuffle=True, seed=args.seed + epoch)
        if tqdm is not None:
            batch_iter = tqdm.tqdm(list(batch_iter), desc=f"Epoch {epoch}/{args.epochs}")

        for xb, yb in batch_iter:
            logits = model.forward(xb)
            loss = loss_fn.forward(logits, yb)
            dlogits = loss_fn.backward()
            model.backward(dlogits)

            clip_gradients(model.params(), args.grad_clip)
            sgd_momentum_update(model.params(), velocities, args.lr, args.momentum)

            train_losses.append(loss)
            train_accs.append(accuracy(logits, yb))

        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(train_accs))

        val_losses = []
        val_accs = []
        for xb, yb in iterate_minibatches(x_val, y_val, args.batch_size, shuffle=False):
            logits = model.forward(xb)
            val_losses.append(loss_fn.forward(logits, yb))
            val_accs.append(accuracy(logits, yb))

        val_loss = float(np.mean(val_losses))
        val_acc = float(np.mean(val_accs))

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} | "
            f"time: {epoch_time:.1f}s"
        )

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                args.lr *= args.lr_decay
                epochs_no_improve = 0
                print(f"Reducing learning rate to {args.lr:.5f}")

    with open(os.path.join("outputs", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if matplotlib is not None:
        import matplotlib.pyplot as plt

        epochs = np.arange(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["train_loss"], label="train")
        plt.plot(epochs, history["val_loss"], label="val")
        plt.title("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history["train_acc"], label="train")
        plt.plot(epochs, history["val_acc"], label="val")
        plt.title("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join("outputs", "training_curves.png"))
        plt.close()

    # Save model parameters for visualization and reuse
    np.savez(os.path.join("outputs", "model.npz"), **model.get_state())

    # Final test evaluation
    test_accs = []
    for xb, yb in iterate_minibatches(x_test, y_test, args.batch_size, shuffle=False):
        logits = model.forward(xb)
        test_accs.append(accuracy(logits, yb))
    test_acc = float(np.mean(test_accs))
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
