# Convolutional Neural Network From First Principles (NumPy Only)

This project implements a complete CNN **from scratch** in pure NumPy (no PyTorch/TensorFlow for the NumPy model). It is designed to demonstrate deep understanding of convolution, backpropagation, and training dynamics. A PyTorch baseline is included for comparison.

## Goals (Brief)
- Implement a CNN from scratch in NumPy (forward + backward for all layers)
- Train on MNIST to exceed 94% accuracy
- Compare with a PyTorch baseline

## Project Structure
- `src/layers.py`: Core layers (Conv2D, ReLU, MaxPool2D, Linear, Flatten, SoftmaxCE)
- `src/cnn.py`: CNN architecture and parameter updates
- `src/mnist.py`: MNIST downloader/loader (IDX format)
- `src/train_numpy.py`: NumPy training pipeline
- `src/pytorch_baseline.py`: PyTorch reference model
- `docs/math.md`: Mathematical explanations
- `scripts/`: Convenience PowerShell scripts

## Requirements
- Python 3.10+
- NumPy
- Optional: `matplotlib`, `tqdm` for plots/progress
- Optional (baseline): `torch`, `torchvision`

## Usage
### 1) NumPy CNN (from scratch)
```powershell
python src/train_numpy.py --epochs 15 --batch-size 64 --lr 0.01
```

### 2) PyTorch baseline
```powershell
python src/pytorch_baseline.py --epochs 15 --batch-size 64 --lr 0.01
```

## Notes
- MNIST is downloaded automatically into `data/` (ignored by git).
- The NumPy model uses im2col for convolution efficiency.
- The PyTorch baseline uses the same architecture for fair comparison.

## Expected Results
With reasonable hyperparameters, the NumPy model should reach 94%+ validation accuracy on MNIST.

## Results (Example Run)
From your latest run:
- Validation accuracy: **0.9899**
- Test accuracy: **0.9904**

Artifacts:
- `outputs/training_curves.png`
- `outputs/model.npz`
- `outputs/conv1_filters.png`
- `outputs/predictions.png`

To generate visuals after training:
```powershell
python src/visualize.py --model outputs\\model.npz
```

## License
MIT. See `LICENSE`.
