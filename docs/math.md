# Mathematical Notes

## Convolution (Forward)
Given input `x` and filter `W`, the 2D convolution output at position `(i, j)` is:

```
Y[n, k, i, j] = sum_{c=0}^{C-1} sum_{u=0}^{KH-1} sum_{v=0}^{KW-1}
               X[n, c, i+u, j+v] * W[k, c, u, v] + b[k]
```

We implement this efficiently using **im2col** (unfold the input into columns) and a matrix multiply.

## Backpropagation Through Convolution
If `dY` is the upstream gradient:

- Gradient w.r.t. weights:
```
dW = dY_col^T * X_col
```
- Gradient w.r.t. input:
```
dX = col2im(dY_col * W_col^T)
```

## ReLU
```
Y = max(0, X)
```
Gradient:
```
dX = dY * (X > 0)
```

## Max Pooling
Forward selects the max in each window. Backward routes gradient **only** to the max location (tracked by argmax).

## Fully Connected Layer
```
Y = XW + b
```
Gradients:
```
dW = X^T dY,  db = sum(dY),  dX = dY W^T
```

## Softmax + Cross-Entropy
For logits `z`:
```
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```
Loss for true class `y`:
```
L = -log(softmax(z_y))
```
Gradient:
```
dz = (softmax(z) - one_hot(y)) / N
```

## Architecture Shapes
Input: `1 x 28 x 28`
- Conv(8, 3x3, pad=1) -> `8 x 28 x 28`
- MaxPool(2x2) -> `8 x 14 x 14`
- Conv(16, 3x3, pad=1) -> `16 x 14 x 14`
- MaxPool(2x2) -> `16 x 7 x 7`
- Flatten -> `784`
- FC(128) -> `128`
- FC(10) -> `10`

Total parameters printed by `CNN.summary()`.
