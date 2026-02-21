# API

This page shows the public Python API for **AutoNeuroNet**.

**Core types**

- `Var`: Scalar value with gradient tracking and reverse-mode automatic differentiation.
- `Matrix`: 2D matrix of `Var` values with math, activation functions, gradients, and more.

**Neural Network**

- `Layer` (base class)
- `Linear`, `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `SiLU`, `ELU`, `Softmax`
- `NeuralNetwork`

**Optimizers**

- `GradientDescentOptimizer`
- `SGDOptimizer`
- `AdagradOptimizer`
- `RMSPropOptimizer`
- `AdamOptimizer`
- `AdamWOptimizer`

**Losses**

- `MSELoss`
- `MAELoss`
- `BCELoss`
- `CrossEntropyLoss`
- `CrossEntropyLossWithLogits`

**Utilities**

- `matmul(A, B)`
- `numpy_to_matrix(array, as_column=False)`
- `from_numpy(array, as_column=False)` (alias)

## Automatic Differentiation

**Var**

- Create a value: `Var(1.0, requires_grad=True)`
- Read/write value: `v.val`
- Read/write gradient: `v.grad`
- Disable grad: `v.noGrad()`
- Detach: `v.detach()`
- Backpropagation: `v.setGrad(1.0); v.backward()`

**Matrix**

- Create: `Matrix(rows, cols, requires_grad=True)`
- Index: `M[i, j]` or row slices `M[i]`
- Math: `+`, `-`, `*`, `/`, `@`, `pow`, `sin`, `cos`, `tanh`, `relu`, `softmax`, etc.
- Disable grad: `M.noGrad()`
- Detach: `M.detach()`

## Neural Networks

**NeuralNetwork**

- Construct with a list of layers.
- Forward pass: `model.forward(x)`
- Save/restore weights: `model.saveWeights(path)`, `model.loadWeights(path)`

**Optimizers**

- Create with `learning_rate` and a model.
- Call `optimize()` after backprop.
- Call `resetGrad()` between steps if needed.
