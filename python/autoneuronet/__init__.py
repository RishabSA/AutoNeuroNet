from . import _autoneuronet as _core

Var = _core.Var
Matrix = _core.Matrix
Layer = _core.Layer
Linear = _core.Linear
ReLU = _core.ReLU
LeakyReLU = _core.LeakyReLU
Sigmoid = _core.Sigmoid
Tanh = _core.Tanh
SiLU = _core.SiLU
ELU = _core.ELU
Softmax = _core.Softmax
NeuralNetwork = _core.NeuralNetwork
Optimizer = _core.Optimizer
GradientDescentOptimizer = _core.GradientDescentOptimizer
SGDOptimizer = _core.SGDOptimizer
matmul = _core.matmul
MSELoss = _core.MSELoss
MAELoss = _core.MAELoss
BCELoss = _core.BCELoss
CrossEntropyLoss = _core.CrossEntropyLoss
CrossEntropyLossWithLogits = _core.CrossEntropyLossWithLogits
operations = _core.operations


def numpy_to_matrix(array: any, *, as_column: bool = False) -> Matrix:
    """Convert a numpy array or sequence to an AutoNeuroNet 2D Matrix."""
    import numpy as np

    numpy_array = np.asarray(array, dtype=float)
    if numpy_array.ndim == 0:
        numpy_array = numpy_array.reshape(1, 1)
    elif numpy_array.ndim == 1:
        numpy_array = (
            numpy_array.reshape((-1, 1)) if as_column else numpy_array.reshape((1, -1))
        )
    elif numpy_array.ndim != 2:
        raise ValueError("Expected a 1D or 2D array")

    rows, cols = numpy_array.shape
    matrix = Matrix(int(rows), int(cols))
    for i in range(rows):
        matrix[i] = numpy_array[i].tolist()

    return matrix


from_numpy = numpy_to_matrix

__all__ = [
    "Var",
    "Matrix",
    "Layer",
    "Linear",
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "SiLU",
    "ELU",
    "Softmax",
    "NeuralNetwork",
    "Optimizer",
    "GradientDescentOptimizer",
    "SGDOptimizer",
    "matmul",
    "MSELoss",
    "MAELoss",
    "BCELoss",
    "CrossEntropyLoss",
    "CrossEntropyLossWithLogits",
    "operations",
    "numpy_to_matrix",
    "from_numpy",
]
