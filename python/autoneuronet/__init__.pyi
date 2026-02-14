"""AutoNeuroNet is a fully implemented automatic differentiation engine with custom matrices and a full neural network architecture and training pipeline. It comes with Python bindings through PyBind11, allowing for quick and easy development of networks through Python, backed with C++ for enhanced speed and performance."""

from types import ModuleType
from typing import overload

class Var:
    """
    A scalar value used for reverse-mode automatic differentiation.

    Build expressions with Var objects, then call `backward()` on the final output after setting its gradient to 1.0 to accumulate gradients.
    Var supports basic arithmetic operations, exponential operations, trigonometric functions, and activation functions.
    """

    def __init__(self, initial: float) -> None:
        """Create a Var with an initial value."""
        ...

    def getVal(self) -> float: ...
    def setVal(self, v: float) -> None: ...
    @property
    def val(self) -> float:
        """Current value of the variable."""
        ...

    @val.setter
    def val(self, v: float) -> None: ...
    def getGrad(self) -> float: ...
    def setGrad(self, v: float) -> None: ...
    @property
    def grad(self) -> float:
        """Current gradient of the variable"""
        ...

    @grad.setter
    def grad(self, v: float) -> None: ...
    @overload
    def add(self, other: Var) -> Var:
        """Add another variable to the current variable."""
        ...

    @overload
    def add(self, other: float) -> Var:
        """Add a double to the current variable."""
        ...

    def __add__(self, other: Var | float) -> Var: ...
    def __radd__(self, other: float) -> Var: ...
    @overload
    def subtract(self, other: Var) -> Var:
        """Subtract another variable from the current variable."""
        ...

    @overload
    def subtract(self, other: float) -> Var:
        """Subtract a double from the current variable."""
        ...

    def __sub__(self, other: Var | float) -> Var: ...
    def __rsub__(self, other: float) -> Var: ...
    @overload
    def multiply(self, other: Var) -> Var:
        """Multiply another variable with the current variable."""
        ...

    @overload
    def multiply(self, other: float) -> Var:
        """Multiply a double with the current variable."""
        ...

    def __mul__(self, other: Var | float) -> Var: ...
    def __rmul__(self, other: float) -> Var: ...
    def __neg__(self) -> Var: ...
    @overload
    def divide(self, other: Var) -> Var:
        """Divide another variable from the current variable."""
        ...

    @overload
    def divide(self, other: float) -> Var:
        """Divide a double from the current variable."""
        ...

    def __truediv__(self, other: Var | float) -> Var: ...
    def __rtruediv__(self, other: float) -> Var: ...
    def pow(self, power: int) -> Var:
        """Take the power of the current variable to an integer."""
        ...

    def __pow__(self, power: int) -> Var: ...
    def sin(self) -> Var:
        """Take the sine of the current variable."""
        ...

    def cos(self) -> Var:
        """Take the cosine of the current variable."""
        ...

    def tan(self) -> Var:
        """Take the tangent of the current variable."""
        ...

    def sec(self) -> Var:
        """Take the secant of the current variable."""
        ...

    def csc(self) -> Var:
        """Take the cosecant of the current variable."""
        ...

    def cot(self) -> Var:
        """Take the cotangent of the current variable."""
        ...

    def log(self) -> Var:
        """Take the natural logarithm ln(x) of the current variable."""
        ...

    def exp(self) -> Var:
        """Get the exponential value exp(var) of the current variable."""
        ...

    def abs(self) -> Var:
        """Take the absolute value |x| of the current variable."""
        ...

    def relu(self) -> Var:
        """Apply the Rectified Linear Unit (ReLU) activation function to the current variable."""
        ...

    def leakyRelu(self, alpha: float = 0.01) -> Var:
        """Apply the Leaky Rectified Exponential Linear Unit (Leaky ReLU) activation function to the current variable."""
        ...

    def tanh(self) -> Var:
        """Apply the Hyperbolic Tangent (Tanh) activation function to the current variable."""
        ...

    def sigmoid(self) -> Var:
        """Apply the Sigmoid activation function to the current variable."""
        ...

    def silu(self) -> Var:
        """Apply the Sigmoid Linear Unit (SiLU) activation function to the current variable."""
        ...

    def elu(self, alpha: float = 1.0) -> Var:
        """Apply the Exponential Linear Unit (ELU) activation function to the current variable."""
        ...

    def resetGradAndParents(self) -> None:
        """Set the current gradient to 0.0 and clear any parent node references."""
        ...

    def backward(self) -> None:
        """Perofrm backpropagation from the current variable, and accumulate gradients."""
        ...

    def __repr__(self) -> str: ...

class Matrix:
    """
    A 2D Matrix of Var objects.

    Matrices support all the same operations as Var objects and accumulate gradients on all variables in the matrix.
    """

    def __init__(self, rows: int, cols: int) -> None:
        """Create a 2D Matrix of size (rows x cols) filled with 0s."""
        ...

    @property
    def rows(self) -> int:
        """Number of rows in the current 2D matrix."""
        ...

    @property
    def cols(self) -> int:
        """Number of columns in the current 2D matrix."""
        ...

    @overload
    def __getitem__(self, i: int) -> Matrix:
        """Get a row from the Matrix."""
        ...

    @overload
    def __getitem__(self, idx: tuple[int, int]) -> Var:
        """Get an item from the Matrix."""
        ...

    @overload
    def __setitem__(self, index: int, value: list[Var | float]) -> None:
        """Set a row in the Matrix."""
        ...

    @overload
    def __setitem__(self, index: tuple[int, int], value: float) -> None:
        """Set an item in the Matrix to a new double."""
        ...

    @overload
    def __setitem__(self, index: tuple[int, int], value: Var) -> None:
        """Set an item from the Matrix to a new variable."""
        ...

    def resetGradAndParents(self) -> None:
        """Set all gradients in the matrix to 0.0 and clear any parent node references."""
        ...

    def randomInit(self) -> None:
        """Randomly initialize all variables in the matrix with small values."""
        ...

    def getValsMatrix(self) -> str:
        """Get all of the values in the matrix."""
        ...

    def getGradsMatrix(self) -> str:
        """Get all of the gradients in the matrix."""
        ...

    @overload
    def add(self, other: Matrix) -> Matrix:
        """Add another matrix to the current matrix."""
        ...

    @overload
    def add(self, other: float) -> Matrix:
        """Add a double to each variable in the current matrix."""
        ...

    def __add__(self, other: Matrix | float) -> Matrix: ...
    def __radd__(self, other: float) -> Matrix: ...
    @overload
    def subtract(self, other: Matrix) -> Matrix:
        """Subtract another matrix from the current matrix."""
        ...

    @overload
    def subtract(self, other: float) -> Matrix:
        """Subtract a double from each variable in the current matrix."""
        ...

    def __sub__(self, other: Matrix | float) -> Matrix: ...
    def multiply(self, other: float) -> Matrix:
        """Multiply a double with each variable in the current matrix."""
        ...

    def __mul__(self, other: float) -> Matrix: ...
    def __rmul__(self, other: float) -> Matrix: ...
    def matmul(self, other: Matrix) -> Matrix:
        """Matrix multiply the current matrix with another matrix."""
        ...

    def __matmul__(self, other: Matrix) -> Matrix: ...
    def divide(self, other: float) -> Matrix:
        """Divide a double from each variable in teh current matrix."""
        ...

    def __truediv__(self, other: float) -> Matrix: ...
    def pow(self, power: int) -> Matrix:
        """Take the power of each variable in the current matrix to an integer."""
        ...

    def __pow__(self, power: int) -> Matrix: ...
    def sin(self) -> Matrix:
        """Take the sine of each variable in the current matrix."""
        ...

    def cos(self) -> Matrix:
        """Take the cosine of each variable in the current matrix."""
        ...

    def tan(self) -> Matrix:
        """Take the tangent of each variable in the current matrix."""
        ...

    def sec(self) -> Matrix:
        """Take the secant of each variable in the current matrix."""
        ...

    def csc(self) -> Matrix:
        """Take the cosecant of each variable in the current matrix."""
        ...

    def cot(self) -> Matrix:
        """Take the cotangent of each variable in the current matrix."""
        ...

    def log(self) -> Matrix:
        """Take the natural logarithm ln(x) of each variable in the current matrix."""
        ...

    def exp(self) -> Matrix:
        """Get the exponential value exp(var) of each variable in the current matrix."""
        ...

    def abs(self) -> Matrix:
        """Take the absolute value |x| of each variable in the current matrix."""
        ...

    def relu(self) -> Matrix:
        """Apply the Rectified Linear Unit (ReLU) activation function to each value in the current matrix."""
        ...

    def leakyRelu(self, alpha: float = 0.01) -> Matrix:
        """Apply the Leaky Rectified Exponential Linear Unit (Leaky ReLU) activation function to each value in the current matrix."""
        ...

    def tanh(self) -> Matrix:
        """Apply the Hyperbolic Tangent (Tanh) activation function to each value in the current matrix."""
        ...

    def sigmoid(self) -> Matrix:
        """Apply the Sigmoid activation function to each value in the current matrix."""
        ...

    def silu(self) -> Matrix:
        """Apply the Sigmoid Linear Unit (SiLU) activation function to each value in the current matrix."""
        ...

    def elu(self, alpha: float = 1.0) -> Matrix:
        """Apply the Exponential Linear Unit (ELU) activation function to each value in the current matrix."""
        ...

    def softmax(self) -> Matrix:
        """Apply the Softmax activation function to the current matrix."""
        ...

    def __repr__(self) -> str: ...

class Layer:
    """Base class for all layers that can be added to a Neural Network."""

    @property
    def name(self) -> str:
        """Name of the current layer."""
        ...

    @property
    def trainable(self) -> bool:
        """Is the current layer trainable."""
        ...

class Linear(Layer):
    """Fully-connected Linear layer that can be added to a Neural Network."""

    def __init__(self, in_dim: int, out_dim: int, init: str = "he") -> None:
        """Intitialize the weight matrix and bias vector of the current Linear layer with an initialization method for use in a Neural Network."""
        ...

    def forward(self, input: Matrix) -> Matrix:
        """Perform a single forward pass through the current Linear layer."""
        ...

    def optimizeWeights(self, learning_rate: float) -> None:
        """Once gardients have been accumulated from a loss function and backpropagation, optimize the weights by moving in the direction of the negative gradient."""
        ...

    def resetGrad(self) -> None:
        """Set all gradients in the weight matrix and bias vector to 0.0 and clear any parent node references."""
        ...

    @property
    def W(self) -> Matrix:
        """Weight matrix."""
        ...

    @property
    def b(self) -> Matrix:
        """Bias vector."""
        ...

class ReLU(Layer):
    def __init__(self) -> None:
        """Initialize the ReLU activation function for use in a Neural Network."""
        ...

    def forward(self, input: Matrix) -> Matrix:
        """Perform a single forward pass through the ReLU activation function."""
        ...

class LeakyReLU(Layer):
    def __init__(self, alpha: float) -> None:
        """Initialize the Leaky ReLU activation function for use in a Neural Network."""
        ...

    def forward(self, input: Matrix) -> Matrix:
        """Perform a single forward pass through the Leaky ReLU activation function."""
        ...

class Sigmoid(Layer):
    def __init__(self) -> None:
        """Initialize the Sigmoid activation function for use in a Neural Network."""
        ...

    def forward(self, input: Matrix) -> Matrix:
        """Perform a single forward pass through the Sigmoid activation function."""
        ...

class Tanh(Layer):
    def __init__(self) -> None:
        """Initialize the Tanh activation function for use in a Neural Network."""
        ...

    def forward(self, input: Matrix) -> Matrix:
        """Perform a single forward pass through the Tanh activation function."""
        ...

class SiLU(Layer):
    def __init__(self) -> None:
        """Initialize the SiLU activation function for use in a Neural Network."""
        ...

    def forward(self, input: Matrix) -> Matrix:
        """Perform a single forward pass through the SiLU activation function."""
        ...

class ELU(Layer):
    def __init__(self, alpha: float) -> None:
        """Initialize the ELU activation function for use in a Neural Network."""
        ...

    def forward(self, input: Matrix) -> Matrix:
        """Perform a single forward pass through the ELU activation function."""
        ...

class Softmax(Layer):
    def __init__(self) -> None:
        """Initialize the Softmax activation function for use in a Neural Network."""
        ...

    def forward(self, input: Matrix) -> Matrix:
        """Perform a single forward pass through the Softmax activation function."""
        ...

class NeuralNetwork:
    """A simple feed-forward neural network built from Matrix layers."""

    def __init__(self, layers: list[Layer]) -> None:
        """Initialize the Neural Network with a list of Layers."""
        ...

    def getLayers(self) -> list[Layer]: ...
    @property
    def layers(self) -> list[Layer]:
        """Neural network layers to use in the forward pass and to optimize."""
        ...

    def addLayer(self, layer: Layer) -> None:
        """Add a layer to the end of the Neural Network."""
        ...

    def forward(self, input: Matrix) -> Matrix:
        """Perform a forward pass through the entire Neural Network, through each Layer."""
        ...

    def getNetworkArchitecture(self) -> str:
        """Get the full architecture of the Neural Network with each Layer specified."""
        ...

    def saveWeights(self, path: str) -> None:
        """Save the weight matrices and bias vectors of every Linear layer in the Neural Network to a .bin file."""
        ...

    def loadWeights(self, path: str) -> None:
        """Load the weight matrices and bias vectors of every Linear layer into the Neural Network from a .bin file."""
        ...

    def __repr__(self) -> str: ...

class Optimizer:
    """Base class for all optimizers for a Neural Network."""

    def optimize(self) -> None:
        """Optimize the weight matrices and bias vectors of every layer in a Neural Network once gradients have been accumulated through backpropagation."""
        ...

    def resetGrad(self) -> None:
        """Set all gradients for every layer's weight matrix and bias vector to 0.0 and clear any parent node references."""
        ...

class GradientDescentOptimizer(Optimizer):
    """Gradient descent optimizer for a Neural Network to update weight matrices and bias vectors."""

    def __init__(self, learning_rate: float, model: NeuralNetwork) -> None:
        """Intiialize the Gradient Descent optimizer with a learning rate and the Neural Network model to optimize."""
        ...

    def optimize(self) -> None:
        """Optimize the weight matrices and bias vectors of every layer in a Neural Network using the Gradient Descent algorithm once gradients have been accumulated through backpropagation."""
        ...

    def resetGrad(self) -> None:
        """Set all gradients for every layer's weight matrix and bias vector to 0.0 and clear any parent node references."""
        ...

class SGDOptimizer(Optimizer):
    """Stochastic gradient descent (SGD) optimizer with momentum/weight decay."""

    def __init__(
        self,
        learning_rate: float,
        model: NeuralNetwork,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        """Intiialize the Stochastic Gradient Descent (SGD) optimizer with a learning rate, the Neural Network model to optimize, momentum, and weight decay."""
        ...

    def optimize(self) -> None:
        """Optimize the weight matrices and bias vectors of every layer in a Neural Network using the Stochastic Gradient Descent (SGD) algorithm once gradients have been accumulated through backpropagation."""
        ...

    def resetGrad(self) -> None:
        """Set all gradients for every layer's weight matrix and bias vector to 0.0 and clear any parent node references."""
        ...

def matmul(A: Matrix, B: Matrix) -> Matrix:
    """Matrix multiply a matrix with another matrix."""
    ...

def MSELoss(labels: Matrix, preds: Matrix) -> Var:
    """Compute the Mean Squared Error (MSE) Loss given a matrix of labels and a matrix of predictions."""
    ...

def MAELoss(labels: Matrix, preds: Matrix) -> Var:
    """Compute the Mean Absolute Error (MAE) Loss given a matrix of labels and a matrix of predictions."""
    ...

def BCELoss(labels: Matrix, preds: Matrix, eps: float = 1e-7) -> Var:
    """Compute the Binary Cross-Entropy (BCE) Loss given a matrix of labels and a matrix of predictions."""
    ...

def CrossEntropyLoss(labels: Matrix, preds: Matrix, eps: float = 1e-9) -> Var:
    """Compute the Cross Entropy Loss given a matrix of labels and a matrix of predictions."""
    ...

def CrossEntropyLossWithLogits(
    labels: Matrix, logits: Matrix, eps: float = 1e-9
) -> Var:
    """Compute the Cross Entropy with Logits Loss given a matrix of labels and a matrix of logits."""
    ...

class _Operations(ModuleType):
    @overload
    def sin(self, var: Var) -> Var:
        """Take the sine of a variable."""
        ...

    @overload
    def sin(self, matrix: Matrix) -> Matrix:
        """Take the sine of each variable in a matrix."""
        ...

    @overload
    def cos(self, var: Var) -> Var:
        """Take the cosine of a variable."""
        ...

    @overload
    def cos(self, matrix: Matrix) -> Matrix:
        """Take the cosine of each variable in a matrix."""
        ...

    @overload
    def tan(self, var: Var) -> Var:
        """Take the tangent of a variable."""
        ...

    @overload
    def tan(self, matrix: Matrix) -> Matrix:
        """Take the tangent of each variable in a matrix."""
        ...

    @overload
    def sec(self, var: Var) -> Var:
        """Take the secant of a variable."""
        ...

    @overload
    def sec(self, matrix: Matrix) -> Matrix:
        """Take the secant of each variable in a matrix."""
        ...

    @overload
    def csc(self, var: Var) -> Var:
        """Take the cosecant of a variable."""
        ...

    @overload
    def csc(self, matrix: Matrix) -> Matrix:
        """Take the cosecant of each variable in a matrix."""
        ...

    @overload
    def cot(self, var: Var) -> Var:
        """Take the cotangent of a variable."""
        ...

    @overload
    def cot(self, matrix: Matrix) -> Matrix:
        """Take the cotangent of each variable in a matrix."""
        ...

    @overload
    def log(self, var: Var) -> Var:
        """Take the natural logarithm ln(x) of a variable."""
        ...

    @overload
    def log(self, matrix: Matrix) -> Matrix:
        """Take the natural logarithm ln(x) of each variable in a matrix."""
        ...

    @overload
    def exp(self, var: Var) -> Var:
        """Get the exponential value exp(var) of a variable."""
        ...

    @overload
    def exp(self, matrix: Matrix) -> Matrix:
        """Get the exponential value exp(var) of each variable in a matrix."""
        ...

    @overload
    def abs(self, var: Var) -> Var:
        """Take the absolute value |x| of a variable."""
        ...

    @overload
    def abs(self, matrix: Matrix) -> Matrix:
        """Take the absolute value |x| of each variable in a matrix."""
        ...

operations: _Operations

def numpy_to_matrix(array: any, *, as_column: bool = False) -> Matrix:
    """Convert a numpy array or list to an AutoNeuroNet 2D Matrix."""
    ...

def from_numpy(array: any, *, as_column: bool = False) -> Matrix:
    """Convert a numpy array or list to an AutoNeuroNet 2D Matrix."""
    ...

__all__: list[str]
