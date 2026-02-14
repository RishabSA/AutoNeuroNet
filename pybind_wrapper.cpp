#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "include/Var.hpp"
#include "include/Matrix.hpp"
#include "include/NeuralNetwork.hpp"
#include "include/Optimizers.hpp"
#include "include/LossFunctions.hpp"

namespace py = pybind11;

static int normalize_index(int idx, int size) {
    if (idx < 0) idx += size;
    return idx;
}

PYBIND11_MODULE(_autoneuronet, m) {
    m.doc() = "AutoNeuroNet is a fully implemented automatic differentiation engine with custom matrices and a full neural network architecture and training pipeline. It comes with Python bindings through PyBind11, allowing for quick and easy development of networks through Python, backed with C++ for enhanced speed and performance.";

    py::class_<Var>(m, "Var",
        R"doc(
A scalar value used for reverse-mode automatic differentiation.

Build expressions with Var objects, then call `backward()` on the final output after setting its gradient to 1.0 to accumulate gradients.
Var supports basic arithmetic operations, exponential operations, trigonometric functions, and activation functions.
)doc")
        .def(py::init<double>(), py::arg("initial"), "Create a Var with an initial value.")

        .def("getVal", &Var::getVal)
        .def("setVal", &Var::setVal, py::arg("v"))
        .def_property("val", &Var::getVal, &Var::setVal, "Current value of the variable.")

        .def("getGrad", &Var::getGrad)
        .def("setGrad", &Var::setGrad, py::arg("v"))
        .def_property("grad", &Var::getGrad, &Var::setGrad, "Current gradient of the variable")

        .def("add", py::overload_cast<Var&>(&Var::add), py::arg("other"), "Add another variable to the current variable.")
        .def("add", py::overload_cast<double>(&Var::add), py::arg("other"), "Add a double to the current variable.")
        .def("__add__", [](Var &a, Var &b) { return a.add(b); }, py::is_operator())
        .def("__add__", [](Var &a, double s) { return a.add(s); }, py::is_operator())
        .def("__radd__", [](Var &a, double s) { return a.add(s); }, py::is_operator())

        .def("subtract", py::overload_cast<Var&>(&Var::subtract), py::arg("other"), "Subtract another variable from the current variable.")
        .def("subtract", py::overload_cast<double>(&Var::subtract), py::arg("other"), "Subtract a double from the current variable.")
        .def("__sub__", [](Var &a, Var &b) { return a.subtract(b); }, py::is_operator())
        .def("__sub__", [](Var &a, double s) { return a.subtract(s); }, py::is_operator())
        .def("__rsub__", [](Var &a, double s) { return Var(s).subtract(a); }, py::is_operator())

        .def("multiply", py::overload_cast<Var&>(&Var::multiply), py::arg("other"), "Multiply another variable with the current variable.")
        .def("multiply", py::overload_cast<double>(&Var::multiply), py::arg("other"), "Multiply a double with the current variable.")
        .def("__mul__", [](Var &a, Var &b) { return a.multiply(b); }, py::is_operator())
        .def("__mul__", [](Var &a, double s) { return a.multiply(s); }, py::is_operator())
        .def("__rmul__", [](Var &a, double s) { return a.multiply(s); }, py::is_operator())
        .def("__neg__", [](Var &a) { Var negative(-1.0); return a.multiply(negative); }, py::is_operator())

        .def("divide", py::overload_cast<Var&>(&Var::divide), py::arg("other"), "Divide another variable from the current variable.")
        .def("divide", py::overload_cast<double>(&Var::divide), py::arg("other"), "Divide a double from the current variable.")
        .def("__truediv__", [](Var &a, Var &b) { return a.divide(b); }, py::is_operator())
        .def("__truediv__", [](Var &a, double s) { return a.divide(s); }, py::is_operator())
        .def("__rtruediv__", [](Var &a, double s) { return Var(s).divide(a); }, py::is_operator())

        .def("pow", &Var::pow, py::arg("power"), "Take the power of the current variable to an integer.")
        .def("__pow__", [](Var &a, int power) { return a.pow(power); }, py::is_operator(), py::arg("power"))

        .def("sin", &Var::sin, "Take the sine of the current variable.")
        .def("cos", &Var::cos, "Take the cosine of the current variable.")
        .def("tan", &Var::tan, "Take the tangent of the current variable.")
        .def("sec", &Var::sec, "Take the secant of the current variable.")
        .def("csc", &Var::csc, "Take the cosecant of the current variable.")
        .def("cot", &Var::cot, "Take the cotangent of the current variable.")

        .def("log", &Var::log, "Take the natural logarithm ln(x) of the current variable.")

        .def("exp", &Var::exp, "Get the exponential value exp(var) of the current variable.")

        .def("abs", &Var::abs, "Take the absolute value |x| of the current variable.")

        .def("relu", &Var::relu, "Apply the Rectified Linear Unit (ReLU) activation function to the current variable.")
        .def("leakyRelu", &Var::leakyRelu, py::arg("alpha") = 0.01, "Apply the Leaky Rectified Exponential Linear Unit (Leaky ReLU) activation function to the current variable.")
        .def("tanh", &Var::tanh, "Apply the Hyperbolic Tangent (Tanh) activation function to the current variable.")
        .def("sigmoid", &Var::sigmoid, "Apply the Sigmoid activation function to the current variable.")
        .def("silu", &Var::silu, "Apply the Sigmoid Linear Unit (SiLU) activation function to the current variable.")
        .def("elu", &Var::elu, py::arg("alpha") = 1.0, "Apply the Exponential Linear Unit (ELU) activation function to the current variable.")

        .def("resetGradAndParents", &Var::resetGradAndParents, "Set the current gradient to 0.0 and clear any parent node references.")
        .def("backward", &Var::backward, "Perofrm backpropagation from the current variable, and accumulate gradients.")

        .def("__repr__", [](const Var& v) {
            return "Var(val=" + std::to_string(v.getVal()) + ", grad=" + std::to_string(v.getGrad()) + ")";
        });

    py::class_<Matrix>(m, "Matrix", R"doc(
A 2D Matrix of Var objects.

Matrices support all the same operations as Var objects and accumulate gradients on all variables in the matrix.
)doc")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"), "Create a 2D Matrix of size (rows x cols) filled with 0s.")
        
        .def_readonly("rows", &Matrix::rows, "Number of rows in the current 2D matrix.")
        .def_readonly("cols", &Matrix::cols, "Number of columns in the current 2D matrix.")

        .def("__getitem__", [](Matrix &M, int i) {
                int normalized_i = normalize_index(i, M.rows);
                if (normalized_i < 0 || normalized_i >= M.rows) throw std::out_of_range("Matrix row index " + std::to_string(i) + " out of range");

                Matrix row(1, M.cols);

                for (int j = 0; j < M.cols; j++) {
                    row(0, j) = M(normalized_i, j);
                }

                return row;
            }, "Get a row from the Matrix.")

        .def("__setitem__", [](Matrix &M, int i, py::sequence seq) {
                int normalized_i = normalize_index(i, M.rows);
                if (normalized_i < 0 || normalized_i >= M.rows) throw std::out_of_range("Matrix row index " + std::to_string(i) + " out of range");
                if (seq.size() != M.cols) throw std::runtime_error("Row size mismatch: " + std::to_string(seq.size()) + " is not equal to " + std::to_string(M.cols));

                for (int j = 0; j < M.cols; j++) {
                    py::handle item = seq[j];

                    if (py::isinstance<Var>(item)) {
                        M(normalized_i, j) = item.cast<Var>();
                    } else {
                        M(normalized_i, j) = Var(item.cast<double>());
                    }
                }
            }, py::arg("index"), py::arg("value"), "Set a row in the Matrix.")
        
        .def("__getitem__", [](Matrix &M, py::tuple idx) -> Var& {
                if (idx.size() != 2) throw std::runtime_error("Use M[i, j]");

                int i = idx[0].cast<int>();
                int j = idx[1].cast<int>();
                int normalized_i = normalize_index(i, M.rows);
                int normalized_j = normalize_index(j, M.cols);

                if (normalized_i < 0 || normalized_i >= M.rows || normalized_j < 0 || normalized_j >= M.cols) throw std::out_of_range("Matrix indices " + std::to_string(i) + " and " + std::to_string(j) + " out of range");

                return M(normalized_i, normalized_j);
            },
            py::return_value_policy::reference_internal, "Get an item from the Matrix.")

        .def("__setitem__", [](Matrix &M, py::tuple idx, double v) {
                if (idx.size() != 2) throw std::runtime_error("Use M[i, j]");

                int i = idx[0].cast<int>();
                int j = idx[1].cast<int>();
                int normalized_i = normalize_index(i, M.rows);
                int normalized_j = normalize_index(j, M.cols);

                if (normalized_i < 0 || normalized_i >= M.rows || normalized_j < 0 || normalized_j >= M.cols) throw std::out_of_range("Matrix indices " + std::to_string(i) + " and " + std::to_string(j) + " out of range");
                    
                M(normalized_i, normalized_j) = Var(v);
            },
            py::arg("index"), py::arg("value"), "Set an item in the Matrix to a new double.")
        .def("__setitem__", [](Matrix &M, py::tuple idx, const Var &v) {
                if (idx.size() != 2) throw std::runtime_error("Use M[i, j]");

                int i = idx[0].cast<int>();
                int j = idx[1].cast<int>();
                int normalized_i = normalize_index(i, M.rows);
                int normalized_j = normalize_index(j, M.cols);

                if (normalized_i < 0 || normalized_i >= M.rows || normalized_j < 0 || normalized_j >= M.cols) throw std::out_of_range("Matrix indices " + std::to_string(i) + " and " + std::to_string(j) + " out of range");

                M(normalized_i, normalized_j) = v;
            },
            py::arg("index"), py::arg("value"), "Set an item from the Matrix to a new variable.")

        .def("resetGradAndParents", &Matrix::resetGradAndParents,  "Set all gradients in the matrix to 0.0 and clear any parent node references.")
        .def("randomInit", &Matrix::randomInit, "Randomly initialize all variables in the matrix with small values.")

        .def("getValsMatrix", &Matrix::getValsMatrix, "Get all of the values in the matrix.")
        .def("getGradsMatrix", &Matrix::getGradsMatrix, "Get all of the gradients in the matrix.")

        .def("add", static_cast<Matrix (Matrix::*)(Matrix&)>(&Matrix::add), py::arg("other"), "Add another matrix to the current matrix.")
        .def("__add__", [](Matrix &A, Matrix &B) { return A.add(B); }, py::is_operator(), py::arg("other"))

        .def("add", static_cast<Matrix (Matrix::*)(double)>(&Matrix::add), py::arg("other"), "Add a double to each variable in the current matrix.")
        .def("__add__", [](Matrix &A, double s) { return A.add(s); }, py::is_operator(), py::arg("other"))
        .def("__radd__", [](Matrix &A, double s) { return A.add(s); }, py::is_operator(), py::arg("other"))

        .def("subtract", static_cast<Matrix (Matrix::*)(Matrix&)>(&Matrix::subtract), py::arg("other"), "Subtract another matrix from the current matrix.")
        .def("__sub__", [](Matrix &A, Matrix &B) { return A.subtract(B); }, py::is_operator(), py::arg("other"))

        .def("subtract", static_cast<Matrix (Matrix::*)(double)>(&Matrix::subtract), py::arg("other"), "Subtract a double from each variable in the current matrix.")
        .def("__sub__", [](Matrix &A, double s) { return A.subtract(s); }, py::is_operator(), py::arg("other"))

        .def("multiply", &Matrix::multiply, py::arg("other"), "Multiply a double with each variable in the current matrix.")
        .def("__mul__", [](Matrix &A, double s) { return A.multiply(s); }, py::is_operator(), py::arg("other"))
        .def("__rmul__", [](Matrix &A, double s) { return A.multiply(s); }, py::is_operator(), py::arg("other"))

        .def("matmul", &Matrix::matmul, py::arg("other"), "Matrix multiply the current matrix with another matrix.")
        .def("__matmul__", [](Matrix &A, Matrix &B) { return A.matmul(B); }, py::is_operator(), py::arg("other"))

        .def("divide", &Matrix::divide, py::arg("other"), "Divide a double from each variable in teh current matrix.")
        .def("__truediv__", [](Matrix &A, double s) { return A.divide(s); }, py::is_operator(), py::arg("other"))

        .def("pow", &Matrix::pow, py::arg("power"), "Take the power of each variable in the current matrix to an integer.")
        .def("__pow__", [](Matrix &A, int p) { return A.pow(p); }, py::is_operator(), py::arg("power"))

        .def("sin", &Matrix::sin, "Take the sine of each variable in the current matrix.")
        .def("cos", &Matrix::cos, "Take the cosine of each variable in the current matrix.")
        .def("tan", &Matrix::tan, "Take the tangent of each variable in the current matrix.")
        .def("sec", &Matrix::sec, "Take the secant of each variable in the current matrix.")
        .def("csc", &Matrix::csc, "Take the cosecant of each variable in the current matrix.")
        .def("cot", &Matrix::cot, "Take the cotangent of each variable in the current matrix.")

        .def("log", &Matrix::log, "Take the natural logarithm ln(x) of each variable in the current matrix.")

        .def("exp", &Matrix::exp, "Get the exponential value exp(var) of each variable in the current matrix.")

        .def("abs", &Matrix::abs, "Take the absolute value |x| of each variable in the current matrix.")

        .def("relu", &Matrix::relu, "Apply the Rectified Linear Unit (ReLU) activation function to each value in the current matrix.")
        .def("leakyRelu", &Matrix::leakyRelu, py::arg("alpha") = 0.01, "Apply the Leaky Rectified Exponential Linear Unit (Leaky ReLU) activation function to each value in the current matrix.")
        .def("tanh", &Matrix::tanh, "Apply the Hyperbolic Tangent (Tanh) activation function to each value in the current matrix.")
        .def("sigmoid", &Matrix::sigmoid, "Apply the Sigmoid activation function to each value in the current matrix.")
        .def("silu", &Matrix::silu, "Apply the Sigmoid Linear Unit (SiLU) activation function to each value in the current matrix.")
        .def("elu", &Matrix::elu, py::arg("alpha") = 1.0, "Apply the Exponential Linear Unit (ELU) activation function to each value in the current matrix.")
        .def("softmax", &Matrix::softmax, "Apply the Softmax activation function to the current matrix.")

        .def("__repr__", [](const Matrix &M) {
            return "Matrix(" + std::to_string(M.rows) + ", " + std::to_string(M.cols) + ") = \n" + M.getValsMatrix();
        });

    py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer", R"doc(
Base class for all layers that can be added to a Neural Network.
)doc")
        .def_property_readonly("name", [](const Layer& layer) { return layer.name; }, "Name of the current layer.")
        .def_property_readonly("trainable", [](const Layer& layer) { return layer.trainable; }, "Is the current layer trainable.");

    py::class_<Linear, Layer, std::shared_ptr<Linear>>(m, "Linear", R"doc(
Fully-connected Linear layer that can be added to a Neural Network.
)doc")
        .def(py::init<int, int, std::string>(), py::arg("in_dim"), py::arg("out_dim"), py::arg("init") = "he", "Intitialize the weight matrix and bias vector of the current Linear layer with an initialization method for use in a Neural Network.")
        .def("forward", &Linear::forward, py::arg("input"), "Perform a single forward pass through the current Linear layer.")
        .def("optimizeWeights", &Linear::optimizeWeights, py::arg("learning_rate"), "Once gardients have been accumulated from a loss function and backpropagation, optimize the weights by moving in the direction of the negative gradient.")
        .def("resetGrad", &Linear::resetGrad,  "Set all gradients in the weight matrix and bias vector to 0.0 and clear any parent node references.")
        .def_readonly("W", &Linear::W, "Weight matrix.")
        .def_readonly("b", &Linear::b, "Bias vector.");

    py::class_<ReLU, Layer, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>(), "Initialize the ReLU activation function for use in a Neural Network.")
        .def("forward", &ReLU::forward, py::arg("input"), "Perform a single forward pass through the ReLU activation function.");

    py::class_<LeakyReLU, Layer, std::shared_ptr<LeakyReLU>>(m, "LeakyReLU")
        .def(py::init<double>(), py::arg("alpha"), "Initialize the Leaky ReLU activation function for use in a Neural Network.")
        .def("forward", &LeakyReLU::forward, py::arg("input"), "Perform a single forward pass through the Leaky ReLU activation function.");

    py::class_<Sigmoid, Layer, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>(), "Initialize the Sigmoid activation function for use in a Neural Network.")
        .def("forward", &Sigmoid::forward, py::arg("input"), "Perform a single forward pass through the Sigmoid activation function.");

    py::class_<Tanh, Layer, std::shared_ptr<Tanh>>(m, "Tanh")
        .def(py::init<>(), "Initialize the Tanh activation function for use in a Neural Network.")
        .def("forward", &Tanh::forward, py::arg("input"), "Perform a single forward pass through the Tanh activation function.");

    py::class_<SiLU, Layer, std::shared_ptr<SiLU>>(m, "SiLU")
        .def(py::init<>(), "Initialize the SiLU activation function for use in a Neural Network.")
        .def("forward", &SiLU::forward, py::arg("input"), "Perform a single forward pass through the SiLU activation function.");

    py::class_<ELU, Layer, std::shared_ptr<ELU>>(m, "ELU")
        .def(py::init<double>(), py::arg("alpha"), "Initialize the ELU activation function for use in a Neural Network.")
        .def("forward", &ELU::forward, py::arg("input"), "Perform a single forward pass through the ELU activation function.");

    py::class_<Softmax, Layer, std::shared_ptr<Softmax>>(m, "Softmax")
        .def(py::init<>(), "Initialize the Softmax activation function for use in a Neural Network.")
        .def("forward", &Softmax::forward, py::arg("input"), "Perform a single forward pass through the Softmax activation function.");

    py::class_<NeuralNetwork>(m, "NeuralNetwork", R"doc(
A simple feed-forward neural network built from Matrix layers.
)doc")
        .def(py::init<std::vector<std::shared_ptr<Layer>>>(), py::arg("layers"), "Initialize the Neural Network with a list of Layers.")

        .def("getLayers", py::overload_cast<>(&NeuralNetwork::getLayers, py::const_))
        .def_property_readonly("layers", py::overload_cast<>(&NeuralNetwork::getLayers, py::const_), "Neural network layers to use in the forward pass and to optimize.")

        .def("addLayer", &NeuralNetwork::addLayer, py::arg("layer"), "Add a layer to the end of the Neural Network.")
        .def("forward", &NeuralNetwork::forward, py::arg("input"), "Perform a forward pass through the entire Neural Network, through each Layer.")
        .def("getNetworkArchitecture", &NeuralNetwork::getNetworkArchitecture, "Get the full architecture of the Neural Network with each Layer specified.")

        .def("saveWeights", &NeuralNetwork::saveWeights, py::arg("path"), "Save the weight matrices and bias vectors of every Linear layer in the Neural Network to a .bin file.")
        .def("loadWeights", &NeuralNetwork::loadWeights, py::arg("path"), "Load the weight matrices and bias vectors of every Linear layer into the Neural Network from a .bin file.")
        
        .def("__repr__", [](const NeuralNetwork &model) {
            return "NeuralNetwork =\n" + model.getNetworkArchitecture();
        });

    py::class_<Optimizer>(m, "Optimizer", R"doc(
Base class for all optimizers for a Neural Network.
)doc")
        .def("optimize", &Optimizer::optimize, "Optimize the weight matrices and bias vectors of every layer in a Neural Network once gradients have been accumulated through backpropagation.")
        .def("resetGrad", &Optimizer::resetGrad, "Set all gradients for every layer's weight matrix and bias vector to 0.0 and clear any parent node references.");

    py::class_<GradientDescentOptimizer, Optimizer>(m, "GradientDescentOptimizer", R"doc(
Gradient descent optimizer for a Neural Network to update weight matrices and bias vectors.
)doc")
        .def(py::init<double, NeuralNetwork*>(), py::arg("learning_rate"), py::arg("model"), py::keep_alive<1, 2>(), "Intiialize the Gradient Descent optimizer with a learning rate and the Neural Network model to optimize.")
        .def("optimize", &GradientDescentOptimizer::optimize, "Optimize the weight matrices and bias vectors of every layer in a Neural Network using the Gradient Descent algorithm once gradients have been accumulated through backpropagation.")
        .def("resetGrad", &GradientDescentOptimizer::resetGrad, "Set all gradients for every layer's weight matrix and bias vector to 0.0 and clear any parent node references.");

    py::class_<SGDOptimizer, Optimizer>(m, "SGDOptimizer", R"doc(
Stochastic gradient descent (SGD) optimizer with momentum/weight decay.
)doc")
        .def(py::init<double, NeuralNetwork*, double, double>(), py::arg("learning_rate"), py::arg("model"), py::arg("momentum") = 0.0, py::arg("weight_decay") = 0.0, py::keep_alive<1, 2>(), "Intiialize the Stochastic Gradient Descent (SGD) optimizer with a learning rate, the Neural Network model to optimize, momentum, and weight decay.")
        .def("optimize", &SGDOptimizer::optimize, "Optimize the weight matrices and bias vectors of every layer in a Neural Network using the Stochastic Gradient Descent (SGD) algorithm once gradients have been accumulated through backpropagation.")
        .def("resetGrad", &SGDOptimizer::resetGrad, "Set all gradients for every layer's weight matrix and bias vector to 0.0 and clear any parent node references.");

    m.def("matmul", &matmul, py::arg("A"), py::arg("B"), "Matrix multiply a matrix with another matrix.");
    m.def("MSELoss", &MSELoss, py::arg("labels"), py::arg("preds"), "Compute the Mean Squared Error (MSE) Loss given a matrix of labels and a matrix of predictions.");
    m.def("MAELoss", &MAELoss, py::arg("labels"), py::arg("preds"), "Compute the Mean Absolute Error (MAE) Loss given a matrix of labels and a matrix of predictions.");
    m.def("BCELoss", &BCELoss, py::arg("labels"), py::arg("preds"), py::arg("eps") = 1e-7, "Compute the Binary Cross-Entropy (BCE) Loss given a matrix of labels and a matrix of predictions.");
    m.def("CrossEntropyLoss", &CrossEntropyLoss, py::arg("labels"), py::arg("preds"), py::arg("eps") = 1e-9, "Compute the Cross Entropy Loss given a matrix of labels and a matrix of predictions.");
    m.def("CrossEntropyLossWithLogits", &CrossEntropyLossWithLogits, py::arg("labels"), py::arg("logits"), py::arg("eps") = 1e-9, "Compute the Cross Entropy with Logits Loss given a matrix of labels and a matrix of logits.");

    py::module_ operations = m.def_submodule("operations");
    operations.def("sin", [](Var& v) { return v.sin(); }, py::arg("var"), "Take the sine of a variable.");
    operations.def("sin", [](Matrix& matrix) { return matrix.sin(); }, py::arg("matrix"), "Take the sine of each variable in a matrix.");

    operations.def("cos", [](Var& v) { return v.cos(); }, py::arg("var"), "Take the cosine of a variable.");
    operations.def("cos", [](Matrix& matrix) { return matrix.cos(); }, py::arg("matrix"), "Take the cosine of each variable in a matrix.");

    operations.def("tan", [](Var& v) { return v.tan(); }, py::arg("var"), "Take the tangent of a variable.");
    operations.def("tan", [](Matrix& matrix) { return matrix.tan(); }, py::arg("matrix"), "Take the tangent of each variable in a matrix.");

    operations.def("sec", [](Var& v) { return v.sec(); }, py::arg("var"), "Take the secant of a variable.");
    operations.def("sec", [](Matrix& matrix) { return matrix.sec(); }, py::arg("matrix"), "Take the secant of each variable in a matrix.");
    
    operations.def("csc", [](Var& v) { return v.csc(); }, py::arg("var"), "Take the cosecant of a variable.");
    operations.def("csc", [](Matrix& matrix) { return matrix.csc(); }, py::arg("matrix"), "Take the cosecant of each variable in a matrix.");

    operations.def("cot", [](Var& v) { return v.cot(); }, py::arg("var"), "Take the cotangent of a variable.");
    operations.def("cot", [](Matrix& matrix) { return matrix.cot(); }, py::arg("matrix"), "Take the cotangent of each variable in a matrix.");

    operations.def("log", [](Var& v) { return v.log(); }, py::arg("var"), "Take the natural logarithm ln(x) of a variable.");
    operations.def("log", [](Matrix& matrix) { return matrix.log(); }, py::arg("matrix"), "Take the natural logarithm ln(x) of each variable in a matrix.");

    operations.def("exp", [](Var& v) { return v.exp(); }, py::arg("var"), "Get the exponential value exp(var) of a variable.");
    operations.def("exp", [](Matrix& matrix) { return matrix.exp(); }, py::arg("matrix"), "Get the exponential value exp(var) of each variable in a matrix.");
    
    operations.def("abs", [](Var& v) { return v.abs(); }, py::arg("var"), "Take the absolute value |x| of a variable.");
    operations.def("abs", [](Matrix& matrix) { return matrix.abs(); }, py::arg("matrix"), "Take the absolute value |x| of each variable in a matrix.");
}
