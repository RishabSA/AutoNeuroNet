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
    m.doc() = "AutoNeuroNet is a library for automatic differentiation and neural networks.";

    py::class_<Var>(m, "Var",
        R"doc(
A scalar value used for reverse-mode automatic differentiation.

You can build expressions with Var objects, then call `backward()` on the final output
after setting its gradient to 1.0 to accumulate gradients.
)doc")
        .def(py::init<double>(), py::arg("initial"))

        .def("getVal", &Var::getVal)
        .def("setVal", &Var::setVal, py::arg("v"))
        .def_property("val", &Var::getVal, &Var::setVal)

        .def("getGrad", &Var::getGrad)
        .def("setGrad", &Var::setGrad, py::arg("v"))
        .def_property("grad", &Var::getGrad, &Var::setGrad)

        .def("add", py::overload_cast<Var&>(&Var::add), py::arg("other"))
        .def("add", py::overload_cast<double>(&Var::add), py::arg("other"))
        .def("__add__", [](Var &a, Var &b) { return a.add(b); }, py::is_operator())
        .def("__add__", [](Var &a, double s) { return a.add(s); }, py::is_operator())
        .def("__radd__", [](Var &a, double s) { return a.add(s); }, py::is_operator())

        .def("subtract", py::overload_cast<Var&>(&Var::subtract), py::arg("other"))
        .def("subtract", py::overload_cast<double>(&Var::subtract), py::arg("other"))
        .def("__sub__", [](Var &a, Var &b) { return a.subtract(b); }, py::is_operator())
        .def("__sub__", [](Var &a, double s) { return a.subtract(s); }, py::is_operator())
        .def("__rsub__", [](Var &a, double s) { return Var(s).subtract(a); }, py::is_operator())

        .def("multiply", py::overload_cast<Var&>(&Var::multiply), py::arg("other"))
        .def("multiply", py::overload_cast<double>(&Var::multiply), py::arg("other"))
        .def("__mul__", [](Var &a, Var &b) { return a.multiply(b); }, py::is_operator())
        .def("__mul__", [](Var &a, double s) { return a.multiply(s); }, py::is_operator())
        .def("__rmul__", [](Var &a, double s) { return a.multiply(s); }, py::is_operator())
        .def("__neg__", [](Var &a) { Var neg(-1.0); return a.multiply(neg); }, py::is_operator())

        .def("divide", py::overload_cast<Var&>(&Var::divide), py::arg("other"))
        .def("divide", py::overload_cast<double>(&Var::divide), py::arg("other"))
        .def("__truediv__", [](Var &a, Var &b) { return a.divide(b); }, py::is_operator())
        .def("__truediv__", [](Var &a, double s) { return a.divide(s); }, py::is_operator())
        .def("__rtruediv__", [](Var &a, double s) { return Var(s).divide(a); }, py::is_operator())

        .def("pow", &Var::pow, py::arg("power"))
        .def("__pow__", [](Var &a, int p) { return a.pow(p); }, py::is_operator(), py::arg("power"))

        .def("sin", &Var::sin)
        .def("cos", &Var::cos)
        .def("tan", &Var::tan)
        .def("sec", &Var::sec)
        .def("csc", &Var::csc)
        .def("cot", &Var::cot)

        .def("log", &Var::log)

        .def("exp", &Var::exp)

        .def("abs", &Var::abs)

        .def("relu", &Var::relu)
        .def("leakyRelu", &Var::leakyRelu, py::arg("alpha") = 0.01)
        .def("tanh", &Var::tanh)
        .def("sigmoid", &Var::sigmoid)
        .def("silu", &Var::silu)
        .def("elu", &Var::elu, py::arg("alpha") = 1.0)

        .def("resetGradAndParents", &Var::resetGradAndParents)
        .def("backward", &Var::backward)

        .def("__repr__", [](const Var& v) {
            return "Var(val=" + std::to_string(v.getVal()) + ", grad=" + std::to_string(v.getGrad()) + ")";
        });

    py::class_<Matrix>(m, "Matrix", R"doc(
A matrix of Var objects.
)doc")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
        
        .def_readonly("rows", &Matrix::rows)
        .def_readonly("cols", &Matrix::cols)

        .def("__getitem__", [](Matrix &M, int i) {
                int normalized_i = normalize_index(i, M.rows);
                if (normalized_i < 0 || normalized_i >= M.rows) throw std::out_of_range("Matrix row index " + std::to_string(i) + " out of range");

                Matrix row(1, M.cols);

                for (int j = 0; j < M.cols; j++) {
                    row(0, j) = M(normalized_i, j);
                }

                return row;
            })

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
            }, py::arg("index"), py::arg("value"))
        
        .def("__getitem__", [](Matrix &M, py::tuple idx) -> Var& {
                if (idx.size() != 2) throw std::runtime_error("Use M[i, j]");

                int i = idx[0].cast<int>();
                int j = idx[1].cast<int>();
                int normalized_i = normalize_index(i, M.rows);
                int normalized_j = normalize_index(j, M.cols);

                if (normalized_i < 0 || normalized_i >= M.rows || normalized_j < 0 || normalized_j >= M.cols) throw std::out_of_range("Matrix indices " + std::to_string(i) + " and " + std::to_string(j) + " out of range");

                return M(normalized_i, normalized_j);
            },
            py::return_value_policy::reference_internal)

        .def("__setitem__", [](Matrix &M, py::tuple idx, double v) {
                if (idx.size() != 2) throw std::runtime_error("Use M[i, j]");

                int i = idx[0].cast<int>();
                int j = idx[1].cast<int>();
                int normalized_i = normalize_index(i, M.rows);
                int normalized_j = normalize_index(j, M.cols);

                if (normalized_i < 0 || normalized_i >= M.rows || normalized_j < 0 || normalized_j >= M.cols) throw std::out_of_range("Matrix indices " + std::to_string(i) + " and " + std::to_string(j) + " out of range");
                    
                M(normalized_i, normalized_j) = Var(v);
            },
            py::arg("index"), py::arg("value"))
        .def("__setitem__", [](Matrix &M, py::tuple idx, const Var &v) {
                if (idx.size() != 2) throw std::runtime_error("Use M[i, j]");

                int i = idx[0].cast<int>();
                int j = idx[1].cast<int>();
                int normalized_i = normalize_index(i, M.rows);
                int normalized_j = normalize_index(j, M.cols);

                if (normalized_i < 0 || normalized_i >= M.rows || normalized_j < 0 || normalized_j >= M.cols) throw std::out_of_range("Matrix indices " + std::to_string(i) + " and " + std::to_string(j) + " out of range");

                M(normalized_i, normalized_j) = v;
            },
            py::arg("index"), py::arg("value"))

        .def("resetGradAndParents", &Matrix::resetGradAndParents)
        .def("randomInit", &Matrix::randomInit)

        .def("getValsMatrix", &Matrix::getValsMatrix)
        .def("getGradsMatrix", &Matrix::getGradsMatrix)

        .def("add", static_cast<Matrix (Matrix::*)(Matrix&)>(&Matrix::add), py::arg("other"))
        .def("__add__", [](Matrix &A, Matrix &B) { return A.add(B); }, py::is_operator(), py::arg("other"))

        .def("add", static_cast<Matrix (Matrix::*)(double)>(&Matrix::add), py::arg("other"))
        .def("__add__", [](Matrix &A, double s) { return A.add(s); }, py::is_operator(), py::arg("other"))
        .def("__radd__", [](Matrix &A, double s) { return A.add(s); }, py::is_operator(), py::arg("other"))

        .def("subtract", static_cast<Matrix (Matrix::*)(Matrix&)>(&Matrix::subtract), py::arg("other"))
        .def("__sub__", [](Matrix &A, Matrix &B) { return A.subtract(B); }, py::is_operator(), py::arg("other"))

        .def("subtract", static_cast<Matrix (Matrix::*)(double)>(&Matrix::subtract), py::arg("other"))
        .def("__sub__", [](Matrix &A, double s) { return A.subtract(s); }, py::is_operator(), py::arg("other"))

        .def("multiply", &Matrix::multiply, py::arg("other"))
        .def("__mul__", [](Matrix &A, double s) { return A.multiply(s); }, py::is_operator(), py::arg("other"))
        .def("__rmul__", [](Matrix &A, double s) { return A.multiply(s); }, py::is_operator(), py::arg("other"))

        .def("matmul", &Matrix::matmul, py::arg("other"))
        .def("__matmul__", [](Matrix &A, Matrix &B) { return A.matmul(B); }, py::is_operator(), py::arg("other"))

        .def("divide", &Matrix::divide, py::arg("other"))
        .def("__truediv__", [](Matrix &A, double s) { return A.divide(s); }, py::is_operator(), py::arg("other"))

        .def("pow", &Matrix::pow, py::arg("power"))
        .def("__pow__", [](Matrix &A, int p) { return A.pow(p); }, py::is_operator(), py::arg("power"))

        .def("sin", &Matrix::sin)
        .def("cos", &Matrix::cos)
        .def("tan", &Matrix::tan)
        .def("sec", &Matrix::sec)
        .def("csc", &Matrix::csc)
        .def("cot", &Matrix::cot)

        .def("log", &Matrix::log)

        .def("exp", &Matrix::exp)

        .def("abs", &Matrix::abs)

        .def("relu", &Matrix::relu)
        .def("leakyRelu", &Matrix::leakyRelu, py::arg("alpha") = 0.01)
        .def("tanh", &Matrix::tanh)
        .def("sigmoid", &Matrix::sigmoid)
        .def("silu", &Matrix::silu)
        .def("elu", &Matrix::elu, py::arg("alpha") = 1.0)
        .def("softmax", &Matrix::softmax)

        .def("__repr__", [](const Matrix &M) {
            return "Matrix(" + std::to_string(M.rows) + " x " + std::to_string(M.cols) + ") = \n" + M.getValsMatrix();
        });

    py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer", R"doc(
Base class for all layers.
)doc")
        .def_property_readonly("name", [](const Layer& layer) { return layer.name; })
        .def_property_readonly("trainable", [](const Layer& layer) { return layer.trainable; });

    py::class_<Linear, Layer, std::shared_ptr<Linear>>(m, "Linear", R"doc(
Linear layer
)doc")
        .def(py::init<int, int, std::string>(), py::arg("in_dim"), py::arg("out_dim"), py::arg("init") = "he")
        .def("forward", &Linear::forward, py::arg("input"))
        .def("optimizeWeights", &Linear::optimizeWeights, py::arg("learning_rate"))
        .def("resetGrad", &Linear::resetGrad)
        .def_readonly("W", &Linear::W)
        .def_readonly("b", &Linear::b);

    py::class_<ReLU, Layer, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward, py::arg("input"));

    py::class_<LeakyReLU, Layer, std::shared_ptr<LeakyReLU>>(m, "LeakyReLU")
        .def(py::init<double>(), py::arg("alpha"))
        .def("forward", &LeakyReLU::forward, py::arg("input"));

    py::class_<Sigmoid, Layer, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward", &Sigmoid::forward, py::arg("input"));

    py::class_<Tanh, Layer, std::shared_ptr<Tanh>>(m, "Tanh")
        .def(py::init<>())
        .def("forward", &Tanh::forward, py::arg("input"));

    py::class_<SiLU, Layer, std::shared_ptr<SiLU>>(m, "SiLU")
        .def(py::init<>())
        .def("forward", &SiLU::forward, py::arg("input"));

    py::class_<ELU, Layer, std::shared_ptr<ELU>>(m, "ELU")
        .def(py::init<double>(), py::arg("alpha"))
        .def("forward", &ELU::forward, py::arg("input"));

    py::class_<Softmax, Layer, std::shared_ptr<Softmax>>(m, "Softmax")
        .def(py::init<>())
        .def("forward", &Softmax::forward, py::arg("input"));

    py::class_<NeuralNetwork>(m, "NeuralNetwork", R"doc(
A simple feed-forward neural network built from Matrix layers.
)doc")
        .def(py::init<std::vector<std::shared_ptr<Layer>>>(), py::arg("layers"))

        .def("getLayers", py::overload_cast<>(&NeuralNetwork::getLayers, py::const_))
        .def_property_readonly("layers", py::overload_cast<>(&NeuralNetwork::getLayers, py::const_))

        .def("addLayer", &NeuralNetwork::addLayer, py::arg("layer"))
        .def("forward", &NeuralNetwork::forward, py::arg("input"))
        .def("getNetworkArchitecture", &NeuralNetwork::getNetworkArchitecture)

        .def("saveWeights", &NeuralNetwork::saveWeights, py::arg("path"))
        .def("loadWeights", &NeuralNetwork::loadWeights, py::arg("path"))
        
        .def("__repr__", [](const NeuralNetwork &model) {
            return "NeuralNetwork =\n" + model.getNetworkArchitecture();
        });

    py::class_<Optimizer>(m, "Optimizer", R"doc(
Base class for all optimizers for a NeuralNetwork.
)doc")
        .def("optimize", &Optimizer::optimize)
        .def("resetGrad", &Optimizer::resetGrad);

    py::class_<GradientDescentOptimizer, Optimizer>(m, "GradientDescentOptimizer", R"doc(
Simple gradient descent optimizer for a NeuralNetwork.
)doc")
        .def(py::init<double, NeuralNetwork*>(), py::arg("learning_rate"), py::arg("model"), py::keep_alive<1, 2>())
        .def("optimize", &GradientDescentOptimizer::optimize)
        .def("resetGrad", &GradientDescentOptimizer::resetGrad);

    py::class_<SGDOptimizer, Optimizer>(m, "SGDOptimizer", R"doc(
Stochastic gradient descent optimizer with momentum/weight decay.
)doc")
        .def(py::init<double, NeuralNetwork*, double, double>(), py::arg("learning_rate"), py::arg("model"), py::arg("momentum") = 0.0, py::arg("weight_decay") = 0.0, py::keep_alive<1, 2>())
        .def("optimize", &SGDOptimizer::optimize)
        .def("resetGrad", &SGDOptimizer::resetGrad);

    m.def("matmul", &matmul, py::arg("A"), py::arg("B"));
    m.def("MSELoss", &MSELoss, py::arg("labels"), py::arg("preds"));
    m.def("MAELoss", &MAELoss, py::arg("labels"), py::arg("preds"));
    m.def("BCELoss", &BCELoss, py::arg("labels"), py::arg("preds"), py::arg("eps") = 1e-7);
    m.def("CrossEntropyLoss", &CrossEntropyLoss, py::arg("labels"), py::arg("preds"), py::arg("eps") = 1e-9);
    m.def("CrossEntropyLossWithLogits", &CrossEntropyLossWithLogits, py::arg("labels"), py::arg("logits"), py::arg("eps") = 1e-9);

    py::module_ operations = m.def_submodule("operations");
    operations.def("sin", [](Var& v) { return v.sin(); }, py::arg("var"));
    operations.def("cos", [](Var& v) { return v.cos(); }, py::arg("var"));
    operations.def("tan", [](Var& v) { return v.tan(); }, py::arg("var"));
    operations.def("sec", [](Var& v) { return v.sec(); }, py::arg("var"));
    operations.def("csc", [](Var& v) { return v.csc(); }, py::arg("var"));
    operations.def("cot", [](Var& v) { return v.cot(); }, py::arg("var"));
    operations.def("log", [](Var& v) { return v.log(); }, py::arg("var"));
    operations.def("exp", [](Var& v) { return v.exp(); }, py::arg("var"));
    operations.def("abs", [](Var& v) { return v.abs(); }, py::arg("var"));
}
