#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "include/Var.hpp"
#include "include/Matrix.hpp"
#include "include/NeuralNetwork.hpp"

namespace py = pybind11;

PYBIND11_MODULE(autodiff, m) {
    m.doc() = "Reverse-mode automatic differentiation scalar variable.";

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

        .def("__getitem__", [](Matrix &M, py::tuple idx) -> Var& {
                if (idx.size() != 2) throw std::runtime_error("Use M[i, j]");

                int i = idx[0].cast<int>();
                int j = idx[1].cast<int>();

                if (i < 0 || i >= M.rows || j < 0 || j >= M.cols)
                    throw std::out_of_range("Matrix index out of range");

                return M(i, j);
            },
            py::return_value_policy::reference_internal)

        .def("__setitem__", [](Matrix &M, py::tuple idx, double v) {
                if (idx.size() != 2) throw std::runtime_error("Use M[i, j]");

                int i = idx[0].cast<int>();
                int j = idx[1].cast<int>();

                if (i < 0 || i >= M.rows || j < 0 || j >= M.cols)
                    throw std::out_of_range("Matrix index out of range");
                    
                M(i, j) = Var(v);
            },
            py::arg("index"), py::arg("value"))
        .def("__setitem__", [](Matrix &M, py::tuple idx, const Var &v) {
                if (idx.size() != 2) throw std::runtime_error("Use M[i, j]");

                int i = idx[0].cast<int>();
                int j = idx[1].cast<int>();

                if (i < 0 || i >= M.rows || j < 0 || j >= M.cols)
                    throw std::out_of_range("Matrix index out of range");

                M(i, j) = v;
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

        .def("__repr__", [](const Matrix &M) {
            return "Matrix(" + std::to_string(M.rows) + " x " + std::to_string(M.cols) + ") = \n" + M.getValsMatrix();
        });

    py::class_<NeuralNetwork>(m, "NeuralNetwork", R"doc(
A simple feed-forward neural network built from Matrix layers.
)doc")
        .def(py::init<std::vector<std::pair<int, int>>>(), py::arg("layers"))

        .def("getLayers", &NeuralNetwork::getLayers)
        .def_property_readonly("layers", &NeuralNetwork::getLayers)

        .def("addLayer", &NeuralNetwork::addLayer, py::arg("layer"))
        .def("forward", &NeuralNetwork::forward, py::arg("input"))
        .def("getNetworkArchitecture", &NeuralNetwork::getNetworkArchitecture)
        
        .def("__repr__", [](const NeuralNetwork &model) {
            return "NeuralNetwork =\n" + model.getNetworkArchitecture();
        });

    m.def("matmul", &matmul, py::arg("A"), py::arg("B"));
    m.def("computeMSELoss", &computeMSELoss, py::arg("labels"), py::arg("preds"));
}
