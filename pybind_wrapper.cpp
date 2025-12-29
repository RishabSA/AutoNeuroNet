#include <pybind11/pybind11.h>
#include "include/Var.hpp"

namespace py = pybind11;

PYBIND11_MODULE(Var, m) {
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

        .def("getGradVal", &Var::getGradVal)
        .def("setGradVal", &Var::setGradVal, py::arg("v"))
        .def_property("gradVal", &Var::getGradVal, &Var::setGradVal)

        .def("add", &Var::add, py::arg("other"))
        .def("subtract", &Var::subtract, py::arg("other"))
        .def("multiply", &Var::multiply, py::arg("other"))
        .def("divide", &Var::divide, py::arg("other"))
        .def("pow", &Var::pow, py::arg("power"))

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

        // Operator overloads
        .def("__add__", [](Var &a, Var &b) { return a.add(b); }, py::is_operator())
        .def("__sub__", [](Var &a, Var &b) { return a.subtract(b); }, py::is_operator())
        .def("__mul__", [](Var &a, Var &b) { return a.multiply(b); }, py::is_operator())
        .def("__truediv__", [](Var &a, Var &b) { return a.divide(b); }, py::is_operator())
        .def("__pow__", [](Var &a, int p) { return a.pow(p); }, py::is_operator(), py::arg("power"))
        .def("__neg__", [](Var &a) { Var neg(-1.0); return a.multiply(neg); }, py::is_operator())

        .def("__repr__", [](const Var& v) {
            return "Var(val=" + std::to_string(v.getVal()) + ", grad=" + std::to_string(v.getGradVal()) + ")";
        });
}