#include <iostream>
#include "Var.hpp"

// g++ automatic_differentiation.cpp Var.cpp -o automatic_differentiation && ./automatic_differentiation

int main () {
    // Reverse-Mode Automatic Differentiation

    Var x0(5.0);
    Var x1(10.0);

    Var z = x0.pow(2);
    Var y = x1 * z;
    y.setGradVal(1.0);

    std::cout << "f(x) = " << y.getVal() << std::endl; // 250

    std::cout << "df/dx = " << x0.grad() << std::endl; // 100
    std::cout << "df/dy = " << x1.grad() << std::endl; // 25

    return 0;
}