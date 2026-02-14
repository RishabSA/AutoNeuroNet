#include <iostream>
#include "Var.hpp"

// g++ demos/automatic_differentiation.cpp src/Var.cpp -I include -o demos/automatic_differentiation && ./demos/automatic_differentiation

int main () {
    // Reverse-Mode Automatic Differentiation

    Var x0(5.0);
    Var x1(10.0);

    Var z = x0.pow(2);
    Var y = x1 * z;
    y.setGrad(1.0);
    y.backward();

    std::cout << "y = " << y.getVal() << std::endl; // 250

    std::cout << "∂f/∂x_0 = " << x0.getGrad() << std::endl; // 100
    std::cout << "∂f/∂x_1 = " << x1.getGrad() << std::endl; // 25

    return 0;
}