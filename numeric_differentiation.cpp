#include <iostream>
#include <cmath>

double quadratic(double x) {
    // x^2
    return pow(x, 2);
}

double forward_numeric_differentiation(double (*f)(double), double x) {
    // Limit definition of the derivative: lim_{h -> 0}(f(x + h) - f(x) / h)
    
    double h = 1e-8;
    return (f(x + h) - f(x)) / h;
}

double central_numeric_differentiation(double (*f)(double), double x) {
    // Central difference derivative: lim_{h -> 0}((f(x + h) - f(x - h)) / 2h)

    double h = 1e-5;
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

int main() {
    double x = 5;

    std::cout << "f(" << x << ") = " << quadratic(x) << std::endl;
    std::cout << "forward f'(" << x << ") = " << forward_numeric_differentiation(quadratic, x) << std::endl;
    std::cout << "central f'(" << x << ") = " << central_numeric_differentiation(quadratic, x) << std::endl;

    return 0;
}