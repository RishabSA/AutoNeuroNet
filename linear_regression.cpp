#include <iostream>
#include "autodiff/Matrix.hpp"

// g++ linear_regression.cpp src/autodiff/Var.cpp src/autodiff/Matrix.cpp -I include -o linear_regression && ./linear_regression

int main () {
    int out_dim = 1;
    int in_dim = 2;

    int N = 10;

    Matrix x(N, in_dim);
    Matrix W(out_dim, in_dim);
    Matrix b(out_dim, 1);

    Matrix MSELoss(1, 1);

    x.randomInit();
    W.randomInit();

    Matrix y = matmul(W, x) + b;

    std::string linearOutputMatrix = y.getValsMatrix();
    std::cout << linearOutputMatrix << std::endl;

    return 0;
}