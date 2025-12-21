#include <iostream>
#include "autodiff/Matrix.hpp"

// g++ matrices.cpp src/autodiff/Var.cpp src/autodiff/Matrix.cpp -I include -o matrices && ./matrices

int main() {
    Matrix mat(5, 5);

    std::string valsMatrix = mat.getValsMatrix();
    std::cout << valsMatrix << std::endl;

    std::string gradsMatrix = mat.getGradsMatrix();
    std::cout << gradsMatrix << std::endl;

    return 0;
}