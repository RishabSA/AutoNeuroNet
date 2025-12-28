#include <iostream>
#include "Matrix.hpp"

// g++ matrices.cpp src/Var.cpp src/Matrix.cpp -I include -o matrices && ./matrices

int main() {
    Matrix mat(5, 5);

    std::string valsMatrix = mat.getValsMatrix();
    std::cout << valsMatrix << std::endl;

    std::string gradsMatrix = mat.getGradsMatrix();
    std::cout << gradsMatrix << std::endl;

    return 0;
}