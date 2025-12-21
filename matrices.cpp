#include <iostream>
#include "Matrix.hpp"

// g++ matrices.cpp Matrix.cpp Var.cpp -o matrices && ./matrices

int main() {
    Matrix mat(5, 5);

    std::string valsMatrix = mat.getValsMatrix();
    std::cout << valsMatrix << std::endl;

    std::string gradsMatrix = mat.getGradsMatrix();
    std::cout << gradsMatrix << std::endl;

    return 0;
}