#pragma once

#include <vector>
#include <string>
#include <random>
#include "Var.hpp"

class Matrix {
public:
    int rows, cols;
    std::vector<std::vector<Var>> data;

    Matrix(int r, int c) {
        rows = r;
        cols = c;

        data = std::vector<std::vector<Var>>(
            r, std::vector<Var>(c, Var(0.0))
        );
    }

    std::string getValsMatrix();

    std::string getGradsMatrix();

    void randomInit();

    Matrix add(Matrix& other);
    Matrix operator+(Matrix& other) { return add(other); };

    Matrix add(double other);
    Matrix operator+(double other) { return add(other); };

    Matrix subtract(Matrix& other);
    Matrix operator-(Matrix& other) { return subtract(other); };

    Matrix subtract(double other);
    Matrix operator-(double other) { return subtract(other); };

    Matrix multiply(double other);
    Matrix operator*(double other) { return multiply(other); };

    Matrix matmul(Matrix& other);

    Matrix divide(double other);
    Matrix operator/(double other) { return divide(other); };

    Matrix pow(int power);
};

Matrix matmul(Matrix& X0, Matrix& X1);