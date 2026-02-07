#pragma once

#include <vector>
#include <string>
#include <random>
#include "Var.hpp"

class Matrix {
public:
    int rows, cols;
    std::vector<std::vector<Var>> data;

    Matrix(int r, int c);

    Var& operator()(int row, int col) {
        return data[row][col];
    };

    void resetGradAndParents();

    std::string getValsMatrix() const;
    std::string getGradsMatrix() const;

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
