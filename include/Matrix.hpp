#pragma once

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <string>
#include <stdexcept>

#include "Var.hpp"

/*
Matrix is a 2D container of differentiable Var objects.

All scalar operations on Var can also be done on Matrix by applying them element-wise to each variable in a matrix.
*/
class Matrix {
public:
    int rows, cols;
    bool requires_grad;
    std::vector<std::vector<Var>> data;

    Matrix();

    Matrix(int r, int c, bool requires_grad = true);

    Var& operator()(int row, int col) {
        return data[row][col];
    };

    void resetGradAndParents();
    void noGrad();
    Matrix detach() const;

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

    // Trigonometric functions
    Matrix sin();
    Matrix cos();
    Matrix tan();
    Matrix sec();
    Matrix csc();
    Matrix cot();

    Matrix log();

    Matrix exp();

    Matrix abs();

    // Activation functions
    Matrix relu();
    Matrix leakyRelu(double alpha = 0.01);
    Matrix sigmoid();
    Matrix tanh();
    Matrix silu();
    Matrix elu(double alpha = 1.0);
    Matrix softmax();
};

Matrix matmul(Matrix& A, Matrix& B);
