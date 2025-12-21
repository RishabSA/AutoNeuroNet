#include "Matrix.hpp"

std::string Matrix::getValsMatrix() {
    std::string out;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out += std::to_string(data[i][j].getVal());
            out += " ";
        }
        out += "\n";
    }

    return out;
};

std::string Matrix::getGradsMatrix() {
    std::string out = "";

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out += std::to_string(data[i][j].getGradVal());
            out += " ";
        }
        out += "\n";
    }

    return out;
};

void Matrix::randomInit() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> unif(-0.01, 0.01);

            data[i][j] = unif(gen);
        }
    }
};

Matrix Matrix::add(Matrix& other) {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j] + other.data[i][j];
        }
    }

    return Y;
};

Matrix Matrix::add(double other) {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Var val(other);
            Y.data[i][j] = data[i][j] + val;
        }
    }

    return Y;
};

Matrix Matrix::subtract(Matrix& other) {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j] - other.data[i][j];
        }
    }

    return Y;
};

Matrix Matrix::subtract(double other) {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Var val(other);
            Y.data[i][j] = data[i][j] - val;
        }
    }

    return Y;
};

Matrix Matrix::multiply(double other) {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Var val(other);
            Y.data[i][j] = data[i][j] * val;
        }
    }

    return Y;
};

Matrix Matrix::matmul(Matrix& other) {
    Matrix Y(rows, other.cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            Var sum(0.0);

            for (int t = 0; t < cols; t++) {
                Var current(data[i][t] * other.data[t][j]);
                sum = sum + current;
            }
            Y.data[i][j] = sum;
        }
    }

    return Y;
};

Matrix matmul(Matrix& X0, Matrix& X1) {
    Matrix Y(X0.rows, X1.cols);

    for (int i = 0; i < X0.rows; i++) {
        for (int j = 0; j < X1.cols; j++) {
            Var sum(0.0);

            for (int t = 0; t < X0.cols; t++) {
                Var current(X0.data[i][t] * X1.data[t][j]);
                sum = sum + current;
            }
            Y.data[i][j] = sum;
        }
    }

    return Y;
};

Matrix Matrix::divide(double other) {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Var val(other);
            Y.data[i][j] = data[i][j] / val;
        }
    }

    return Y;
};

Matrix Matrix::pow(int power) {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].pow(power);
        }
    }

    return Y;
};