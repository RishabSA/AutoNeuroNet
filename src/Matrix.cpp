#include "Matrix.hpp"

Matrix::Matrix() {
    rows = 0;
    cols = 0;
};

Matrix::Matrix(int r, int c) {
    rows = r;
    cols = c;

    data.resize(r);
    for (int i = 0; i < r; ++i) {
        data[i].resize(c);
    }
};

void Matrix::resetGradAndParents() {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j].resetGradAndParents();
        }
    }
}

std::string Matrix::getValsMatrix() const {
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

std::string Matrix::getGradsMatrix() const {
    std::string out = "";

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out += std::to_string(data[i][j].getGrad());
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

    if (rows == other.rows && cols == other.cols) {
        // Add element-wise values for matrices with the same shape
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Y.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
    } else if (other.rows == 1 && other.cols == 1) {
        // Broadcast the scalar for matrix addition when the other has shape (1, 1)
        Var& val = other.data[0][0];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Y.data[i][j] = data[i][j] + val;
            }
        }
    } else if (other.rows == 1 && other.cols == cols) {
        // Broadcast a row vector across rows
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Y.data[i][j] = data[i][j] + other.data[0][j];
            }
        }
    } else if (other.cols == 1 && other.rows == rows) {
        // Broadcast a column vector across columns
        for (int i = 0; i < rows; i++) {
            Var& val = other.data[i][0];
            for (int j = 0; j < cols; j++) {
                Y.data[i][j] = data[i][j] + val;
            }
        }
    } else {
        throw std::runtime_error("Dimension mismatch when attempting to add matrices - (" + std::to_string(rows) + ", " + std::to_string(cols) + ") + (" + std::to_string(other.rows) + ", " + std::to_string(other.cols) + ")");
    }

    return Y;
};

Matrix Matrix::add(double other) {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j] + other;
        }
    }

    return Y;
};

Matrix Matrix::subtract(Matrix& other) {
    Matrix Y(rows, cols);

    if (rows == other.rows && cols == other.cols) {
        // Add element-wise values for matrices with the same shape
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Y.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
    } else if (other.rows == 1 && other.cols == 1) {
        // Broadcast the scalar for matrix addition when the other has shape (1, 1)
        Var& val = other.data[0][0];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Y.data[i][j] = data[i][j] - val;
            }
        }
    } else if (other.rows == 1 && other.cols == cols) {
        // Broadcast a row vector across rows
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Y.data[i][j] = data[i][j] - other.data[0][j];
            }
        }
    } else if (other.cols == 1 && other.rows == rows) {
        // Broadcast a column vector across columns
        for (int i = 0; i < rows; i++) {
            Var& val = other.data[i][0];
            for (int j = 0; j < cols; j++) {
                Y.data[i][j] = data[i][j] - val;
            }
        }
    } else {
        throw std::runtime_error("Dimension mismatch when attempting to add matrices - (" + std::to_string(rows) + ", " + std::to_string(cols) + ") + (" + std::to_string(other.rows) + ", " + std::to_string(other.cols) + ")");
    }

    return Y;
};

Matrix Matrix::subtract(double other) {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j] - other;
        }
    }

    return Y;
};

Matrix Matrix::multiply(double other) {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j] * other;
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
            Y.data[i][j] = data[i][j] / other;
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

Matrix Matrix::sin() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].sin();
        }
    }

    return Y;
};


Matrix Matrix::cos() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].cos();
        }
    }

    return Y;
};

Matrix Matrix::tan() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].tan();
        }
    }

    return Y;
};

Matrix Matrix::sec() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].sec();
        }
    }

    return Y;
};

Matrix Matrix::csc() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].csc();
        }
    }

    return Y;
};

Matrix Matrix::cot() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].cot();
        }
    }

    return Y;
};

Matrix Matrix::log() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].log();
        }
    }

    return Y;
};

Matrix Matrix::exp() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].exp();
        }
    }

    return Y;
};

Matrix Matrix::abs() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].abs();
        }
    }

    return Y;
};

Matrix Matrix::relu() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].relu();
        }
    }

    return Y;
};

Matrix Matrix::leakyRelu(double alpha) {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].leakyRelu(alpha);
        }
    }

    return Y;
};

Matrix Matrix::sigmoid() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].sigmoid();
        }
    }

    return Y;
};

Matrix Matrix::tanh() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].tanh();
        }
    }

    return Y;
};

Matrix Matrix::silu() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].silu();
        }
    }

    return Y;
};

Matrix Matrix::elu(double alpha) {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = data[i][j].elu(alpha);
        }
    }

    return Y;
};

Matrix Matrix::softmax() {
    Matrix Y(rows, cols);

    for (int i = 0; i < rows; i++) {
        double row_max = data[i][0].getVal();
        for (int j = 1; j < cols; j++) {
            row_max = std::max(row_max, data[i][j].getVal());
        }

        Var sum(0.0);
        for (int j = 0; j < cols; j++) {
            Var e = (data[i][j] - row_max).exp();
            Y.data[i][j] = e;
            sum = sum + e;
        }

        for (int j = 0; j < cols; j++) {
            Y.data[i][j] = Y.data[i][j] / sum;
        }
    }
    return Y;
}