#include "NeuralNetwork.hpp"

void NeuralNetwork::addLayer(std::pair<int, int> l) {
    // Input: layer with (in_dim, out_dim)
    int in_dim = l.first;
    int out_dim = l.second;

    Matrix W = Matrix(in_dim, out_dim);
    W.randomInit();

    Matrix b = Matrix(1, out_dim);

    layers.emplace_back(W, b);
};

Matrix NeuralNetwork::forward(const Matrix& input) {
    Matrix output = input;

    for (auto& layer : layers) {
        Matrix& W = layer.first;
        Matrix& b = layer.second;

        output = matmul(output, W) + b;
    }

    return output;
};

Var computeMSELoss(Matrix& labels, Matrix& preds) {
    if (labels.rows != preds.rows || labels.cols != preds.cols) {
        throw std::runtime_error("Dimension mismatch when attempting to compute loss");
    }

    Var loss(0.0);
    int N = 0;

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            Var errors = labels.data[i][j] - preds.data[i][j];
            Var squared_errors = errors.pow(2);
            loss = loss + squared_errors; 
            N++;
        }
    }

    Var total(N);
    loss = loss / total;

    return loss;
};