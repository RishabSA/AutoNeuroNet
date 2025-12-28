#pragma once

#include <vector>
#include <utility>
#include "Matrix.hpp"

class NeuralNetwork {
public:
    std::vector<std::pair<Matrix, Matrix>> layers;

    NeuralNetwork(std::vector<std::pair<int, int>> l) {
        // Input: vector of layers, each of (in_dim, out_dim)

        for (auto i : l) {
            int in_dim = i.first;
            int out_dim = i.second;

            Matrix W = Matrix(in_dim, out_dim);
            W.randomInit();

            Matrix b = Matrix(1, out_dim);

            layers.emplace_back(W, b);
        }
    };

    std::pair<const Matrix&, const Matrix&> getLayer(int idx) { return layers.at(idx); };

    void addLayer(std::pair<int, int> l);

    Matrix forward(const Matrix& input);

    std::string getNetworkArchitecture();
};

Var computeMSELoss(Matrix& labels, Matrix& preds);