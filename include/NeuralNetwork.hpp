#pragma once

#include <vector>
#include <utility>
#include "Matrix.hpp"

class NeuralNetwork {
public:
    std::vector<std::pair<Matrix, Matrix>> layers;

    std::vector<std::pair<Matrix, Matrix>> getLayers() const;

    NeuralNetwork(std::vector<std::pair<int, int>> l);

    std::pair<const Matrix&, const Matrix&> getLayer(int idx) { return layers.at(idx); };

    void addLayer(std::pair<int, int> l);

    Matrix forward(const Matrix& input);

    std::string getNetworkArchitecture() const;
};

Var computeMSELoss(Matrix& labels, Matrix& preds);