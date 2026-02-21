#pragma once

#include <vector>
#include <utility>
#include <cctype>
#include <cmath>
#include <random>
#include <fstream>
#include <stdexcept>

#include "Matrix.hpp"

class Layer {
public:
    std::string name;
    bool trainable;

    virtual ~Layer() = default;

    virtual Matrix forward(Matrix& input) = 0;

    virtual void resetGrad() = 0;
};

class Linear : public Layer {
public:
    Matrix W, b;

    // Weight initialization can be done with "kaiming"/"he" or "xavier"/"glorot"
    Linear(int inDim, int outDim, const std::string& init = "kaiming");

    Matrix forward(Matrix& input) override;

    void resetGrad() override;
};

class ReLU : public Layer {
public:
    ReLU();
    
    Matrix forward(Matrix& input) override;

    // Do not do anything
    void resetGrad() override;
};

class LeakyReLU : public Layer {
public:
    double alpha;

    LeakyReLU(double a);
    
    Matrix forward(Matrix& input) override;

    // Do not do anything
    void resetGrad() override;
};

class Sigmoid : public Layer {
public:
    Sigmoid();
    
    Matrix forward(Matrix& input) override;

    // Do not do anything
    void resetGrad() override;
};

class Tanh : public Layer {
public:
    Tanh();
    
    Matrix forward(Matrix& input) override;

    // Do not do anything
    void resetGrad() override;
};

class SiLU : public Layer {
public:
    SiLU();
    
    Matrix forward(Matrix& input) override;

    // Do not do anything
    void resetGrad() override;
};

class ELU : public Layer {
public:
    double alpha;

    ELU(double a);
    
    Matrix forward(Matrix& input) override;

    // Do not do anything
    void resetGrad() override;
};

class Softmax : public Layer {
public:
    Softmax();
    
    Matrix forward(Matrix& input) override;

    // Do not do anything
    void resetGrad() override;
};

class NeuralNetwork {
public:
    std::vector<std::shared_ptr<Layer>> layers;

    NeuralNetwork(std::vector<std::shared_ptr<Layer>> network);

    std::vector<std::shared_ptr<Layer>> getLayers();
    const std::vector<std::shared_ptr<Layer>> getLayers() const;

    void addLayer(std::shared_ptr<Layer> layer);

    Matrix forward(Matrix input);

    std::string getNetworkArchitecture() const;

    void saveWeights(const std::string& path);
    void loadWeights(const std::string& path);
};

void initWeights(Matrix& W, int inDim, int outDim, const std::string& init);