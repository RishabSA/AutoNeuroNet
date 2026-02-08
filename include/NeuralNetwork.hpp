#pragma once

#include <vector>
#include <utility>
#include "Matrix.hpp"

class Layer {
public:
    std::string name;
    bool trainable = false;

    virtual ~Layer() = default;

    virtual Matrix forward(Matrix& input) = 0;

    virtual void optimizeWeights(double learning_rate) = 0;
    virtual void resetGrad() = 0;
};

class Linear : public Layer {
public:
    Matrix W;
    Matrix b;

    Linear(int inDim, int outDim);

    Matrix forward(Matrix& input) override;

    void optimizeWeights(double learning_rate) override;
    void resetGrad() override;
};

class ReLU : public Layer {
public:
    ReLU();
    
    Matrix forward(Matrix& input) override;
    void optimizeWeights(double learning_rate) override;
    void resetGrad() override;
};

class LeakyReLU : public Layer {
public:
    double alpha;

    LeakyReLU(double a);
    
    Matrix forward(Matrix& input) override;
    void optimizeWeights(double learning_rate) override;
    void resetGrad() override;
};

class Sigmoid : public Layer {
public:
    Sigmoid();
    
    Matrix forward(Matrix& input) override;
    void optimizeWeights(double learning_rate) override;
    void resetGrad() override;
};

class Tanh : public Layer {
public:
    Tanh();
    
    Matrix forward(Matrix& input) override;
    void optimizeWeights(double learning_rate) override;
    void resetGrad() override;
};

class SiLU : public Layer {
public:
    SiLU();
    
    Matrix forward(Matrix& input) override;
    void optimizeWeights(double learning_rate) override;
    void resetGrad() override;
};

class ELU : public Layer {
public:
    double alpha;

    ELU(double a);
    
    Matrix forward(Matrix& input) override;
    void optimizeWeights(double learning_rate) override;
    void resetGrad() override;
};

class Softmax : public Layer {
public:
    Softmax();
    
    Matrix forward(Matrix& input) override;
    void optimizeWeights(double learning_rate) override;
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
};
