#pragma once

#include <vector>
#include <utility>
#include <memory>

#include "NeuralNetwork.hpp"

class Optimizer {
public:
    double learning_rate;
    NeuralNetwork* neural_network;

    virtual ~Optimizer();

    virtual void optimize() = 0;
    void resetGrad();
};

class GradientDescentOptimizer : public Optimizer {
public:
    GradientDescentOptimizer(double lr, NeuralNetwork* model);

    void optimize() override;
};

class SGDOptimizer : public Optimizer {
public:
    double momentum;
    double weight_decay;

    SGDOptimizer(double lr, NeuralNetwork* model, double momentum = 0.0, double weight_decay = 0.0);

    void optimize() override;

private:
    struct Velocity {
        std::vector<std::vector<double>> W;
        std::vector<std::vector<double>> b;
    };

    bool velocities_initialized = false;
    std::vector<Linear*> linear_layers;
    std::vector<Velocity> velocities;

    void initVelocities();
};
