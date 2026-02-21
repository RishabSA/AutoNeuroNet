#pragma once

#include <vector>
#include <utility>
#include <memory>
#include <cmath>

#include "NeuralNetwork.hpp"

class Optimizer {
public:
    double learning_rate;
    NeuralNetwork* neural_network;
    std::vector<Linear*> linear_layers;

    virtual ~Optimizer() = default;

    virtual void initializeOptimizer() = 0;
    virtual void optimize() = 0;
    void resetGrad();
};

class GradientDescentOptimizer : public Optimizer {
public:
    GradientDescentOptimizer(double lr, NeuralNetwork* model);

    void initializeOptimizer() override;
    void optimize() override;
};

class SGDOptimizer : public Optimizer {
public:
    double momentum, weight_decay;

    SGDOptimizer(double lr, NeuralNetwork* model, double momentum = 0.0, double weight_decay = 0.0);

    void initializeOptimizer() override;
    void optimize() override;

    struct Velocity {
        std::vector<std::vector<double>> W, b;
    };

    std::vector<Velocity> velocities;
};

class AdagradOptimizer : public Optimizer {
public:
    double epsilon;

    AdagradOptimizer(double lr, NeuralNetwork* model, double epsilon = 1e-8);

    void initializeOptimizer() override;
    void optimize() override;

    struct Accumulator {
        std::vector<std::vector<double>> W, b;
    };

    std::vector<Accumulator> accumulators;
};

class RMSPropOptimizer : public Optimizer {
public:
    double decay_rate, epsilon;

    RMSPropOptimizer(double lr, NeuralNetwork* model, double decay_rate = 0.9, double epsilon = 1e-8);

    void initializeOptimizer() override;
    void optimize() override;

    struct Accumulator {
        std::vector<std::vector<double>> W, b;
    };

    std::vector<Accumulator> accumulators;
};

class AdamOptimizer : public Optimizer {
public:
    double beta1, beta2, epsilon;

    AdamOptimizer(double lr, NeuralNetwork* model, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

    void initializeOptimizer() override;
    void optimize() override;

    struct Moments {
        std::vector<std::vector<double>> mW, vW, mb, vb;
    };

    int timestep = 0;
    std::vector<Moments> moments;
};

class AdamWOptimizer : public Optimizer {
public:
    double beta1, beta2, epsilon, weight_decay;

    AdamWOptimizer(double lr, NeuralNetwork* model, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double weight_decay = 0.0);

    void initializeOptimizer() override;
    void optimize() override;

    struct Moments {
        std::vector<std::vector<double>> mW, vW, mb, vb;
    };

    int timestep = 0;
    std::vector<Moments> moments;
};
