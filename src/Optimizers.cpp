#include "Optimizers.hpp"

Optimizer::~Optimizer() = default;

void Optimizer::resetGrad() {
    // Reset gradients and the old graph on everything
    for (auto& layer : neural_network->layers) {
        if (layer->trainable) {
            layer->resetGrad();
        }
    }
}

GradientDescentOptimizer::GradientDescentOptimizer(double lr, NeuralNetwork* model) {
    learning_rate = lr;
    neural_network = model;
};

void GradientDescentOptimizer::optimize() {
    // Backpropagation and Gradient Descent for each parameter
    for (auto& layer : neural_network->layers) {
        if (layer->trainable) {
            layer->optimizeWeights(learning_rate);
        }
    }
};

void SGDOptimizer::initVelocities() {
    linear_layers.clear();
    velocities.clear();

    for (auto& layer : neural_network->layers) {
        auto linear = std::dynamic_pointer_cast<Linear>(layer);
        if (!linear) {
            continue;
        }

        linear_layers.push_back(linear.get());

        Velocity v;
        v.W.assign(linear->W.rows, std::vector<double>(linear->W.cols, 0.0));
        v.b.assign(linear->b.rows, std::vector<double>(linear->b.cols, 0.0));
        velocities.push_back(std::move(v));
    }

    velocities_initialized = true;
}

SGDOptimizer::SGDOptimizer(double lr, NeuralNetwork* model, double momentum, double weight_decay) {
    learning_rate = lr;
    neural_network = model;
    this->momentum = momentum;
    this->weight_decay = weight_decay;
};

void SGDOptimizer::optimize() {
    if (!velocities_initialized) {
        initVelocities();
    }

    for (size_t idx = 0; idx < linear_layers.size(); idx++) {
        Linear* layer = linear_layers[idx];
        Velocity& v = velocities[idx];

        // Update W
        for (int i = 0; i < layer->W.rows; i++) {
            for (int j = 0; j < layer->W.cols; j++) {
                Var& weight_param = layer->W.data[i][j];
                double grad = weight_param.getGrad();

                if (weight_decay != 0.0) {
                    grad += weight_decay * weight_param.getVal();
                }

                if (momentum != 0.0) {
                    v.W[i][j] = momentum * v.W[i][j] + grad;
                    grad = v.W[i][j];
                }

                weight_param.setVal(weight_param.getVal() - learning_rate * grad);
            }
        }

        // Update b (no weight decay by default)
        for (int i = 0; i < layer->b.rows; i++) {
            for (int j = 0; j < layer->b.cols; j++) {
                Var& bias_param = layer->b.data[i][j];
                double grad = bias_param.getGrad();

                if (momentum != 0.0) {
                    v.b[i][j] = momentum * v.b[i][j] + grad;
                    grad = v.b[i][j];
                }

                bias_param.setVal(bias_param.getVal() - learning_rate * grad);
            }
        }
    }
};