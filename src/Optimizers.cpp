#include "Optimizers.hpp"

GradientDescentOptimizer::GradientDescentOptimizer(double lr, NeuralNetwork* model) {
    learning_rate = lr;
    neural_network = model;
};

void GradientDescentOptimizer::optimizeModelWeights() {
    // Backpropagation and Gradient Descent for each parameter
    for (auto& layer :neural_network->layers) {
        if (layer->trainable) {
            layer->optimizeWeights(learning_rate);
        }
    }
};

void GradientDescentOptimizer::resetGrad() {
    // Reset gradients and the old graph on everything
    for (auto& layer : neural_network->layers) {
        if (layer->trainable) {
            layer->resetGrad();
        }
    }
}
