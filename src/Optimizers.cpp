#include "Optimizers.hpp"

void Optimizer::resetGrad() {
    // Reset gradients and the old graph on everything
    for (std::shared_ptr<Layer> layer : neural_network->layers) {
        if (layer->trainable) {
            layer->resetGrad();
        }
    }
};

GradientDescentOptimizer::GradientDescentOptimizer(double lr, NeuralNetwork* model) {
    learning_rate = lr;
    neural_network = model;

    initializeOptimizer();
};

void GradientDescentOptimizer::initializeOptimizer() {
    linear_layers.clear();

    for (std::shared_ptr<Layer> layer : neural_network->layers) {
        std::shared_ptr<Linear> linear_layer = std::dynamic_pointer_cast<Linear>(layer);
        if (!layer->trainable || !linear_layer) continue;

        linear_layers.push_back(linear_layer.get());
    }
};

void GradientDescentOptimizer::optimize() {
    // Gradient Descent optimization for each parameter
    // Moves parameters in the direction of their negative gradient

    for (int layerIdx = 0; layerIdx < linear_layers.size(); layerIdx++) {
        Linear* linear_layer = linear_layers[layerIdx];

        // Update W
        for (int i = 0; i < linear_layer->W.rows; i++) {
            for (int j = 0; j < linear_layer->W.cols; j++) {
                Var& weight_param = linear_layer->W.data[i][j];

                // Partial derivative of the loss function with respect to the weight parameter (∂L/∂W)
                double grad = weight_param.getGrad();

                // W_t = W_{t - 1} - (lr * ∂L/∂W)
                weight_param.setVal(weight_param.getVal() - learning_rate * grad);
            }
        }

        // Update b
        for (int i = 0; i < linear_layer->b.rows; i++) {
            for (int j = 0; j < linear_layer->b.cols; j++) {
                Var& bias_param = linear_layer->b.data[i][j];

                // Partial derivative of the loss function with respect to the bias parameter (∂L/∂b)
                double grad = bias_param.getGrad();

                // b_t = b_{t - 1} - (lr * ∂L/∂b)
                bias_param.setVal(bias_param.getVal() - learning_rate * grad);
            }
        }
    }
};

SGDOptimizer::SGDOptimizer(double lr, NeuralNetwork* model, double momentum, double weight_decay) {
    learning_rate = lr;
    neural_network = model;
    this->momentum = momentum;
    this->weight_decay = weight_decay;

    initializeOptimizer();
};

void SGDOptimizer::initializeOptimizer() {
    linear_layers.clear();
    velocities.clear();

    for (std::shared_ptr<Layer> layer : neural_network->layers) {
        std::shared_ptr<Linear> linear_layer = std::dynamic_pointer_cast<Linear>(layer);
        if (!layer->trainable || !linear_layer) continue;

        linear_layers.push_back(linear_layer.get());

        Velocity velocity;
        velocity.W.assign(linear_layer->W.rows, std::vector<double>(linear_layer->W.cols, 0.0));
        velocity.b.assign(linear_layer->b.rows, std::vector<double>(linear_layer->b.cols, 0.0));

        velocities.push_back(std::move(velocity));
    }
};

void SGDOptimizer::optimize() {
    // Stochastic Gradient Descent (SGD) with momentum optimization for each parameter
    // Momentum smooths and accelerates convergence by accumulating a velocity vector

    for (int layerIdx = 0; layerIdx < linear_layers.size(); layerIdx++) {
        Linear* linear_layer = linear_layers[layerIdx];
        Velocity& velocity = velocities[layerIdx];

        // Update W
        for (int i = 0; i < linear_layer->W.rows; i++) {
            for (int j = 0; j < linear_layer->W.cols; j++) {
                Var& weight_param = linear_layer->W.data[i][j];

                // Partial derivative of the loss function with respect to the weight parameter (∂L/∂W)
                double grad = weight_param.getGrad();

                if (weight_decay != 0.0) {
                    grad += weight_decay * weight_param.getVal();
                }

                if (momentum != 0.0) {
                    // v_W_t = momentum * v_W_{t - 1} + grad_t
                    velocity.W[i][j] = momentum * velocity.W[i][j] + grad;
                    grad = velocity.W[i][j];
                }

                // W_t = W_{t - 1} - (lr * ∂L/∂W)
                weight_param.setVal(weight_param.getVal() - learning_rate * grad);
            }
        }

        // Update b
        for (int i = 0; i < linear_layer->b.rows; i++) {
            for (int j = 0; j < linear_layer->b.cols; j++) {
                Var& bias_param = linear_layer->b.data[i][j];

                // Partial derivative of the loss function with respect to the bias parameter (∂L/∂b)
                double grad = bias_param.getGrad();

                if (momentum != 0.0) {
                    // v_b_t = momentum * v_b_{t - 1} + grad_t
                    velocity.b[i][j] = momentum * velocity.b[i][j] + grad;
                    grad = velocity.b[i][j];
                }

                // b_t = b_{t - 1} - (lr * ∂L/∂b)
                bias_param.setVal(bias_param.getVal() - learning_rate * grad);
            }
        }
    }
};

AdagradOptimizer::AdagradOptimizer(double lr, NeuralNetwork* model, double epsilon) {
    learning_rate = lr;
    neural_network = model;
    this->epsilon = epsilon;

    initializeOptimizer();
};

void AdagradOptimizer::initializeOptimizer() {
    linear_layers.clear();
    accumulators.clear();

    for (std::shared_ptr<Layer> layer : neural_network->layers) {
        std::shared_ptr<Linear> linear_layer = std::dynamic_pointer_cast<Linear>(layer);
        if (!layer->trainable || !linear_layer) continue;

        linear_layers.push_back(linear_layer.get());

        Accumulator accumulator;
        accumulator.W.assign(linear_layer->W.rows, std::vector<double>(linear_layer->W.cols, 0.0));
        accumulator.b.assign(linear_layer->b.rows, std::vector<double>(linear_layer->b.cols, 0.0));

        accumulators.push_back(std::move(accumulator));
    }
};

void AdagradOptimizer::optimize() {
    // Adaptive Gradient Algorithm (AdaGrad) optimization for each parameter
    // Adapts the learning rate per-parameter by accumulating the sum of squared past gradients
    // Parameters with large cumulative gradients get smaller effective ste

    for (int layerIdx = 0; layerIdx < linear_layers.size(); layerIdx++) {
        Linear* linear_layer = linear_layers[layerIdx];
        Accumulator& accumulator = accumulators[layerIdx];

        // Update W
        for (int i = 0; i < linear_layer->W.rows; i++) {
            for (int j = 0; j < linear_layer->W.cols; j++) {
                Var& weight_param = linear_layer->W.data[i][j];

                // Partial derivative of the loss function with respect to the weight parameter (∂L/∂W)
                double grad = weight_param.getGrad();

                // accumulator_W_t = accumulator_W_{t - 1} + (∂L/∂W * ∂L/∂W)
                accumulator.W[i][j] += grad * grad;

                // adjusted_lr = lr / (sqrt{accumulator_W_t} + eps)
                double adjusted_lr = learning_rate / (std::sqrt(accumulator.W[i][j]) + epsilon);

                // W_t = W_{t - 1} - (adjusted_lr * ∂L/∂W)
                weight_param.setVal(weight_param.getVal() - adjusted_lr * grad);
            }
        }

        // Update b
        for (int i = 0; i < linear_layer->b.rows; i++) {
            for (int j = 0; j < linear_layer->b.cols; j++) {
                Var& bias_param = linear_layer->b.data[i][j];

                // Partial derivative of the loss function with respect to the bias parameter (∂L/∂b)
                double grad = bias_param.getGrad();

                // accumulator_b_t = accumulator_b_{t - 1} + (∂L/∂b * ∂L/∂b)
                accumulator.b[i][j] += grad * grad;

                // adjusted_lr = lr / (sqrt{accumulator_b_t} + eps)
                double adjusted_lr = learning_rate / (std::sqrt(accumulator.b[i][j]) + epsilon);

                // b_t = b_{t - 1} - (adjusted_lr * ∂L/∂b)
                bias_param.setVal(bias_param.getVal() - adjusted_lr * grad);
            }
        }
    }
};

RMSPropOptimizer::RMSPropOptimizer(double lr, NeuralNetwork* model, double decay_rate, double epsilon) {
    learning_rate = lr;
    neural_network = model;
    this->decay_rate = decay_rate;
    this->epsilon = epsilon;

    initializeOptimizer();
};

void RMSPropOptimizer::initializeOptimizer() {
    linear_layers.clear();
    accumulators.clear();

    for (std::shared_ptr<Layer> layer : neural_network->layers) {
        std::shared_ptr<Linear> linear_layer = std::dynamic_pointer_cast<Linear>(layer);
        if (!layer->trainable || !linear_layer) continue;

        linear_layers.push_back(linear_layer.get());

        Accumulator accumulator;
        accumulator.W.assign(linear_layer->W.rows, std::vector<double>(linear_layer->W.cols, 0.0));
        accumulator.b.assign(linear_layer->b.rows, std::vector<double>(linear_layer->b.cols, 0.0));

        accumulators.push_back(std::move(accumulator));
    }
};

void RMSPropOptimizer::optimize() {
    // Root Mean Square Propagation (RMSProp) optimization for each parameter
    // Fixes AdaGrad’s ever-increasing accumulator by using an exponential moving average of past squared gradients

    for (int layerIdx = 0; layerIdx < linear_layers.size(); layerIdx++) {
        Linear* linear_layer = linear_layers[layerIdx];
        Accumulator& accumulator = accumulators[layerIdx];

        // Update W
        for (int i = 0; i < linear_layer->W.rows; i++) {
            for (int j = 0; j < linear_layer->W.cols; j++) {
                Var& weight_param = linear_layer->W.data[i][j];

                // Partial derivative of the loss function with respect to the weight parameter (∂L/∂W)
                double grad = weight_param.getGrad();

                // accumulator_W_t = decay_rate * accumulator_W_{t - 1} + (1 - decay_rate) * (∂L/∂W * ∂L/∂W)
                accumulator.W[i][j] = decay_rate * accumulator.W[i][j] + (1.0 - decay_rate) * grad * grad;

                // adjusted_lr = lr / (sqrt{accumulator_W_t} + eps)
                double adjusted_lr = learning_rate / (std::sqrt(accumulator.W[i][j]) + epsilon);

                // W_t = W_{t - 1} - (adjusted_lr * ∂L/∂W)
                weight_param.setVal(weight_param.getVal() - adjusted_lr * grad);
            }
        }

        // Update b
        for (int i = 0; i < linear_layer->b.rows; i++) {
            for (int j = 0; j < linear_layer->b.cols; j++) {
                Var& bias_param = linear_layer->b.data[i][j];

                // Partial derivative of the loss function with respect to the bias parameter (∂L/∂b)
                double grad = bias_param.getGrad();

                // accumulator_b_t = decay_rate * accumulator_b_{t - 1} + (1 - decay_rate) * (∂L/∂b * ∂L/∂b)
                accumulator.b[i][j] = decay_rate * accumulator.b[i][j] + (1.0 - decay_rate) * grad * grad;

                // adjusted_lr = lr / (sqrt{accumulator_W_t} + eps)
                double adjusted_lr = learning_rate / (std::sqrt(accumulator.b[i][j]) + epsilon);

                // b_t = b_{t - 1} - (adjusted_lr * ∂L/∂b)
                bias_param.setVal(bias_param.getVal() - adjusted_lr * grad);
            }
        }
    }
};

AdamOptimizer::AdamOptimizer(double lr, NeuralNetwork* model, double beta1, double beta2, double epsilon) {
    learning_rate = lr;
    neural_network = model;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->epsilon = epsilon;

    initializeOptimizer();
};

void AdamOptimizer::initializeOptimizer() {
    linear_layers.clear();
    moments.clear();
    timestep = 0;

    for (std::shared_ptr<Layer> layer : neural_network->layers) {
        std::shared_ptr<Linear> linear_layer = std::dynamic_pointer_cast<Linear>(layer);
        if (!layer->trainable || !linear_layer) continue;

        linear_layers.push_back(linear_layer.get());

        Moments moment;
        moment.mW.assign(linear_layer->W.rows, std::vector<double>(linear_layer->W.cols, 0.0));
        moment.vW.assign(linear_layer->W.rows, std::vector<double>(linear_layer->W.cols, 0.0));
        moment.mb.assign(linear_layer->b.rows, std::vector<double>(linear_layer->b.cols, 0.0));
        moment.vb.assign(linear_layer->b.rows, std::vector<double>(linear_layer->b.cols, 0.0));

        moments.push_back(std::move(moment));
    }
};

void AdamOptimizer::optimize() {
    // Adaptive Moment Estimation (Adam) optimization for each parameter
    // Combines RMSProp’s adaptive second-moment scaling with momentum’s first-moment (mean) tracking
    // Both direction smoothing and per-parameter step sizing

    timestep++;

    for (int layerIdx = 0; layerIdx < linear_layers.size(); layerIdx++) {
        Linear* linear_layer = linear_layers[layerIdx];
        Moments& moment = moments[layerIdx];

        // Update W
        for (int i = 0; i < linear_layer->W.rows; i++) {
            for (int j = 0; j < linear_layer->W.cols; j++) {
                Var& weight_param = linear_layer->W.data[i][j];

                // Partial derivative of the loss function with respect to the weight parameter (∂L/∂W)
                double grad = weight_param.getGrad();

                // mW_t = beta_1 * m_W_{t - 1} + (1 - beta_1) * ∂L/∂W
                moment.mW[i][j] = beta1 * moment.mW[i][j] + (1.0 - beta1) * grad;

                // vW_t = beta_2 * v_W_{t - 1} + (1 - beta_2) * (∂L/∂W * ∂L/∂W)
                moment.vW[i][j] = beta2 * moment.vW[i][j] + (1.0 - beta2) * grad * grad;

                // corrected_mW = mW_t / (1 - beta_1^t)
                double corrected_mW = moment.mW[i][j] / (1.0 - std::pow(beta1, timestep));

                // corrected_vW = vW_t / (1 - beta_2^t)
                double corrected_vW = moment.vW[i][j] / (1.0 - std::pow(beta2, timestep));

                // W_t = W_{t - 1} - (lr * (corrected_mW / (sqrt{corrected_vW} + eps)))
                weight_param.setVal(weight_param.getVal() - learning_rate * corrected_mW / (std::sqrt(corrected_vW) + epsilon));
            }
        }

        // Update b
        for (int i = 0; i < linear_layer->b.rows; i++) {
            for (int j = 0; j < linear_layer->b.cols; j++) {
                Var& bias_param = linear_layer->b.data[i][j];

                // Partial derivative of the loss function with respect to the bias parameter (∂L/∂b)
                double grad = bias_param.getGrad();

                // mb_t = beta_1 * m_b_{t - 1} + (1 - beta_1) * ∂L/∂b
                moment.mb[i][j] = beta1 * moment.mb[i][j] + (1.0 - beta1) * grad;

                // vb_t = beta_2 * v_b_{t - 1} + (1 - beta_2) * (∂L/∂b * ∂L/∂b)
                moment.vb[i][j] = beta2 * moment.vb[i][j] + (1.0 - beta2) * grad * grad;

                // corrected_mb = mb_t / (1 - beta_1^t)
                double corrected_mb = moment.mb[i][j] / (1.0 - std::pow(beta1, timestep));

                // corrected_vb = vb_t / (1 - beta_2^t)
                double corrected_vb = moment.vb[i][j] / (1.0 - std::pow(beta2, timestep));

                // b_t = b_{t - 1} - (lr * (corrected_mb / (sqrt{corrected_vb} + eps)))
                bias_param.setVal(bias_param.getVal() - learning_rate * corrected_mb / (std::sqrt(corrected_vb) + epsilon));
            }
        }
    }
};

AdamWOptimizer::AdamWOptimizer(double lr, NeuralNetwork* model, double beta1, double beta2, double epsilon, double weight_decay) {
    learning_rate = lr;
    neural_network = model;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->epsilon = epsilon;
    this->weight_decay = weight_decay;

    initializeOptimizer();
};

void AdamWOptimizer::initializeOptimizer() {
    linear_layers.clear();
    moments.clear();
    timestep = 0;

    for (std::shared_ptr<Layer> layer : neural_network->layers) {
        std::shared_ptr<Linear> linear_layer = std::dynamic_pointer_cast<Linear>(layer);
        if (!layer->trainable || !linear_layer) continue;

        linear_layers.push_back(linear_layer.get());

        Moments moment;
        moment.mW.assign(linear_layer->W.rows, std::vector<double>(linear_layer->W.cols, 0.0));
        moment.vW.assign(linear_layer->W.rows, std::vector<double>(linear_layer->W.cols, 0.0));
        moment.mb.assign(linear_layer->b.rows, std::vector<double>(linear_layer->b.cols, 0.0));
        moment.vb.assign(linear_layer->b.rows, std::vector<double>(linear_layer->b.cols, 0.0));

        moments.push_back(std::move(moment));
    }
};

void AdamWOptimizer::optimize() {
    // Adam with Weight Decay (AdamW) optimization for each parameter
    // AdamW decouples weight decay from the gradient update, applying it directly to the model parameters rather than adding it separately as an L2 penalty to the loss

    timestep++;

    for (int layerIdx = 0; layerIdx < linear_layers.size(); layerIdx++) {
        Linear* linear_layer = linear_layers[layerIdx];
        Moments& moment = moments[layerIdx];

        // Update W
        for (int i = 0; i < linear_layer->W.rows; i++) {
            for (int j = 0; j < linear_layer->W.cols; j++) {
                Var& weight_param = linear_layer->W.data[i][j];

                // Partial derivative of the loss function with respect to the weight parameter (∂L/∂W)
                double grad = weight_param.getGrad();

                // mW_t = beta_1 * m_W_{t - 1} + (1 - beta_1) * ∂L/∂W
                moment.mW[i][j] = beta1 * moment.mW[i][j] + (1.0 - beta1) * grad;

                // vW_t = beta_2 * v_W_{t - 1} + (1 - beta_2) * (∂L/∂W * ∂L/∂W)
                moment.vW[i][j] = beta2 * moment.vW[i][j] + (1.0 - beta2) * grad * grad;

                // corrected_mW = mW_t / (1 - beta_1^t)
                double corrected_mW = moment.mW[i][j] / (1.0 - std::pow(beta1, timestep));

                // corrected_vW = vW_t / (1 - beta_2^t)
                double corrected_vW = moment.vW[i][j] / (1.0 - std::pow(beta2, timestep));

                // W_t = (W_{t - 1} - (lr * weight_decay * W_{t - 1})) - (lr * (corrected_mW / (sqrt{corrected_vW} + eps)))
                weight_param.setVal(weight_param.getVal() - (learning_rate * weight_decay * weight_param.getVal()) - learning_rate * corrected_mW / (std::sqrt(corrected_vW) + epsilon));
            }
        }

        // Update b
        for (int i = 0; i < linear_layer->b.rows; i++) {
            for (int j = 0; j < linear_layer->b.cols; j++) {
                Var& bias_param = linear_layer->b.data[i][j];

                // Partial derivative of the loss function with respect to the bias parameter (∂L/∂b)
                double grad = bias_param.getGrad();

                // mb_t = beta_1 * m_b_{t - 1} + (1 - beta_1) * ∂L/∂b
                moment.mb[i][j] = beta1 * moment.mb[i][j] + (1.0 - beta1) * grad;

                // vb_t = beta_2 * v_b_{t - 1} + (1 - beta_2) * (∂L/∂b * ∂L/∂b)
                moment.vb[i][j] = beta2 * moment.vb[i][j] + (1.0 - beta2) * grad * grad;

                // corrected_mb = mb_t / (1 - beta_1^t)
                double corrected_mb = moment.mb[i][j] / (1.0 - std::pow(beta1, timestep));

                // corrected_vb = vb_t / (1 - beta_2^t)
                double corrected_vb = moment.vb[i][j] / (1.0 - std::pow(beta2, timestep));

                // b_t = b_{t - 1} - (lr * (corrected_mb / (sqrt{corrected_vb} + eps)))
                bias_param.setVal(bias_param.getVal() - learning_rate * corrected_mb / (std::sqrt(corrected_vb) + epsilon));
            }
        }
    }
};