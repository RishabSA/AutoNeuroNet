#include "NeuralNetwork.hpp"

void initWeights(Matrix& W, int fan_in, int fan_out, const std::string& init) {
    double stddev = 0.0;
    if (init == "xavier" || init == "glorot") {
        stddev = std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
    } else {
        // default to He/Kaiming
        stddev = std::sqrt(2.0 / static_cast<double>(fan_in));
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, stddev);

    for (int i = 0; i < W.rows; i++) {
        for (int j = 0; j < W.cols; j++) {
            W.data[i][j] = dist(gen);
        }
    }
}


Linear::Linear(int inDim, int outDim, const std::string& init) {
    name = "Linear(" + std::to_string(inDim) + ", " + std::to_string(outDim) + ")";
    trainable = true;

    W = Matrix(inDim, outDim);
    initWeights(W, inDim, outDim, init);

    b = Matrix(1, outDim);
}

Matrix Linear::forward(Matrix& input) {
    Matrix output = matmul(input, W) + b;
    return output;
};

void Linear::optimizeWeights(double learning_rate) {
    // Backpropagation and Gradient Descent for each parameter

    // Update W
    for (int i = 0; i < W.rows; i++) {
        for (int j = 0; j < W.cols; j++) {
            Var& weight_param = W.data[i][j];

            // Partial derivative of the Loss function with respect to the weight parameter
            double gradient = weight_param.getGrad();
            weight_param.setVal(weight_param.getVal() - learning_rate * gradient);
        }
    }

    // Update b
    for (int i = 0; i < b.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            Var& bias_param = b.data[i][j];

            // Partial derivative of the Loss function with respect to the bias parameter
            double gradient = bias_param.getGrad();
            bias_param.setVal(bias_param.getVal() - learning_rate * gradient);
        }
    }
};

void Linear::resetGrad() {
    W.resetGradAndParents();
    b.resetGradAndParents();
};

ReLU::ReLU() {
    name = "ReLU()";
    trainable = false;
}

Matrix ReLU::forward(Matrix& input) {
    Matrix output = input.relu();
    return output;
};

void ReLU::optimizeWeights(double learning_rate) {
    (void)learning_rate;
}

void ReLU::resetGrad() {}

LeakyReLU::LeakyReLU(double a) {
    alpha = a;
    name = "LeakyReLU(alpha=" + std::to_string(alpha) + ")";
    trainable = false;
}

Matrix LeakyReLU::forward(Matrix& input) {
    Matrix output = input.leakyRelu(alpha);
    return output;
};

void LeakyReLU::optimizeWeights(double learning_rate) {
    (void)learning_rate;
}

void LeakyReLU::resetGrad() {}

Sigmoid::Sigmoid() {
    name = "Sigmoid()";
    trainable = false;
}

Matrix Sigmoid::forward(Matrix& input) {
    Matrix output = input.sigmoid();
    return output;
};

void Sigmoid::optimizeWeights(double learning_rate) {
    (void)learning_rate;
}

void Sigmoid::resetGrad() {}

Tanh::Tanh() {
    name = "Tanh()";
    trainable = false;
}

Matrix Tanh::forward(Matrix& input) {
    Matrix output = input.tanh();
    return output;
};

void Tanh::optimizeWeights(double learning_rate) {
    (void)learning_rate;
}

void Tanh::resetGrad() {}

SiLU::SiLU() {
    name = "SiLU()";
    trainable = false;
}

Matrix SiLU::forward(Matrix& input) {
    Matrix output = input.silu();
    return output;
};

void SiLU::optimizeWeights(double learning_rate) {
    (void)learning_rate;
}

void SiLU::resetGrad() {}

ELU::ELU(double a) {
    alpha = a;
    name = "ELU(alpha=" + std::to_string(alpha) + ")";
    trainable = false;
}

Matrix ELU::forward(Matrix& input) {
    Matrix output = input.elu(alpha);
    return output;
};

void ELU::optimizeWeights(double learning_rate) {
    (void)learning_rate;
}

void ELU::resetGrad() {}

Softmax::Softmax() {
    name = "Softmax()";
    trainable = false;
}

Matrix Softmax::forward(Matrix& input) {
    Matrix output = input.softmax();
    return output;
};

void Softmax::optimizeWeights(double learning_rate) {
    (void)learning_rate;
}

void Softmax::resetGrad() {}

NeuralNetwork::NeuralNetwork(std::vector<std::shared_ptr<Layer>> network) {
    layers = std::move(network);
};

std::vector<std::shared_ptr<Layer>> NeuralNetwork::getLayers() {
    return layers;
}

const std::vector<std::shared_ptr<Layer>> NeuralNetwork::getLayers() const {
    return layers;
}

void NeuralNetwork::addLayer(std::shared_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

Matrix NeuralNetwork::forward(Matrix input) {
    for (auto& layer : layers) {
        input = layer->forward(input);
    }
    return input;
};

std::string NeuralNetwork::getNetworkArchitecture() const {
    if (layers.empty()) {
        return "[]";
    }

    std::string architecture = "";

    for (auto& layer : layers) {
        architecture += layer->name + "\n";
    }

    return architecture;
};

void NeuralNetwork::saveWeights(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file");

    uint32_t num = static_cast<uint32_t>(layers.size());
    out.write(reinterpret_cast<char*>(&num), sizeof(num));

    for (auto& layer : layers) {
        auto linear = std::dynamic_pointer_cast<Linear>(layer);
        if (!linear) continue;

        int rows = linear->W.rows;
        int cols = linear->W.cols;
        out.write(reinterpret_cast<char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<char*>(&cols), sizeof(cols));

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double v = linear->W.data[i][j].getVal();
                out.write(reinterpret_cast<char*>(&v), sizeof(v));
            }
        }

        int brow = linear->b.rows, bcol = linear->b.cols;
        out.write(reinterpret_cast<char*>(&brow), sizeof(brow));
        out.write(reinterpret_cast<char*>(&bcol), sizeof(bcol));

        for (int i = 0; i < brow; i++) {
            for (int j = 0; j < bcol; j++) {
                double v = linear->b.data[i][j].getVal();
                out.write(reinterpret_cast<char*>(&v), sizeof(v));
            }
        }
    }
}

void NeuralNetwork::loadWeights(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file");

    uint32_t num = 0;
    in.read(reinterpret_cast<char*>(&num), sizeof(num));

    for (auto& layer : layers) {
        auto linear = std::dynamic_pointer_cast<Linear>(layer);
        if (!linear) continue;

        int rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        in.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        if (rows != linear->W.rows || cols != linear->W.cols)
            throw std::runtime_error("Weight shape mismatch");

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double v;
                in.read(reinterpret_cast<char*>(&v), sizeof(v));
                linear->W.data[i][j].setVal(v);
            }
        }

        int brow, bcol;
        in.read(reinterpret_cast<char*>(&brow), sizeof(brow));
        in.read(reinterpret_cast<char*>(&bcol), sizeof(bcol));

        if (brow != linear->b.rows || bcol != linear->b.cols)
            throw std::runtime_error("Bias shape mismatch");

        for (int i = 0; i < brow; ++i) {
            for (int j = 0; j < bcol; ++j) {
                double v;
                in.read(reinterpret_cast<char*>(&v), sizeof(v));
                linear->b.data[i][j].setVal(v);
            }
        }
    }
}
