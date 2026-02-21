#include "NeuralNetwork.hpp"

void initWeights(Matrix& W, int inDim, int outDim, const std::string& init) {
    // Initialize a weight matrix with random weights using a specific algorithm's distribution

    double std = 0.0;
    if (init == "xavier" || init == "glorot") {
        std = std::sqrt(2.0 / static_cast<double>(inDim + outDim));
    } else {
        // default to Kaiming/He
        std = std::sqrt(2.0 / static_cast<double>(inDim));
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, std);

    for (int i = 0; i < W.rows; i++) {
        for (int j = 0; j < W.cols; j++) {
            W.data[i][j] = dist(gen);
        }
    }
};


Linear::Linear(int inDim, int outDim, const std::string& init) {
    name = "Linear(" + std::to_string(inDim) + ", " + std::to_string(outDim) + ")";
    trainable = true;

    W = Matrix(inDim, outDim);
    initWeights(W, inDim, outDim, init);

    b = Matrix(1, outDim);
};

Matrix Linear::forward(Matrix& input) {
    Matrix output = matmul(input, W) + b;
    return output;
};

void Linear::resetGrad() {
    W.resetGradAndParents();
    b.resetGradAndParents();
};

ReLU::ReLU() {
    name = "ReLU()";
    trainable = false;
};

Matrix ReLU::forward(Matrix& input) {
    Matrix output = input.relu();
    return output;
};

void ReLU::resetGrad() {}

LeakyReLU::LeakyReLU(double a) {
    alpha = a;
    name = "LeakyReLU(alpha=" + std::to_string(alpha) + ")";
    trainable = false;
};

Matrix LeakyReLU::forward(Matrix& input) {
    Matrix output = input.leakyRelu(alpha);
    return output;
};

void LeakyReLU::resetGrad() {};

Sigmoid::Sigmoid() {
    name = "Sigmoid()";
    trainable = false;
};

Matrix Sigmoid::forward(Matrix& input) {
    Matrix output = input.sigmoid();
    return output;
};

void Sigmoid::resetGrad() {};

Tanh::Tanh() {
    name = "Tanh()";
    trainable = false;
};

Matrix Tanh::forward(Matrix& input) {
    Matrix output = input.tanh();
    return output;
};

void Tanh::resetGrad() {};

SiLU::SiLU() {
    name = "SiLU()";
    trainable = false;
};

Matrix SiLU::forward(Matrix& input) {
    Matrix output = input.silu();
    return output;
};

void SiLU::resetGrad() {};

ELU::ELU(double a) {
    alpha = a;
    name = "ELU(alpha=" + std::to_string(alpha) + ")";
    trainable = false;
};

Matrix ELU::forward(Matrix& input) {
    Matrix output = input.elu(alpha);
    return output;
};

void ELU::resetGrad() {};

Softmax::Softmax() {
    name = "Softmax()";
    trainable = false;
};

Matrix Softmax::forward(Matrix& input) {
    Matrix output = input.softmax();
    return output;
};

void Softmax::resetGrad() {};

NeuralNetwork::NeuralNetwork(std::vector<std::shared_ptr<Layer>> network) {
    layers = std::move(network);
};

std::vector<std::shared_ptr<Layer>> NeuralNetwork::getLayers() {
    return layers;
};

const std::vector<std::shared_ptr<Layer>> NeuralNetwork::getLayers() const {
    return layers;
};

void NeuralNetwork::addLayer(std::shared_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
};

Matrix NeuralNetwork::forward(Matrix input) {
    for (std::shared_ptr<Layer> layer : layers) {
        input = layer->forward(input);
    }

    return input;
};

std::string NeuralNetwork::getNetworkArchitecture() const {
    if (layers.empty()) {
        return "[]";
    }

    std::string architecture = "";

    for (std::shared_ptr<Layer> layer : layers) {
        architecture += layer->name + "\n";
    }

    return architecture;
};

void NeuralNetwork::saveWeights(const std::string& path) {
    // Save all neural network model weights to a .bin binary file

    std::ofstream outFile(path, std::ios::binary);
    if (!outFile) throw std::runtime_error("Failed to open file");

    int num_layers = layers.size();

    // reinterpret_cast is used to treat the memory address of the struct as a character array (bytes)
    outFile.write(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    for (std::shared_ptr<Layer> layer : layers) {
        std::shared_ptr<Linear> linear = std::dynamic_pointer_cast<Linear>(layer);
        if (!linear) continue;

        int weight_rows = linear->W.rows;
        int weight_cols = linear->W.cols;

        outFile.write(reinterpret_cast<char*>(&weight_rows), sizeof(weight_rows));
        outFile.write(reinterpret_cast<char*>(&weight_cols), sizeof(weight_cols));

        for (int i = 0; i < weight_rows; i++) {
            for (int j = 0; j < weight_cols; j++) {
                double val = linear->W.data[i][j].getVal();
                outFile.write(reinterpret_cast<char*>(&val), sizeof(val));
            }
        }

        int bias_rows = linear->b.rows;
        int bias_cols = linear->b.cols;

        outFile.write(reinterpret_cast<char*>(&bias_rows), sizeof(bias_rows));
        outFile.write(reinterpret_cast<char*>(&bias_cols), sizeof(bias_cols));

        for (int i = 0; i < bias_rows; i++) {
            for (int j = 0; j < bias_cols; j++) {
                double val = linear->b.data[i][j].getVal();
                outFile.write(reinterpret_cast<char*>(&val), sizeof(val));
            }
        }
    }
};

void NeuralNetwork::loadWeights(const std::string& path) {
    // Load all neural network model weights from a .bin binary file

    std::ifstream inFile(path, std::ios::binary);
    if (!inFile) throw std::runtime_error("Failed to open file");

    int num_layers = 0;
    inFile.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    for (std::shared_ptr<Layer> layer : layers) {
        std::shared_ptr<Linear> linear = std::dynamic_pointer_cast<Linear>(layer);
        if (!linear) continue;

        int weight_rows;
        int weight_cols;

        inFile.read(reinterpret_cast<char*>(&weight_rows), sizeof(weight_rows));
        inFile.read(reinterpret_cast<char*>(&weight_cols), sizeof(weight_cols));

        if (weight_rows != linear->W.rows || weight_cols != linear->W.cols) throw std::runtime_error("Weight shape mismatch");

        for (int i = 0; i < weight_rows; i++) {
            for (int j = 0; j < weight_cols; j++) {
                double val;
                inFile.read(reinterpret_cast<char*>(&val), sizeof(val));
                linear->W.data[i][j].setVal(val);
            }
        }

        int bias_rows;
        int bias_cols;

        inFile.read(reinterpret_cast<char*>(&bias_rows), sizeof(bias_rows));
        inFile.read(reinterpret_cast<char*>(&bias_cols), sizeof(bias_cols));

        if (bias_rows != linear->b.rows || bias_cols != linear->b.cols) throw std::runtime_error("Bias shape mismatch");

        for (int i = 0; i < bias_rows; i++) {
            for (int j = 0; j < bias_cols; j++) {
                double val;
                inFile.read(reinterpret_cast<char*>(&val), sizeof(val));
                linear->b.data[i][j].setVal(val);
            }
        }
    }
};