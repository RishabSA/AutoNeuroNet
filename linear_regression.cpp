#include <iostream>
#include "NeuralNetwork.hpp"

// g++ linear_regression.cpp src/Var.cpp src/Matrix.cpp src/NeuralNetwork.cpp -I include -o linear_regression && ./linear_regression

int main () {
    int in_dim = 1;
    int out_dim = 1;

    int N = 10;

    Matrix X(N, in_dim); // shape: (N, 1)
    Matrix Y_true(N, out_dim); // shape: (N, 1)

    // Initialize training data
    for (int i = 0; i < N; i++) {
        X(i, 0) = Var(i);
        Y_true(i, 0) = 5.0 * i + 3.0; // y = 5x + 3
    }

    // std::cout << "Training Data:" << std::endl;
    // std::cout << X.getValsMatrix() << std::endl;

    NeuralNetwork model({ std::make_tuple(in_dim, out_dim) });

    double lr = 0.001;
    int epochs = 1000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Reset gradients and the old graph on everything
        X.resetGradAndParents();
        Y_true.resetGradAndParents();

        for (auto& layer : model.layers) {
            Matrix& W = layer.first;
            Matrix& b = layer.second;

            W.resetGradAndParents();
            b.resetGradAndParents();
        }

        // Forward pass
        Matrix Y_pred = model.forward(X);

        // Calculate the loss
        Var loss = computeMSELoss(Y_true, Y_pred);
        double loss_val = loss.getVal();

        // Backpropagation (Reverse-Mode Automatic Differentiation)
        loss.setGradVal(1.0);
        loss.backward();

        // Backpropagation and Gradient Descent for each parameter
        for (auto& layer : model.layers) {
            Matrix& W = layer.first;
            Matrix& b = layer.second;

            // Update W
            for (int i = 0; i < W.rows; ++i) {
                for (int j = 0; j < W.cols; ++j) {
                    Var& weight_param = W.data[i][j];

                    // Partial derivative of the Loss function with respect to the weight parameter
                    double gradient = weight_param.getGradVal();
                    weight_param.setVal(weight_param.getVal() - lr * gradient);
                }
            }

            // Update b
            for (int i = 0; i < b.rows; ++i) {
                for (int j = 0; j < b.cols; ++j) {
                    Var& bias_param = b.data[i][j];

                    // Partial derivative of the Loss function with respect to the bias parameter
                    double gradient = bias_param.getGradVal();
                    bias_param.setVal(bias_param.getVal() - lr * gradient);
                }
            }
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch + 1 << " | train loss: " << loss_val << "\n";
        }
    }

    // Make Predictions
    Matrix Y_pred_final = model.forward(X);

    std::cout << "Ground Truth Labels:\n" << std::endl;
    std::cout << Y_true.getValsMatrix() << std::endl;

    std::cout << "Final Model Predictions:\n" << std::endl;
    std::cout << Y_pred_final.getValsMatrix() << std::endl;

    auto& first_layer = model.layers[0];
    Matrix& W_learned = first_layer.first;
    Matrix& b_learned = first_layer.second;

    std::cout << "Learned W(0, 0) = " << W_learned.data[0][0].getVal() << "\n";
    std::cout << "Learned b(0, 0) = " << b_learned.data[0][0].getVal();

    return 0;
}