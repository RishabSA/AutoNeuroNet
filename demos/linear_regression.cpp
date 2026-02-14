#include <iostream>
#include <memory>
#include "NeuralNetwork.hpp"
#include "Optimizers.hpp"
#include "LossFunctions.hpp"

// g++ demos/linear_regression.cpp src/Var.cpp src/Matrix.cpp src/NeuralNetwork.cpp src/Optimizers.cpp src/LossFunctions.cpp -I include -o demos/linear_regression && ./demos/linear_regression

int main () {
    int inDim = 1;
    int outDim = 1;

    int N = 10;

    Matrix X(N, inDim); // shape: (N, 1)
    Matrix Y_true(N, outDim); // shape: (N, 1)

    // Initialize training data
    for (int i = 0; i < N; i++) {
        X(i, 0) = Var(i);
        Y_true(i, 0) = 5.0 * i + 3.0; // y = 5x + 3
    }

    NeuralNetwork model({ std::make_shared<Linear>(inDim, outDim) });
    double lr = 0.01;
    GradientDescentOptimizer optimizer(lr, &model);
    
    int epochs = 1000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        optimizer.resetGrad();

        // Forward pass
        Matrix Y_pred = model.forward(X);

        // Calculate the loss
        Var loss = MSELoss(Y_true, Y_pred);
        double loss_val = loss.getVal();

        // Backpropagation (Reverse-Mode Automatic Differentiation)
        loss.setGrad(1.0);
        loss.backward();
        optimizer.optimize();

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch + 1 << " | Train Loss: " << loss_val << "\n";
        }
    }

    // Make Predictions
    Matrix Y_pred_final = model.forward(X);

    std::cout << "Ground Truth Labels:\n" << std::endl;
    std::cout << Y_true.getValsMatrix() << std::endl;

    std::cout << "Final Model Predictions:\n" << std::endl;
    std::cout << Y_pred_final.getValsMatrix() << std::endl;

    auto linear_layer = std::dynamic_pointer_cast<Linear>(model.layers[0]);
    Matrix& W_learned = linear_layer->W;
    Matrix& b_learned = linear_layer->b;

    std::cout << "Learned W(0, 0) = " << W_learned.data[0][0].getVal() << "\n";
    std::cout << "Learned b(0, 0) = " << b_learned.data[0][0].getVal();

    return 0;
}
