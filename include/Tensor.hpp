#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>

class Tensor {
public:
    int rows = 0;
    int cols = 0;

    Tensor();
    Tensor(int r, int c, bool requiresGrad = true);

    double& operator()(int row, int col);
    double operator()(int row, int col) const;

    double gradAt(int row, int col) const;
    void setGradAt(int row, int col, double v);

    void resetGradAndParents();
    void zeroGrad();

    std::string getValsTensor() const;
    std::string getGradsTensor() const;

    void randomInit(double low = -0.01, double high = 0.01);

    Tensor add(Tensor& other);
    Tensor operator+(Tensor& other) { return add(other); };

    Tensor add(double other);
    Tensor operator+(double other) { return add(other); };

    Tensor subtract(Tensor& other);
    Tensor operator-(Tensor& other) { return subtract(other); };

    Tensor subtract(double other);
    Tensor operator-(double other) { return subtract(other); };

    Tensor multiply(Tensor& other);
    Tensor operator*(Tensor& other) { return multiply(other); };

    Tensor multiply(double other);
    Tensor operator*(double other) { return multiply(other); };

    Tensor matmul(Tensor& other);

    Tensor divide(double other);
    Tensor operator/(double other) { return divide(other); };

    Tensor pow(int power);

    Tensor relu();
    Tensor leakyRelu(double alpha = 0.01);
    Tensor sigmoid();
    Tensor tanh();
    Tensor silu();
    Tensor elu(double alpha = 1.0);
    Tensor softmax();

    Tensor log();
    Tensor exp();
    Tensor abs();

    Tensor sum();
    Tensor mean();

    void backward();

private:
    struct Node;

    struct Edge {
        std::shared_ptr<Node> parent;
        std::vector<double> local_grad;
        std::function<void(const Node& self, Node& parent)> backward;
    };

    struct Node {
        int rows = 0;
        int cols = 0;
        bool requiresGrad = true;
        std::vector<double> val;
        std::vector<double> grad;
        int pending_children = 0;
        std::vector<Edge> parents;
    };

    std::shared_ptr<Node> node;

    int index(int row, int col) const;
};

Tensor matmul(Tensor& A, Tensor& B);
