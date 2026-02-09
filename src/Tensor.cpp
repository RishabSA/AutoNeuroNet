#include "Tensor.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

Tensor::Tensor() : node(std::make_shared<Node>()) {
    rows = 0;
    cols = 0;
    node->rows = 0;
    node->cols = 0;
    node->requiresGrad = true;
}

Tensor::Tensor(int r, int c, bool requiresGrad) : node(std::make_shared<Node>()) {
    rows = r;
    cols = c;
    node->rows = r;
    node->cols = c;
    node->requiresGrad = requiresGrad;

    const int n = r * c;
    node->val.assign(n, 0.0);
    node->grad.assign(n, 0.0);
}

int Tensor::index(int row, int col) const {
    return row * cols + col;
}

double& Tensor::operator()(int row, int col) {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("Tensor index out of range");
    }
    return node->val[index(row, col)];
}

double Tensor::operator()(int row, int col) const {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("Tensor index out of range");
    }
    return node->val[index(row, col)];
}

double Tensor::gradAt(int row, int col) const {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("Tensor grad index out of range");
    }
    return node->grad[index(row, col)];
}

void Tensor::setGradAt(int row, int col, double v) {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("Tensor grad index out of range");
    }
    node->grad[index(row, col)] = v;
}

void Tensor::zeroGrad() {
    std::fill(node->grad.begin(), node->grad.end(), 0.0);
}

void Tensor::resetGradAndParents() {
    zeroGrad();
    node->pending_children = 0;
    node->parents.clear();
}

std::string Tensor::getValsTensor() const {
    std::string out;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out += std::to_string(node->val[index(i, j)]);
            out += " ";
        }
        out += "\n";
    }

    return out;
}

std::string Tensor::getGradsTensor() const {
    std::string out;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out += std::to_string(node->grad[index(i, j)]);
            out += " ";
        }
        out += "\n";
    }

    return out;
}

void Tensor::randomInit(double low, double high) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> unif(low, high);

    for (int i = 0; i < rows * cols; i++) {
        node->val[i] = unif(gen);
    }
}

Tensor Tensor::add(Tensor& other) {
    Tensor Y(rows, cols, node->requiresGrad || other.node->requiresGrad);

    if (rows == other.rows && cols == other.cols) {
        for (int i = 0; i < rows * cols; i++) {
            Y.node->val[i] = node->val[i] + other.node->val[i];
        }

        if (node->requiresGrad) {
            std::vector<double> local(rows * cols, 1.0);
            Y.node->parents.push_back({node, std::move(local), {}});
            node->pending_children += 1;
        }

        if (other.node->requiresGrad) {
            std::vector<double> local(rows * cols, 1.0);
            Y.node->parents.push_back({other.node, std::move(local), {}});
            other.node->pending_children += 1;
        }
    } else if (other.rows == 1 && other.cols == 1) {
        const double val = other.node->val[0];
        for (int i = 0; i < rows * cols; i++) {
            Y.node->val[i] = node->val[i] + val;
        }

        if (node->requiresGrad) {
            std::vector<double> local(rows * cols, 1.0);
            Y.node->parents.push_back({node, std::move(local), {}});
            node->pending_children += 1;
        }

        if (other.node->requiresGrad) {
            Y.node->parents.push_back({
                other.node,
                {},
                [](const Node& self, Node& parent) {
                    double sum = 0.0;
                    for (double g : self.grad) sum += g;
                    parent.grad[0] += sum;
                }
            });
            other.node->pending_children += 1;
        }
    } else if (other.rows == 1 && other.cols == cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Y.node->val[index(i, j)] = node->val[index(i, j)] + other.node->val[j];
            }
        }

        if (node->requiresGrad) {
            std::vector<double> local(rows * cols, 1.0);
            Y.node->parents.push_back({node, std::move(local), {}});
            node->pending_children += 1;
        }

        if (other.node->requiresGrad) {
            Y.node->parents.push_back({
                other.node,
                {},
                [rows = rows, cols = cols](const Node& self, Node& parent) {
                    for (int j = 0; j < cols; j++) {
                        double sum = 0.0;
                        for (int i = 0; i < rows; i++) {
                            sum += self.grad[i * cols + j];
                        }
                        parent.grad[j] += sum;
                    }
                }
            });
            other.node->pending_children += 1;
        }
    } else if (other.cols == 1 && other.rows == rows) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Y.node->val[index(i, j)] = node->val[index(i, j)] + other.node->val[i];
            }
        }

        if (node->requiresGrad) {
            std::vector<double> local(rows * cols, 1.0);
            Y.node->parents.push_back({node, std::move(local), {}});
            node->pending_children += 1;
        }

        if (other.node->requiresGrad) {
            Y.node->parents.push_back({
                other.node,
                {},
                [rows = rows, cols = cols](const Node& self, Node& parent) {
                    for (int i = 0; i < rows; i++) {
                        double sum = 0.0;
                        for (int j = 0; j < cols; j++) {
                            sum += self.grad[i * cols + j];
                        }
                        parent.grad[i] += sum;
                    }
                }
            });
            other.node->pending_children += 1;
        }
    } else {
        throw std::runtime_error("Dimension mismatch when attempting to add tensors - (" + std::to_string(rows) + ", " + std::to_string(cols) + ") + (" + std::to_string(other.rows) + ", " + std::to_string(other.cols) + ")");
    }

    return Y;
}

Tensor Tensor::add(double other) {
    Tensor Y(rows, cols, node->requiresGrad);

    for (int i = 0; i < rows * cols; i++) {
        Y.node->val[i] = node->val[i] + other;
    }

    if (node->requiresGrad) {
        std::vector<double> local(rows * cols, 1.0);
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::subtract(Tensor& other) {
    Tensor Y(rows, cols, node->requiresGrad || other.node->requiresGrad);

    if (rows == other.rows && cols == other.cols) {
        for (int i = 0; i < rows * cols; i++) {
            Y.node->val[i] = node->val[i] - other.node->val[i];
        }

        if (node->requiresGrad) {
            std::vector<double> local(rows * cols, 1.0);
            Y.node->parents.push_back({node, std::move(local), {}});
            node->pending_children += 1;
        }

        if (other.node->requiresGrad) {
            std::vector<double> local(rows * cols, -1.0);
            Y.node->parents.push_back({other.node, std::move(local), {}});
            other.node->pending_children += 1;
        }
    } else if (other.rows == 1 && other.cols == 1) {
        const double val = other.node->val[0];
        for (int i = 0; i < rows * cols; i++) {
            Y.node->val[i] = node->val[i] - val;
        }

        if (node->requiresGrad) {
            std::vector<double> local(rows * cols, 1.0);
            Y.node->parents.push_back({node, std::move(local), {}});
            node->pending_children += 1;
        }

        if (other.node->requiresGrad) {
            Y.node->parents.push_back({
                other.node,
                {},
                [](const Node& self, Node& parent) {
                    double sum = 0.0;
                    for (double g : self.grad) sum += g;
                    parent.grad[0] -= sum;
                }
            });
            other.node->pending_children += 1;
        }
    } else if (other.rows == 1 && other.cols == cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Y.node->val[index(i, j)] = node->val[index(i, j)] - other.node->val[j];
            }
        }

        if (node->requiresGrad) {
            std::vector<double> local(rows * cols, 1.0);
            Y.node->parents.push_back({node, std::move(local), {}});
            node->pending_children += 1;
        }

        if (other.node->requiresGrad) {
            Y.node->parents.push_back({
                other.node,
                {},
                [rows = rows, cols = cols](const Node& self, Node& parent) {
                    for (int j = 0; j < cols; j++) {
                        double sum = 0.0;
                        for (int i = 0; i < rows; i++) {
                            sum += self.grad[i * cols + j];
                        }
                        parent.grad[j] -= sum;
                    }
                }
            });
            other.node->pending_children += 1;
        }
    } else if (other.cols == 1 && other.rows == rows) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Y.node->val[index(i, j)] = node->val[index(i, j)] - other.node->val[i];
            }
        }

        if (node->requiresGrad) {
            std::vector<double> local(rows * cols, 1.0);
            Y.node->parents.push_back({node, std::move(local), {}});
            node->pending_children += 1;
        }

        if (other.node->requiresGrad) {
            Y.node->parents.push_back({
                other.node,
                {},
                [rows = rows, cols = cols](const Node& self, Node& parent) {
                    for (int i = 0; i < rows; i++) {
                        double sum = 0.0;
                        for (int j = 0; j < cols; j++) {
                            sum += self.grad[i * cols + j];
                        }
                        parent.grad[i] -= sum;
                    }
                }
            });
            other.node->pending_children += 1;
        }
    } else {
        throw std::runtime_error("Dimension mismatch when attempting to subtract tensors - (" + std::to_string(rows) + ", " + std::to_string(cols) + ") - (" + std::to_string(other.rows) + ", " + std::to_string(other.cols) + ")");
    }

    return Y;
}

Tensor Tensor::subtract(double other) {
    Tensor Y(rows, cols, node->requiresGrad);

    for (int i = 0; i < rows * cols; i++) {
        Y.node->val[i] = node->val[i] - other;
    }

    if (node->requiresGrad) {
        std::vector<double> local(rows * cols, 1.0);
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::multiply(Tensor& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::runtime_error("Dimension mismatch when attempting to multiply tensors - (" + std::to_string(rows) + ", " + std::to_string(cols) + ") * (" + std::to_string(other.rows) + ", " + std::to_string(other.cols) + ")");
    }

    Tensor Y(rows, cols, node->requiresGrad || other.node->requiresGrad);

    for (int i = 0; i < rows * cols; i++) {
        Y.node->val[i] = node->val[i] * other.node->val[i];
    }

    if (node->requiresGrad) {
        std::vector<double> local = other.node->val;
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    if (other.node->requiresGrad) {
        std::vector<double> local = node->val;
        Y.node->parents.push_back({other.node, std::move(local), {}});
        other.node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::multiply(double other) {
    Tensor Y(rows, cols, node->requiresGrad);

    for (int i = 0; i < rows * cols; i++) {
        Y.node->val[i] = node->val[i] * other;
    }

    if (node->requiresGrad) {
        std::vector<double> local(rows * cols, other);
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::matmul(Tensor& other) {
    if (cols != other.rows) {
        throw std::runtime_error("Dimension mismatch when attempting to matmul tensors - (" + std::to_string(rows) + ", " + std::to_string(cols) + ") @ (" + std::to_string(other.rows) + ", " + std::to_string(other.cols) + ")");
    }

    const int m = rows;
    const int k = cols;
    const int n = other.cols;

    Tensor Y(m, n, node->requiresGrad || other.node->requiresGrad);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int t = 0; t < k; t++) {
                sum += node->val[i * k + t] * other.node->val[t * n + j];
            }
            Y.node->val[i * n + j] = sum;
        }
    }

    if (node->requiresGrad) {
        auto right = other.node;
        Y.node->parents.push_back({
            node,
            {},
            [right, m, k, n](const Node& self, Node& parent) {
                for (int i = 0; i < m; i++) {
                    for (int t = 0; t < k; t++) {
                        double sum = 0.0;
                        for (int j = 0; j < n; j++) {
                            sum += self.grad[i * n + j] * right->val[t * n + j];
                        }
                        parent.grad[i * k + t] += sum;
                    }
                }
            }
        });
        node->pending_children += 1;
    }

    if (other.node->requiresGrad) {
        auto left = node;
        Y.node->parents.push_back({
            other.node,
            {},
            [left, m, k, n](const Node& self, Node& parent) {
                for (int t = 0; t < k; t++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int i = 0; i < m; i++) {
                            sum += left->val[i * k + t] * self.grad[i * n + j];
                        }
                        parent.grad[t * n + j] += sum;
                    }
                }
            }
        });
        other.node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::divide(double other) {
    Tensor Y(rows, cols, node->requiresGrad);

    for (int i = 0; i < rows * cols; i++) {
        Y.node->val[i] = node->val[i] / other;
    }

    if (node->requiresGrad) {
        std::vector<double> local(rows * cols, 1.0 / other);
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::pow(int power) {
    Tensor Y(rows, cols, node->requiresGrad);

    std::vector<double> local(rows * cols, 0.0);
    for (int i = 0; i < rows * cols; i++) {
        double base = node->val[i];
        Y.node->val[i] = std::pow(base, power);
        if (power != 0) {
            local[i] = power * std::pow(base, power - 1);
        }
    }

    if (node->requiresGrad) {
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::relu() {
    Tensor Y(rows, cols, node->requiresGrad);

    std::vector<double> local(rows * cols, 0.0);
    for (int i = 0; i < rows * cols; i++) {
        double x = node->val[i];
        Y.node->val[i] = x > 0.0 ? x : 0.0;
        local[i] = x > 0.0 ? 1.0 : 0.0;
    }

    if (node->requiresGrad) {
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::leakyRelu(double alpha) {
    Tensor Y(rows, cols, node->requiresGrad);

    std::vector<double> local(rows * cols, 0.0);
    for (int i = 0; i < rows * cols; i++) {
        double x = node->val[i];
        Y.node->val[i] = x > 0.0 ? x : alpha * x;
        local[i] = x > 0.0 ? 1.0 : alpha;
    }

    if (node->requiresGrad) {
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::sigmoid() {
    Tensor Y(rows, cols, node->requiresGrad);

    std::vector<double> local(rows * cols, 0.0);
    for (int i = 0; i < rows * cols; i++) {
        double s = 1.0 / (1.0 + std::exp(-node->val[i]));
        Y.node->val[i] = s;
        local[i] = s * (1.0 - s);
    }

    if (node->requiresGrad) {
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::tanh() {
    Tensor Y(rows, cols, node->requiresGrad);

    std::vector<double> local(rows * cols, 0.0);
    for (int i = 0; i < rows * cols; i++) {
        double t = std::tanh(node->val[i]);
        Y.node->val[i] = t;
        local[i] = 1.0 - t * t;
    }

    if (node->requiresGrad) {
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::silu() {
    Tensor Y(rows, cols, node->requiresGrad);

    std::vector<double> local(rows * cols, 0.0);
    for (int i = 0; i < rows * cols; i++) {
        double x = node->val[i];
        double s = 1.0 / (1.0 + std::exp(-x));
        Y.node->val[i] = x * s;
        local[i] = s + x * s * (1.0 - s);
    }

    if (node->requiresGrad) {
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::elu(double alpha) {
    Tensor Y(rows, cols, node->requiresGrad);

    std::vector<double> local(rows * cols, 0.0);
    for (int i = 0; i < rows * cols; i++) {
        double x = node->val[i];
        Y.node->val[i] = x > 0.0 ? x : alpha * (std::exp(x) - 1.0);
        local[i] = x > 0.0 ? 1.0 : alpha * std::exp(x);
    }

    if (node->requiresGrad) {
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::softmax() {
    Tensor Y(rows, cols, node->requiresGrad);

    for (int i = 0; i < rows; i++) {
        double max_val = node->val[i * cols];
        for (int j = 1; j < cols; j++) {
            max_val = std::max(max_val, node->val[i * cols + j]);
        }

        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += std::exp(node->val[i * cols + j] - max_val);
        }

        for (int j = 0; j < cols; j++) {
            Y.node->val[i * cols + j] = std::exp(node->val[i * cols + j] - max_val) / sum;
        }
    }

    if (node->requiresGrad) {
        Y.node->parents.push_back({
            node,
            {},
            [rows = rows, cols = cols](const Node& self, Node& parent) {
                for (int i = 0; i < rows; i++) {
                    double dot = 0.0;
                    for (int j = 0; j < cols; j++) {
                        dot += self.grad[i * cols + j] * self.val[i * cols + j];
                    }
                    for (int j = 0; j < cols; j++) {
                        double y = self.val[i * cols + j];
                        parent.grad[i * cols + j] += y * (self.grad[i * cols + j] - dot);
                    }
                }
            }
        });
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::log() {
    Tensor Y(rows, cols, node->requiresGrad);

    std::vector<double> local(rows * cols, 0.0);
    for (int i = 0; i < rows * cols; i++) {
        double x = node->val[i];
        Y.node->val[i] = std::log(x);
        local[i] = 1.0 / x;
    }

    if (node->requiresGrad) {
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::exp() {
    Tensor Y(rows, cols, node->requiresGrad);

    std::vector<double> local(rows * cols, 0.0);
    for (int i = 0; i < rows * cols; i++) {
        double y = std::exp(node->val[i]);
        Y.node->val[i] = y;
        local[i] = y;
    }

    if (node->requiresGrad) {
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::abs() {
    Tensor Y(rows, cols, node->requiresGrad);

    std::vector<double> local(rows * cols, 0.0);
    for (int i = 0; i < rows * cols; i++) {
        double x = node->val[i];
        Y.node->val[i] = std::abs(x);
        local[i] = (x > 0.0) ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
    }

    if (node->requiresGrad) {
        Y.node->parents.push_back({node, std::move(local), {}});
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::sum() {
    Tensor Y(1, 1, node->requiresGrad);

    double total = 0.0;
    for (double v : node->val) {
        total += v;
    }
    Y.node->val[0] = total;

    if (node->requiresGrad) {
        Y.node->parents.push_back({
            node,
            {},
            [](const Node& self, Node& parent) {
                double g = self.grad[0];
                for (double& pg : parent.grad) {
                    pg += g;
                }
            }
        });
        node->pending_children += 1;
    }

    return Y;
}

Tensor Tensor::mean() {
    Tensor Y(1, 1, node->requiresGrad);

    const double n = static_cast<double>(node->val.size());
    double total = 0.0;
    for (double v : node->val) {
        total += v;
    }
    Y.node->val[0] = total / n;

    if (node->requiresGrad) {
        Y.node->parents.push_back({
            node,
            {},
            [n](const Node& self, Node& parent) {
                double g = self.grad[0] / n;
                for (double& pg : parent.grad) {
                    pg += g;
                }
            }
        });
        node->pending_children += 1;
    }

    return Y;
}

void Tensor::backward() {
    if (!node) {
        return;
    }

    if (node->val.size() == 1) {
        if (node->grad.size() == 1 && node->grad[0] == 0.0) {
            node->grad[0] = 1.0;
        }
    }

    std::vector<std::shared_ptr<Node>> nodes;
    nodes.push_back(node);

    while (!nodes.empty()) {
        auto back_node = nodes.back();
        nodes.pop_back();

        for (auto& edge : back_node->parents) {
            auto parent = edge.parent;

            if (edge.backward) {
                edge.backward(*back_node, *parent);
            } else if (!edge.local_grad.empty()) {
                if (edge.local_grad.size() != parent->grad.size() || back_node->grad.size() != parent->grad.size()) {
                    throw std::runtime_error("Tensor backward: gradient size mismatch");
                }
                for (size_t i = 0; i < parent->grad.size(); i++) {
                    parent->grad[i] += back_node->grad[i] * edge.local_grad[i];
                }
            }

            parent->pending_children -= 1;
            if (parent->pending_children == 0) {
                nodes.push_back(parent);
            }
        }
    }
}

Tensor matmul(Tensor& A, Tensor& B) {
    return A.matmul(B);
}
