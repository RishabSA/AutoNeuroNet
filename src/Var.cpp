#include "Var.hpp"

Var::Var() {
    node = std::make_shared<Node>();
};

Var::Var(double initial, bool requires_grad) {
    node = std::make_shared<Node>();

    node->val = initial;
    node->grad = 0.0;
    node->requires_grad = requires_grad;
};

double Var::getVal() const {
    return node->val;
};

void Var::setVal(double val) {
    node->val = val;
};

double Var::getGrad() const {
    return node->grad;
};

void Var::setGrad(double grad) {
    node->grad = grad;
};

bool Var::requiresGrad() const {
    return node->requires_grad;
};

void Var::resetGradAndParents() {
    // Resets the accumulated gradient for this variable, the counter used to track nodes that contribute to this variable, and any parent links
    // Detaches this node from its gradient computation graph

    node->grad = 0.0;
    node->pending_children = 0;
    node->parents.clear();
};

void Var::noGrad() {
    node->requires_grad = false;
    resetGradAndParents();
};

Var Var::detach() const {
    return Var(node->val, false);
};

Var Var::add(Var& other) {
    Var y(node->val + other.node->val, node->requires_grad || other.node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = 1.0
        y.node->parents.emplace_back(1.0, node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    if (other.node->requires_grad) {
        // Track the other variable as a parent of y

        // ∂y/other = 1.0
        y.node->parents.emplace_back(1.0, other.node);

        // Other variable has one more child contributing to its gradient
        other.node->pending_children += 1;
    }

    return y;
};

Var Var::add(double other) {
    Var y(node->val + other, node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = 1.0
        y.node->parents.emplace_back(1.0, node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::subtract(Var& other) {
    Var y(node->val - other.node->val, node->requires_grad || other.node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = 1.0
        y.node->parents.emplace_back(1.0, node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    if (other.node->requires_grad) {
        // Track the other variable as a parent of y

        // ∂y/∂other = -1.0
        y.node->parents.emplace_back(-1.0, other.node);

        // Other variable has one more child contributing to its gradient
        other.node->pending_children += 1;
    }

    return y;
};

Var Var::subtract(double other) {
    Var y(node->val - other, node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = 1.0
        y.node->parents.emplace_back(1.0, node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::multiply(Var& other) {
    Var y(node->val * other.node->val, node->requires_grad || other.node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = other.val
        y.node->parents.emplace_back(other.node->val, node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    if (other.node->requires_grad) {
        // Track the other variable as a parent of y

        // ∂y/other = val
        y.node->parents.emplace_back(node->val, other.node);

        // Other variable has one more child contributing to its gradient
        other.node->pending_children += 1;
    }

    return y;
};

Var Var::multiply(double other) {
    Var y(node->val * other, node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = other.val
        y.node->parents.emplace_back(other, node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::divide(Var& other) {
    Var y(node->val / other.node->val, node->requires_grad || other.node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = 1 / other.val
        y.node->parents.emplace_back(1.0 / other.node->val, node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    if (other.node->requires_grad) {
        // Track the other variable as a parent of y

        // ∂y/other = -value / other.val^2
        y.node->parents.emplace_back(-node->val / std::pow(other.node->val, 2), other.node);

        // Other variable has one more child contributing to its gradient
        other.node->pending_children += 1;
    }

    return y;
};

Var Var::divide(double other) {
    Var y(node->val / other, node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = 1 / other.val
        y.node->parents.emplace_back(1.0 / other, node);
        
        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::pow(int power) {
    Var y(std::pow(node->val, power), node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = power * val ** (power - 1)
        y.node->parents.emplace_back(power * std::pow(node->val, power - 1), node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::sin() {
    Var y(std::sin(node->val), node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = cos(val)
        y.node->parents.emplace_back(std::cos(node->val), node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::cos() {
    Var y(std::cos(node->val), node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = -sin(val)
        y.node->parents.emplace_back(-std::sin(node->val), node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::tan() {
    Var y(std::tan(node->val), node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = sec^2(val)
        y.node->parents.emplace_back(std::pow(1 / std::cos(node->val), 2), node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::sec() {
    double secant_val = 1 / std::cos(node->val);
    
    Var y(secant_val, node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = sec(val) * tan(val)
        y.node->parents.emplace_back(secant_val * std::tan(node->val), node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::csc() {
    double cosecant_val = 1 / std::sin(node->val);
    
    Var y(cosecant_val, node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = - csc(val) * cot(val)
        y.node->parents.emplace_back(-cosecant_val * (1 / std::tan(node->val)), node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::cot() {
    Var y(1 / std::tan(node->val), node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = -csc^2(val)
        y.node->parents.emplace_back(-std::pow(1 / std::sin(node->val), 2), node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

// Natural Log - base e
Var Var::log() {
    Var y(std::log(node->val), node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = 1/val
        y.node->parents.emplace_back(1 / node->val, node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::exp() {
    Var y(std::exp(node->val), node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = e^x
        y.node->parents.emplace_back(std::exp(node->val), node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::abs() {
    // |val|
    Var y(std::abs(node->val), node->requires_grad);

    double abs_derivative = 0.0;
    if (node->val > 0.0) abs_derivative = 1.0;
    else if (node->val < 0.0) abs_derivative = -1.0;

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = abs_derivative
        y.node->parents.emplace_back(abs_derivative, node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::relu() {
    // ReLU(val)= max(0, val)
    Var y(node->val > 0.0 ? node->val : 0.0, node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = 1 if val > 0 else 0
        y.node->parents.emplace_back(node->val > 0.0 ? 1.0 : 0.0, node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::leakyRelu(double alpha) {
    // LeakyReLU(val) = max(\alpha val, val)
    Var y(node->val > 0.0 ? node->val : alpha * node->val, node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = 1 if val > 0 else alpha
        y.node->parents.emplace_back(node->val > 0.0 ? 1.0 : alpha, node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::sigmoid() {
    // sigmoid(val) = \frac{1}{1 + e^{-val}}
    double sigmoid_val = 1.0 / (1.0 + std::exp(-node->val));

    Var y(sigmoid_val, node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = sigmoid(val) (1 - sigmoid(val))
        y.node->parents.emplace_back(sigmoid_val * (1.0 - sigmoid_val), node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::tanh() {
    // tanh(val) = \frac{e^val - e^{-val}}{e^val + e^{-val}}
    double tanh_val = std::tanh(node->val);
    
    Var y(tanh_val, node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = 1 - tanh^2(val)
        y.node->parents.emplace_back(1.0 - tanh_val * tanh_val, node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::silu() {
    // SiLU(val) = val * sigmoid(val)
    double sigmoid_val = 1.0 / (1.0 + std::exp(-node->val));
    
    Var y(node->val * sigmoid_val, node->requires_grad);
    
    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = sigmoid(val) + val * sigmoid(val) * (1 - sigmoid(val))
        y.node->parents.emplace_back(sigmoid_val + node->val * sigmoid_val * (1.0 - sigmoid_val), node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

Var Var::elu(double alpha) {
    // ELU(val) = val if val >= 0 else \alpha (e^x - 1)
    Var y(node->val > 0.0 ? node->val : alpha * (std::exp(node->val) - 1.0), node->requires_grad);

    if (node->requires_grad) {
        // Track the current variable as a parent of y

        // ∂y/∂this = 1 if val > 0 else alpha * exp(val)
        y.node->parents.emplace_back((node->val > 0.0) ? 1.0 : alpha * std::exp(node->val), node);

        // Current variable has one more child contributing to its gradient
        node->pending_children += 1;
    }

    return y;
};

void Var::backward() {
    if (!node) {
        return;
    }

    // Stack of nodes (LIFO) - start from the current variable
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.push_back(node);

    // Depth-first Search (DFS) through the stack
    while (!nodes.empty()) {
        std::shared_ptr<Node> current_node = nodes.back();
        nodes.pop_back();

        for (auto& p : current_node->parents) {
            double local_grad = p.first; // ∂current/∂parent
            std::shared_ptr<Node> parent = p.second;

            if (!parent->requires_grad) {
                parent->pending_children -= 1;
                continue;
            }

            // Accumulate gradients by the chain rule
            // ∂L/∂parent += ∂L/∂current * ∂current/∂parent
            parent->grad += current_node->grad * local_grad;

            parent->pending_children -= 1;

            // Now that gradients for the current node (child node) have been accumulated, add the parent node to the nodes stack to process it next
            if (parent->pending_children == 0) {
                nodes.push_back(parent);
            }
        }
    }
};