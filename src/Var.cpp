#include "Var.hpp"

Var::Var() {
    node = std::make_shared<Node>();
}

Var::Var(double initial) {
    node = std::make_shared<Node>();
    node->val = initial;
    node->grad = 0.0;
}

double Var::getVal() const {
    return node->val;
}

void Var::setVal(double v) {
    node->val = v;
}

double Var::getGrad() const {
    return node->grad;
}

void Var::setGrad(double v) {
    node->grad = v;
}

void Var::resetGradAndParents() {
    node->grad = 0.0;
    node->pending_children = 0;
    node->parents.clear();
}

Var Var::add(Var& other) {
    Var y(node->val + other.node->val);

    // ∂y/∂this = 1.0
    y.node->parents.emplace_back(1.0, node);
    node->pending_children += 1;

    // ∂y/other = 1.0
    y.node->parents.emplace_back(1.0, other.node);
    other.node->pending_children += 1;

    return y;
}

Var Var::add(double other) {
    Var y(node->val + other);

    // ∂y/∂this = 1.0
    y.node->parents.emplace_back(1.0, node);
    node->pending_children += 1;

    return y;
}

Var Var::subtract(Var& other) {
    Var y(node->val - other.node->val);

    // ∂y/∂this = 1.0
    y.node->parents.emplace_back(1.0, node);
    node->pending_children += 1;

    // ∂y/∂other = -1.0
    y.node->parents.emplace_back(-1.0, other.node);
    other.node->pending_children += 1;

    return y;
}

Var Var::subtract(double other) {
    Var y(node->val - other);

    // ∂y/∂this = 1.0
    y.node->parents.emplace_back(1.0, node);
    node->pending_children += 1;

    return y;
}

Var Var::multiply(Var& other) {
    Var y(node->val * other.node->val);

    // ∂y/∂this = other.val
    y.node->parents.emplace_back(other.node->val, node);
    node->pending_children += 1;

    // ∂y/other = val
    y.node->parents.emplace_back(node->val, other.node);
    other.node->pending_children += 1;

    return y;
}

Var Var::multiply(double other) {
    Var y(node->val * other);

    // ∂y/∂this = other.val
    y.node->parents.emplace_back(other, node);
    node->pending_children += 1;

    return y;
}

Var Var::divide(Var& other) {
    Var y(node->val / other.node->val);

    // ∂y/∂this = 1 / other.val
    y.node->parents.emplace_back(1.0 / other.node->val, node);
    node->pending_children += 1;

    // ∂y/other = -value / other.val^2
    y.node->parents.emplace_back(-node->val / std::pow(other.node->val, 2), other.node);
    other.node->pending_children += 1;

    return y;
}

Var Var::divide(double other) {
    Var y(node->val / other);

    // ∂y/∂this = 1 / other.val
    y.node->parents.emplace_back(1.0 / other, node);
    node->pending_children += 1;

    return y;
}

Var Var::pow(int power) {
    Var y(std::pow(node->val, power));

    // ∂y/∂this = power * val ** (power - 1)
    y.node->parents.emplace_back(power * std::pow(node->val, power - 1), node);
    node->pending_children += 1;

    return y;
}

Var Var::sin() {
    Var y(std::sin(node->val));

    // ∂y/∂this = cos(val)
    y.node->parents.emplace_back(std::cos(node->val), node);
    node->pending_children += 1;

    return y;
}

Var Var::cos() {
    Var y(std::cos(node->val));

    // ∂y/∂this = -sin(val)
    y.node->parents.emplace_back(-std::sin(node->val), node);
    node->pending_children += 1;

    return y;
}

Var Var::tan() {
    Var y(std::tan(node->val));

    // ∂y/∂this = sec^2(val)
    y.node->parents.emplace_back(std::pow(1 / std::cos(node->val), 2), node);
    node->pending_children += 1;

    return y;
}

Var Var::sec() {
    double secant_val = 1 / std::cos(node->val);
    Var y(secant_val);

    // ∂y/∂this = sec(val) * tan(val)
    y.node->parents.emplace_back(secant_val * std::tan(node->val), node);
    node->pending_children += 1;

    return y;
}

Var Var::csc() {
    double cosecant_val = 1 / std::sin(node->val);
    Var y(cosecant_val);

    // ∂y/∂this = - csc(val) * cot(val)
    y.node->parents.emplace_back(-cosecant_val * (1 / std::tan(node->val)), node);
    node->pending_children += 1;

    return y;
}

Var Var::cot() {
    Var y(1 / std::tan(node->val));

    // ∂y/∂this = -csc^2(val)
    y.node->parents.emplace_back(-std::pow(1 / std::sin(node->val), 2), node);
    node->pending_children += 1;

    return y;
}

// Natural Log - base e
Var Var::log() {
    Var y(std::log(node->val));

    // ∂y/∂this = 1/val
    y.node->parents.emplace_back(1 / node->val, node);
    node->pending_children += 1;

    return y;
}

Var Var::exp() {
    Var y(std::exp(node->val));

    // ∂y/∂this = e^x
    y.node->parents.emplace_back(std::exp(node->val), node);
    node->pending_children += 1;

    return y;
}


Var Var::relu() {
    Var y(node->val > 0.0 ? node->val : 0.0);

    // ∂y/∂this = 1 if val > 0 else 0
    y.node->parents.emplace_back(node->val > 0.0 ? 1.0 : 0.0, node);
    node->pending_children += 1;

    return y;
}

Var Var::leakyRelu(double alpha) {
    Var y(node->val > 0.0 ? node->val : alpha * node->val);

    // ∂y/∂this = 1 if val > 0 else alpha
    y.node->parents.emplace_back(node->val > 0.0 ? 1.0 : alpha, node);
    node->pending_children += 1;

    return y;
}

Var Var::sigmoid() {
    double simoid_val = 1.0 / (1.0 + std::exp(-node->val));
    Var y(simoid_val);

    // ∂y/∂this = s * (1 - s)
    y.node->parents.emplace_back(simoid_val * (1.0 - simoid_val), node);
    node->pending_children += 1;

    return y;
}

Var Var::tanh() {
    double tanh_val = std::tanh(node->val);
    Var y(tanh_val);

    // ∂y/∂this = 1 - tanh^2(val)
    y.node->parents.emplace_back(1.0 - tanh_val * tanh_val, node);
    node->pending_children += 1;

    return y;
}

Var Var::silu() {
    double silu_val = 1.0 / (1.0 + std::exp(-node->val));
    Var y(node->val * silu_val);

    // ∂y/∂this = silu_val + x * silu_val * (1 - silu_val)
    double grad = silu_val + node->val * silu_val * (1.0 - silu_val);
    y.node->parents.emplace_back(grad, node);
    node->pending_children += 1;

    return y;
}

Var Var::elu(double alpha) {
    Var y(node->val > 0.0 ? node->val : alpha * (std::exp(node->val) - 1.0));

    // ∂y/∂this = 1 if val > 0 else alpha * exp(val)
    double grad = (node->val > 0.0) ? 1.0 : alpha * std::exp(node->val);
    y.node->parents.emplace_back(grad, node);
    node->pending_children += 1;

    return y;
}

void Var::backward() {
    if (!node) {
        return;
    }

    std::vector<std::shared_ptr<Node>> nodes;
    nodes.push_back(node);

    while (!nodes.empty()) {
        std::shared_ptr<Node> back_node = nodes.back();
        nodes.pop_back();

        for (auto& p : back_node->parents) {
            double local_grad = p.first; // ∂this/∂parent
            std::shared_ptr<Node> parent = p.second;

            parent->grad += back_node->grad * local_grad;  // dL/dparent += dL/dthis * dthis/dparent
            parent->pending_children -= 1;

            if (parent->pending_children == 0) {
                nodes.push_back(parent);
            }
        }
    }
}
