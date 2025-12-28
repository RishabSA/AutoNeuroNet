#include "Var.hpp"

void Var::resetGradAndParents() {
    visited = false;
    gradVal = 0.0;
    parents.clear();
}

Var Var::add(Var& other) {
    Var y(val + other.val);

    // ∂y/∂this = 1.0
    y.parents.emplace_back(1.0, this);

    // ∂y/∂other = 1.0
    y.parents.emplace_back(1.0, &other);

    return y;
};

Var Var::subtract(Var& other) {
    Var y(val - other.val);

    // ∂y/∂this = 1.0
    y.parents.emplace_back(1.0, this);

    // ∂y/∂other = -1.0
    y.parents.emplace_back(-1.0, &other);

    return y;
};

Var Var::multiply(Var& other) {
    Var y(val * other.val);

    // ∂y/∂this = other.val
    y.parents.emplace_back(other.val, this);

    // ∂y/∂other = val
    y.parents.emplace_back(val, &other);

    return y;
};

Var Var::divide(Var& other) {
    Var y(val / other.val);

    // ∂y/∂this = 1 / other.val
    y.parents.emplace_back(1.0 / other.val, this);

    // ∂y/∂other = -value / other.val^2
    y.parents.emplace_back(-val / std::pow(other.val, 2), &other);

    return y;
};

Var Var::pow(int power) {
    Var y(std::pow(val, power));

    // ∂y/∂this = power * val ** (power - 1)
    y.parents.emplace_back(power * std::pow(val, power - 1), this);

    return y;
};

// All trig functions are in radians

Var Var::sin() {
    Var y(std::sin(val));

    // ∂y/∂this = cos(val)
    y.parents.emplace_back(std::cos(val), this);

    return y;
};

Var Var::cos() {
    Var y(std::cos(val));

    // ∂y/∂this = -sin(val)
    y.parents.emplace_back(-std::sin(val), this);

    return y;
};

Var Var::tan() {
    Var y(std::tan(val));

    // ∂y/∂this = sec**2(val)
    y.parents.emplace_back(std::pow(1 / std::cos(val), 2), this);

    return y;
};

Var Var::sec() {
    double secant_val = 1 / std::cos(val);

    Var y(secant_val);

    // ∂y/∂this = sec(val) * tan(val)
    y.parents.emplace_back(secant_val * std::tan(val), this);

    return y;
};

Var Var::csc() {
    double cosecant_val = 1 / std::sin(val);

    Var y(cosecant_val);

    // ∂y/∂this = - csc(val) * cot(val)
    y.parents.emplace_back(-cosecant_val * (1 / std::tan(val)), this);

    return y;
};

Var Var::cot() {
    Var y(1 / std::tan(val));

    // ∂y/∂this = -csc**2(val)
    y.parents.emplace_back(-std::pow(1 / std::sin(val), 2), this);

    return y;
};

// Natural Log - base e
Var Var::log() {
    Var y(std::log(val));

    // ∂y/∂this = 1/val
    y.parents.emplace_back(1 / val, this);

    return y;
};

// log - base n
Var Var::log(int base) {
    if (val <= 0.0 || base <= 0.0 || base == 1.0) {
        return std::numeric_limits<double>::quiet_NaN(); 
    }

    // Use log base change rule with natural log
    Var y(std::log(val) / std::log(base));

    // ∂y/∂this = 1/(ln(base) * val)
    y.parents.emplace_back(1 / (std::log(base) * val), this);

    return y;
};

Var Var::exp() {
    Var y(std::exp(val));

    // ∂y/∂this = e^x
    y.parents.emplace_back(std::exp(val), this);

    return y;
};

void Var::backward() {
    if (visited) return;
    visited = true;

    for (auto& p : parents) {
        double local_grad = p.first;   // ∂this/∂parent
        Var* parent = p.second;

        parent->gradVal += gradVal * local_grad;  // dL/dparent += dL/dthis * dthis/dparent
        parent->backward();
    }
}
