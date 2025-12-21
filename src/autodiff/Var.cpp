#include "autodiff/Var.hpp"

Var Var::add(Var& other) {
    Var y(val + other.val);

    // dy/d_self = 1.0
    children.emplace_back(1.0, &y);

    // dy/d_other = 1.0
    other.children.emplace_back(1.0, &y);

    return y;
};

Var Var::subtract(Var& other) {
    Var y(val - other.val);

    // dy/d_self = 1.0
    children.emplace_back(1.0, &y);

    // dy/d_other = -1.0
    other.children.emplace_back(-1.0, &y);

    return y;
};

Var Var::multiply(Var& other) {
    Var y(val * other.val);

    // dy/d_self = other.val
    children.emplace_back(other.val, &y);

    // dy/d_other = val
    other.children.emplace_back(val, &y);

    return y;
};

Var Var::divide(Var& other) {
    Var y(val / other.val);

    // dy/d_self = 1 / other.val
    children.emplace_back(1.0 / other.val, &y);

    // dy/d_other = -value / other.val^2
    other.children.emplace_back(-val / std::pow(other.val, 2), &y);

    return y;
};

Var Var::pow(int power) {
    Var y(std::pow(val, power));

    // dy/d_self = power * val ** (power - 1)
    children.emplace_back(power * std::pow(val, power - 1), &y);

    return y;
};

// All trig functions are in radians

Var Var::sin() {
    Var y(std::sin(val));

    // dy/d_self = cos(val)
    children.emplace_back(std::cos(val), &y);

    return y;
};

Var Var::cos() {
    Var y(std::cos(val));

    // dy/d_self = -sin(val)
    children.emplace_back(-std::sin(val), &y);

    return y;
};

Var Var::tan() {
    Var y(std::tan(val));

    // dy/d_self = sec**2(val)
    children.emplace_back(std::pow(1 / std::cos(val), 2), &y);

    return y;
};

Var Var::sec() {
    double secant_val = 1 / std::cos(val);

    Var y(secant_val);

    // dy/d_self = sec(val) * tan(val)
    children.emplace_back(secant_val * std::tan(val), &y);

    return y;
};

Var Var::csc() {
    double cosecant_val = 1 / std::sin(val);

    Var y(cosecant_val);

    // dy/d_self = - csc(val) * cot(val)
    children.emplace_back(-cosecant_val * (1 / std::tan(val)), &y);

    return y;
};

Var Var::cot() {
    Var y(1 / std::tan(val));

    // dy/d_self = -csc**2(val)
    children.emplace_back(-std::pow(1 / std::sin(val), 2), &y);

    return y;
};

// Natural Log - base e
Var Var::log() {
    Var y(std::log(val));

    // dy/d_self = 1/val
    children.emplace_back(1 / val, &y);

    return y;
};

// log - base n
Var Var::log(int base) {
    if (val <= 0.0 || base <= 0.0 || base == 1.0) {
        return std::numeric_limits<double>::quiet_NaN(); 
    }

    // Use log base change rule with natural log
    Var y(std::log(val) / std::log(base));

    // dy/d_self = 1/(ln(base) * val)
    children.emplace_back(1 / (std::log(base) * val), &y);

    return y;
};

Var Var::exp() {
    Var y(std::exp(val));

    // dy/d_self = e^x
    children.emplace_back(std::exp(val), &y);

    return y;
};

double Var::grad() {
    if (std::isnan(gradVal)) {
        // Compute derivative with the chain rule
        double sum = 0.0;
        
        for (auto child : children) {
            sum += std::get<0>(child) * std::get<1>(child)->grad();
        }

        gradVal = sum;
    }

    return gradVal;
};
