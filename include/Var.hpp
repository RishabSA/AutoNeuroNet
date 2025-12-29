#pragma once

#include <vector>
#include <utility>
#include <cmath>

class Var {
public:
    Var(double initial) {
        val = initial;
        visited = false;
        gradVal = 0.0;
    };

    ~Var() = default;

    double getVal() const { return val; };
    void setVal(double v) {
        val = v;
    };

    double getGradVal() const { return gradVal; };
    void setGradVal(double v) {
        gradVal = v;
    };

    void resetGradAndParents();

    Var add(Var& other);
    Var operator+(Var& other) { return add(other); };

    Var subtract(Var& other);
    Var operator-(Var& other) { return subtract(other); };

    Var multiply(Var& other);
    Var operator*(Var& other) { return multiply(other); };

    Var divide(Var& other);
    Var operator/(Var& other) { return divide(other); };

    Var pow(int power);

    Var sin();
    Var cos();
    Var tan();
    Var sec();
    Var csc();
    Var cot();

    Var log();

    Var exp();

    void backward();

private:
    double val;
    double gradVal;
    bool visited;
    std::vector<std::pair<double, Var*>> parents;
};