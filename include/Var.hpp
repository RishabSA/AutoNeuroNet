#pragma once

#include <vector>
#include <utility>
#include <cmath>
#include <memory>

class Var {
public:
    struct Node {
        double val = 0.0;
        double grad = 0.0;
        int pending_children = 0;

        // Have to use shared_ptr because it keeps each Node alive until no Var refers to it, allowing for intermediate/temporary Var objects
        std::vector<std::pair<double, std::shared_ptr<Node>>> parents;
    };

    Var();
    Var(double initial);

    ~Var() = default;

    double getVal() const;
    void setVal(double v);

    double getGrad() const;
    void setGrad(double v);

    void resetGradAndParents();

    Var add(Var& other);
    Var operator+(Var& other) { return add(other); };

    Var subtract(Var& other);
    Var operator-(Var& other) { return subtract(other); };

    Var multiply(Var& other);
    Var operator*(Var& other) { return multiply(other); };

    Var divide(Var& other);
    Var operator/(Var& other) { return divide(other); };

    Var add(double other);
    Var operator+(double other) { return add(other); };

    Var subtract(double other);
    Var operator-(double other) { return subtract(other); };

    Var multiply(double other);
    Var operator*(double other) { return multiply(other); };

    Var divide(double other);
    Var operator/(double other) { return divide(other); };

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
    std::shared_ptr<Node> node;
};
