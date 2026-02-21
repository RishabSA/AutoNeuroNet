#pragma once

#include <vector>
#include <utility>
#include <cmath>
#include <memory>

/*
Var is a scalar value capable of reverse-mode automatic differentiation and several operations.
Var objects can perform arithmetic operations, trigonometric functions, exponentials, absolute values, and activation functions while tracking all derivatives.

Each Var contains a Node that stores:
    - value
    - gradient
    - counter for backpropagation order
    - list of parent nodes with their local derivatives

A Node struct is used within each Var because graph edges are built using pointers to parent nodes, and those pointers must stay stable and refer to a single shared node per variable.

When any operation is performed:
- A new Var is created with its own Node
    - The new node records its parent (the creating variable(s)) and the local gradient for each parent

- Forms a directed acylcic graoh (DAG) from the output variable back to the input variable(s).
    - The graph flows in one direction with no cycles

Backpropagation:
- Set the final output gardient to 1.0
- Perform backpropagation through the graph
    - pending_children is used to ensure that all children have contributed to the gradient
    - For each edge, the parent gradient is incremented by child.grad * local_grad  
        - ∂L/∂parent += ∂L/∂this * ∂this/∂parent
- If requires_grad is false for a variable, it is skipped in the backpropagation graph
    - noGrad() and detach() break the gradient history
*/
class Var {
public:
    struct Node {
        double val = 0.0;
        double grad = 0.0;
        int pending_children = 0;
        bool requires_grad = true;

        /*
        Why is shared_ptr used?

        shared_ptr keeps each Node alive until no Var refers to it, allowing for intermediate/temporary Var objects
        It keeps the computation graph alive across any temporary Vars that are created
        Any node referenced as a parent stays alive until all downstream Var objects are done with it

        If nodes were stored as raw pointers, parent nodes could be destroyed as their owning Var objects go out of scope, leaving dangling pointers in the graph and messing up backward()
        */

        // Stores the local derivative ∂current/∂parent and the parent variable's node
        std::vector<std::pair<double, std::shared_ptr<Node>>> parents;
    };

    Var();
    Var(double initial, bool requires_grad = true);

    ~Var() = default;

    double getVal() const;
    void setVal(double val);

    double getGrad() const;
    void setGrad(double grad);
    bool requiresGrad() const;

    void resetGradAndParents();
    void noGrad();
    Var detach() const;

    // For arithmetic operations (+, -, *, /), operator overloads are given for Var and double

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

    // Trigonometric functions
    Var sin();
    Var cos();
    Var tan();
    Var sec();
    Var csc();
    Var cot();

    Var log();

    Var exp();

    Var abs();

    // Activation functions
    Var relu();
    Var leakyRelu(double alpha = 0.01);
    Var sigmoid();
    Var tanh();
    Var silu();
    Var elu(double alpha = 1.0);

    void backward();

private:
    std::shared_ptr<Node> node;
};
