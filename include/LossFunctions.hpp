#pragma once

#include <vector>
#include <utility>
#include <stdexcept>

#include "Matrix.hpp"

Var MSELoss(Matrix& labels, Matrix& preds);
Var MAELoss(Matrix& labels, Matrix& preds);
Var BCELoss(Matrix& labels, Matrix& preds, double eps = 1e-7);
Var CrossEntropyLoss(Matrix& labels, Matrix& preds, double eps = 1e-9);
Var CrossEntropyLossWithLogits(Matrix& labels, Matrix& logits, double eps = 1e-9);