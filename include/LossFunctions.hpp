#pragma once

#include <vector>
#include <utility>
#include "Matrix.hpp"

Var MSELoss(Matrix& labels, Matrix& preds);
Var MAELoss(Matrix& labels, Matrix& preds);
Var BCELoss(Matrix& labels, Matrix& preds, double eps = 1e-7);
Var CrossEntropyLoss(Matrix& labels, Matrix& preds, double eps = 1e-9);