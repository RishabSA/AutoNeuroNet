#include "LossFunctions.hpp"

Var MSELoss(Matrix& labels, Matrix& preds) {
    if (labels.rows != preds.rows || labels.cols != preds.cols) {
        throw std::runtime_error("Dimension mismatch when attempting to compute loss - labels: (" + std::to_string(labels.rows) + ", " + std::to_string(labels.cols) + ") preds: (" + std::to_string(preds.rows) + ", " + std::to_string(preds.cols) + ")");
    }

    // MSE(\hat{y}, y) = \frac{1}{m}\sum_{i = 1}^m(\hat{y}_i - y_i)^2
    Var loss(0.0, false);

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            Var errors = labels.data[i][j] - preds.data[i][j];
            Var squared_errors = errors.pow(2);
            loss = loss + squared_errors;
        }
    }

    loss = loss / (labels.rows * labels.cols);
    return loss;
};

Var MAELoss(Matrix& labels, Matrix& preds) {
    if (labels.rows != preds.rows || labels.cols != preds.cols) {
        throw std::runtime_error("Dimension mismatch when attempting to compute loss - labels: (" + std::to_string(labels.rows) + ", " + std::to_string(labels.cols) + ") preds: (" + std::to_string(preds.rows) + ", " + std::to_string(preds.cols) + ")");
    }

    // MAE(\hat{y}, y) = \frac{1}{m}\sum_{i = 1}^m | \hat{y}_i - y_i |
    Var loss(0.0, false);

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            Var errors = labels.data[i][j] - preds.data[i][j];
            Var absolute_errors = errors.abs();
            loss = loss + absolute_errors;
        }
    }

    loss = loss / (labels.rows * labels.cols);
    return loss;
};

Var BCELoss(Matrix& labels, Matrix& preds, double eps) {
    if (labels.rows != preds.rows || labels.cols != preds.cols) {
        throw std::runtime_error("Dimension mismatch when attempting to compute loss - labels: (" + std::to_string(labels.rows) + ", " + std::to_string(labels.cols) + ") preds: (" + std::to_string(labels.rows) + ", " + std::to_string(labels.cols) + ")");
    }

    // BCE(\hat{y}, y) = - \frac{1}{m}\sum_{i = 1}^m[y_i \ln (\hat{y}_i) + (1-y_i) \ln(1 - \hat{y}_i)]
    Var loss(0.0, false);

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            Var& y = labels.data[i][j];
            Var& p = preds.data[i][j];

            Var log_p = (p + eps).log();
            Var log_one_minus_p = ((Var(1.0, false) - p) + eps).log();
            Var one_minus_y = Var(1.0, false) - y;

            Var term1 = y * log_p;
            Var term2 = one_minus_y * log_one_minus_p;
            Var sum = term1 + term2;

            loss = loss - sum;
        }
    }

    loss = loss / (labels.rows * labels.cols);
    return loss;
};

Var CrossEntropyLoss(Matrix& labels, Matrix& preds, double eps) {
    if (labels.rows != preds.rows) {
        throw std::runtime_error("Dimension mismatch in CrossEntropyLoss - labels rows: (" + std::to_string(labels.rows) + ") preds rows: (" + std::to_string(preds.rows) + ")");
    }

    if (labels.cols != 1) {
        throw std::runtime_error("CrossEntropyLoss expects labels shape (N, 1) with class indices");
    }

    // CE(\hat{y}, y) = - \frac{1}{m}\sum_{i = 1}^m \sum_{k = 1}^K [y_{i, k} \ln(\hat{y}_{i, k})]
    Var loss(0.0, false);

    for (int i = 0; i < labels.rows; i++) {
        int class_idx = static_cast<int>(labels.data[i][0].getVal());
        if (class_idx < 0 || class_idx >= preds.cols) {
            throw std::runtime_error("CrossEntropyLoss class index out of range at row " + std::to_string(i));
        }

        Var p = preds.data[i][class_idx];
        Var log_p = (p + eps).log();

        loss = loss - log_p;
    }

    loss = loss / labels.rows;
    return loss;
};

Var CrossEntropyLossWithLogits(Matrix& labels, Matrix& logits, double eps) {
    if (labels.rows != logits.rows) {
        throw std::runtime_error("Dimension mismatch in CrossEntropyLoss - labels rows: (" + std::to_string(labels.rows) + ") logits rows: (" + std::to_string(logits.rows) + ")");
    }

    if (labels.cols != 1) {
        throw std::runtime_error("CrossEntropyLoss expects labels shape (N, 1) with class indices");
    }

    // CE(\hat{y}, y) = - \frac{1}{m}\sum_{i = 1}^m \sum_{k = 1}^K [y_{i, k} \ln(\hat{y}_{i, k})]
    Var loss(0.0, false);

    for (int i = 0; i < labels.rows; i++) {
        int class_idx = static_cast<int>(labels.data[i][0].getVal());
        if (class_idx < 0 || class_idx >= logits.cols) {
            throw std::runtime_error("CrossEntropyLoss class index out of range at row " + std::to_string(i));
        }

        double row_max = logits.data[i][0].getVal();
        for (int j = 1; j < logits.cols; j++) {
            row_max = std::max(row_max, logits.data[i][j].getVal());
        }

        // Softmax
        Var sum(0.0, false);
        for (int j = 0; j < logits.cols; j++) {
            Var exp_raw_score((logits.data[i][j] - row_max).exp());
            sum = sum + exp_raw_score;
        }

        Var log_sum_exp = sum.log() + row_max;
        Var log_p = logits.data[i][class_idx] - log_sum_exp;

        loss = loss - log_p;
    }

    loss = loss / labels.rows;
    return loss;
};