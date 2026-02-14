#include "LossFunctions.hpp"

#include <stdexcept>

Var MSELoss(Matrix& labels, Matrix& preds) {
    if (labels.rows != preds.rows || labels.cols != preds.cols) {
        throw std::runtime_error("Dimension mismatch when attempting to compute loss - labels: (" + std::to_string(labels.rows) + ", " + std::to_string(labels.cols) + ") preds: (" + std::to_string(preds.rows) + ", " + std::to_string(preds.cols) + ")");
    }

    Var loss(0.0);

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            Var errors = labels.data[i][j] - preds.data[i][j];
            Var squared_errors = errors.pow(2);
            loss = loss + squared_errors;
        }
    }

    Var total(labels.rows * labels.cols);
    loss = loss / total;

    return loss;
};

Var MAELoss(Matrix& labels, Matrix& preds) {
    if (labels.rows != preds.rows || labels.cols != preds.cols) {
        throw std::runtime_error("Dimension mismatch when attempting to compute loss - labels: (" + std::to_string(labels.rows) + ", " + std::to_string(labels.cols) + ") preds: (" + std::to_string(preds.rows) + ", " + std::to_string(preds.cols) + ")");
    }

    Var loss(0.0);

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            Var errors = labels.data[i][j] - preds.data[i][j];
            Var absolute_errors = errors.abs();
            loss = loss + absolute_errors;
        }
    }

    Var total(labels.rows * labels.cols);
    loss = loss / total;

    return loss;
};

Var BCELoss(Matrix& labels, Matrix& preds, double eps) {
    if (labels.rows != preds.rows || labels.cols != preds.cols) {
        throw std::runtime_error("Dimension mismatch when attempting to compute loss - labels: (" + std::to_string(labels.rows) + ", " + std::to_string(labels.cols) + ") preds: (" + std::to_string(labels.rows) + ", " + std::to_string(labels.cols) + ")");
    }

    Var loss(0.0);

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            Var& y = labels.data[i][j];
            Var& p = preds.data[i][j];

            Var p_eps = p + eps;
            Var one(1.0);
            Var one_minus_p = one - p;
            Var one_minus_p_eps = one_minus_p + eps;

            Var logp = p_eps.log();
            Var log_one_minus_p = one_minus_p_eps.log();

            Var term1 = y.multiply(logp);
            Var one_minus_y = one - y;
            Var term2 = one_minus_y.multiply(log_one_minus_p);

            Var sum = term1 + term2;
            loss = loss - sum;
        }
    }

    Var total(labels.rows * labels.cols);
    loss = loss / total;

    return loss;
};

Var CrossEntropyLoss(Matrix& labels, Matrix& preds, double eps) {
    if (labels.rows != preds.rows) {
        throw std::runtime_error("Dimension mismatch in CrossEntropyLoss - labels rows: (" + std::to_string(labels.rows) + ") preds rows: (" + std::to_string(preds.rows) + ")");
    }

    if (labels.cols != 1) {
        throw std::runtime_error("CrossEntropyLoss expects labels shape (N, 1) with class indices");
    }

    Var loss(0.0);

    for (int i = 0; i < labels.rows; i++) {
        int cls = static_cast<int>(labels.data[i][0].getVal());
        if (cls < 0 || cls >= preds.cols) {
            throw std::runtime_error("CrossEntropyLoss class index out of range at row " + std::to_string(i));
        }

        Var p = preds.data[i][cls];
        Var p_eps = p + eps;
        Var logp = p_eps.log();
        loss = loss - logp;
    }

    Var rows(labels.rows);
    loss = loss / rows;

    return loss;
}

Var CrossEntropyLossWithLogits(Matrix& labels, Matrix& logits, double eps) {
    if (labels.rows != logits.rows) { /* ... */ }
    if (labels.cols != 1) { /* ... */ }

    Var loss(0.0);

    for (int i = 0; i < labels.rows; i++) {
        int cls = static_cast<int>(labels.data[i][0].getVal());
        if (cls < 0 || cls >= logits.cols) { /* ... */ }

        double row_max = logits.data[i][0].getVal();
        for (int j = 1; j < logits.cols; j++) {
            row_max = std::max(row_max, logits.data[i][j].getVal());
        }

        Var sum(0.0);
        for (int j = 0; j < logits.cols; j++) {
            Var e((logits.data[i][j] - row_max).exp());
            sum = sum + e;
        }

        Var logsumexp = sum.log() + row_max;
        Var logp = logits.data[i][cls] - logsumexp;
        loss = loss - logp;
    }

    Var rows(labels.rows);
    loss = loss / rows;

    return loss;
}
