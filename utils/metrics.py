from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def compute_classification_metrics(y_true: list | np.ndarray, y_pred: list | np.ndarray) -> dict[str, float]:
    """Compute classification metrics: accuracy, f1_macro, precision, recall."""
    y_true_str = [str(x) for x in y_true]
    y_pred_str = [str(x) for x in y_pred]

    acc = float(accuracy_score(y_true_str, y_pred_str))
    f1 = float(f1_score(y_true_str, y_pred_str, average="macro", zero_division=0))
    prec = float(precision_score(y_true_str, y_pred_str, average="macro", zero_division=0))
    rec = float(recall_score(y_true_str, y_pred_str, average="macro", zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }


def compute_regression_metrics(y_true: list | np.ndarray, y_pred: list | np.ndarray) -> dict[str, float]:
    """Compute regression metrics: rmse, mae, r2, mse."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)

    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "mse": round(mse, 4),
    }
