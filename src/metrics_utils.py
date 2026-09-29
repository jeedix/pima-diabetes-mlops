"""Метрики с бутстрэп-интервалами."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Считает метрики и их 95% доверительные интервалы через бутстрэп."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def _metrics(yt, yp, ypr):
        return {
            "accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "roc_auc": roc_auc_score(yt, ypr),
        }

    point = _metrics(y_true, y_pred, y_proba)

    boots = {k: [] for k in point}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = _metrics(y_true[idx], y_pred[idx], y_proba[idx])
        for k, v in m.items():
            boots[k].append(v)

    result = {}
    for k, v in point.items():
        lo = float(np.percentile(boots[k], 100 * alpha / 2))
        hi = float(np.percentile(boots[k], 100 * (1 - alpha / 2)))
        result[k] = {"value": float(v), "ci_low": lo, "ci_high": hi}
    return result


def naive_baseline(y_true: np.ndarray) -> dict:
    """Наивный ориентир: всегда предсказываем мажоритарный класс."""
    majority = int(np.round(np.mean(y_true)))
    y_pred = np.full_like(y_true, majority)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
