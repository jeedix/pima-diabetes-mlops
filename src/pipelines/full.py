"""Обучение на полном датасете (в памяти)."""

from sklearn.linear_model import LogisticRegression

from src.monitoring import measure_resources
from src.pipelines.metrics import bootstrap_ci
from src.pipelines.preprocessing import build_preprocessor


def train_full(X_train, X_test, y_train, y_test, cfg) -> dict:
    """Обучает модель на полном датасете, возвращает метрики и ресурсы."""
    pre = build_preprocessor(
        imputer_strategy=cfg.pipeline.imputer_strategy,
        scaler=cfg.pipeline.scaler,
    )
    model = LogisticRegression(**cfg.pipeline.model_params)

    # Замер: время + память + CPU через psutil
    with measure_resources() as res:
        X_train_t = pre.fit_transform(X_train)
        X_test_t = pre.transform(X_test)
        model.fit(X_train_t, y_train)

    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]
    metrics = bootstrap_ci(y_test.values, y_pred, y_proba, cfg.training.n_bootstrap)

    return {
        "model": model,
        "preprocessor": pre,
        "X_test_t": X_test_t,
        "metrics": metrics,
        "perf": res,
    }
