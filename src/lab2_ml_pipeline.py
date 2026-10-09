"""ЛР2: пайплайн ML-модели — полный датасет и чанки."""

import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Lab2Config
from src.data_utils import iter_chunks, load_pima, split_data
from src.metrics_utils import bootstrap_ci, naive_baseline

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports" / "LAB2"
ARTIFACTS = ROOT / "artifacts"

REPORTS.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Lab2Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Lab2Config(**raw)


def build_pipeline(cfg: Lab2Config):
    """Собирает sklearn-пайплайн: импутация → масштабирование → модель."""
    steps = [
        ("imputer", SimpleImputer(strategy=cfg.pipeline.imputer_strategy)),
    ]
    if cfg.pipeline.scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif cfg.pipeline.scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        steps.append(("scaler", MinMaxScaler()))

    if cfg.pipeline.model == "logreg":
        model = LogisticRegression(**cfg.pipeline.model_params)
    elif cfg.pipeline.model == "sgd":
        model = SGDClassifier(**cfg.pipeline.model_params)
    else:
        raise ValueError(f"Неизвестная модель: {cfg.pipeline.model}")

    steps.append(("model", model))
    return Pipeline(steps)


def run_full(cfg: Lab2Config, X_train, X_test, y_train, y_test):
    """Обучение на полном датасете."""
    tracemalloc.start()
    t0 = time.perf_counter()
    pipe = build_pipeline(cfg)
    pipe.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = bootstrap_ci(y_test.values, y_pred, y_proba, cfg.training.n_bootstrap)
    return pipe, metrics, {"time_sec": t_fit, "peak_mem_mb": peak / 1e6}


def run_chunks(cfg: Lab2Config, X_train, X_test, y_train, y_test):
    """Обучение на чанках через partial_fit (только для SGD)."""
    if cfg.pipeline.model != "sgd":
        raise ValueError("Режим чанков требует model=sgd (partial_fit)")

    tracemalloc.start()
    t0 = time.perf_counter()

    imputer = SimpleImputer(strategy=cfg.pipeline.imputer_strategy)
    scaler = StandardScaler() if cfg.pipeline.scaler == "standard" else None

    # Первый проход: статистики импутера и scaler
    X_train_arr = X_train.to_numpy(dtype=float)
    y_train_arr = y_train.to_numpy()

    # Простая реализация: сначала импутация по train, затем partial_fit
    X_train_imp = imputer.fit_transform(X_train_arr)
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train_imp)
    else:
        X_train_scaled = X_train_imp

    classes = np.unique(y_train_arr)
    model = SGDClassifier(**cfg.pipeline.model_params)

    for X_chunk, y_chunk in iter_chunks(
        X_train_scaled, y_train_arr, cfg.data.chunk_size
    ):
        model.partial_fit(X_chunk, y_chunk, classes=classes)

    t_fit = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    X_test_arr = X_test.to_numpy(dtype=float)
    X_test_imp = imputer.transform(X_test_arr)
    X_test_scaled = scaler.transform(X_test_imp) if scaler is not None else X_test_imp

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics = bootstrap_ci(y_test.values, y_pred, y_proba, cfg.training.n_bootstrap)
    return model, metrics, {"time_sec": t_fit, "peak_mem_mb": peak / 1e6}


def save_model_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    full_cfg = load_config(CONFIGS / "lab2_full.yaml")
    chunks_cfg = load_config(CONFIGS / "lab2_chunks.yaml")

    df = load_pima(full_cfg.data.raw_path)
    X_train, X_test, y_train, y_test = split_data(
        df,
        full_cfg.data.target,
        full_cfg.data.test_size,
        full_cfg.data.seed,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    mlflow.set_experiment(full_cfg.training.mlflow_experiment)

    results = {}

    # --- Полный датасет ---
    with mlflow.start_run(run_name="full") as run_full_id:
        pipe, metrics, perf = run_full(full_cfg, X_train, X_test, y_train, y_test)
        naive = naive_baseline(y_test.values)
        mlflow.log_params({"mode": "full", **full_cfg.pipeline.model_params})
        for k, v in metrics.items():
            mlflow.log_metric(f"{k}_value", v["value"])
            mlflow.log_metric(f"{k}_ci_low", v["ci_low"])
            mlflow.log_metric(f"{k}_ci_high", v["ci_high"])
        mlflow.log_metric("time_sec", perf["time_sec"])
        mlflow.log_metric("peak_mem_mb", perf["peak_mem_mb"])
        mlflow.log_dict(naive, "naive_baseline.json")

        model_path = ARTIFACTS / "ml_model.joblib"
        joblib.dump(pipe, model_path)
        model_hash = save_model_hash(model_path)
        mlflow.log_param("model_hash", model_hash)

        results["full"] = {
            "run_id": run_full_id.info.run_id,
            "metrics": metrics,
            "naive": naive,
            "perf": perf,
            "model_hash": model_hash,
        }
        print(f"[full] run_id={run_full_id.info.run_id}")

    # --- Чанки ---
    with mlflow.start_run(run_name="chunks") as run_chunks_id:
        _, metrics, perf = run_chunks(chunks_cfg, X_train, X_test, y_train, y_test)
        mlflow.log_params({"mode": "chunks", **chunks_cfg.pipeline.model_params})
        for k, v in metrics.items():
            mlflow.log_metric(f"{k}_value", v["value"])
            mlflow.log_metric(f"{k}_ci_low", v["ci_low"])
            mlflow.log_metric(f"{k}_ci_high", v["ci_high"])
        mlflow.log_metric("time_sec", perf["time_sec"])
        mlflow.log_metric("peak_mem_mb", perf["peak_mem_mb"])

        results["chunks"] = {
            "run_id": run_chunks_id.info.run_id,
            "metrics": metrics,
            "perf": perf,
        }
        print(f"[chunks] run_id={run_chunks_id.info.run_id}")

    # --- Сохранение метрик в CSV ---
    import pandas as pd

    rows = []
    for mode, r in results.items():
        for metric, vals in r["metrics"].items():
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "value": vals["value"],
                    "ci_low": vals["ci_low"],
                    "ci_high": vals["ci_high"],
                    "run_id": r["run_id"],
                }
            )
    pd.DataFrame(rows).to_csv(REPORTS / "ml_metrics.csv", index=False)
    print(f"Метрики сохранены: {REPORTS / 'ml_metrics.csv'}")

    # --- Итоговый JSON ---
    (REPORTS / "lab2_summary.json").write_text(
        json.dumps(results, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
