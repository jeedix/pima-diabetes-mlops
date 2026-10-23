"""Точка входа ЛР4: валидатор в режимах full и chunks."""

import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.validator_config import Lab4Config
from src.validator import PimaValidator, fit_chunked


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Lab4Config:
    with open(path, encoding="utf-8") as f:
        return Lab4Config(**yaml.safe_load(f))


def make_foreign_examples(X_test: np.ndarray, seed: int = 42):
    """Генерирует чужеродные примеры: сдвиг средних и выбросы."""
    rng = np.random.default_rng(seed)

    # 1. Сдвиг средних (все признаки)
    shifted = X_test + rng.normal(loc=3.0, scale=1.0, size=X_test.shape)

    # 2. Выбросы: только первые 8 числовых признаков, индикаторы NaN (последние 5) не трогаем
    outliers = X_test.copy()
    n_numeric = 8  # Pima: 8 признаков
    outliers[:10, :n_numeric] = outliers[:10, :n_numeric] * 3.0 + 20.0

    return {"shifted": shifted, "outliers": outliers}


def run_validator(cfg: Lab4Config, model_path: Path, mode: str = "full") -> dict:
    """Обучает валидатор, оценивает на test и чужеродных примерах."""
    df = pd.read_parquet(cfg.data.parquet_path)

    y = df[cfg.data.target].values
    X = df.drop(columns=[cfg.data.target]).values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.seed,
        stratify=y,
    )

    # Обучаем модель (или загружаем сохранённую из ЛР2)
    model = joblib.load(model_path) if model_path.exists() else None
    if model is None:
        pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        pipe.fit(X_train, y_train)
        model = pipe
        joblib.dump(model, model_path)

    # Обучаем валидатор
    if mode == "full":
        validator = PimaValidator(
            distance_threshold=cfg.rules.distance_threshold,
            confidence_threshold=cfg.rules.confidence_threshold,
        ).fit(X_train)
    else:
        validator, _ = fit_chunked(
            cfg.data.parquet_path,
            cfg.data.target,
            cfg.data.chunk_size,
        )

    # Предсказания на тесте — берём вероятность класса 1
    proba = model.predict_proba(X_test)[:, 1]
    reject_test = validator.reject(X_test, proba)

    # Чужеродные примеры
    foreign = make_foreign_examples(X_test, cfg.data.seed)
    foreign_stats = {}
    for name, X_foreign in foreign.items():
        proba_f = model.predict_proba(X_foreign)[:, 1]
        reject_f = validator.reject(X_foreign, proba_f)

        # Для выбросов считаем reject_rate только по первым 10 (реальным выбросам)
        if name == "outliers":
            reject_rate = float(reject_f[:10].mean())
        else:
            reject_rate = float(reject_f.mean())

        dist_f = validator.mahalanobis(X_foreign)
        conf_f = np.maximum(proba_f, 1 - proba_f)
        print(
            f"  {name}: dist_max={dist_f.max():.2f}, "
            f"dist_mean={dist_f.mean():.2f}, "
            f"conf_mean={conf_f.mean():.2f}, "
            f"reject_rate={reject_rate:.4f}"
        )

        foreign_stats[name] = {
            "n": int(len(X_foreign)),
            "reject_rate": reject_rate,
        }

    false_reject_rate = float(reject_test.mean())
    foreign_reject_rates = {k: v["reject_rate"] for k, v in foreign_stats.items()}

    # ROC-AUC валидатора
    X_all = [X_test] + [X_foreign for X_foreign in foreign.values()]
    y_binary = [np.zeros(len(X_test))] + [
        np.ones(len(X_foreign)) for X_foreign in foreign.values()
    ]

    X_all = np.vstack(X_all)
    y_binary = np.concatenate(y_binary)

    proba_all = model.predict_proba(X_all)[:, 1]
    scores = validator.score(X_all, proba_all)
    roc_auc = float(roc_auc_score(y_binary, scores))

    return {
        "validator": validator,
        "metrics": {
            "false_reject_rate": false_reject_rate,
            "foreign_reject_rates": foreign_reject_rates,
            "roc_auc": roc_auc,
        },
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def main():
    parser = argparse.ArgumentParser(description="ЛР4: валидатор")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "chunks"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.training.mode
    print(f"Mode: {mode}")

    model_path = ARTIFACTS / "ml_model.joblib"
    result = run_validator(cfg, model_path, mode=mode)

    # Сохраняем параметры
    params_path = ARTIFACTS / "validator_params.npz"
    result["validator"].save(params_path)

    # MLflow
    mlflow.set_experiment(cfg.training.mlflow_experiment)
    with mlflow.start_run(run_name=mode) as run:
        mlflow.log_params(
            {
                "mode": mode,
                "distance_threshold": cfg.rules.distance_threshold,
                "confidence_threshold": cfg.rules.confidence_threshold,
            }
        )
        mlflow.log_metric("false_reject_rate", result["metrics"]["false_reject_rate"])
        mlflow.log_metric("roc_auc", result["metrics"]["roc_auc"])
        for k, v in result["metrics"]["foreign_reject_rates"].items():
            mlflow.log_metric(f"foreign_reject_{k}", v)
        mlflow.log_artifact(str(params_path))

        print(f"[{mode}] run_id={run.info.run_id}")
        print(f"  false_reject_rate: {result['metrics']['false_reject_rate']:.4f}")
        print(f"  foreign_reject_rates: {result['metrics']['foreign_reject_rates']}")
        print(f"  roc_auc: {result['metrics']['roc_auc']:.4f}")
        print(f"  n_train: {result['n_train']}, n_test: {result['n_test']}")


if __name__ == "__main__":
    main()
