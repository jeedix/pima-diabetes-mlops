"""Проверка воспроизводимости ЛР2.

4 сценария:
[1] Обучение seed=42, сохранение модели
[2] Обучение seed=42 повторно → должно совпасть с [1]
[3] Загрузка модели из [1] → предсказания совпадают с [1]
[4] Обучение seed=123 → должно отличаться от [1]
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.config import Lab2Config
from src.data_utils import split_data
from src.pipelines import train_full

PARQUET = "data/processed/pima.parquet"
CONFIG = "configs/lab2_full.yaml"
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def train_with_seed(df, cfg_raw, seed):
    """Обучает модель с заданным seed, возвращает модель и предсказания."""
    cfg_raw["data"]["seed"] = seed
    cfg = Lab2Config(**cfg_raw)
    X_train, X_test, y_train, y_test = split_data(
        df,
        cfg.data.target,
        cfg.data.test_size,
        seed,
    )
    result = train_full(X_train, X_test, y_train, y_test, cfg)
    preds = result["model"].predict(result["X_test_t"])  # ← X_test_t вместо X_test
    return result["model"], preds


def compare(a: np.ndarray, b: np.ndarray) -> float:
    """Доля совпадающих предсказаний."""
    return float((a == b).mean())


def main():
    df = pd.read_parquet(PARQUET)
    with open(CONFIG, encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)

    # [1] Обучение seed=42, сохранение
    cfg_raw["data"]["seed"] = 42
    cfg = Lab2Config(**cfg_raw)
    X_train, X_test, y_train, y_test = split_data(
        df,
        cfg.data.target,
        cfg.data.test_size,
        42,
    )
    result1 = train_full(X_train, X_test, y_train, y_test, cfg)
    preds1 = result1["model"].predict(result1["X_test_t"])
    joblib.dump(result1["model"], ARTIFACTS / "repro_model_42.joblib")
    print("[1] Модель обучена с seed=42, сохранена в repro_model_42.joblib")

    # [2] Обучение seed=42 повторно → сравнение с [1]
    result2 = train_full(X_train, X_test, y_train, y_test, cfg)
    preds2 = result2["model"].predict(result2["X_test_t"])
    print(
        f"[2] Совпадение предсказаний (seed=42 vs seed=42): {compare(preds1, preds2):.4f}"
    )

    # [3] Загрузка модели из файла → сравнение с [1]
    loaded = joblib.load(ARTIFACTS / "repro_model_42.joblib")
    preds_loaded = loaded.predict(result1["X_test_t"])
    print(
        f"[3] Совпадение предсказаний (loaded vs original): {compare(preds1, preds_loaded):.4f}"
    )

    # [4] Обучение seed=123 → сравнение с [1]
    cfg_raw["data"]["seed"] = 123
    cfg123 = Lab2Config(**cfg_raw)
    X_train_123, X_test_123, y_train_123, y_test_123 = split_data(
        df,
        cfg123.data.target,
        cfg123.data.test_size,
        123,
    )
    result4 = train_full(X_train_123, X_test_123, y_train_123, y_test_123, cfg123)
    preds4 = result4["model"].predict(result4["X_test_t"])
    print(
        f"[4] Совпадение предсказаний (seed=42 vs seed=123): {compare(preds1, preds4):.4f}"
    )


if __name__ == "__main__":
    main()
