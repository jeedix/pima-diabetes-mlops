"""Чтение данных и генерация чанков."""

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ZERO_AS_NAN_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_pima(path: Path) -> pd.DataFrame:
    """Читает CSV Pima, заменяет 0 в физиологически значимых колонках на NaN."""
    columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    ]
    df = pd.read_csv(path, header=None, names=columns)
    df[ZERO_AS_NAN_COLUMNS] = df[ZERO_AS_NAN_COLUMNS].replace(0, np.nan)
    return df


def split_data(df: pd.DataFrame, target: str, test_size: float, seed: int):
    """Стратифицированное разбиение train/test."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def iter_chunks(
    X: np.ndarray, y: np.ndarray, chunk_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Отдаёт чанки (X, y) последовательно."""
    n = len(X)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield X[start:end], y[start:end]
