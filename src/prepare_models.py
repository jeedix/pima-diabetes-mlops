"""Создаёт две версии моделей для теста переключения (ЛР6)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_version(seed: int) -> Pipeline:
    df = pd.read_parquet("data/processed/pima.parquet")
    y = df["Outcome"].values
    X = df.drop(columns=["Outcome"]).values
    X = np.nan_to_num(X, nan=0.0)
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


if __name__ == "__main__":
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    for v, seed in [("v1", 42), ("v2", 123)]:
        model = train_version(seed)
        joblib.dump(model, f"artifacts/models/{v}.joblib")
        print(f"Сохранено: artifacts/models/{v}.joblib")
