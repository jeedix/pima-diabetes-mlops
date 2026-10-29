"""Генератор синтетических данных Pima.

Режимы:
- «similar» — гауссы по классам (близко к реальным данным).
- «random» — шум в допустимых пределах (далеко от реальных).
"""

import numpy as np
import pandas as pd


ZERO_AS_NAN_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def fit_class_gaussians(df: pd.DataFrame, target: str) -> dict:
    """Оценивает mean и std по классам и признакам."""
    params = {}
    for cls in sorted(df[target].unique()):
        sub = df[df[target] == cls]
        params[int(cls)] = {
            "mean": sub.drop(columns=[target]).mean().to_dict(),
            "std": sub.drop(columns=[target]).std().to_dict(),
            "weight": len(sub) / len(df),
        }
    return params


def generate_similar(
    params: dict, n_samples: int, seed: int, real_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Генерирует данные бутстрэпом из реальных (близко к реальным)."""
    if real_df is None:
        # fallback — гауссы
        return _generate_gaussian(params, n_samples, seed)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(real_df), size=n_samples, replace=True)
    return real_df.iloc[idx].reset_index(drop=True)


def _generate_gaussian(params: dict, n_samples: int, seed: int) -> pd.DataFrame:
    """Запасной вариант — гауссы по классам."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    n_class0 = int(round(n_samples * params[0]["weight"]))
    n_class1 = n_samples - n_class0
    for cls, n_cls in [(0, n_class0), (1, n_class1)]:
        means = params[cls]["mean"]
        stds = params[cls]["std"]
        for _ in range(n_cls):
            rows.append({c: rng.normal(means[c], stds[c]) for c in means})
            labels.append(cls)
    df = pd.DataFrame(rows)
    df["Outcome"] = labels
    return df


def generate_random(df_real: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    """Генерирует шум в допустимых пределах (далеко от реальных)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()

    for col in df_real.columns:
        lo = df_real[col].min()
        hi = df_real[col].max()
        df[col] = rng.uniform(lo, hi, size=n_samples)

    # Бинарный target
    df["Outcome"] = rng.integers(0, 2, size=n_samples)
    return df


def apply_drift(df: pd.DataFrame, drift: dict, target: str = "Outcome") -> pd.DataFrame:
    """Применяет ручки дрейфа."""
    df = df.copy()

    # Сдвиг средних
    if drift.get("shift_mean", 0) != 0:
        for col in df.columns:
            if col != target:
                df[col] = df[col] + drift["shift_mean"]

    # Масштабирование
    if drift.get("scale_features", 1.0) != 1.0:
        for col in df.columns:
            if col != target:
                df[col] = df[col] * drift["scale_features"]

    # Переворот меток
    if drift.get("flip_labels", 0) > 0:
        rng = np.random.default_rng(42)
        flip_mask = rng.random(len(df)) < drift["flip_labels"]
        df[target] = df[target].where(~flip_mask, 1 - df[target])

    return df
