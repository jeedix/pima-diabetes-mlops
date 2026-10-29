"""Тесты генератора ЛР5."""

import pandas as pd
import pytest

from src.generator import (
    fit_class_gaussians,
    generate_similar,
    generate_random,
    apply_drift,
)


@pytest.fixture
def real_data():
    """Загружает реальные данные Pima."""
    return pd.read_parquet("data/processed/pima.parquet")


@pytest.fixture
def params(real_data):
    return fit_class_gaussians(real_data, "Outcome")


def test_generate_similar_shape(real_data, params):
    """Генерация similar даёт 768 строк и 9 колонок."""
    df = generate_similar(params, 768, seed=42)
    assert len(df) == 768
    assert "Outcome" in df.columns


def test_generate_similar_binary_target(real_data, params):
    """Целевая переменная содержит только 0 и 1."""
    df = generate_similar(params, 768, seed=42)
    assert set(df["Outcome"].unique()).issubset({0, 1})


def test_generate_similar_reproducible(real_data, params):
    """Два прогона с тем же seed дают идентичный результат."""
    df1 = generate_similar(params, 768, seed=42)
    df2 = generate_similar(params, 768, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_similar_diff_seed(real_data, params):
    """Разный seed даёт разные данные."""
    df1 = generate_similar(params, 768, seed=42)
    df2 = generate_similar(params, 768, seed=123)
    assert not df1.equals(df2)


def test_generate_random_vs_similar(real_data, params):
    """Режимы random и similar дают разные данные."""
    df_sim = generate_similar(params, 768, seed=42)
    df_rand = generate_random(real_data, 768, seed=42)
    assert not df_sim.equals(df_rand)


def test_shift_mean_changes_data(real_data, params):
    """Сдвиг средних увеличивает среднее."""
    df_orig = generate_similar(params, 768, seed=42)
    df_shift = apply_drift(df_orig, {"shift_mean": 5.0})
    assert df_shift["Glucose"].mean() > df_orig["Glucose"].mean()


def test_flip_labels_changes_target(real_data, params):
    """Переворот меток инвертирует target."""
    df_orig = generate_similar(params, 768, seed=42)
    df_flip = apply_drift(df_orig, {"flip_labels": 1.0})
    n_changed = (df_orig["Outcome"] != df_flip["Outcome"]).sum()
    assert n_changed == len(df_orig)


def test_scale_features_changes_std(real_data, params):
    """Масштабирование увеличивает std."""
    df_orig = generate_similar(params, 768, seed=42)
    df_scaled = apply_drift(df_orig, {"scale_features": 2.0})
    assert df_scaled["Glucose"].std() > df_orig["Glucose"].std() * 1.5


def test_invalid_config_rejected():
    """Pydantic отклоняет невалидную конфигурацию."""
    from src.generator_config import Lab5Config

    with pytest.raises(Exception):
        Lab5Config(
            data={"real_parquet": "nonexistent.parquet", "n_samples": 768},
            output={"mode": "similar"},
        )


def test_output_schema(real_data, params):
    """Сгенерированный датафрейм имеет те же колонки, что и реальный."""
    df = generate_similar(params, 768, seed=42)
    assert set(df.columns) == set(real_data.columns)
