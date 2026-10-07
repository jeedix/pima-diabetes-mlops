"""Построение препроцессора: импутация + масштабирование."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def build_preprocessor(
    imputer_strategy: str = "median", scaler: str = "standard"
) -> Pipeline:
    """Собирает шаги препроцессинга.

    Параметры
    ---------
    imputer_strategy : str
        'mean' | 'median' | 'most_frequent'
    scaler : str
        'standard' | 'minmax' | 'none'

    Возвращает
    ----------
    sklearn.pipeline.Pipeline
    """
    steps = [("imputer", SimpleImputer(strategy=imputer_strategy))]

    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "minmax":
        steps.append(("scaler", MinMaxScaler()))

    return Pipeline(steps)
