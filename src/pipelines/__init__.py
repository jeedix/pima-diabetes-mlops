"""Пакет пайплайнов ЛР2: препроцессинг, обучение, метрики."""

from .preprocessing import build_preprocessor
from .full import train_full
from .chunks import train_chunks
from .metrics import bootstrap_ci, naive_baseline

__all__ = [
    "build_preprocessor",
    "train_full",
    "train_chunks",
    "bootstrap_ci",
    "naive_baseline",
]
