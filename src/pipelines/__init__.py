"""Пакет пайплайнов ЛР2: препроцессинг, обучение, метрики."""

from .chunks import train_chunks
from .full import train_full
from .metrics import bootstrap_ci, naive_baseline
from .preprocessing import build_preprocessor

__all__ = [
    "bootstrap_ci",
    "build_preprocessor",
    "naive_baseline",
    "train_chunks",
    "train_full",
]
