"""Pydantic-схема настроек генератора ЛР5."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class GeneratorDataConfig(BaseModel):
    """Пути и параметры данных."""

    real_parquet: Path
    output_parquet: Path = Path("artifacts/synthetic_pima.parquet")
    target: str = "Outcome"
    n_samples: int = Field(768, ge=100, le=100_000)
    seed: int = 42

    @field_validator("real_parquet")
    @classmethod
    def path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Файл не найден: {v}")
        return v


class DriftConfig(BaseModel):
    """Ручки дрейфа."""

    shift_mean: float = Field(0.0, ge=-10.0, le=10.0)
    flip_labels: float = Field(0.0, ge=0.0, le=1.0)
    scale_features: float = Field(1.0, gt=0.0, le=10.0)


class GeneratorOutputConfig(BaseModel):
    """Режим генерации."""

    mode: Literal["random", "similar"]
    drift: DriftConfig = Field(default_factory=DriftConfig)


class Lab5Config(BaseModel):
    """Полный конфиг ЛР5."""

    data: GeneratorDataConfig
    output: GeneratorOutputConfig
    mlflow_experiment: str = "lab5_pima_generator"
