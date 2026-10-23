"""Pydantic-схема настроек валидатора ЛР4."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ValidatorDataConfig(BaseModel):
    """Пути и параметры данных."""

    parquet_path: Path
    target: str = "Outcome"
    test_size: float = Field(0.2, ge=0.1, le=0.4)
    seed: int = 42
    chunk_size: int = Field(128, ge=16, le=1024)

    @field_validator("parquet_path")
    @classmethod
    def path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Файл не найден: {v}")
        return v


class ValidatorRuleConfig(BaseModel):
    """Правила отказа."""

    distance_threshold: float = Field(5.0, gt=0)
    confidence_threshold: float = Field(0.6, gt=0, lt=1)
    use_mahalanobis: bool = True


class ValidatorTrainingConfig(BaseModel):
    """Режим обучения валидатора."""

    mode: Literal["full", "chunks"]
    mlflow_experiment: str = "lab4_pima_validator"


class Lab4Config(BaseModel):
    """Полный конфиг ЛР4."""

    data: ValidatorDataConfig
    rules: ValidatorRuleConfig
    training: ValidatorTrainingConfig
