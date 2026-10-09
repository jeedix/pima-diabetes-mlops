"""Pydantic-схемы конфигураций для ЛР3 (DL)."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DLDataConfig(BaseModel):
    """Пути и параметры данных."""

    parquet_path: Path
    target: str = "Outcome"
    test_size: float = Field(0.2, ge=0.1, le=0.4)
    seed: int = 42
    batch_size: int = Field(64, ge=8, le=512)

    @field_validator("parquet_path")
    @classmethod
    def path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Файл не найден: {v}")
        return v


class ArchitectureConfig(BaseModel):
    """Архитектура MLP."""

    hidden_sizes: list[int] = Field(default=[64, 32])
    dropout: float = Field(0.2, ge=0.0, le=0.9)
    use_missing_indicators: bool = True
    activation: Literal["relu", "tanh", "gelu"] = "relu"


class TrainingConfig(BaseModel):
    """Условия обучения."""

    mode: Literal["full", "chunks"]
    epochs: int = Field(100, ge=1, le=1000)
    lr: float = Field(1e-3, gt=0, le=1)
    weight_decay: float = Field(0.0, ge=0.0, le=1.0)
    early_stopping_patience: int = Field(10, ge=1, le=100)
    seed: int = 42
    mlflow_experiment: str = "lab3_pima_dl"


class Lab3Config(BaseModel):
    """Полный конфиг ЛР3."""

    data: DLDataConfig
    architecture: ArchitectureConfig
    training: TrainingConfig
