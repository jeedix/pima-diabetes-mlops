"""Pydantic-схемы конфигураций для ЛР2."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Пути и параметры данных."""

    raw_path: Path
    target: str = "Outcome"
    test_size: float = Field(0.2, ge=0.1, le=0.4)
    seed: int = 42
    chunk_size: int = Field(128, ge=16, le=1024)

    @field_validator("raw_path")
    @classmethod
    def path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Файл не найден: {v}")
        return v


class PipelineConfig(BaseModel):
    """Состав препроцессинга и модель."""

    imputer_strategy: Literal["mean", "median", "most_frequent"] = "median"
    scaler: Literal["standard", "minmax", "none"] = "standard"
    model: Literal["logreg", "sgd"] = "logreg"
    model_params: dict = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Режим обучения и метрики."""

    mode: Literal["full", "chunks"]
    n_bootstrap: int = Field(1000, ge=100, le=10000)
    mlflow_experiment: str = "lab2_pima"


class Lab2Config(BaseModel):
    """Полный конфиг ЛР2."""

    data: DataConfig
    pipeline: PipelineConfig
    training: TrainingConfig
