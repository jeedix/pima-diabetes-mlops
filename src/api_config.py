"""Pydantic-схемы для FastAPI-сервиса ЛР6."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Запрос на предсказание: строка из 8 признаков Pima."""

    pregnancies: float = Field(..., ge=0, le=20, description="Число беременностей")
    glucose: float = Field(..., ge=0, le=300, description="Уровень глюкозы, мг/дл")
    blood_pressure: float = Field(
        ..., ge=0, le=200, description="Диастолическое давление"
    )
    skin_thickness: float = Field(..., ge=0, le=100, description="Толщина складки, мм")
    insulin: float = Field(..., ge=0, le=1000, description="Инсулин, мкЕд/мл")
    bmi: float = Field(..., ge=0, le=100, description="Индекс массы тела")
    diabetes_pedigree: float = Field(..., ge=0, le=3, description="Наследственность")
    age: float = Field(..., ge=0, le=120, description="Возраст, лет")


class PredictResponse(BaseModel):
    """Ответ: предсказание + вероятность + версия модели."""

    prediction: int = Field(..., description="0 — нет диабета, 1 — диабет")
    probability: float = Field(..., ge=0, le=1, description="Вероятность класса 1")
    model_version: str = Field(..., description="Версия модели")


class HealthResponse(BaseModel):
    status: str
    active_model: str
    uptime_seconds: float


class ModelInfo(BaseModel):
    version: str
    path: str
    hash: str
    metrics: dict
    status: Literal["ready", "trial", "working", "retired"]


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    active: str


class PromoteRequest(BaseModel):
    version: str


class RollbackRequest(BaseModel):
    pass


class ServiceConfig(BaseModel):
    """Конфиг сервиса."""

    models_dir: Path = Path("artifacts/models")
    registry_path: Path = Path("reports/LAB6/registry.json")
    log_path: Path = Path("reports/LAB6/service.log")

    @field_validator("models_dir")
    @classmethod
    def dir_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v
