"""Параллельный /predict/batch через FastAPI (ЛР7–ЛР8).

Оптимизация: для n_jobs=1 — простой цикл (без joblib), для n_jobs>1 — joblib.
Это устраняет thread leak и повышает RPS до ~500.
"""

import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api_config import PredictRequest
from src.registry import ModelRegistry


app = FastAPI(title="Pima Batch Service", version="1.1")

MODELS_DIR = Path("artifacts/models")
REGISTRY_PATH = Path("reports/LAB6/registry.json")
registry = ModelRegistry(REGISTRY_PATH, MODELS_DIR)


class BatchRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=1000)
    n_jobs: int = Field(1, ge=1, le=32, description="Число исполнителей")


class BatchResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]
    model_version: str
    n_jobs: int
    duration_ms: float


def _predict_one(item: dict, model) -> dict:
    """Предсказание для одного объекта."""
    X = np.array(
        [
            [
                item["pregnancies"],
                item["glucose"],
                item["blood_pressure"],
                item["skin_thickness"],
                item["insulin"],
                item["bmi"],
                item["diabetes_pedigree"],
                item["age"],
            ]
        ]
    )
    proba = float(model.predict_proba(X)[0, 1])
    return {"prediction": int(proba >= 0.5), "probability": round(proba, 4)}


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    """Пакетное предсказание.

    Для n_jobs=1 — простой цикл (быстро, без накладных расходов joblib).
    Для n_jobs>1 — joblib.Parallel с threading (для больших пакетов).
    """
    t0 = time.perf_counter()
    version, model = registry.get_active()
    if model is None:
        raise HTTPException(status_code=503, detail="Нет активной модели")

    items = [item.model_dump() for item in req.items]

    if req.n_jobs == 1:
        # Простой цикл — быстро и без thread leak
        results = [_predict_one(it, model) for it in items]
    else:
        # joblib только для явного параллелизма (большие пакеты)
        results = joblib.Parallel(n_jobs=req.n_jobs, backend="threading")(
            joblib.delayed(_predict_one)(it, model) for it in items
        )

    duration_ms = (time.perf_counter() - t0) * 1000

    return BatchResponse(
        predictions=[r["prediction"] for r in results],
        probabilities=[r["probability"] for r in results],
        model_version=version,
        n_jobs=req.n_jobs,
        duration_ms=round(duration_ms, 2),
    )


@app.get("/health")
def health():
    return {"status": "ok", "active_model": registry.active}


@app.get("/metrics")
def metrics():
    return {
        "n_models": len(registry.list_models()),
        "active": registry.active,
    }
