"""FastAPI-сервис модели Pima (ЛР6)."""

import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.api_config import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    ModelsResponse,
    PromoteRequest,
    RollbackRequest,
)
from src.registry import ModelRegistry
from src.service_monitor import ServiceMonitor


START_TIME = time.time()

app = FastAPI(title="Pima Diabetes Service", version="1.0")

# Загрузка конфигурации
MODELS_DIR = Path("artifacts/models")
REGISTRY_PATH = Path("reports/LAB6/registry.json")
LOG_PATH = Path("reports/LAB6/service.log")

registry = ModelRegistry(REGISTRY_PATH, MODELS_DIR)
monitor = ServiceMonitor(LOG_PATH)


@app.get("/health", response_model=HealthResponse)
def health():
    """Статус сервиса: активная версия, uptime."""
    return HealthResponse(
        status="ok" if registry.active else "no_active_model",
        active_model=registry.active or "none",
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Предсказание для одного объекта."""
    t0 = time.perf_counter()
    version, model = registry.get_active()
    if model is None:
        monitor.log(
            "/predict", 503, (time.perf_counter() - t0) * 1000, error="no_active_model"
        )
        raise HTTPException(status_code=503, detail="Нет активной модели")

    X = np.array(
        [
            [
                req.pregnancies,
                req.glucose,
                req.blood_pressure,
                req.skin_thickness,
                req.insulin,
                req.bmi,
                req.diabetes_pedigree,
                req.age,
            ]
        ]
    )

    try:
        proba = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        monitor.log("/predict", 500, (time.perf_counter() - t0) * 1000, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ошибка модели: {e}")

    monitor.log(
        "/predict",
        200,
        (time.perf_counter() - t0) * 1000,
        version=version,
        prediction=int(proba >= 0.5),
    )

    return PredictResponse(
        prediction=int(proba >= 0.5),
        probability=round(proba, 4),
        model_version=version,
    )


@app.get("/models", response_model=ModelsResponse)
def list_models():
    """Список всех версий моделей."""
    return ModelsResponse(
        models=registry.list_models(),
        active=registry.active or "none",
    )


@app.post("/promote")
def promote(req: PromoteRequest):
    """Продвигает версию до рабочей."""
    t0 = time.perf_counter()
    try:
        registry.promote(req.version)
    except ValueError as e:
        monitor.log("/promote", 404, (time.perf_counter() - t0) * 1000, error=str(e))
        raise HTTPException(status_code=404, detail=str(e))

    monitor.log("/promote", 200, (time.perf_counter() - t0) * 1000, version=req.version)
    return {"status": "ok", "active": registry.active}


@app.post("/rollback")
def rollback(req: RollbackRequest):
    """Откат к предыдущей рабочей версии."""
    t0 = time.perf_counter()
    models = registry.list_models()
    retired = [m for m in models if m.status == "retired"]
    if not retired:
        monitor.log(
            "/rollback",
            404,
            (time.perf_counter() - t0) * 1000,
            error="no_retired_versions",
        )
        raise HTTPException(status_code=404, detail="Нет версий для отката")

    prev = sorted(retired, key=lambda m: m.version)[-1]
    registry.rollback(prev.version)

    monitor.log(
        "/rollback", 200, (time.perf_counter() - t0) * 1000, version=prev.version
    )
    return {"status": "ok", "active": registry.active}


@app.get("/metrics")
def metrics():
    """Метрики сервиса."""
    return {
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "active_model": registry.active,
        "n_models": len(registry.list_models()),
    }


@app.exception_handler(Exception)
async def generic_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )
