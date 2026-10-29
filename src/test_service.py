"""Контрактные тесты FastAPI-сервиса ЛР6."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.service import app
from src.registry import ModelRegistry


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_registry():
    """Убеждаемся, что модели зарегистрированы и v1 активна."""
    reg = ModelRegistry(Path("reports/LAB6/registry.json"), Path("artifacts/models"))
    versions = [m.version for m in reg.list_models()]
    if "v1" not in versions:
        reg.register("v1", Path("artifacts/models/v1.joblib"), {"accuracy": 0.73})
    if "v2" not in versions:
        reg.register("v2", Path("artifacts/models/v2.joblib"), {"accuracy": 0.72})
    reg.promote("v1")
    yield


VALID_INPUT = {
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree": 0.627,
    "age": 50,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid(client):
    r = client.post("/predict", json=VALID_INPUT)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in (0, 1)
    assert 0 <= data["probability"] <= 1
    assert data["model_version"] == "v1"


def test_predict_missing_field(client):
    payload = VALID_INPUT.copy()
    del payload["glucose"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_wrong_type(client):
    payload = VALID_INPUT.copy()
    payload["glucose"] = "abc"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_out_of_range(client):
    payload = VALID_INPUT.copy()
    payload["pregnancies"] = -5
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_models_list(client):
    r = client.get("/models")
    assert r.status_code == 200
    versions = [m["version"] for m in r.json()["models"]]
    assert "v1" in versions
    assert "v2" in versions


def test_promote_and_rollback(client):
    r = client.post("/promote", json={"version": "v2"})
    assert r.status_code == 200
    assert r.json()["active"] == "v2"

    r = client.get("/health")
    assert r.json()["active_model"] == "v2"

    r = client.post("/rollback", json={})
    assert r.status_code == 200
    assert r.json()["active"] == "v1"


def test_promote_unknown_version(client):
    r = client.post("/promote", json={"version": "v99"})
    assert r.status_code == 404


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "uptime_seconds" in r.json()


def test_load_100_predict(client):
    """Нагрузка: 100 запросов к /predict."""
    n = 100
    success = 0
    for _ in range(n):
        r = client.post("/predict", json=VALID_INPUT)
        if r.status_code == 200:
            success += 1
    assert success == n
    print(f"\nНагрузка /predict: {success}/{n} успешных (100%)")


def test_switch_under_load(client):
    """Переключение версии под нагрузкой: 0 потерянных запросов."""
    import threading
    import time

    results = []
    errors = []

    def worker():
        for _ in range(20):
            r = client.post("/predict", json=VALID_INPUT)
            if r.status_code == 200:
                results.append(r.json()["model_version"])
            else:
                errors.append(r.status_code)

    # 5 потоков × 20 запросов = 100
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()

    # Переключение под нагрузкой
    time.sleep(0.005)
    client.post("/promote", json={"version": "v2"})
    time.sleep(0.005)
    client.post("/rollback", json={})

    for t in threads:
        t.join()

    total = len(results) + len(errors)
    success_rate = len(results) / total if total > 0 else 0
    print(
        f"\nНагрузка при переключении: {len(results)}/{total} успешных ({success_rate * 100:.2f}%)"
    )
    assert len(errors) == 0, f"Потерянные запросы: {errors}"
    assert success_rate >= 0.999
