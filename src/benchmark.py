"""Бенчмарк /predict/batch при разном числе исполнителей (ЛР7)."""

import statistics
import time
from pathlib import Path

import pandas as pd
import requests
import yaml


BASE_URL = "http://127.0.0.1:8001"


def make_payload(n: int = 100) -> dict:
    """Формирует пакет из n одинаковых объектов."""
    return {
        "items": [
            {
                "pregnancies": 6,
                "glucose": 148,
                "blood_pressure": 72,
                "skin_thickness": 35,
                "insulin": 0,
                "bmi": 33.6,
                "diabetes_pedigree": 0.627,
                "age": 50,
            }
        ]
        * n,
        "n_jobs": 1,
    }


def measure(n_jobs: int, n_requests: int, n_warmup: int, repeats: int) -> dict:
    """Измеряет задержки при заданном числе исполнителей.

    Делает n_warmup прогревов, потом repeats замеров.
    Возвращает медиану, min, max, throughput.
    """
    # Прогрев — не учитываем в замерах
    for _ in range(n_warmup):
        payload = make_payload(n_requests)
        payload["n_jobs"] = n_jobs
        r = requests.post(f"{BASE_URL}/predict/batch", json=payload, timeout=120)
        r.raise_for_status()

    latencies = []
    for _ in range(repeats):
        payload = make_payload(n_requests)
        payload["n_jobs"] = n_jobs
        t0 = time.perf_counter()
        r = requests.post(f"{BASE_URL}/predict/batch", json=payload, timeout=120)
        dt = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        latencies.append(dt)

    median_ms = statistics.median(latencies)
    return {
        "n_jobs": n_jobs,
        "n_requests": n_requests,
        "repeats": repeats,
        "median_ms": round(median_ms, 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "throughput_rps": round(n_requests / (median_ms / 1000), 2),
    }


def main():
    with open("configs/lab7_monitoring.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Создаём папку для результатов
    Path(cfg["drift"]["output_benchmark"]).parent.mkdir(parents=True, exist_ok=True)

    bc = cfg["benchmark"]
    rows = []
    for n_jobs in bc["n_workers"]:
        print(f"n_jobs={n_jobs}...")
        rows.append(
            measure(
                n_jobs=n_jobs,
                n_requests=bc["n_requests"],
                n_warmup=bc["n_warmup"],
                repeats=bc["repeats"],
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(cfg["drift"]["output_benchmark"], index=False)
    print()
    print(df.to_string(index=False))
    print(f"\nСохранено: {cfg['drift']['output_benchmark']}")


if __name__ == "__main__":
    main()
