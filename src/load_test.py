"""Нагрузочное тестирование: 10 000 запросов к сервису (ЛР8)."""

import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


BASE_URL = "http://127.0.0.1:8001"
N_REQUESTS = 10_000


def make_payload():
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
        ],
        "n_jobs": 1,
    }


def measure_rps(n_requests: int) -> dict:
    """10 000 запросов через /predict/batch, замер RPS и задержек.

    Каждый запрос измеряется отдельно: t_req — до, после — разница.
    Накопленное время (duration) — от начала до конца теста.
    """
    payload = make_payload()
    latencies = []
    errors = 0

    t_start = time.perf_counter()

    for _ in tqdm(range(n_requests), desc="requests"):
        t_req = time.perf_counter()
        try:
            r = requests.post(f"{BASE_URL}/predict/batch", json=payload, timeout=10)
            if r.status_code != 200:
                errors += 1
        except Exception:
            errors += 1
        latencies.append((time.perf_counter() - t_req) * 1000)

    duration = time.perf_counter() - t_start
    s = pd.Series(latencies)

    return {
        "n_requests": n_requests,
        "duration_sec": round(duration, 2),
        "errors": errors,
        "success_rate": round((n_requests - errors) / n_requests * 100, 2),
        "rps": round(n_requests / duration, 2),
        "p50_ms": round(float(s.quantile(0.50)), 2),
        "p95_ms": round(float(s.quantile(0.95)), 2),
        "p99_ms": round(float(s.quantile(0.99)), 2),
    }


def main():
    out = Path("reports/LAB8/load_test.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Запуск нагрузочного теста: 10 000 запросов к /predict/batch")
    print(f"Сервис: {BASE_URL}")
    print(f"Запросов: {N_REQUESTS}")

    result = measure_rps(N_REQUESTS)
    print()
    for k, v in result.items():
        print(f"  {k}: {v}")

    df = pd.DataFrame([result])
    df.to_csv(out, index=False)
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
