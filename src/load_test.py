"""Нагрузочное тестирование: 6 этапов × 10 000 запросов × 10 минут (ЛР8)."""

import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


BASE_URL = "http://127.0.0.1:8001"
N_STAGES = 6
N_REQUESTS_PER_STAGE = 10_000
TOTAL = N_STAGES * N_REQUESTS_PER_STAGE

PAUSE_BETWEEN_STAGES_SEC = 5
REQUEST_TIMEOUT_SEC = 30
DELAY_BETWEEN_REQUESTS_SEC = 0.06  # 60 мс → 10000 × 60 мс = 600 сек = 10 минут


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


def measure_stage(stage: int, n_requests: int) -> dict:
    """Один этап: n_requests запросов с паузой 60 мс (10 минут)."""
    payload = make_payload()
    latencies = []
    errors = 0

    session = requests.Session()

    t_start = time.perf_counter()

    for _ in tqdm(range(n_requests), desc=f"stage {stage + 1}"):
        t_req = time.perf_counter()
        try:
            r = session.post(f"{BASE_URL}/predict/batch", json=payload, timeout=REQUEST_TIMEOUT_SEC)
            if r.status_code != 200:
                errors += 1
        except Exception:
            errors += 1
        latencies.append((time.perf_counter() - t_req) * 1000)

        # Искусственная пауза — чтобы этап длился ~10 минут
        time.sleep(DELAY_BETWEEN_REQUESTS_SEC)

    session.close()

    duration = time.perf_counter() - t_start
    s = pd.Series(latencies)

    return {
        "stage": stage + 1,
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

    print(f"Нагрузочный тест: {N_STAGES} этапов × {N_REQUESTS_PER_STAGE} = {TOTAL} запросов")
    print(f"Сервис: {BASE_URL}")
    print(f"Таймаут: {REQUEST_TIMEOUT_SEC} сек")
    print(f"Пауза между запросами: {DELAY_BETWEEN_REQUESTS_SEC * 1000:.0f} мс")
    print(f"Пауза между этапами: {PAUSE_BETWEEN_STAGES_SEC} сек")
    print(f"Ожидаемое время: {N_STAGES} × 10 мин ≈ {N_STAGES * 10} мин")

    results = []
    for stage in range(N_STAGES):
        print(f"\n=== Этап {stage + 1}/{N_STAGES} ===")
        result = measure_stage(stage, N_REQUESTS_PER_STAGE)
        results.append(result)
        print(
            f"  RPS: {result['rps']}, p95: {result['p95_ms']} мс, "
            f"errors: {result['errors']}, duration: {result['duration_sec']} сек"
        )
        if stage < N_STAGES - 1:
            print(f"  Пауза {PAUSE_BETWEEN_STAGES_SEC} сек...")
            time.sleep(PAUSE_BETWEEN_STAGES_SEC)

    total_errors = sum(r["errors"] for r in results)
    total_duration = sum(r["duration_sec"] for r in results)
    total_requests = sum(r["n_requests"] for r in results)
    total_success = total_requests - total_errors

    print("\n=== ИТОГО ===")
    print(f"  Всего запросов: {total_requests}")
    print(f"  Успешных: {total_success} ({total_success / total_requests * 100:.2f}%)")
    print(f"  Ошибок: {total_errors}")
    print(f"  Время: {total_duration:.2f} сек ({total_duration / 60:.2f} мин)")
    print(f"  Средний RPS: {total_requests / total_duration:.2f}")

    df = pd.DataFrame(results)
    df.to_csv(out, index=False)
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
