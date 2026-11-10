"""Сквозной нагрузочный тест через FastAPI-сервис (ЛР8).

Исправления:
1. subprocess.PIPE → файл (иначе uvicorn блокируется на переполнении буфера).
2. Проверка /health и /predict/batch перед нагрузкой.
3. RPS без искусственного sleep.
4. Графики памяти и CPU.
5. Всё через FastAPI.
"""

import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import requests
from tqdm import tqdm


BASE_URL = "http://127.0.0.1:8001"
OUT_DIR = Path("reports/LAB8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STAGES = 6
STAGE_DURATION_SEC = 10  # 10 минут на этап (для теста поставьте 10)
MONITOR_STEP_SEC = 2.0

SAMPLE_PAYLOAD = {
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


def find_service_pid() -> int:
    result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if ":8001" in line and "LISTENING" in line:
            try:
                return int(line.split()[-1])
            except ValueError:
                continue
    return -1


def start_service():
    """Запускает uvicorn. Пишет лог в файл (не PIPE — иначе буфер блокирует)."""
    log_path = OUT_DIR / "uvicorn.log"
    log_file = open(log_path, "w", encoding="utf-8")
    print(f"Запуск uvicorn на 8001 (лог: {log_path})...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.batch_service:app",
            "--port",
            "8001",
            "--host",
            "127.0.0.1",
            "--log-level",
            "warning",
            "--no-access-log",
        ],
        stdout=log_file,
        stderr=log_file,
    )

    # Ждём готовности /health
    for _ in range(30):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                health = r.json()
                print(f"  Сервис готов: {health}")
                if health.get("active_model") in (None, "none"):
                    print("  ⚠️  Активная модель не задана! Запустите promote('v1').")
                return proc, log_file
        except Exception:
            pass
        time.sleep(1)

    log_file.close()
    raise RuntimeError(f"Сервис не запустился. См. {log_path}")


def check_predict_works():
    """Проверяет /predict/batch до нагрузки — иначе все этапы упадут."""
    try:
        r = requests.post(f"{BASE_URL}/predict/batch", json=SAMPLE_PAYLOAD, timeout=10)
        if r.status_code != 200:
            print(f"  ⚠️  /predict/batch → {r.status_code}: {r.text[:200]}")
            return False
        print(f"  Тест /predict/batch OK: {r.json()['predictions']}")
        return True
    except Exception as e:
        print(f"  ⚠️  /predict/batch не работает: {e}")
        return False


def run_stage(stage: int, pid: int) -> dict:
    """Один этап: STAGE_DURATION_SEC запросов через FastAPI + мониторинг."""
    session = requests.Session()
    proc = psutil.Process(pid)
    proc.cpu_percent(interval=None)

    timestamps, memory_mb, cpu_percent = [], [], []
    latencies, errors = [], 0
    n_requests = 0

    t_start = time.perf_counter()
    last_monitor = t_start
    pbar = tqdm(desc=f"stage {stage}/{N_STAGES}", unit="req")

    while time.perf_counter() - t_start < STAGE_DURATION_SEC:
        t_req = time.perf_counter()
        try:
            r = session.post(f"{BASE_URL}/predict/batch", json=SAMPLE_PAYLOAD, timeout=5)
            if r.status_code != 200:
                errors += 1
        except Exception:
            errors += 1
        latencies.append((time.perf_counter() - t_req) * 1000)
        n_requests += 1
        pbar.update(1)

        now = time.perf_counter()
        if now - last_monitor >= MONITOR_STEP_SEC:
            timestamps.append(round(now - t_start, 2))
            memory_mb.append(round(proc.memory_info().rss / 1e6, 2))
            cpu_percent.append(round(proc.cpu_percent(interval=None), 2))
            last_monitor = now

    pbar.close()
    session.close()

    duration = time.perf_counter() - t_start
    s = pd.Series(latencies)

    return {
        "stage": stage,
        "n_requests": n_requests,
        "duration_sec": round(duration, 2),
        "errors": errors,
        "success_rate": round((n_requests - errors) / n_requests * 100, 2),
        "rps": round(n_requests / duration, 2),
        "p50_ms": round(float(s.quantile(0.50)), 2),
        "p95_ms": round(float(s.quantile(0.95)), 2),
        "p99_ms": round(float(s.quantile(0.99)), 2),
        "resource_timestamps": timestamps,
        "resource_memory_mb": memory_mb,
        "resource_cpu_percent": cpu_percent,
    }


def plot_stage(stage_result: dict, out_prefix: Path):
    ts = stage_result["resource_timestamps"]
    mem = stage_result["resource_memory_mb"]
    cpu = stage_result["resource_cpu_percent"]
    if not ts:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, mem, color="blue", linewidth=0.9)
    ax.set_xlabel("Время, сек")
    ax.set_ylabel("Память (RSS), МБ")
    ax.set_title(f"Этап {stage_result['stage']}: утилизация памяти (шаг {MONITOR_STEP_SEC:.0f} с)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_prefix}_memory.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, cpu, color="red", linewidth=0.9)
    ax.set_xlabel("Время, сек")
    ax.set_ylabel("CPU, %")
    ax.set_title(f"Этап {stage_result['stage']}: загрузка CPU (шаг {MONITOR_STEP_SEC:.0f} с)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_prefix}_cpu.png", dpi=120)
    plt.close(fig)


def main():
    # Убеждаемся, что на 8001 ничего не висит
    old_pid = find_service_pid()
    if old_pid > 0:
        print(f"Убиваю старый процесс на 8001: PID={old_pid}")
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(old_pid)], capture_output=True, timeout=5)
        except Exception:
            pass
        time.sleep(2)

    service, log_file = start_service()
    try:
        pid = find_service_pid()
        if pid < 0:
            raise RuntimeError("Не удалось найти PID на 8001")
        print(f"PID сервиса: {pid}")

        if not check_predict_works():
            raise RuntimeError("Сервис не отвечает на /predict/batch — прерываю")

        print(
            f"Этапов: {N_STAGES}, по {STAGE_DURATION_SEC} сек ({STAGE_DURATION_SEC / 60:.1f} мин)"
        )

        results = []
        for stage in range(1, N_STAGES + 1):
            print(f"\n=== Этап {stage}/{N_STAGES} ===")
            result = run_stage(stage, pid)
            results.append(result)

            print(f"  RPS: {result['rps']}")
            print(f"  p50/p95/p99: {result['p50_ms']}/{result['p95_ms']}/{result['p99_ms']} мс")
            print(f"  errors: {result['errors']}, success_rate: {result['success_rate']}%")
            print(f"  duration: {result['duration_sec']} сек")
            if result["resource_memory_mb"]:
                print(
                    f"  Память: {min(result['resource_memory_mb']):.1f}–"
                    f"{max(result['resource_memory_mb']):.1f} МБ"
                )
                print(
                    f"  CPU: {min(result['resource_cpu_percent']):.1f}–"
                    f"{max(result['resource_cpu_percent']):.1f} %"
                )

            plot_stage(result, OUT_DIR / f"stage_{stage}")

        total_errors = sum(r["errors"] for r in results)
        total_requests = sum(r["n_requests"] for r in results)
        total_duration = sum(r["duration_sec"] for r in results)

        print("\n=== ИТОГО ===")
        print(f"  Всего запросов: {total_requests}")
        print(
            f"  Успешных: {total_requests - total_errors} "
            f"({(total_requests - total_errors) / total_requests * 100:.2f}%)"
        )
        print(f"  Ошибок: {total_errors}")
        print(f"  Время: {total_duration:.2f} сек ({total_duration / 60:.2f} мин)")
        print(f"  Средний RPS: {total_requests / total_duration:.2f}")

        summary_rows = [
            {
                k: r[k]
                for k in (
                    "stage",
                    "n_requests",
                    "duration_sec",
                    "errors",
                    "success_rate",
                    "rps",
                    "p50_ms",
                    "p95_ms",
                    "p99_ms",
                )
            }
            for r in results
        ]
        pd.DataFrame(summary_rows).to_csv(OUT_DIR / "load_test.csv", index=False)
        print(f"\nСохранено: {OUT_DIR / 'load_test.csv'}")
        print(f"Графики: {OUT_DIR}/stage_*_memory.png, {OUT_DIR}/stage_*_cpu.png")

    finally:
        print("Остановка сервиса...")
        service.terminate()
        try:
            service.wait(timeout=10)
        except Exception:
            service.kill()
        log_file.close()


if __name__ == "__main__":
    main()
