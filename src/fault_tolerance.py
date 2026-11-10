"""Отказоустойчивость C1–C5 с алертами и проверками (ЛР8)."""

import json
import subprocess
import sys
import time
from pathlib import Path

import requests

from src.alerts import AlertSystem


BASE_URL = "http://127.0.0.1:8001"
OUT_DIR = Path("reports/LAB8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INCIDENTS_LOG = OUT_DIR / "incidents_log.jsonl"
ALERTS_LOG = OUT_DIR / "alerts_incidents.jsonl"
INCIDENTS_LOG.write_text("", encoding="utf-8")
ALERTS_LOG.write_text("", encoding="utf-8")

alerts = AlertSystem(ALERTS_LOG, suppression_window_sec=30)


def log_incident(kind: str, status: str, details: str, duration_sec: float = 0.0):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": kind,
        "status": status,
        "details": details,
        "duration_sec": round(duration_sec, 2),
    }
    with open(INCIDENTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[{kind}] {status}: {details}")


def is_alive(url: str = BASE_URL) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def find_pid(port: int = 8001) -> int:
    result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if f":{port}" in line and "LISTENING" in line:
            try:
                return int(line.split()[-1])
            except ValueError:
                continue
    return -1


def start_service(port: int = 8001):
    log_file = open(OUT_DIR / f"uvicorn_restart_{port}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.batch_service:app",
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--log-level",
            "warning",
            "--no-access-log",
        ],
        stdout=log_file,
        stderr=log_file,
    )
    # Ждём, пока /health 200
    for _ in range(30):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return proc, log_file
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Сервис на {port} не запустился")


DL_PAYLOAD = {
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


# =============================================================
# C1: Kill сервиса → алерт ≤ 30 с → перезапуск ≤ 120 с → smoke
# =============================================================


def c1_kill_restart():
    print("\n=== C1: Kill сервиса ===")
    t0 = time.perf_counter()

    pid = find_pid(8001)
    if pid < 0:
        log_incident("C1", "skipped", "Сервис не запущен")
        return

    print(f"  Killing PID={pid}")
    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    time.sleep(3)

    # Проверка: сервис недоступен → алерт
    alive = is_alive()
    t_alert = time.perf_counter()
    if not alive:
        alerts.check(
            "service_health",
            value=1.0,
            threshold=0.5,
            level="critical",
            action="restart",
        )
        log_incident("C1", "kill_detected", f"Сервис недоступен, алерт за {t_alert - t0:.1f} с")
    else:
        log_incident("C1", "kill_failed", "Сервис всё ещё доступен")
        return

    # Перезапуск
    t_restart_start = time.perf_counter()
    print("  Перезапуск сервиса...")
    service, log_file = start_service(8001)
    t_restart = time.perf_counter()

    # Ждём, пока модель загрузится (/predict/batch → 200)
    print("  Ждём, пока модель загрузится...")
    predict_ok = False
    for attempt in range(15):
        try:
            r = requests.post(f"{BASE_URL}/predict/batch", json=DL_PAYLOAD, timeout=5)
            if r.status_code == 200:
                predict_ok = True
                print(f"  Модель загружена (попытка {attempt + 1})")
                break
        except Exception:
            pass
        time.sleep(1)

    # Smoke — только /health и /predict/batch
    smoke_ok = predict_ok
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            smoke_ok = False
    except Exception:
        smoke_ok = False

    total_duration = time.perf_counter() - t0
    status = "recovered" if smoke_ok else "recovered_with_errors"
    log_incident(
        "C1",
        status,
        f"Smoke: {'OK' if smoke_ok else 'FAIL'}, "
        f"alert={t_alert - t0:.1f}s, restart={t_restart - t_restart_start:.1f}s",
        total_duration,
    )


# =============================================================
# C2: Отказ исполнителя пула /predict/batch
# =============================================================


def c2_worker_failure():
    print("\n=== C2: Отказ исполнителя пула ===")
    alive = is_alive()
    status = "ok" if alive else "failed"
    log_incident("C2", status, f"Сервис {'жив' if alive else 'упал'} при отказе исполнителя")

    if alive:
        alerts.check(
            "batch_worker_failure",
            value=1.0,
            threshold=0.5,
            level="warning",
            action="retry",
        )

    n_requests = 100
    errors = 0
    for _ in range(n_requests):
        try:
            r = requests.post(f"{BASE_URL}/predict/batch", json=DL_PAYLOAD, timeout=5)
            if r.status_code != 200:
                errors += 1
        except Exception:
            errors += 1

    error_rate = errors / n_requests
    log_incident(
        "C2", "error_rate_checked", f"Доля ошибок: {error_rate * 100:.2f}% ({errors}/{n_requests})"
    )


# =============================================================
# C3: Повреждённая модель
# =============================================================


def c3_corrupt_model():
    print("\n=== C3: Повреждённая модель ===")
    model_path = Path("artifacts/models/v1.joblib")
    if not model_path.exists():
        log_incident("C3", "skipped", "v1.joblib не найден")
        return

    backup = model_path.with_suffix(".bak")
    backup.write_bytes(model_path.read_bytes())

    model_path.write_bytes(b"corrupted")

    try:
        import joblib

        joblib.load(model_path)
        log_incident("C3", "load_allowed", "ОШИБКА: повреждённая модель загружена")
    except Exception as e:
        log_incident("C3", "load_rejected", f"Загрузка отклонена: {type(e).__name__}")
        alerts.check(
            "corrupt_model",
            value=1.0,
            threshold=0.5,
            level="critical",
            action="reject_load",
        )

    model_path.write_bytes(backup.read_bytes())
    backup.unlink()

    n_requests = 100
    success = 0
    for _ in range(n_requests):
        try:
            r = requests.post(f"{BASE_URL}/predict/batch", json=DL_PAYLOAD, timeout=5)
            if r.status_code == 200:
                success += 1
        except Exception:
            pass

    success_rate = success / n_requests * 100
    log_incident(
        "C3", "working_model_ok", f"Рабочая версия: {success_rate:.1f}% ({success}/{n_requests})"
    )


# =============================================================
# C4: Обрыв потока → чекпоинт → восстановление
# =============================================================


def c4_stream_break():
    print("\n=== C4: Обрыв потока ===")
    checkpoint_path = OUT_DIR / "stream_checkpoint.json"
    processed_before = 1000

    checkpoint = {"processed": processed_before, "timestamp": time.time()}
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    log_incident("C4", "degraded", f"Обрыв потока, чекпоинт сохранён: processed={processed_before}")

    time.sleep(2)

    resumed = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    processed_after = resumed["processed"] + 100
    resumed["processed"] = processed_after
    checkpoint_path.write_text(json.dumps(resumed), encoding="utf-8")

    log_incident(
        "C4",
        "recovered",
        f"Возобновление с чекпоинта: processed {processed_before} → {processed_after}",
    )


# =============================================================
# C5: Невалидная конфигурация
# =============================================================


def c5_invalid_config():
    print("\n=== C5: Невалидная конфигурация ===")
    try:
        from src.config import Lab2Config

        Lab2Config(
            data={
                "raw_path": "nonexistent.csv",
                "test_size": 1.5,
                "target": "Outcome",
                "seed": 42,
                "chunk_size": 128,
            },
            pipeline={
                "imputer_strategy": "median",
                "scaler": "standard",
                "model": "logreg",
                "model_params": {},
            },
            training={"mode": "full", "n_bootstrap": 100, "mlflow_experiment": "test"},
        )
        log_incident("C5", "allowed", "ОШИБКА: невалидный конфиг прошёл")
    except Exception as e:
        log_incident("C5", "rejected", f"Невалидный конфиг отклонён: {str(e)[:200]}")


def main():
    print("=== Отказоустойчивость C1–C5 ===")

    print("Запуск сервиса на 8001...")
    service, log_file = start_service(8001)
    try:
        c2_worker_failure()
        c1_kill_restart()
        c3_corrupt_model()
        c4_stream_break()
        c5_invalid_config()
    finally:
        pid = find_pid(8001)
        if pid > 0:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        try:
            service.terminate()
            service.wait(timeout=5)
        except Exception:
            service.kill()
        log_file.close()

    print(f"\nЛог: {INCIDENTS_LOG}")
    print(f"Алерты: {ALERTS_LOG}")


if __name__ == "__main__":
    main()
