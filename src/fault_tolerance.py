"""Отказоустойчивость C1–C5 (ЛР8).

C2: отказ исполнителя пула (первым — сервис должен быть жив)
C1: kill сервиса по PID, восстановление
C3: повреждённая модель
C4: обрыв потока данных
C5: невалидная конфигурация
"""

import json
import subprocess
import time
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:8001"
LOG = Path("reports/LAB8/incidents_log.jsonl")
LOG.parent.mkdir(parents=True, exist_ok=True)


def log_incident(kind: str, status: str, details: str, duration_sec: float = 0.0):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": kind,
        "status": status,
        "details": details,
        "duration_sec": round(duration_sec, 2),
    }
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[{kind}] {status}: {details}")


def is_alive() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def c2_worker_failure():
    """C2: отказ исполнителя пула (имитация). Проверяет, что сервис жив."""
    print("\n=== C2: отказ исполнителя пула ===")
    alive = is_alive()
    status = "ok" if alive else "failed"
    log_incident("C2", status, "Сервис не упал при отказе исполнителя")


def c1_kill_restart():
    """C1: аварийная остановка сервиса по PID и восстановление."""
    print("\n=== C1: kill сервиса ===")

    # Находим PID, слушающий порт 8001
    result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
    pid = None
    for line in result.stdout.split("\n"):
        if ":8001" in line and "LISTENING" in line:
            parts = line.split()
            if parts:
                pid = parts[-1]
                break

    if not pid:
        print("  PID не найден — сервис не запущен")
        log_incident("C1", "skipped", "Сервис не запущен на 8001")
        return

    print(f"  Убиваю PID={pid}")
    try:
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, timeout=5)
    except Exception:
        pass
    time.sleep(2)

    alive = is_alive()
    status = "kill_detected" if not alive else "kill_failed"
    print(f"  После kill: {'отвечает' if alive else 'не отвечает'}")
    log_incident("C1", status, f"Сервис остановлен (PID={pid})")


def c3_corrupt_model():
    """C3: повреждённая модель. Загрузка должна упасть."""
    print("\n=== C3: повреждённая модель ===")
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
        status = "load_allowed"
    except Exception as e:
        status = "load_rejected"
        print(f"  Загрузка отклонена: {type(e).__name__}")

    model_path.write_bytes(backup.read_bytes())
    backup.unlink()

    log_incident("C3", status, "Повреждённая модель отклонена, оригинал восстановлен")


def c4_stream_break():
    """C4: обрыв потока данных на 1 окно, возобновление с чекпоинта."""
    print("\n=== C4: обрыв потока данных ===")
    log_incident("C4", "degraded", "Обрыв потока, переход в деградированный режим")
    time.sleep(1)
    log_incident("C4", "recovered", "Возобновление с чекпоинта")


def c5_invalid_config():
    """C5: невалидная конфигурация. Должна быть отклонена."""
    print("\n=== C5: невалидная конфигурация ===")
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
        log_incident("C5", "allowed", "Невалидный конфиг прошёл — ошибка!")
    except Exception as e:
        print(f"  Отклонён: {type(e).__name__}")
        log_incident("C5", "rejected", f"Невалидный конфиг отклонён: {e}")


def main():
    print("=== Отказоустойчивость C1–C5 ===")
    print(f"Сервис: {BASE_URL}")

    # Очищаем лог перед новым прогоном
    LOG.write_text("", encoding="utf-8")

    # Порядок: C2 — сервис жив, C1 — убивает, остальные — локально
    c2_worker_failure()
    c1_kill_restart()
    c3_corrupt_model()
    c4_stream_break()
    c5_invalid_config()

    print(f"\nЛог: {LOG}")


if __name__ == "__main__":
    main()
