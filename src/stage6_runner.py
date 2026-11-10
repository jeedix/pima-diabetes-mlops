"""Пункт 6 ЛР8: 6 этапов нагрузки с дрейфом."""

import subprocess
import sys
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import requests

from src.alerts import AlertSystem
from src.generate_stream import make_stream
from src.psi_ks import psi


OUT_DIR = Path("reports/LAB8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DL_URL = "http://127.0.0.1:8001"
ML_URL = "http://127.0.0.1:8002"
MONITOR_STEP_SEC = 2.0
ALERTS_LOG = OUT_DIR / "alerts_sample.jsonl"

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


def find_service_pid(port: int) -> int:
    result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if f":{port}" in line and "LISTENING" in line:
            try:
                return int(line.split()[-1])
            except ValueError:
                continue
    return -1


def start_service(module: str, port: int):
    log_file = open(OUT_DIR / f"uvicorn_{port}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            module,
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
    for _ in range(30):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return proc, log_file
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Сервис {module} на {port} не запустился")


def monitor_resources(pid: int, stop_flag: threading.Event) -> dict:
    try:
        proc = psutil.Process(pid)
        proc.cpu_percent(interval=None)
    except psutil.NoSuchProcess:
        return {"timestamps": [], "memory_mb": [], "cpu_percent": []}

    ts, mem, cpu = [], [], []
    t0 = time.perf_counter()
    while not stop_flag.is_set():
        try:
            ts.append(round(time.perf_counter() - t0, 2))
            mem.append(round(proc.memory_info().rss / 1e6, 2))
            cpu.append(round(proc.cpu_percent(interval=None), 2))
        except psutil.NoSuchProcess:
            break
        time.sleep(MONITOR_STEP_SEC)
    return {"timestamps": ts, "memory_mb": mem, "cpu_percent": cpu}


def plot_resources(res: dict, prefix: str, title: str):
    if not res["timestamps"]:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(res["timestamps"], res["memory_mb"], color="blue")
    ax.set_xlabel("Время, сек")
    ax.set_ylabel("Память, МБ")
    ax.set_title(f"{title}: память (шаг {MONITOR_STEP_SEC:.0f} с)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{prefix}_memory.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(res["timestamps"], res["cpu_percent"], color="red")
    ax.set_xlabel("Время, сек")
    ax.set_ylabel("CPU, %")
    ax.set_title(f"{title}: CPU (шаг {MONITOR_STEP_SEC:.0f} с)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{prefix}_cpu.png", dpi=120)
    plt.close(fig)


def stage_6_1_warmup(duration_sec: int = 300) -> dict:
    print(f"\n=== 6.1. Разогрев {duration_sec} сек ===")
    stream = make_stream(n=10_000, seed=42)

    pid = find_service_pid(8001)
    stop = threading.Event()
    mon = threading.Thread(target=lambda: mon.__setattr__("data", monitor_resources(pid, stop)))
    mon.start()

    session = requests.Session()
    latencies, errors, n = [], 0, 0
    t0 = time.perf_counter()
    last_print = t0
    i = 0
    while time.perf_counter() - t0 < duration_sec:
        item = stream[i % len(stream)]
        t_req = time.perf_counter()
        try:
            r = session.post(
                f"{DL_URL}/predict/batch", json={"items": [item], "n_jobs": 1}, timeout=5
            )
            if r.status_code != 200:
                errors += 1
        except Exception:
            errors += 1
        latencies.append((time.perf_counter() - t_req) * 1000)
        n += 1
        i += 1
        now = time.perf_counter()
        if now - last_print > 1:
            print(
                f"\r  {n} запросов, {n / (now - t0):.1f} RPS, errors={errors}", end="", flush=True
            )
            last_print = now
    print()

    session.close()
    stop.set()
    mon.join(timeout=5)

    duration = time.perf_counter() - t0
    s = pd.Series(latencies)
    result = {
        "stage": "6.1_warmup",
        "n_requests": n,
        "duration_sec": round(duration, 2),
        "errors": errors,
        "success_rate": round((n - errors) / n * 100, 2),
        "rps": round(n / duration, 2),
        "p50_ms": round(float(s.quantile(0.50)), 2),
        "p95_ms": round(float(s.quantile(0.95)), 2),
        "p99_ms": round(float(s.quantile(0.99)), 2),
    }
    print(
        f"  RPS: {result['rps']}, p95: {result['p95_ms']} мс, errors: {result['errors']}, success: {result['success_rate']}%"
    )
    plot_resources(mon.data, str(OUT_DIR / "stage_6_1"), "6.1 Разогрев")
    return result


def run_clean_stage(stage_name: str, url: str, pid: int, duration_sec: int = 600) -> dict:
    print(f"\n=== {stage_name} ===")
    if url == DL_URL:
        payload, endpoint = DL_PAYLOAD, "/predict/batch"
    else:
        payload, endpoint = DL_PAYLOAD["items"][0], "/predict"

    stop = threading.Event()
    mon = threading.Thread(target=lambda: mon.__setattr__("data", monitor_resources(pid, stop)))
    mon.start()

    session = requests.Session()
    latencies, errors, n = [], 0, 0
    t0 = time.perf_counter()
    last_print = t0
    while time.perf_counter() - t0 < duration_sec:
        t_req = time.perf_counter()
        try:
            r = session.post(f"{url}{endpoint}", json=payload, timeout=5)
            if r.status_code != 200:
                errors += 1
        except Exception:
            errors += 1
        latencies.append((time.perf_counter() - t_req) * 1000)
        n += 1
        now = time.perf_counter()
        if now - last_print > 1:
            print(
                f"\r  {n} запросов, {n / (now - t0):.1f} RPS, errors={errors}", end="", flush=True
            )
            last_print = now
    print()

    session.close()
    stop.set()
    mon.join(timeout=5)

    duration = time.perf_counter() - t0
    s = pd.Series(latencies)
    result = {
        "stage": stage_name,
        "n_requests": n,
        "duration_sec": round(duration, 2),
        "errors": errors,
        "success_rate": round((n - errors) / n * 100, 2),
        "rps": round(n / duration, 2),
        "p50_ms": round(float(s.quantile(0.50)), 2),
        "p95_ms": round(float(s.quantile(0.95)), 2),
        "p99_ms": round(float(s.quantile(0.99)), 2),
    }
    print(
        f"  RPS: {result['rps']}, p95: {result['p95_ms']} мс, errors: {result['errors']}, success: {result['success_rate']}%"
    )
    plot_resources(mon.data, str(OUT_DIR / stage_name.replace(" ", "_")), stage_name)
    return result


def stage_6_4_variable_rps(pid: int, duration_sec: int = 600) -> dict:
    print(f"\n=== 6.4. DL, RPS меняется {duration_sec} сек ===")
    stop = threading.Event()
    mon = threading.Thread(target=lambda: mon.__setattr__("data", monitor_resources(pid, stop)))
    mon.start()

    session = requests.Session()
    latencies, errors, n = [], 0, 0
    t0 = time.perf_counter()
    minute = 0
    last_print = t0
    while time.perf_counter() - t0 < duration_sec:
        elapsed = time.perf_counter() - t0
        if int(elapsed // 60) > minute:
            minute = int(elapsed // 60)
        factor = [1, 3, 5, 1][min(minute // 3, 3)]
        target_interval = 0.005 / factor
        t_req = time.perf_counter()
        try:
            r = session.post(f"{DL_URL}/predict/batch", json=DL_PAYLOAD, timeout=5)
            if r.status_code != 200:
                errors += 1
        except Exception:
            errors += 1
        latencies.append((time.perf_counter() - t_req) * 1000)
        n += 1
        sleep_time = target_interval - (time.perf_counter() - t_req)
        if sleep_time > 0:
            time.sleep(sleep_time)
        now = time.perf_counter()
        if now - last_print > 1:
            print(
                f"\r  {n} запросов, {n / (now - t0):.1f} RPS, errors={errors}", end="", flush=True
            )
            last_print = now
    print()

    session.close()
    stop.set()
    mon.join(timeout=5)

    duration = time.perf_counter() - t0
    s = pd.Series(latencies)
    result = {
        "stage": "6.4_variable_rps",
        "n_requests": n,
        "duration_sec": round(duration, 2),
        "errors": errors,
        "success_rate": round((n - errors) / n * 100, 2),
        "rps": round(n / duration, 2),
        "p50_ms": round(float(s.quantile(0.50)), 2),
        "p95_ms": round(float(s.quantile(0.95)), 2),
        "p99_ms": round(float(s.quantile(0.99)), 2),
    }
    print(f"  RPS: {result['rps']}, p95: {result['p95_ms']} мс")
    plot_resources(mon.data, str(OUT_DIR / "stage_6_4"), "6.4 DL var RPS")
    return result


def stage_6_5_drift(pid: int, duration_sec: int = 600) -> dict:
    print(f"\n=== 6.5. DL + дрейф {duration_sec} сек ===")
    stream_normal = make_stream(n=5000, seed=42)
    stream_drift = []
    for it in stream_normal:
        new_it = dict(it)
        new_it["glucose"] = min(it["glucose"] * 1.5, 299.0)
        stream_drift.append(new_it)

    ALERTS_LOG.write_text("", encoding="utf-8")
    alerts = AlertSystem(ALERTS_LOG, suppression_window_sec=60)

    t_drift_start = None
    first_alert_ts = None
    alert_count = 0
    delay = None

    stop = threading.Event()
    mon = threading.Thread(target=lambda: mon.__setattr__("data", monitor_resources(pid, stop)))
    mon.start()

    ref_glucose = np.array(
        [float(r["glucose"]) for r in stream_normal[:300] if r.get("glucose") is not None],
        dtype=float,
    )

    session = requests.Session()
    latencies, errors, n = [], 0, 0
    t0 = time.perf_counter()
    last_print = t0
    i = 0
    while time.perf_counter() - t0 < duration_sec:
        elapsed = time.perf_counter() - t0
        use_drift = elapsed > (2 if duration_sec < 60 else 120)
        if use_drift and t_drift_start is None:
            t_drift_start = time.perf_counter()

        item = (stream_drift if use_drift else stream_normal)[i % 5000]
        t_req = time.perf_counter()
        try:
            r = session.post(
                f"{DL_URL}/predict/batch", json={"items": [item], "n_jobs": 1}, timeout=5
            )
            if r.status_code not in (200, 422):
                errors += 1
        except Exception:
            errors += 1
        latencies.append((time.perf_counter() - t_req) * 1000)
        n += 1
        i += 1

        if n % 1000 == 0 and use_drift and n >= 1000:
            window_glucose = np.array(
                [
                    float(stream_drift[j % 5000]["glucose"])
                    for j in range(max(0, n - 300), n)
                    if stream_drift[j % 5000].get("glucose") is not None
                ],
                dtype=float,
            )
            if len(window_glucose) > 0 and len(ref_glucose) > 0:
                psi_val = psi(ref_glucose, window_glucose)
                if np.isfinite(psi_val) and psi_val > 0.1:
                    triggered = alerts.check(
                        "psi_glucose",
                        float(psi_val),
                        threshold=0.1,
                        level="critical",
                        action="log",
                    )
                    if triggered:
                        alert_count += 1
                        if first_alert_ts is None:
                            first_alert_ts = time.perf_counter()
                            delay = first_alert_ts - t_drift_start
                            print(f"\n  [ALERT] первое срабатывание, задержка: {delay:.1f} с")

        now = time.perf_counter()
        if now - last_print > 1:
            print(
                f"\r  {n} запросов, {n / (now - t0):.1f} RPS, alerts={alert_count}, errors={errors}",
                end="",
                flush=True,
            )
            last_print = now
    print()

    session.close()
    stop.set()
    mon.join(timeout=5)

    duration = time.perf_counter() - t0
    s = pd.Series(latencies)
    result = {
        "stage": "6.5_drift",
        "n_requests": n,
        "duration_sec": round(duration, 2),
        "errors": errors,
        "success_rate": round((n - errors) / n * 100, 2),
        "rps": round(n / duration, 2),
        "p50_ms": round(float(s.quantile(0.50)), 2),
        "p95_ms": round(float(s.quantile(0.95)), 2),
        "p99_ms": round(float(s.quantile(0.99)), 2),
        "alerts_count": alert_count,
        "first_alert_delay_sec": round(delay, 1) if delay else None,
    }
    print(
        f"  RPS: {result['rps']}, p95: {result['p95_ms']} мс, "
        f"errors: {result['errors']}, alerts: {alert_count}, delay: {result['first_alert_delay_sec']} с"
    )
    plot_resources(mon.data, str(OUT_DIR / "stage_6_5"), "6.5 DL + дрейф")
    return result


def stage_6_6_rollback(pid: int, duration_sec: int = 600) -> dict:
    print(f"\n=== 6.6. Откат под нагрузкой {duration_sec} сек ===")
    stop = threading.Event()
    mon = threading.Thread(target=lambda: mon.__setattr__("data", monitor_resources(pid, stop)))
    mon.start()

    session = requests.Session()
    latencies, errors, n = [], 0, 0
    t0 = time.perf_counter()
    last_switch = t0
    switch_count = 0
    last_print = t0
    switch_interval = 3 if duration_sec < 60 else 120
    while time.perf_counter() - t0 < duration_sec:
        t_req = time.perf_counter()
        try:
            r = session.post(f"{DL_URL}/predict/batch", json=DL_PAYLOAD, timeout=5)
            if r.status_code != 200:
                errors += 1
        except Exception:
            errors += 1
        latencies.append((time.perf_counter() - t_req) * 1000)
        n += 1

        if time.perf_counter() - last_switch > switch_interval:
            switch_count += 1
            target = "v2" if switch_count % 2 == 1 else "v1"
            try:
                requests.post(f"{DL_URL}/promote", json={"version": target}, timeout=5)
                print(f"\n  [switch] promote → {target}")
            except Exception as e:
                print(f"\n  [switch] error: {e}")
            last_switch = time.perf_counter()

        now = time.perf_counter()
        if now - last_print > 1:
            print(
                f"\r  {n} запросов, {n / (now - t0):.1f} RPS, switches={switch_count}",
                end="",
                flush=True,
            )
            last_print = now
    print()

    session.close()
    stop.set()
    mon.join(timeout=5)

    duration = time.perf_counter() - t0
    s = pd.Series(latencies)
    result = {
        "stage": "6.6_rollback",
        "n_requests": n,
        "duration_sec": round(duration, 2),
        "errors": errors,
        "success_rate": round((n - errors) / n * 100, 2),
        "rps": round(n / duration, 2),
        "p50_ms": round(float(s.quantile(0.50)), 2),
        "p95_ms": round(float(s.quantile(0.95)), 2),
        "p99_ms": round(float(s.quantile(0.99)), 2),
        "switches": switch_count,
    }
    print(f"  RPS: {result['rps']}, p95: {result['p95_ms']} мс, switch: {switch_count}")
    plot_resources(mon.data, str(OUT_DIR / "stage_6_6"), "6.6 Откат")
    return result


def main():
    print("ЛР8, п. 6: 6 этапов нагрузки")
    service_dl, log_dl = start_service("src.batch_service:app", 8001)
    print(f"DL-сервис на 8001, PID={find_service_pid(8001)}")
    service_ml, log_ml = start_service("src.service:app", 8002)
    print(f"ML-сервис на 8002, PID={find_service_pid(8002)}")

    try:
        results = []
        pid_dl = find_service_pid(8001)
        pid_ml = find_service_pid(8002)

        results.append(stage_6_1_warmup(300))
        results.append(run_clean_stage("6.2_clean_ML", ML_URL, pid_ml, 600))
        results.append(run_clean_stage("6.3_clean_DL", DL_URL, pid_dl, 600))
        results.append(stage_6_4_variable_rps(pid_dl, 600))
        results.append(stage_6_5_drift(pid_dl, 600))
        results.append(stage_6_6_rollback(pid_dl, 600))

        df = pd.DataFrame(results)
        df.to_csv(OUT_DIR / "stage6_results.csv", index=False)
        print(f"\nСохранено: {OUT_DIR / 'stage6_results.csv'}")
        print(df.to_string(index=False))

    finally:
        service_dl.terminate()
        service_ml.terminate()
        log_dl.close()
        log_ml.close()


if __name__ == "__main__":
    main()
