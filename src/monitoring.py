"""Замер ресурсов: время, реальная память (RSS), загрузка CPU.

Использует psutil для получения реального потребления процесса,
а не только Python-аллокаций (tracemalloc).

Использование
-------------
    from src.monitoring import measure_resources

    with measure_resources() as res:
        model.fit(X_train, y_train)

    print(res["time_sec"], res["peak_mem_mb"], res["cpu_percent"])
"""

import time
from contextlib import contextmanager

import psutil


@contextmanager
def measure_resources():
    """Контекстный менеджер: время, пик RSS-памяти, средний CPU процесса.

    Возвращает
    ----------
    dict с ключами:
        time_sec      — время выполнения, секунды (округлено до 4 знаков)
        peak_mem_mb   — пик RSS-памяти процесса, МБ (округлено до 2 знаков)
        mem_delta_mb  — изменение RSS за время блока, МБ
        cpu_percent   — средняя загрузка CPU процессом, %

    Пример
    ------
    with measure_resources() as res:
        model.fit(X_train, y_train)
    print(f"Время: {res['time_sec']} с")
    print(f"Память: {res['peak_mem_mb']} МБ")
    print(f"CPU: {res['cpu_percent']}%")
    """
    process = psutil.Process()

    # Перед началом
    mem_before = process.memory_info().rss / 1e6  # байты → МБ
    process.cpu_percent(interval=None)  # сброс счётчика CPU
    t0 = time.perf_counter()

    result = {}

    try:
        yield result
    finally:
        # После окончания
        t1 = time.perf_counter()
        mem_after = process.memory_info().rss / 1e6
        cpu = process.cpu_percent(interval=None)

        result["time_sec"] = round(t1 - t0, 4)
        result["peak_mem_mb"] = round(max(mem_before, mem_after), 2)
        result["mem_delta_mb"] = round(mem_after - mem_before, 2)
        result["cpu_percent"] = round(cpu, 2)
