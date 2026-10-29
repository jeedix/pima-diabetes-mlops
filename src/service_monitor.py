"""Сервис мониторинга: логирование запросов и метрик (ЛР6)."""

import json
from datetime import datetime, timezone
from pathlib import Path


class ServiceMonitor:
    """Пишет логи в JSONL: время, эндпоинт, статус, время ответа."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, endpoint: str, status: int, duration_ms: float, **extra):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            **extra,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
