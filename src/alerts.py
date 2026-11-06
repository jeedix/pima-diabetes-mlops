"""Система алертов: пороги, подавление повторов, JSONL-лог (ЛР8)."""

import json
from datetime import datetime, timezone
from pathlib import Path


class AlertSystem:
    """Алерты с подавлением повторов.

    Каждый алерт — запись в JSONL: время, показатель, значение, порог, уровень, действие.
    Подавление: повторный алерт того же показателя — не более 1 за период (окно suppression_window).
    """

    def __init__(self, log_path: Path, suppression_window_sec: float = 300.0):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.suppression_window_sec = suppression_window_sec
        self._last_alerts: dict[str, float] = {}

    def check(
        self,
        metric: str,
        value: float,
        threshold: float,
        level: str = "warning",
        action: str = "log",
    ) -> bool:
        """Проверяет метрику. Если value > threshold — пишет алерт (с подавлением).

        Возвращает True, если алерт записан, False — если подавлен или не сработал.
        """
        now = datetime.now(timezone.utc).timestamp()

        if value <= threshold:
            return False

        # Подавление повторов
        if metric in self._last_alerts:
            elapsed = now - self._last_alerts[metric]
            if elapsed < self.suppression_window_sec:
                return False

        self._last_alerts[metric] = now

        entry = {
            "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "metric": metric,
            "value": round(value, 4),
            "threshold": round(threshold, 4),
            "level": level,
            "action": action,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"ALERT [{level}] {metric}={value:.4f} > {threshold:.4f} — {action}")
        return True


def main():
    """Демонстрация: чистый прогон (0 алертов) + дрейф (алерты)."""
    log = Path("reports/LAB8/alerts_sample.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("", encoding="utf-8")  # очищаем

    alerts = AlertSystem(log, suppression_window_sec=60)

    # Чистый прогон — значения ниже порогов
    print("=== Чистый прогон ===")
    for value in [0.05, 0.10, 0.15, 0.08]:
        alerts.check("psi_glucose", value, threshold=0.5, level="warning")

    # Дрейф — значения выше порога
    print("\n=== Дрейф (сдвиг средних) ===")
    for value in [12.4, 12.4, 12.4]:  # 3 алерта, но подавление — 1
        alerts.check(
            "psi_glucose", value, threshold=0.5, level="critical", action="promote_rollback"
        )

    print("\n=== Дрейф (дисбаланс классов) ===")
    for value in [0.25, 0.25]:  # подавление
        alerts.check("psi_outcome", value, threshold=0.1, level="warning", action="log")

    print("\n=== Деградация валидатора ===")
    alerts.check("false_reject_rate", 0.15, threshold=0.10, level="warning", action="notify")

    # Показываем лог
    print(f"\n=== Лог {log} ===")
    print(log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
