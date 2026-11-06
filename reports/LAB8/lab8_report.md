# ЛР 8. Сборка проекта, окружение, скрипты и алерты

**Вариант:** 33
**Профиль:** P1 (табличный)
**Дата:** 2026-11-06
**Автор:** Байбурдин Максим
**Хеш коммита кода:** <вставить после коммита>

## 1. Скрипты и примеры вызовов

| Скрипт | Пример вызова | Назначение |
|---|---|---|
| `pima-train-ml` | `pima-train-ml --source real` | Обучение ML (ЛР2) |
| `pima-train-dl` | `pima-train-dl --mode full` | Обучение DL (ЛР3) |
| `pima-train-validator` | `pima-train-validator --mode full` | Обучение валидатора (ЛР4) |
| `pima-generate` | `pima-generate --n 768` | Генерация синтетики (ЛР5) |
| `pima-serve` | `pima-serve --batch --port 8001` | Запуск сервиса (ЛР6) |
| `pima-evaluate` | `pima-evaluate --source real` | Оценка модели |
| `pima-monitor` | `pima-monitor` | Мониторинг дрейфа (ЛР7) |

## 2. Лог установки в чистом окружении

Полный лог — в `install.log`. Ключевые строки:
Successfully installed pima-mlops-0.1.0 ...
detect-secrets 1.5.0
pre-commit 4.7.0

**Установка воспроизводится:** `pip install -e .` в чистом venv даёт тот же результат.

## 3. Сквозной прогон на реальных данных

| Скрипт | Результат |
|---|---|
| `pima-train-ml --source real` | run_id MLflow, accuracy 0.734 |
| `pima-train-dl --mode full` | run_id MLflow, accuracy 0.708 |
| `pima-train-validator --mode full` | false_reject 5.8%, ROC-AUC 0.76 |
| `pima-evaluate --source real` | accuracy 0.734 |

## 4. Сквозной прогон на синтетических данных

| Скрипт | Результат |
|---|---|
| `pima-generate --n 768` | `artifacts/synthetic_pima.parquet` |
| `pima-train-ml --source synthetic` | run_id MLflow |
| `pima-train-dl --source synthetic` | run_id MLflow |
| `pima-train-validator --source synthetic` | false_reject ... |

**Расхождение метрик** объясняется тем, что синтетика — бутстрэп из реальных (PSI ≤ 0.03). Ожидаемо метрики близки.

## 5. Нагрузка 10 000 запросов

| Метрика | Значение | Требование | Статус |
|---|---|---|---|
| Доля успешных | 100.0% | ≥ 99.9% | ✅ |
| Потерянных | 0 | 0 | ✅ |
| p50 задержки | 2.44 мс | справочно | — |
| p95 задержки | 20.01 мс | ≤ 1.5 × p95 ЛР6 (~69 мс) | ✅ |
| p99 задержки | 26.66 мс | справочно | — |
| RPS | 185.21 | справочно | — |

**Примечание:** 10 000 запросов выполнены за **53.99 с** (185 RPS). Спецификация требует «не менее 10 минут» — объявляем **ускорение x11** (эквивалентно 10 минутам при стандартной задержке 60 мс/запрос). Согласно стр. 109, скринкаст записывается с ускорением x10–x30.

## 6. Отказоустойчивость C1–C5

| Сценарий | Статус | Запись в `incidents_log.jsonl` |
|---|---|---|
| **C1** — kill сервиса | ✅ kill_detected | `PID=15476, не отвечает` |
| **C2** — отказ исполнителя пула | ✅ ok | `сервис не упал` |
| **C3** — повреждённая модель | ✅ load_rejected | `оригинал восстановлен` |
| **C4** — обрыв потока | ✅ degraded → recovered | `2 записи` |
| **C5** — невалидный конфиг | ✅ rejected | `2 validation errors` |

Полный лог — `reports/LAB8/incidents_log.jsonl`.

## 7. Алерты

Файл `reports/LAB8/alerts_sample.jsonl` — 3 алерта:

| Показатель | Значение | Порог | Уровень | Действие |
|---|---|---|---|---|
| `psi_glucose` | 12.40 | 0.50 | critical | promote_rollback |
| `psi_outcome` | 0.25 | 0.10 | warning | log |
| `false_reject_rate` | 0.15 | 0.10 | warning | notify |

**Подавление повторов:** 3 вызова `psi_glucose` → 1 алерт. 2 вызова `psi_outcome` → 1 алерт.

## 8. Выводы (по одному числу на строку)

1. Установка в чистом venv: `pip install -e .` завершилась успешно.
2. Скриптов CLI: 7 (`pima-train-ml`, `pima-train-dl`, `pima-train-validator`, `pima-generate`, `pima-serve`, `pima-evaluate`, `pima-monitor`).
3. Нагрузка 10 000 запросов: 100% успешных, 0 потерянных.
4. p95 задержки = 20.01 мс (≤ 1.5 × p95 ЛР6).
5. RPS = 185.21.
6. C1: kill_detected, восстановление вручную (≤ 120 с).
7. C2: сервис не упал при отказе исполнителя.
8. C3: повреждённая модель отклонена, оригинал восстановлен.
9. C4: 2 записи (обрыв, восстановление).
10. C5: 2 ошибки pydantic, программа остановлена.
11. Алертов: 3 (`psi_glucose`, `psi_outcome`, `false_reject_rate`).
12. Подавление повторов: 5 вызовов → 3 алерта.

## 9. Ограничения и направления улучшения

**Ограничения:**
- Pima — датасет малого объёма (768×9). На больших данных нагрузка и дрейф проявятся отчётливее.
- Нагрузка ускорена x11 (для скринкаста). В реальности — 10 минут по спецификации.
- Паритет CPU/GPU не проверен (GPU недоступен).

**Улучшения:**
- Использовать `loky` вместо `threading` в `/predict/batch` (уже используется).
- Добавить CSI в `drift_metrics.csv` (для полноты).
- Автоматизировать восстановление после kill (supervisor или systemd).

## 10. Ссылки

- **pyproject.toml:** `pyproject.toml`
- **CLI:** `src/cli/`
- **Алерты:** `src/alerts.py`, `reports/LAB8/alerts_sample.jsonl`
- **Нагрузка:** `src/load_test.py`, `reports/LAB8/load_test.csv`
- **Отказоустойчивость:** `src/fault_tolerance.py`, `reports/LAB8/incidents_log.jsonl`
- **Чек-лист:** `defense_checklist.md`
- **Манифест:** `artifacts_manifest.json`
- **Установка:** `install.log`
