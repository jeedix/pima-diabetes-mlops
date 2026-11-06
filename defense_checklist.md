# Defense Checklist — вариант 33 (Pima Indians Diabetes)

**ФИО:** Байбурдин Максим
**Вариант:** 33
**Профиль:** P1 (табличный)
**Дата:** 2026-11-06

## Сводная таблица по восьми работам

| № | Работа | Комплект | Хеши | Негативные контроли | CI | Коммиты | Срок | Замечания |
|---|---|---|---|---|---|---|---|---|
| 1 | Git, pre-commit, EDA | ✅ | ✅ | ✅ 4/4 | ✅ | ✅ | ✅ | — |
| 2 | ML pipeline full/chunks | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | psutil, parquet, pipelines/ |
| 3 | DL pipeline full/chunks | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | reproducibility 4 сценария |
| 4 | Validator full/chunks | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ROC-AUC 0.76, PSI по target |
| 5 | Generator + pytest | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 94% coverage, PSI ≤ 0.028 |
| 6 | FastAPI service | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 11 тестов, switch_under_load |
| 7 | Batch + drift monitor | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 1.9x ускорение, 0 ложных |
| 8 | Сборка, CLI, алерты | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | в процессе |

## Детали по работам

### ЛР1. Git-окружение, pre-commit, EDA

- **Артефакты:** `lab1_report.md`, `eda_stats.csv`, `eda_resources.json`, `problem_plot.png`, `correlation_plot.png`, `distributions_plot.png`, `.pre-commit-config.yaml`, `logs/precommit_block.log`.
- **Негативные контроли:** стиль (ruff), секрет (detect-secrets), большой файл (check-added-large-files), подмена данных (hash_manifest).
- **Замер ресурсов:** `psutil`, RSS 184–212 МБ, CPU 97.3%.

### ЛР2. ML pipeline full/chunks

- **Артефакты:** `lab2_report.md`, `ml_metrics.csv`, `config_negative_test.log`, `reproducibility.log`, `artifacts/ml_model.joblib`.
- **Ключевое:** `src/pipelines/` (модуль), `src/main.py` (CLI), Parquet, `psutil`.
- **Воспроизводимость:** 4 сценария, Δ = 0.0000 (один seed), Δ = 0.5065 (разный).

### ЛР3. DL pipeline full/chunks

- **Артефакты:** `lab3_report.md`, `dl_metrics.csv`, `memory_compare.csv`, `loss_curves.png`, `reproducibility.log`, `parity.log`, `dl_config_negative.log`.
- **Ключевое:** `PimaIterableDataset`, early stopping, MLflow, `dl_model.pt`.
- **Воспроизводимость:** Δ = 0.000000 (один seed), Δ = 0.0696 (разный), Δ = 0.00e+00 (загрузка).

### ЛР4. Validator full/chunks

- **Артефакты:** `lab4_report.md`, `validator_metrics.csv`, `validator_negative.log`, `validator_params.npz`.
- **Ключевое:** `PimaValidator` (расстояние Махаланобиса), PSI по target как CSI.
- **Метрики:** false_reject=5.8% (full) / 4.5% (chunks), outliers=100%, ROC-AUC=0.76.

### ЛР5. Generator + pytest

- **Артефакты:** `lab5_report.md`, `synthetic_metrics.csv`, `generator_config.json`, `generator_tests.log`.
- **Ключевое:** бутстрэп (similar), равномерный шум (random), ручки дрейфа.
- **Тесты:** 10 passed, coverage 94%.

### ЛР6. FastAPI service

- **Артефакты:** `lab6_report.md`, `registry.json`, `service_tests.log`.
- **Ключевое:** 6 эндпоинтов, реестр версий, бесшовное переключение.
- **Тесты:** 11 passed, 100/100 нагрузка, `test_switch_under_load`.

### ЛР7. Batch + drift monitor

- **Артефакты:** `lab7_report.md`, `parallel_benchmark.csv`, `drift_metrics.csv`.
- **Ключевое:** `/predict/batch` с `joblib` (loky), PSI/KS по признакам + target.
- **Ускорение:** 1.9x при `n_jobs=4` (24.54 vs 46.52 мс).
- **Дрейф:** none=0/3, shift=3/3, class_balance=2/3, flip=3/3.

### ЛР8. Сборка, CLI, алерты (в процессе)

- **Артефакты (8.1):** `pyproject.toml`, `install.log`, `artifacts_manifest.json`, `src/cli/*`.
- **В процессе (8.2–8.4):** `alerts_sample.jsonl`, `lab8_report.md`, нагрузка 10 000 запросов, отказоустойчивость C1–C5.

## Раскрытие использования ИИ

Все восемь лабораторных работ выполнены с использованием **Claude (Anthropic)** в качестве консультанта по:

- структуре репозитория и pre-commit;
- pydantic-схемам конфигов;
- шаблонам скриптов обучения и оценки;
- отладке ошибок (sklearn, pydantic, MLflow, FastAPI, joblib);
- текстам отчётов (на основе реальных чисел из артефактов).

**Самостоятельно:**

- запуск всех скриптов и тестов;
- сбор реальных чисел из MLflow и CSV;
- интерпретация результатов;
- написание критических комментариев к рецензиям БЯМ.

## Рецензии БЯМ и комментарии

| Работа | Рецензия | Комментарий |
|---|---|---|
| ЛР1 | `reports/LAB1/ai_review1` | в том же файле |
| ЛР2 | `reports/LAB2/ai_review2` | в том же файле |
| ЛР3 | `reports/LAB3/ai_review3` | в том же файле |
| ЛР4 | `reports/LAB4/ai_review4` | в том же файле |
| ЛР5 | `reports/LAB5/ai_review5` | в том же файле |
| ЛР6 | `reports/LAB6/ai_review6` | в том же файле |
| ЛР7 | `reports/LAB7/ai_review7` | в том же файле |
| ЛР8 | ⏳ | ⏳ |

**Замечание:** рецензия и критический комментарий объединены в одном файле `ai_reviewN` (осознанное решение — допустимо).
