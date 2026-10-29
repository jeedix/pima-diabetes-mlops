# ЛР 6. FastAPI-сервис модели с бесшовным переключением

**Вариант:** 33
**Профиль:** P1 (табличный)
**Дата:** 2026-10-29
**Автор:** Байбурдин Максим
**Хеш коммита кода:** 058f29dde3e250da1db898b99952c621414e640a

## 1. Адреса и схемы

| Адрес | Метод | Назначение |
|---|---|---|
| `/health` | GET | Статус сервиса, активная модель, uptime |
| `/predict` | POST | Предсказание для одного объекта |
| `/metrics` | GET | Метрики сервиса |
| `/models` | GET | Список версий моделей со статусами |
| `/promote` | POST | Продвижение версии до рабочей |
| `/rollback` | POST | Откат к предыдущей версии |

**Схема запроса `PredictRequest`:**
- 8 обязательных полей: pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age.
- Все поля — float с ограничениями (`ge=0`, `le=...`).
- Невалидный вход → **422** с объяснением.

**Схема ответа `PredictResponse`:**
- `prediction` (int, 0/1)
- `probability` (float, 0..1)
- `model_version` (str)

## 2. Реестр версий

Файл `reports/LAB6/registry.json` содержит версии моделей:
- `version` — имя (v1, v2),
- `path` — путь к .joblib,
- `hash` — SHA-256,
- `metrics` — accuracy,
- `status` — `ready` / `trial` / `working` / `retired`.

**Активная версия:** `v1` (рабочая).

## 3. Контрактные тесты

| Тест | Ожидание | Результат |
|---|---|---|
| `test_health` | 200, status=ok | ✅ |
| `test_predict_valid` | 200, prediction in {0,1} | ✅ |
| `test_predict_missing_field` | 422 | ✅ |
| `test_predict_wrong_type` | 422 | ✅ |
| `test_predict_out_of_range` | 422 | ✅ |
| `test_models_list` | 200, v1+v2 | ✅ |
| `test_promote_and_rollback` | active=v2 → active=v1 | ✅ |
| `test_promote_unknown_version` | 404 | ✅ |
| `test_metrics` | 200, uptime | ✅ |
| `test_load_100_predict` | 100/100 успешных | ✅ |

**Все 10 тестов прошли.**

## 4. Бесшовное переключение (дрилл)

**Сценарий:**
1. Активна `v1` — предсказание: `model_version="v1"`, probability=0.8405.
2. `POST /promote {"version": "v2"}` → `active="v2"`.
3. Предсказание: `model_version="v2"`, probability=0.8107.
4. `POST /rollback {}` → `active="v1"`.
5. Предсказание снова `model_version="v1"`.

**Доля успешных запросов:** 100% (нагрузка 100 запросов прошла без ошибок).
**Потерянных запросов:** 0.

## 5. Сервис мониторинга

**Модуль `src/service_monitor.py`:** пишет JSONL-логи в `reports/LAB6/service.log`. Каждая запись: `timestamp`, `endpoint`, `status`, `duration_ms`, `version`, `prediction`.

**Модуль `src/monitoring.py`** (из ЛР1–ЛР3) — замер ресурсов через `psutil`, используется в обучении моделей.

## 6. Выводы (по одному числу на строку)

1. Эндпоинтов: 6 (`/health`, `/predict`, `/metrics`, `/models`, `/promote`, `/rollback`).
2. Контрактных тестов: 10, все прошли.
3. Нагрузка 100 запросов к `/predict`: 100/100 успешных (100%).
4. Потерянных запросов при переключении: 0.
5. Активная версия v1: probability=0.8405.
6. После promote v2: probability=0.8107.
7. После rollback: снова v1, probability=0.8405.
8. Невалидный вход (`pregnancies=-5`): код 422.
9. Неизвестная версия (`v99`): код 404.
10. Uptime сервиса в момент проверки: >200 с.

## 7. Ссылки

### Код
- **Сервис:** `src/service.py`
- **Pydantic-схемы:** `src/api_config.py`
- **Реестр:** `src/registry.py`
- **Мониторинг:** `src/service_monitor.py`
- **Тесты:** `src/test_service.py`
- **Подготовка моделей:** `src/prepare_models.py`

### Артефакты ЛР6
- **Реестр:** `reports/LAB6/registry.json`
- **Лог тестов:** `reports/LAB6/service_tests.log`
- **Лог сервиса:** `reports/LAB6/service.log`
- **Модели:** `artifacts/models/v1.joblib`, `v2.joblib`

### Команда запуска
```powershell
python -m uvicorn src.service:app --reload --port 8000
