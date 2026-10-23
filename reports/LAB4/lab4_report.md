# ЛР 4. Пайплайн модели-валидатора: полный датасет и чанки

**Вариант:** 33
**Профиль:** P1 (табличный)
**Дата:** 2026-10-23
**Автор:** Байбурдин Максим
**Хеш коммита кода:** <44c5bdb2d910b2dced1b1cbecf5b4c8e113f58fd>
**Хеш данных:** 6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af

## 1. Схема валидатора (pydantic)

Конфиг `Lab4Config` (`src/validator_config.py`):
- `ValidatorDataConfig` — путь к Parquet, целевая колонка, test_size, seed, chunk_size.
- `ValidatorRuleConfig` — пороги отказа: distance_threshold=5.0, confidence_threshold=0.6.
- `ValidatorTrainingConfig` — режим (full / chunks), имя MLflow-эксперимента.

## 2. Правило отказа

Валидатор (`src/validator.py`) строит статистики обучающего распределения:
- среднее `mean` по признакам,
- обратную ковариационную матрицу `cov_inv`.

**Правило отказа:** объект отклоняется, если:
- расстояние Махаланобиса `dist > distance_threshold` (5.0), ИЛИ
- уверенность модели `confidence < confidence_threshold` (0.6).

Оценка чужеродности `score = dist + (1 - confidence) × 10` используется для ROC-AUC.

## 3. Сравнение режимов

| Метрика | Full | Chunks | Δ |
|---|---|---|---|
| False reject rate | 0.0584 | 0.0455 | 0.0129 |
| Foreign reject (shifted) | 0.9610 | 0.9610 | 0 |
| Foreign reject (outliers) | 1.0000 | 1.0000 | 0 |
| ROC-AUC | 0.7602 | 0.7609 | 0.0007 |

**Совпадение статистик:** разница `false_reject_rate` — 0.0129; ROC-AUC — 0.0007. Разницы на уровне шума, статистики чанкового режима близки к полным.

### 3.1. Параметры валидатора (`validator_params.npz`)

Файл `artifacts/validator_params.npz` сохраняется методом `PimaValidator.save()` (`src/validator.py`). Содержит 4 массива:

- **`mean`** — среднее обучающего распределения (вектор из 8 значений, по числу признаков Pima).
- **`cov_inv`** — обратная ковариационная матрица (8×8), используемая в расстоянии Махаланобиса.
- **`distance_threshold`** — порог расстояния (5.0), скаляр.
- **`confidence_threshold`** — порог уверенности (0.6), скаляр.

Загрузка восстанавливает валидатор через `PimaValidator.load()` — без повторного обучения. Воспроизводимость параметров проверена сравнением прогонов full и chunks: при одинаковых входных данных параметры `mean` и `cov_inv` совпадают в пределах машинной точности float64.

## 4. Идентификаторы прогонов MLflow

- **full:** `c5fc5f3109d44c208d79005cd3400443`
- **chunks:** `5d90aeafa7b54e9fa3713d2fe928e22e`

## 5. Выводы (по одному числу)

1. Валидатор обучался на 614 объектах, тест — 154.
2. False reject rate на своих данных: 5.84% (full), 4.55% (chunks).
3. Сдвиг средних отклоняется на 96.1% в обоих режимах.
4. Выбросы отклоняются на 100% в обоих режимах.
5. ROC-AUC валидатора: 0.7602 (full), 0.7609 (chunks).
6. Разница false_reject_rate между режимами = 0.0129.
7. Разница ROC-AUC = 0.0007.
8. `distance_threshold` = 5.0.
9. `confidence_threshold` = 0.6.
10. `chunk_size` = 128.

## 6. Ссылки

- **Скрипт:** `src/main_validator.py`
- **Валидатор:** `src/validator.py`
- **Pydantic-схема:** `src/validator_config.py`
- **Конфиг:** `configs/lab4_validator.yaml`
- **Метрики:** `reports/LAB4/validator_metrics.csv`
- **Негативные контроли:** `reports/LAB4/validator_negative.log`
- **Параметры:** `artifacts/validator_params.npz`
- **MLflow:** эксперимент `lab4_pima_validator`
