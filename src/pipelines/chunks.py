"""Обучение чанками: чтение Parquet с диска порциями."""

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import SGDClassifier

from src.monitoring import measure_resources
from src.pipelines.metrics import bootstrap_ci
from src.pipelines.preprocessing import build_preprocessor


def train_chunks(parquet_path, X_test, y_test, cfg) -> dict:
    """Обучает SGD инкрементально, читая Parquet чанками с диска.

    Датасет целиком в память не загружается — только чанк за раз.
    """
    pre = build_preprocessor(
        imputer_strategy=cfg.pipeline.imputer_strategy,
        scaler=cfg.pipeline.scaler,
    )
    model = SGDClassifier(**cfg.pipeline.model_params)

    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows

    # Первый проход: fit препроцессора на первом чанке
    first_batch = next(parquet_file.iter_batches(batch_size=cfg.data.chunk_size))
    pre.fit(first_batch.to_pandas().drop(columns=[cfg.data.target]))

    # Второй проход: инкрементальное обучение с замером ресурсов
    with measure_resources() as res:
        for batch in parquet_file.iter_batches(batch_size=cfg.data.chunk_size):
            df_chunk = batch.to_pandas()
            X_chunk = pre.transform(df_chunk.drop(columns=[cfg.data.target]))
            y_chunk = df_chunk[cfg.data.target].values
            model.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1]))

    X_test_t = pre.transform(X_test)
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]
    metrics = bootstrap_ci(y_test.values, y_pred, y_proba, cfg.training.n_bootstrap)

    return {
        "model": model,
        "preprocessor": pre,
        "metrics": metrics,
        "perf": res,
        "n_rows": total_rows,
    }
