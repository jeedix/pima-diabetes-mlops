"""Dataset и DataLoader для ЛР3.

Плюс функция prepare_full — готовит train/test split в памяти для full-режима.

Для chunks-режима DataLoader читает данные порциями с диска через
pyarrow.parquet.ParquetFile.iter_batches — без материализации всего датасета.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.pipelines.preprocessing import build_preprocessor


class PimaDataset(Dataset):
    """Torch Dataset для Pima с опциональными индикаторами NaN."""

    def __init__(
        self, X: np.ndarray, y: np.ndarray, indicators: np.ndarray | None = None
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if indicators is not None:
            self.ind = torch.tensor(indicators, dtype=torch.float32)
        else:
            self.ind = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.ind is not None:
            x = torch.cat([self.X[idx], self.ind[idx]])
        else:
            x = self.X[idx]
        return x, self.y[idx]


def make_indicators(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Бинарные индикаторы NaN для указанных колонок."""
    return df[cols].isna().astype(np.float32).values


def prepare_full(
    df: pd.DataFrame, target: str, zero_cols: list[str], seed: int, test_size: float
) -> dict:
    """Готовит массивы для полного режима (train/test split в памяти).

    Возвращает словарь с X_train, X_test, y_train, y_test,
    ind_train, ind_test, preprocessor.
    """
    y = df[target].values.astype(np.float32)
    X_raw = df.drop(columns=[target])

    ind = make_indicators(X_raw, zero_cols) if zero_cols else None

    X_train_raw, X_test_raw, y_train, y_test, ind_train, ind_test = train_test_split(
        X_raw,
        y,
        ind,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    pre = build_preprocessor(imputer_strategy="median", scaler="standard")
    X_train = pre.fit_transform(X_train_raw)
    X_test = pre.transform(X_test_raw)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "ind_train": ind_train,
        "ind_test": ind_test,
        "preprocessor": pre,
    }


def iter_chunks_from_parquet(
    parquet_path, batch_size: int, target: str, zero_cols: list[str], preprocessor
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Генератор: читает Parquet чанками, применяет препроцессор, отдаёт (X, y).

    Датасет целиком в память не загружается — только batch_size строк за раз.
    """
    parquet_file = pq.ParquetFile(parquet_path)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df_chunk = batch.to_pandas()
        y_chunk = df_chunk[target].values.astype(np.float32)
        X_chunk_raw = df_chunk.drop(columns=[target])

        ind_chunk = make_indicators(X_chunk_raw, zero_cols)
        X_chunk = preprocessor.transform(X_chunk_raw)

        # Конкатенируем признаки и индикаторы
        X_aug = np.hstack([X_chunk, ind_chunk]).astype(np.float32)

        yield X_aug, y_chunk


class PimaIterableDataset(torch.utils.data.IterableDataset):
    """Потоковый Dataset: читает Parquet чанками с диска.

    Весь датасет не загружается в память — каждый батч читается
    из Parquet и обрабатывается на лету.
    """

    def __init__(
        self,
        parquet_path: str,
        target: str,
        zero_cols: list[str],
        preprocessor,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.target = target
        self.zero_cols = zero_cols
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        parquet_file = pq.ParquetFile(self.parquet_path)

        # Собираем все строки порциями и выдаём батчами
        X_buf, y_buf = [], []
        for batch in parquet_file.iter_batches(batch_size=self.batch_size):
            df_chunk = batch.to_pandas()
            y_chunk = df_chunk[self.target].values.astype(np.float32)
            X_raw = df_chunk.drop(columns=[self.target])
            ind_chunk = make_indicators(X_raw, self.zero_cols)
            X_chunk = self.preprocessor.transform(X_raw)
            X_aug = np.hstack([X_chunk, ind_chunk]).astype(np.float32)

            for i in range(len(X_aug)):
                X_buf.append(X_aug[i])
                y_buf.append(y_chunk[i])

                if len(X_buf) == self.batch_size:
                    yield (
                        torch.tensor(np.array(X_buf), dtype=torch.float32),
                        torch.tensor(np.array(y_buf), dtype=torch.float32),
                    )
                    X_buf, y_buf = [], []

        if X_buf:
            yield (
                torch.tensor(np.array(X_buf), dtype=torch.float32),
                torch.tensor(np.array(y_buf), dtype=torch.float32),
            )
