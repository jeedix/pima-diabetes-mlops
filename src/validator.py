"""numpy-валидатор: статистики обучающего распределения + правило отказа."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


class PimaValidator:
    """Валидатор для табличных данных Pima."""

    def __init__(
        self, distance_threshold: float = 5.0, confidence_threshold: float = 0.6
    ):
        self.distance_threshold = distance_threshold
        self.confidence_threshold = confidence_threshold
        self.mean = None
        self.cov_inv = None

    def fit(self, X: np.ndarray) -> "PimaValidator":
        """Оценивает статистики: среднее и обратную ковариацию."""
        self.mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        # Регуляризация для устойчивости
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv = np.linalg.inv(cov)
        return self

    def mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Расстояние Махаланобиса каждого объекта до центра."""
        diff = X - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self.cov_inv, diff))

    def reject(self, X: np.ndarray, proba: np.ndarray) -> np.ndarray:
        """Правило отказа: True = отказ, False = принять."""
        dist = self.mahalanobis(X)
        confidence = np.maximum(proba, 1 - proba)
        return (dist > self.distance_threshold) | (
            confidence < self.confidence_threshold
        )

    def score(self, X: np.ndarray, proba: np.ndarray) -> np.ndarray:
        """Оценка чужеродности: чем выше, тем более чужой объект.

        Комбинирует расстояние Махаланобиса и неуверенность модели.
        """
        dist = self.mahalanobis(X)
        confidence = np.maximum(proba, 1 - proba)
        return dist + (1 - confidence) * 10.0

    def save(self, path: Path) -> None:
        """Сохраняет параметры в .npz."""
        np.savez(
            path,
            mean=self.mean,
            cov_inv=self.cov_inv,
            distance_threshold=self.distance_threshold,
            confidence_threshold=self.confidence_threshold,
        )

    @classmethod
    def load(cls, path: Path) -> "PimaValidator":
        """Загружает параметры из .npz."""
        data = np.load(path)
        v = cls(
            distance_threshold=float(data["distance_threshold"]),
            confidence_threshold=float(data["confidence_threshold"]),
        )
        v.mean = data["mean"]
        v.cov_inv = data["cov_inv"]
        return v


def fit_chunked(parquet_path: Path, target: str, chunk_size: int) -> tuple:
    """Оценка статистик чанками из Parquet без полной загрузки."""
    parquet_file = pq.ParquetFile(parquet_path)
    n_total = 0
    sum_x = None
    cov_sum = None

    # Первый проход: среднее и ковариация
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        X = df.drop(columns=[target]).values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0)

        n = len(X)
        if n_total == 0:
            sum_x = X.sum(axis=0)
            cov_sum = np.cov(X, rowvar=False) * (n - 1)
        else:
            sum_x += X.sum(axis=0)
            cov_sum += np.cov(X, rowvar=False) * (n - 1)
        n_total += n

    mean = sum_x / n_total
    cov = cov_sum / max(n_total - 1, 1)
    cov += np.eye(cov.shape[0]) * 1e-6

    validator = PimaValidator()
    validator.mean = mean
    validator.cov_inv = np.linalg.inv(cov)
    return validator, n_total
