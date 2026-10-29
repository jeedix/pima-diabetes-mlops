"""Метрики отличия распределений: PSI и KS."""

import numpy as np
from scipy import stats


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index для одной колонки.

    PSI < 0.1 — распределения близки
    PSI 0.1–0.2 — умеренное отличие
    PSI > 0.2 — значительное отличие
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Ограничение, чтобы избежать log(0)
    eps = 1e-6
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    return float(
        np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    )


def ks(expected: np.ndarray, actual: np.ndarray) -> float:
    """KS-статистика (0 — одинаковые, 1 — совершенно разные)."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    statistic, _ = stats.ks_2samp(expected, actual)
    return float(statistic)
