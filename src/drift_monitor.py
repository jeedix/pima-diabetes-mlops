"""Мониторинг дрейфа: PSI/KS по признакам + PSI по target (ЛР7)."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

from src.psi_ks import psi, ks


def load_baseline(path: str) -> pd.DataFrame:
    """Загружает baseline (эталонное распределение)."""
    return pd.read_parquet(path)


def stream_chunks(parquet_path: str, chunk_size: int, drift_type: str = "none"):
    """Генератор окон данных с заданным типом дрейфа.

    Типы:
    - none: без дрейфа
    - shift: сдвиг средних (+3 к каждому признаку)
    - class_balance: удвоение класса 1 (дисбаланс)
    - flip: переворот меток
    """
    parquet_file = pq.ParquetFile(parquet_path)

    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()

        if drift_type == "shift":
            for col in df.columns:
                if col != "Outcome":
                    df[col] = df[col] + 3.0
        elif drift_type == "class_balance":
            df = pd.concat([df, df[df["Outcome"] == 1]], ignore_index=True)
        elif drift_type == "flip":
            df["Outcome"] = 1 - df["Outcome"]

        yield df


def compute_drift_window(baseline: pd.DataFrame, window: pd.DataFrame) -> list:
    """PSI/KS по признакам + PSI по target.

    Возвращает список словарей: [{"feature": ..., "psi": ..., "ks": ...}, ...].
    """
    rows = []

    for col in baseline.columns:
        if col == "Outcome":
            continue
        rows.append(
            {
                "feature": col,
                "psi": psi(baseline[col].values, window[col].values),
                "ks": ks(baseline[col].values, window[col].values),
            }
        )

    rows.append(
        {
            "feature": "Outcome",
            "psi": psi(baseline["Outcome"].values, window["Outcome"].values),
            "ks": ks(baseline["Outcome"].values, window["Outcome"].values),
        }
    )

    return rows


def main():
    with open("configs/lab7_monitoring.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    Path(cfg["drift"]["output_csv"]).parent.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline(cfg["drift"]["baseline_path"])
    window_size = cfg["drift"]["window_size"]
    psi_threshold = cfg["drift"]["psi_threshold"]
    target_threshold = 0.1  # отдельный порог для PSI по Outcome

    all_metrics = []

    for drift_type in ["none", "shift", "class_balance", "flip"]:
        print(f"\n=== Drift: {drift_type} ===")

        triggered_count = 0
        total_windows = 0

        for i, window in enumerate(
            stream_chunks(cfg["drift"]["baseline_path"], window_size, drift_type)
        ):
            if i > 2:
                break

            metrics = compute_drift_window(baseline, window)

            for m in metrics:
                all_metrics.append(
                    {
                        "drift_type": drift_type,
                        "window_idx": i,
                        "feature": m["feature"],
                        "psi": m["psi"],
                        "ks": m["ks"],
                    }
                )

            # Отдельные пороги: target — 0.1, признаки — psi_threshold
            triggered = any(
                m["psi"]
                > (target_threshold if m["feature"] == "Outcome" else psi_threshold)
                for m in metrics
            )

            max_psi = max(m["psi"] for m in metrics)
            target_psi = next(m["psi"] for m in metrics if m["feature"] == "Outcome")

            if triggered:
                triggered_count += 1
            total_windows += 1

            print(
                f"  window {i}: max PSI={max_psi:.4f} (target={target_psi:.4f}), triggered={triggered}"
            )

        print(f"  Итого: {triggered_count}/{total_windows} окон сработало")

    df = pd.DataFrame(all_metrics)
    df.to_csv(cfg["drift"]["output_csv"], index=False)
    print(f"\nСохранено: {cfg['drift']['output_csv']}")
    print(f"Всего строк: {len(df)}")


if __name__ == "__main__":
    main()
