"""Конвертация CSV Pima в Parquet с заменой нулей на NaN."""

from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("data/raw/pima-indians-diabetes.csv")
PARQUET_PATH = Path("data/processed/pima.parquet")

COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]
ZERO_AS_NAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def convert(csv_path: Path, parquet_path: Path) -> None:
    """Читает CSV, заменяет 0 на NaN в физиологических колонках, пишет Parquet."""
    df = pd.read_csv(csv_path, header=None, names=COLUMNS)
    df[ZERO_AS_NAN] = df[ZERO_AS_NAN].replace(0, np.nan)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    print(f"CSV:     {csv_path} ({csv_path.stat().st_size} байт)")
    print(f"Parquet: {parquet_path} ({parquet_path.stat().st_size} байт)")
    print(f"Строк: {len(df)}, колонок: {len(df.columns)}")
    print(f"NaN в Glucose: {df['Glucose'].isna().sum()}")


if __name__ == "__main__":
    convert(CSV_PATH, PARQUET_PATH)
