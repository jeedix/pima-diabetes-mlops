"""Генератор потока синтетических данных для разогрева (6.1)."""

import pandas as pd

from src.generator import fit_class_gaussians, generate_similar


def make_stream(n: int = 10_000, seed: int = 42) -> list[dict]:
    """Генерирует поток из n объектов (синтетика similar)."""
    df_real = pd.read_parquet("data/processed/pima.parquet")
    params = fit_class_gaussians(df_real, "Outcome")
    df_syn = generate_similar(params, n, seed, df_real)

    return [
        {
            "pregnancies": max(0, float(row["Pregnancies"])),
            "glucose": max(0, float(row["Glucose"])),
            "blood_pressure": max(0, float(row["BloodPressure"])),
            "skin_thickness": max(0, float(row["SkinThickness"])),
            "insulin": max(0, float(row["Insulin"])),
            "bmi": max(0, float(row["BMI"])),
            "diabetes_pedigree": max(0, float(row["DiabetesPedigreeFunction"])),
            "age": max(0, float(row["Age"])),
        }
        for _, row in df_syn.iterrows()
    ]


if __name__ == "__main__":
    stream = make_stream(100)
    print(f"Сгенерировано: {len(stream)} объектов")
    print(stream[0])
