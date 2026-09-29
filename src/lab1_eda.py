"""ЛР1: EDA варианта 33 (Pima — нули-пропуски)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # не открывать окна, сохранять в файлы
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path("data/raw/pima-indians-diabetes.csv")
OUT_DIR = Path("reports/LAB1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_STATS = OUT_DIR / "eda_stats.csv"
OUT_PROBLEM = OUT_DIR / "problem_plot.png"
OUT_CORR = OUT_DIR / "correlation_plot.png"
OUT_DIST = OUT_DIR / "distributions_plot.png"

# Колонки, где 0 — закодированный пропуск
ZERO_AS_NAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Имена колонок (в файле их нет — задаём вручную)
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

sns.set_theme(style="whitegrid")

# --- Чтение CSV без заголовков ---
df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

print(f"Прочитан датасет: {df.shape[0]} строк × {df.shape[1]} колонок")
print(f"Колонки: {list(df.columns)}")

# --- 1. Паспорт ---
n_rows, n_cols = df.shape
class_counts = df["Outcome"].value_counts().to_dict()
class_share = (df["Outcome"].value_counts(normalize=True) * 100).round(2).to_dict()

# --- 2. Доли нулей ---
zero_stats = {}
for col in ZERO_AS_NAN:
    n_zero = int((df[col] == 0).sum())
    zero_stats[col] = {
        "n_zero": n_zero,
        "share_pct": round(n_zero / n_rows * 100, 2),
    }

# --- 3. График проблемы ---
fig, ax = plt.subplots(figsize=(9, 5))
cols = list(zero_stats.keys())
shares = [zero_stats[c]["share_pct"] for c in cols]
bars = ax.bar(cols, shares, color="salmon", edgecolor="black")
for bar, share in zip(bars, shares):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{share}%",
        ha="center",
        va="bottom",
        fontsize=10,
    )
ax.set_ylabel("Доля нулей (закодированных пропусков), %")
ax.set_title("Проблема варианта 33: закодированные пропуски в Pima")
ax.set_ylim(0, max(shares) * 1.25)
plt.tight_layout()
fig.savefig(OUT_PROBLEM, dpi=150)
plt.close(fig)

# --- 4. Корреляции (обязательно для P1) ---
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax
)
ax.set_title("Корреляции признаков Pima")
plt.tight_layout()
fig.savefig(OUT_CORR, dpi=150)
plt.close(fig)

# --- 5. Распределения ключевых величин ---
key_cols = ["Glucose", "BMI", "Age", "Insulin"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), key_cols):
    sns.boxplot(data=df, x="Outcome", y=col, ax=ax)
    ax.set_title(f"{col} по классам Outcome")
plt.tight_layout()
fig.savefig(OUT_DIST, dpi=150)
plt.close(fig)

# --- 6. Сводки в CSV ---
rows = []
for col in df.columns:
    rows.append(
        {
            "column": col,
            "dtype": str(df[col].dtype),
            "n_missing": int(df[col].isna().sum()),
            "n_zero": int((df[col] == 0).sum()) if col in ZERO_AS_NAN else 0,
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
    )
stats_df = pd.DataFrame(rows)
stats_df.to_csv(OUT_STATS, index=False)

# --- 7. Сводный JSON для отчёта ---
summary = {
    "n_rows": n_rows,
    "n_cols": n_cols,
    "class_counts": class_counts,
    "class_share_pct": class_share,
    "zero_stats": zero_stats,
}
(OUT_DIR / "eda_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nАртефакты: {OUT_STATS}, {OUT_PROBLEM}, {OUT_CORR}, {OUT_DIST}")
