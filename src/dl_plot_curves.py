"""Построение кривых потерь для ЛР3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.dl_config import Lab3Config
from src.dl_dataset import prepare_full
from src.dl_pipeline import train_dl

ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
OUT = Path("reports/LAB3/loss_curves.png")


def main():
    df = pd.read_parquet("data/processed/pima.parquet")
    with open("configs/lab3_full.yaml", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)

    cfg = Lab3Config(**cfg_raw)
    data = prepare_full(
        df, cfg.data.target, ZERO_COLS, cfg.data.seed, cfg.data.test_size
    )
    result = train_dl(cfg, data)

    train_losses = result["train_losses"]
    val_losses = result["val_losses"]
    best_epoch = result["best_epoch"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="train loss",
        marker="o",
        markersize=3,
    )
    ax.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label="val loss",
        marker="s",
        markersize=3,
    )
    ax.axvline(
        best_epoch,
        color="green",
        linestyle="--",
        alpha=0.6,
        label=f"best epoch = {best_epoch}",
    )
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("BCE loss")
    ax.set_title("Кривые потерь MLP (full-режим, Pima)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {OUT}")


if __name__ == "__main__":
    main()
