"""Точка входа ЛР5: генерация синтетики и сравнение с реальными."""

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from src.generator_config import Lab5Config
from src.generator import (
    fit_class_gaussians,
    generate_similar,
    generate_random,
    apply_drift,
)
from src.psi_ks import psi, ks


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
REPORTS = ROOT / "reports" / "LAB5"
REPORTS.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Lab5Config:
    with open(path, encoding="utf-8") as f:
        return Lab5Config(**yaml.safe_load(f))


def save_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="ЛР5: генератор синтетики")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Mode: {cfg.output.mode}")
    print(f"Drift: {cfg.output.drift}")

    df_real = pd.read_parquet(cfg.data.real_parquet)

    # Генерация
    if cfg.output.mode == "similar":
        params = fit_class_gaussians(df_real, cfg.data.target)
        df_syn = generate_similar(params, cfg.data.n_samples, cfg.data.seed, df_real)
    else:
        df_syn = generate_random(df_real, cfg.data.n_samples, cfg.data.seed)

    # Дрейф
    df_syn = apply_drift(df_syn, cfg.output.drift.model_dump(), cfg.data.target)

    # Сохраняем
    cfg.data.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_syn.to_parquet(cfg.data.output_parquet, index=False)
    synth_hash = save_hash(cfg.data.output_parquet)
    print(f"Хеш синтетики: {synth_hash}")

    # Метрики PSI/KS по признакам
    metrics_rows = []
    for col in df_real.columns:
        if col == cfg.data.target:
            continue
        psi_val = psi(df_real[col].values, df_syn[col].values)
        ks_val = ks(df_real[col].values, df_syn[col].values)
        metrics_rows.append({"feature": col, "psi": psi_val, "ks": ks_val})
        print(f"  {col}: PSI={psi_val:.4f}, KS={ks_val:.4f}")

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(REPORTS / "synthetic_metrics.csv", index=False)

    # Генератор-конфиг
    (REPORTS / "generator_config.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # MLflow
    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run(run_name=cfg.output.mode) as run:
        mlflow.log_params(
            {
                "mode": cfg.output.mode,
                "n_samples": cfg.data.n_samples,
                "seed": cfg.data.seed,
                "shift_mean": cfg.output.drift.shift_mean,
                "flip_labels": cfg.output.drift.flip_labels,
                "scale_features": cfg.output.drift.scale_features,
                "synth_hash": synth_hash,
            }
        )
        for row in metrics_rows:
            mlflow.log_metric(f"psi_{row['feature']}", row["psi"])
            mlflow.log_metric(f"ks_{row['feature']}", row["ks"])
        mlflow.log_artifact(str(cfg.data.output_parquet))
        mlflow.log_artifact(str(REPORTS / "synthetic_metrics.csv"))

        print(f"run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
