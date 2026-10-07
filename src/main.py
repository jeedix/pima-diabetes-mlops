"""Точка входа ЛР2: запуск ML-пайплайнов (full / chunks)."""

import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml

from src.config import Lab2Config
from src.data_utils import split_data
from src.pipelines import train_full, train_chunks


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Lab2Config:
    """Читает YAML и валидирует через pydantic."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Lab2Config(**raw)


def log_to_mlflow(result: dict, cfg: Lab2Config, run_name: str) -> str:
    """Логирует метрики и ресурсы прогона в MLflow. Возвращает run_id."""
    mlflow.set_experiment(cfg.training.mlflow_experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"mode": run_name, **cfg.pipeline.model_params})

        # Метрики с интервалами
        for k, v in result["metrics"].items():
            mlflow.log_metric(f"{k}_value", v["value"])
            mlflow.log_metric(f"{k}_ci_low", v["ci_low"])
            mlflow.log_metric(f"{k}_ci_high", v["ci_high"])

        # Ресурсы
        for k in ["time_sec", "peak_mem_mb", "mem_delta_mb", "cpu_percent"]:
            if k in result["perf"]:
                mlflow.log_metric(k, result["perf"][k])

        return run.info.run_id


def run_full(cfg: Lab2Config, parquet_path: Path) -> dict:
    """Запускает полный пайплайн (модель в памяти)."""
    df = pd.read_parquet(parquet_path)
    X_train, X_test, y_train, y_test = split_data(
        df,
        cfg.data.target,
        cfg.data.test_size,
        cfg.data.seed,
    )

    result = train_full(X_train, X_test, y_train, y_test, cfg)
    run_id = log_to_mlflow(result, cfg, run_name="full")

    # Сохраняем модель
    model_path = ARTIFACTS / "ml_model.joblib"
    joblib.dump(result["model"], model_path)

    return {"run_id": run_id, **result}


def run_chunks(cfg: Lab2Config, parquet_path: Path) -> dict:
    """Запускает чанковый пайплайн (чтение Parquet с диска)."""
    df = pd.read_parquet(parquet_path)
    _, X_test, _, y_test = split_data(
        df,
        cfg.data.target,
        cfg.data.test_size,
        cfg.data.seed,
    )

    result = train_chunks(parquet_path, X_test, y_test, cfg)
    run_id = log_to_mlflow(result, cfg, run_name="chunks")

    return {"run_id": run_id, **result}


def main():
    parser = argparse.ArgumentParser(description="ЛР2: ML pipeline")
    parser.add_argument(
        "--config-full",
        type=Path,
        default=ROOT / "configs" / "lab2_full.yaml",
        help="Конфиг для полного режима",
    )
    parser.add_argument(
        "--config-chunks",
        type=Path,
        default=ROOT / "configs" / "lab2_chunks.yaml",
        help="Конфиг для чанкового режима",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=ROOT / "data" / "processed" / "pima.parquet",
        help="Путь к Parquet-файлу",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "chunks", "both"],
        default="both",
        help="Что запустить",
    )
    args = parser.parse_args()

    print(f"Parquet: {args.parquet}")
    print(f"Mode: {args.mode}")

    if args.mode in ("full", "both"):
        cfg_full = load_config(args.config_full)
        print(f"Config (full): {args.config_full}")
        r = run_full(cfg_full, args.parquet)
        print(
            f"[full]   run_id={r['run_id']}  "
            f"time={r['perf']['time_sec']}s  mem={r['perf']['peak_mem_mb']}MB  "
            f"cpu={r['perf']['cpu_percent']}%"
        )

    if args.mode in ("chunks", "both"):
        cfg_chunks = load_config(args.config_chunks)
        print(f"Config (chunks): {args.config_chunks}")
        r = run_chunks(cfg_chunks, args.parquet)
        print(
            f"[chunks] run_id={r['run_id']}  "
            f"time={r['perf']['time_sec']}s  mem={r['perf']['peak_mem_mb']}MB  "
            f"cpu={r['perf']['cpu_percent']}%"
        )


if __name__ == "__main__":
    main()
