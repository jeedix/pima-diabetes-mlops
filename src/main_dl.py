"""Точка входа ЛР3: DL пайплайн (full / chunks)."""

import argparse
import hashlib
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml

from src.dl_config import Lab3Config
from src.dl_dataset import prepare_full
from src.dl_pipeline import train_dl, train_dl_streaming

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_config(path: Path) -> Lab3Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Lab3Config(**raw)


def save_model_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _log_mlflow(cfg: Lab3Config, result: dict) -> None:
    """Общее логирование параметров и метрик в MLflow."""
    mlflow.log_params(
        {
            "mode": cfg.training.mode,
            "hidden_sizes": cfg.architecture.hidden_sizes,
            "dropout": cfg.architecture.dropout,
            "activation": cfg.architecture.activation,
            "epochs": cfg.training.epochs,
            "lr": cfg.training.lr,
            "weight_decay": cfg.training.weight_decay,
            "batch_size": cfg.data.batch_size,
            "n_params": result["n_params"],
            "best_epoch": result["best_epoch"],
        }
    )
    for k, v in result["metrics"].items():
        mlflow.log_metric(f"{k}_value", v["value"])
        mlflow.log_metric(f"{k}_ci_low", v["ci_low"])
        mlflow.log_metric(f"{k}_ci_high", v["ci_high"])
    mlflow.log_metric("time_sec", result["perf"]["time_sec"])
    mlflow.log_metric("peak_mem_mb", result["perf"]["peak_mem_mb"])
    mlflow.log_metric("cpu_percent", result["perf"]["cpu_percent"])


def run_dl_full(cfg: Lab3Config, device: str = "cpu") -> dict:
    """Полный режим: весь датасет в памяти."""
    df = pd.read_parquet(cfg.data.parquet_path)
    data = prepare_full(
        df, cfg.data.target, ZERO_COLS, cfg.data.seed, cfg.data.test_size
    )

    mlflow.set_experiment(cfg.training.mlflow_experiment)
    with mlflow.start_run(run_name="full") as run:
        result = train_dl(cfg, data, device=device)
        _log_mlflow(cfg, result)

        model_path = ARTIFACTS / "dl_model.pt"
        torch.save(result["model"].state_dict(), model_path)
        model_hash = save_model_hash(model_path)
        mlflow.log_param("model_hash", model_hash)

        return {"run_id": run.info.run_id, "model_hash": model_hash, **result}


def run_dl_chunks(cfg: Lab3Config, device: str = "cpu") -> dict:
    """Потоковый режим: чтение Parquet чанками с диска."""
    df = pd.read_parquet(cfg.data.parquet_path)
    data = prepare_full(
        df, cfg.data.target, ZERO_COLS, cfg.data.seed, cfg.data.test_size
    )
    preprocessor = data["preprocessor"]

    mlflow.set_experiment(cfg.training.mlflow_experiment)
    with mlflow.start_run(run_name="chunks") as run:
        result = train_dl_streaming(
            cfg,
            data,
            preprocessor,
            parquet_path=str(cfg.data.parquet_path),
            device=device,
        )
        _log_mlflow(cfg, result)

        model_path = ARTIFACTS / "dl_model_chunks.pt"
        torch.save(result["model"].state_dict(), model_path)
        model_hash = save_model_hash(model_path)
        mlflow.log_param("model_hash", model_hash)

        return {"run_id": run.info.run_id, "model_hash": model_hash, **result}


def main():
    parser = argparse.ArgumentParser(description="ЛР3: DL pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config: {args.config}")
    print(f"Mode: {cfg.training.mode}")
    print(f"Device: {args.device}")

    if cfg.training.mode == "full":
        r = run_dl_full(cfg, device=args.device)
    else:
        r = run_dl_chunks(cfg, device=args.device)

    print(f"[{cfg.training.mode}] run_id={r['run_id']}")
    print(f"  n_params: {r['n_params']}")
    print(f"  best_epoch: {r['best_epoch']}")
    print(
        f"  time: {r['perf']['time_sec']}s, "
        f"mem: {r['perf']['peak_mem_mb']}MB, "
        f"cpu: {r['perf']['cpu_percent']}%"
    )
    print(f"  model_hash: {r['model_hash']}")
    print("  metrics:")
    for k, v in r["metrics"].items():
        print(f"    {k}: {v['value']:.4f} (CI: {v['ci_low']:.4f}-{v['ci_high']:.4f})")


if __name__ == "__main__":
    main()
