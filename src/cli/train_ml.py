"""CLI: обучение ML-модели (ЛР2)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.main import run_full  # noqa: E402
from src.main import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train ML (ЛР2)")
    parser.add_argument("--config", type=Path, default=Path("configs/lab2_full.yaml"))
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    args = parser.parse_args()

    parquet = Path("data/processed/pima.parquet")
    if args.source == "synthetic":
        parquet = Path("artifacts/synthetic_pima.parquet")
    if not parquet.exists():
        print(f"Файл не найден: {parquet}. Сначала запустите generate.")
        return 1

    cfg = load_config(args.config)
    result = run_full(cfg, parquet)
    print(f"[train_ml] run_id={result['run_id']}, source={args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
