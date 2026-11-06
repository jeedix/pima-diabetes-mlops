"""CLI: обучение DL-модели (ЛР3)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.main_dl import load_config, run_dl_full, run_dl_chunks  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train DL (ЛР3)")
    parser.add_argument("--config", type=Path, default=Path("configs/lab3_full.yaml"))
    parser.add_argument("--mode", choices=["full", "chunks"], default="full")
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.source == "synthetic":
        cfg.data.parquet_path = Path("artifacts/synthetic_pima.parquet")

    result = run_dl_full(cfg) if args.mode == "full" else run_dl_chunks(cfg)
    print(f"[train_dl] run_id={result['run_id']}, source={args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
