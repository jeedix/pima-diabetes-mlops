"""CLI: обучение валидатора (ЛР4)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.main_validator import load_config, run_validator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train validator (ЛР4)")
    parser.add_argument("--config", type=Path, default=Path("configs/lab4_validator.yaml"))
    parser.add_argument("--mode", choices=["full", "chunks"], default="full")
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.source == "synthetic":
        cfg.data.parquet_path = Path("artifacts/synthetic_pima.parquet")

    result = run_validator(cfg, Path("artifacts/ml_model.joblib"), mode=args.mode)
    print(
        f"[train_validator] mode={args.mode}, false_reject={result['metrics']['false_reject_rate']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
