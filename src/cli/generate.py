"""CLI: генерация синтетики (ЛР5)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402

from src.generator import fit_class_gaussians, generate_similar  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic (ЛР5)")
    parser.add_argument("--mode", choices=["similar", "random"], default="similar")
    parser.add_argument("--n", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df_real = pd.read_parquet("data/processed/pima.parquet")
    params = fit_class_gaussians(df_real, "Outcome")
    df_syn = generate_similar(params, args.n, args.seed, df_real)
    out = Path("artifacts/synthetic_pima.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df_syn.to_parquet(out, index=False)
    print(f"[generate] {out}, n={len(df_syn)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
