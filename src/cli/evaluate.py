"""CLI: оценка модели (ЛР8)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate (ЛР8)")
    parser.add_argument("--model", type=Path, default=Path("artifacts/ml_model.joblib"))
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    args = parser.parse_args()

    parquet = (
        "data/processed/pima.parquet"
        if args.source == "real"
        else "artifacts/synthetic_pima.parquet"
    )
    df = pd.read_parquet(parquet)
    X = np.nan_to_num(df.drop(columns=["Outcome"]).values, nan=0.0)
    y = df["Outcome"].values

    model = joblib.load(args.model)
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    print(f"[evaluate] source={args.source}")
    print(f"  accuracy={accuracy_score(y, preds):.4f}")
    print(f"  roc_auc={roc_auc_score(y, proba):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
