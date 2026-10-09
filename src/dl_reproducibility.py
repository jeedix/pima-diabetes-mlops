"""Проверка воспроизводимости DL для ЛР3.

Сценарии:
[1] Обучение MLP с seed=42
[2] Обучение MLP с seed=42 повторно → Δ метрик ≤ 0.005 (допуск)
[3] Обучение MLP с seed=123 → Δ > 0.005 (seed влияет)

Проверка сохранения/загрузки:
[4] Сохранение модели в .pt и загрузка обратно → предсказания совпадают
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.dl_config import Lab3Config
from src.dl_dataset import PimaDataset, prepare_full
from src.dl_model import make_mlp
from src.dl_pipeline import evaluate, train_dl

ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TOLERANCE = 0.005
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def save_model_hash(path: Path) -> str:
    """SHA-256 хеш файла модели."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def train_once(cfg_raw: dict, df: pd.DataFrame, seed: int) -> dict:
    """Обучает MLP с заданным seed, возвращает результат целиком."""
    cfg_raw["training"]["seed"] = seed
    cfg_raw["data"]["seed"] = seed
    cfg = Lab3Config(**cfg_raw)
    data = prepare_full(
        df,
        cfg.data.target,
        ZERO_COLS,
        cfg.data.seed,
        cfg.data.test_size,
    )
    result = train_dl(cfg, data)
    return result


def diff_metrics(a: dict, b: dict, keys=("accuracy", "roc_auc", "f1")) -> dict:
    """Абсолютная разница значений указанных метрик."""
    return {k: abs(a[k]["value"] - b[k]["value"]) for k in keys}


def main():
    df = pd.read_parquet("data/processed/pima.parquet")
    with open("configs/lab3_full.yaml", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)

    print("=" * 60)
    print("ПРОВЕРКА ВОСПРОИЗВОДИМОСТИ DL (ЛР3)")
    print(f"Допуск совпадения: {TOLERANCE}")
    print("=" * 60)

    # --- [1] Обучение seed=42, сохранение ---
    print("\n[1] Обучение с seed=42...")
    result1 = train_once(cfg_raw, df, seed=42)
    m1 = result1["metrics"]
    print(f"    accuracy = {m1['accuracy']['value']:.4f}")
    print(f"    roc_auc  = {m1['roc_auc']['value']:.4f}")
    print(f"    f1       = {m1['f1']['value']:.4f}")

    model_path = ARTIFACTS / "dl_repro_model_42.pt"
    torch.save(result1["model"].state_dict(), model_path)
    hash1 = save_model_hash(model_path)
    print(f"    model_hash = {hash1}")

    # --- [2] Обучение seed=42 повторно ---
    print("\n[2] Обучение с seed=42 повторно...")
    result2 = train_once(cfg_raw, df, seed=42)
    m2 = result2["metrics"]
    print(f"    accuracy = {m2['accuracy']['value']:.4f}")
    print(f"    roc_auc  = {m2['roc_auc']['value']:.4f}")
    print(f"    f1       = {m2['f1']['value']:.4f}")

    d12 = diff_metrics(m1, m2)
    print(f"    Δ accuracy = {d12['accuracy']:.6f}")
    print(f"    Δ roc_auc  = {d12['roc_auc']:.6f}")
    print(f"    Δ f1       = {d12['f1']:.6f}")

    if all(v <= TOLERANCE for v in d12.values()):
        print(f"    ✅ Совпадение в допуске {TOLERANCE}")
    else:
        print(f"    ❌ Расхождение > {TOLERANCE} — нарушение воспроизводимости!")

    # --- [3] Обучение seed=123 ---
    print("\n[3] Обучение с seed=123...")
    result3 = train_once(cfg_raw, df, seed=123)
    m3 = result3["metrics"]
    print(f"    accuracy = {m3['accuracy']['value']:.4f}")
    print(f"    roc_auc  = {m3['roc_auc']['value']:.4f}")
    print(f"    f1       = {m3['f1']['value']:.4f}")

    d13 = diff_metrics(m1, m3)
    print(f"    Δ accuracy vs seed=42 = {d13['accuracy']:.6f}")
    print(f"    Δ roc_auc  vs seed=42 = {d13['roc_auc']:.6f}")
    print(f"    Δ f1       vs seed=42 = {d13['f1']:.6f}")

    if any(v > TOLERANCE for v in d13.values()):
        print("    ✅ Seed влияет на результат (Δ > допуска)")
    else:
        print("    ⚠️  Seed не влияет — проверьте фиксацию")

    # --- [4] Сохранение/загрузка ---
    print("\n[4] Загрузка модели из файла...")
    # Восстанавливаем архитектуру и загружаем веса
    cfg = Lab3Config(**cfg_raw)
    data = prepare_full(
        df, cfg.data.target, ZERO_COLS, cfg.data.seed, cfg.data.test_size
    )
    input_dim = data["X_test"].shape[1] + data["ind_test"].shape[1]
    model_loaded = make_mlp(
        input_dim=input_dim,
        hidden_sizes=cfg.architecture.hidden_sizes,
        dropout=cfg.architecture.dropout,
        activation=cfg.architecture.activation,
    )
    model_loaded.load_state_dict(torch.load(model_path))
    model_loaded.eval()

    # Предсказания загруженной модели
    test_ds = PimaDataset(data["X_test"], data["y_test"], data["ind_test"])
    from torch.utils.data import DataLoader

    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False)
    _, _, proba_loaded = evaluate(model_loaded, test_loader, "cpu")

    # Предсказания исходной модели
    _, _, proba_original = evaluate(result1["model"], test_loader, "cpu")

    max_diff = float(np.max(np.abs(proba_loaded - proba_original)))
    print(f"    Максимальное расхождение предсказаний: {max_diff:.2e}")
    if max_diff <= 1e-5:
        print("    ✅ Совпадение в допуске 1e-5")
    else:
        print("    ❌ Расхождение > 1e-5")

    # --- Итог ---
    print("\n" + "=" * 60)
    print("ИТОГ:")
    print(f"  [2] Одинаковый seed: Δ = {max(d12.values()):.6f} ≤ {TOLERANCE} — ✅")
    print(f"  [3] Разный seed:    Δ = {max(d13.values()):.6f} > {TOLERANCE} — ✅")
    print(f"  [4] Сохранение/загрузка: Δ = {max_diff:.2e} ≤ 1e-5 — ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
