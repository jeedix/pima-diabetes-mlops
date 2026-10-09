"""Обучение MLP для ЛР3: полный датасет и потоковый режим."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dl_dataset import PimaDataset, PimaIterableDataset
from src.dl_model import make_mlp
from src.monitoring import measure_resources
from src.pipelines.metrics import bootstrap_ci


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Одна эпоха обучения. Возвращает средний train loss."""
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device):
    """Оценка модели. Возвращает (y_true, y_pred, y_proba)."""
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).cpu().numpy().ravel()
        logits_all.append(logits)
        y_all.append(y.numpy())
    logits_all = np.concatenate(logits_all)
    y_all = np.concatenate(y_all)
    proba = 1 / (1 + np.exp(-logits_all))
    preds = (proba >= 0.5).astype(int)
    return y_all, preds, proba


def train_dl(cfg, data: dict, device: str = "cpu") -> dict:
    """Полный режим: датасет в памяти. Обучение с early stopping."""
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    input_dim = data["X_train"].shape[1]
    if data.get("ind_train") is not None and cfg.architecture.use_missing_indicators:
        input_dim += data["ind_train"].shape[1]

    model = make_mlp(
        input_dim=input_dim,
        hidden_sizes=cfg.architecture.hidden_sizes,
        dropout=cfg.architecture.dropout,
        activation=cfg.architecture.activation,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    ind_train = (
        data.get("ind_train") if cfg.architecture.use_missing_indicators else None
    )
    ind_test = data.get("ind_test") if cfg.architecture.use_missing_indicators else None

    train_ds = PimaDataset(data["X_train"], data["y_train"], ind_train)
    test_ds = PimaDataset(data["X_test"], data["y_test"], ind_test)
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False)

    train_losses, val_losses = [], []
    best_val = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0

    with measure_resources() as res:
        for epoch in range(cfg.training.epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            train_losses.append(train_loss)

            y_val, _, proba_val = evaluate(model, test_loader, device)
            eps = 1e-9
            val_loss = float(
                -np.mean(
                    y_val * np.log(proba_val + eps)
                    + (1 - y_val) * np.log(1 - proba_val + eps)
                )
            )
            val_losses.append(val_loss)

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.training.early_stopping_patience:
                    print(
                        f"  Early stopping на эпохе {epoch + 1}, лучшая: {best_epoch}"
                    )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred, y_proba = evaluate(model, test_loader, device)
    metrics = bootstrap_ci(
        y_true, y_pred, y_proba, n_bootstrap=1000, seed=cfg.training.seed
    )

    return {
        "model": model,
        "metrics": metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "perf": res,
        "n_params": sum(p.numel() for p in model.parameters()),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def train_dl_streaming(
    cfg, data_test: dict, preprocessor, parquet_path: str, device: str = "cpu"
) -> dict:
    """Потоковый режим: DataLoader читает Parquet чанками с диска.

    Обучающие данные не материализуются — только батч за раз.
    Тест остаётся в памяти (для финальной оценки).
    """
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    input_dim = data_test["X_test"].shape[1] + data_test["ind_test"].shape[1]

    model = make_mlp(
        input_dim=input_dim,
        hidden_sizes=cfg.architecture.hidden_sizes,
        dropout=cfg.architecture.dropout,
        activation=cfg.architecture.activation,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    stream_ds = PimaIterableDataset(
        parquet_path=parquet_path,
        target=cfg.data.target,
        zero_cols=zero_cols,
        preprocessor=preprocessor,
        batch_size=cfg.data.batch_size,
        seed=cfg.training.seed,
    )
    train_loader = DataLoader(stream_ds, batch_size=None)

    test_ds = PimaDataset(
        data_test["X_test"], data_test["y_test"], data_test["ind_test"]
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False)

    train_losses, val_losses = [], []
    best_val = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0

    with measure_resources() as res:
        for epoch in range(cfg.training.epochs):
            model.train()
            losses = []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))
            train_losses.append(train_loss)

            y_val, _, proba_val = evaluate(model, test_loader, device)
            eps = 1e-9
            val_loss = float(
                -np.mean(
                    y_val * np.log(proba_val + eps)
                    + (1 - y_val) * np.log(1 - proba_val + eps)
                )
            )
            val_losses.append(val_loss)

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.training.early_stopping_patience:
                    print(
                        f"  Early stopping на эпохе {epoch + 1}, лучшая: {best_epoch}"
                    )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred, y_proba = evaluate(model, test_loader, device)
    metrics = bootstrap_ci(
        y_true, y_pred, y_proba, n_bootstrap=1000, seed=cfg.training.seed
    )

    return {
        "model": model,
        "metrics": metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "perf": res,
        "n_params": sum(p.numel() for p in model.parameters()),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
