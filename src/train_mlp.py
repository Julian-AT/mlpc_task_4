from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd

from . import config
from .metrics import macro_ap, micro_ap, per_class_ap
from .train_lr import load_preprocessed


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Iterable[int], out_dim: int, dropout: float = 0.2):
        super().__init__()
        dims = [int(in_dim), *[int(dim) for dim in hidden_dims]]
        layers: list[nn.Module] = []
        for left, right in zip(dims[:-1], dims[1:], strict=True):
            layers.append(nn.Linear(left, right))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], int(out_dim)))
        self.net = nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


def positive_class_weights(labels: np.ndarray, max_weight: float = 20.0) -> np.ndarray:
    y = np.asarray(labels, dtype=np.float32)
    positives = y.sum(axis=0)
    negatives = y.shape[0] - positives
    weights = negatives / np.maximum(positives, 1.0)
    return np.clip(weights, 1.0, max_weight).astype(np.float32)


def weighted_bce_with_logits(logits: mx.array, targets: mx.array, pos_weight: mx.array) -> mx.array:
    positive = targets * pos_weight
    weights = mx.where(targets > 0, positive, mx.ones_like(targets))
    loss = mx.maximum(logits, 0) - logits * targets + mx.log1p(mx.exp(-mx.abs(logits)))
    return mx.mean(loss * weights)


def _batch_indices(n_rows: int, batch_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    return [order[start : start + batch_size] for start in range(0, n_rows, batch_size)]


def predict_proba(model: MLP, features: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    x = np.asarray(features, dtype=np.float32)
    chunks: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        logits = model(mx.array(x[start : start + batch_size]))
        chunks.append(np.asarray(mx.sigmoid(logits), dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)


def train_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: Iterable[int],
    dropout: float,
    lr: float,
    epochs: int = config.MLP_EPOCHS,
    batch_size: int = config.MLP_BATCH,
    patience: int = config.MLP_PATIENCE,
    seed: int = config.SEED,
    model_path: Path | str | None = None,
) -> tuple[MLP, np.ndarray, dict[str, Any]]:
    mx.random.seed(seed)
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.uint8)

    model = MLP(x_train.shape[1], hidden_dims, y_train.shape[1], dropout=dropout)
    optimizer = optim.AdamW(learning_rate=lr)
    pos_weight = mx.array(positive_class_weights(y_train))

    def loss_fn(model: MLP, xb: mx.array, yb: mx.array) -> mx.array:
        return weighted_bce_with_logits(model(xb), yb, pos_weight)

    value_and_grad = nn.value_and_grad(model, loss_fn)
    history: list[dict[str, float]] = []
    best_macro = -np.inf
    best_scores: np.ndarray | None = None
    bad_epochs = 0
    start_time = time.time()

    for epoch in range(int(epochs)):
        model.train()
        losses: list[float] = []
        for indices in _batch_indices(len(x_train), int(batch_size), seed + epoch):
            xb = mx.array(x_train[indices])
            yb = mx.array(y_train[indices])
            loss, grads = value_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(float(loss))

        val_scores = predict_proba(model, x_val)
        val_macro = macro_ap(y_val, val_scores)
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses)), "val_macro_ap": val_macro})

        if val_macro > best_macro:
            best_macro = val_macro
            best_scores = val_scores
            bad_epochs = 0
            if model_path is not None:
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                model.save_weights(str(model_path))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    scores = best_scores if best_scores is not None else predict_proba(model, x_val)
    metrics = {
        "macro_ap": macro_ap(y_val, scores),
        "micro_ap": micro_ap(y_val, scores),
        "runtime_s": time.time() - start_time,
        "epochs": len(history),
        "history": history,
    }
    return model, scores, metrics


def iter_grid(grid: dict[str, Iterable[Any]] | None = None) -> list[dict[str, Any]]:
    source = grid or config.MLP_GRID
    return [
        {"hidden_dims": list(hidden_dims), "dropout": float(dropout), "lr": float(lr)}
        for hidden_dims, dropout, lr in itertools.product(
            source["hidden_dims"], source["dropout"], source["lr"]
        )
    ]


def sweep_mlp(
    data: dict[str, np.ndarray] | None = None,
    grid: dict[str, Iterable[Any]] | None = None,
    sweep_path: Path | str | None = None,
    model_path: Path | str | None = None,
    predictions_path: Path | str | None = None,
    epochs: int = config.MLP_EPOCHS,
    batch_size: int = config.MLP_BATCH,
    patience: int = config.MLP_PATIENCE,
) -> pd.DataFrame:
    dataset = data if data is not None else load_preprocessed()
    x = np.asarray(dataset.get("features_context", dataset.get("features_scaled")), dtype=np.float32)
    y = np.asarray(dataset["labels"], dtype=np.uint8)
    class_names = dataset.get("class_names", np.asarray(config.CLASS_NAMES, dtype=object))
    train_idx = np.asarray(dataset["train_idx"], dtype=np.int64)
    val_idx = np.asarray(dataset["val_idx"], dtype=np.int64)
    test_idx = np.asarray(dataset["test_idx"], dtype=np.int64)

    rows: list[dict[str, Any]] = []
    best_macro = -np.inf
    best_model: MLP | None = None
    best_val_scores: np.ndarray | None = None

    weight_path = Path(model_path) if model_path is not None else config.MLP_BEST_MODEL
    for params in iter_grid(grid):
        candidate_path = weight_path if len(iter_grid(grid)) == 1 else None
        model, val_scores, metrics = train_one(
            x[train_idx],
            y[train_idx],
            x[val_idx],
            y[val_idx],
            hidden_dims=params["hidden_dims"],
            dropout=params["dropout"],
            lr=params["lr"],
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            model_path=candidate_path,
        )
        ap = per_class_ap(y[val_idx], val_scores)
        row = {
            "hidden_dims": "-".join(str(dim) for dim in params["hidden_dims"]),
            "dropout": params["dropout"],
            "lr": params["lr"],
            "macro_ap": metrics["macro_ap"],
            "micro_ap": metrics["micro_ap"],
            "runtime_s": metrics["runtime_s"],
            "epochs": metrics["epochs"],
        }
        row.update({f"ap_{name}": float(value) for name, value in zip(class_names, ap, strict=True)})
        rows.append(row)
        if metrics["macro_ap"] > best_macro:
            best_macro = metrics["macro_ap"]
            best_model = model
            best_val_scores = val_scores
            weight_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_weights(str(weight_path))

    frame = pd.DataFrame(rows).sort_values("macro_ap", ascending=False).reset_index(drop=True)
    output = Path(sweep_path) if sweep_path is not None else config.MLP_SWEEP_CSV
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    if best_model is not None and best_val_scores is not None:
        test_scores = predict_proba(best_model, x[test_idx])
        pred_output = Path(predictions_path) if predictions_path is not None else config.PREDICTIONS_TEST
        existing: dict[str, np.ndarray] = {}
        if pred_output.exists():
            with np.load(pred_output, allow_pickle=True) as loaded:
                existing = {key: loaded[key] for key in loaded.files}
        existing.update(
            {
                "mlp_val_scores": best_val_scores,
                "mlp_test_scores": test_scores,
                "y_val": y[val_idx],
                "y_test": y[test_idx],
                "val_idx": val_idx,
                "test_idx": test_idx,
                "class_names": np.asarray(class_names, dtype=object),
            }
        )
        pred_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_output, **existing)

    return frame


if __name__ == "__main__":
    sweep_mlp()
