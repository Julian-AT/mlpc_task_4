from __future__ import annotations

import itertools
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from . import config
from .metrics import macro_ap, micro_ap, per_class_ap
from .train_lr import load_preprocessed

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ModuleNotFoundError:
    mx = None
    nn = None
    optim = None

HAS_MLX = mx is not None and nn is not None and optim is not None

try:
    import torch
    import torch.nn as torch_nn
except ModuleNotFoundError:
    torch = None
    torch_nn = None

HAS_TORCH_CUDA = torch is not None and torch_nn is not None and torch.cuda.is_available()

if HAS_MLX:

    class MLP(nn.Module):
        def __init__(
            self, in_dim: int, hidden_dims: Iterable[int], out_dim: int, dropout: float = 0.2
        ):
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

        def __call__(self, x: Any) -> Any:
            return self.net(x)

else:

    class MLP:
        def __init__(
            self, in_dim: int, hidden_dims: Iterable[int], out_dim: int, dropout: float = 0.2
        ):
            del dropout
            rng = np.random.default_rng(config.SEED)
            dims = [int(in_dim), *[int(dim) for dim in hidden_dims], int(out_dim)]
            self.weights = [
                rng.normal(0.0, 1.0 / np.sqrt(max(left, 1)), size=(left, right)).astype(np.float32)
                for left, right in zip(dims[:-1], dims[1:], strict=True)
            ]
            self.biases = [np.zeros(right, dtype=np.float32) for right in dims[1:]]

        def train(self) -> None:
            return None

        def eval(self) -> None:
            return None

        def __call__(self, x: np.ndarray) -> np.ndarray:
            output = np.asarray(x, dtype=np.float32)
            for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
                output = output @ weight + bias
                if index < len(self.weights) - 1:
                    output = np.maximum(output, 0.0)
            return output

        def save_weights(self, path: str) -> None:
            arrays = {f"w_{idx}": value for idx, value in enumerate(self.weights)}
            arrays.update({f"b_{idx}": value for idx, value in enumerate(self.biases)})
            np.savez_compressed(path, **arrays)


if torch_nn is not None:

    class TorchMLP(torch_nn.Module):
        def __init__(
            self, in_dim: int, hidden_dims: Iterable[int], out_dim: int, dropout: float = 0.2
        ):
            super().__init__()
            dims = [int(in_dim), *[int(dim) for dim in hidden_dims]]
            layers: list[torch_nn.Module] = []
            for left, right in zip(dims[:-1], dims[1:], strict=True):
                layers.append(torch_nn.Linear(left, right))
                layers.append(torch_nn.ReLU())
                if dropout > 0:
                    layers.append(torch_nn.Dropout(dropout))
            layers.append(torch_nn.Linear(dims[-1], int(out_dim)))
            self.net = torch_nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return self.net(x)

else:
    TorchMLP = None


def positive_class_weights(labels: np.ndarray, max_weight: float = 20.0) -> np.ndarray:
    y = np.asarray(labels, dtype=np.float32)
    positives = y.sum(axis=0)
    negatives = y.shape[0] - positives
    weights = negatives / np.maximum(positives, 1.0)
    return np.clip(weights, 1.0, max_weight).astype(np.float32)


def weighted_bce_with_logits(logits: mx.array, targets: mx.array, pos_weight: mx.array) -> mx.array:
    if not HAS_MLX:
        raise RuntimeError("weighted_bce_with_logits requires mlx")
    positive = targets * pos_weight
    weights = mx.where(targets > 0, positive, mx.ones_like(targets))
    loss = mx.maximum(logits, 0) - logits * targets + mx.log1p(mx.exp(-mx.abs(logits)))
    return mx.mean(loss * weights)


def _batch_indices(n_rows: int, batch_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    return [order[start : start + batch_size] for start in range(0, n_rows, batch_size)]


def _normalize_sklearn_scores(scores: Any) -> np.ndarray:
    if isinstance(scores, list):
        return np.column_stack(
            [score[:, 1] if score.ndim == 2 else score for score in scores]
        ).astype(np.float32)
    array = np.asarray(scores, dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 2:
        return array[:, :, 1].astype(np.float32)
    return array.astype(np.float32)


def _sigmoid(array: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-array))).astype(np.float32)


def predict_proba(model: Any, features: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    if isinstance(model, MLP):
        model.eval()
    if isinstance(model, MLPClassifier):
        return _normalize_sklearn_scores(model.predict_proba(features))
    if HAS_TORCH_CUDA and TorchMLP is not None and isinstance(model, TorchMLP):
        model.eval()
        device = next(model.parameters()).device
        x = np.asarray(features, dtype=np.float32)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                xb = torch.from_numpy(x[start : start + batch_size]).to(device)
                scores = torch.sigmoid(model(xb)).detach().cpu().numpy().astype(np.float32)
                chunks.append(scores)
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)

    x = np.asarray(features, dtype=np.float32)
    chunks: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size]
        if HAS_MLX:
            logits = model(mx.array(batch))
            chunks.append(np.asarray(mx.sigmoid(logits), dtype=np.float32))
        else:
            chunks.append(_sigmoid(np.asarray(model(batch), dtype=np.float32)))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)


def _train_one_mlx(
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
    if not HAS_MLX:
        raise RuntimeError("MLX backend requested but mlx is unavailable")

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
        history.append(
            {"epoch": float(epoch + 1), "loss": float(np.mean(losses)), "val_macro_ap": val_macro}
        )

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


def _train_one_torch(
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
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    if not HAS_TORCH_CUDA or TorchMLP is None:
        raise RuntimeError("PyTorch CUDA backend requested but CUDA is unavailable")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val_uint = np.asarray(y_val, dtype=np.uint8)

    model = TorchMLP(x_train.shape[1], hidden_dims, y_train.shape[1], dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    pos_weight = torch.from_numpy(positive_class_weights(y_train)).to(device)
    criterion = torch_nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history: list[dict[str, float]] = []
    best_macro = -np.inf
    best_scores: np.ndarray | None = None
    best_state: dict[str, Any] | None = None
    bad_epochs = 0
    start_time = time.time()

    for epoch in range(int(epochs)):
        model.train()
        losses: list[float] = []
        for indices in _batch_indices(len(x_train), int(batch_size), seed + epoch):
            xb = torch.from_numpy(x_train[indices]).to(device)
            yb = torch.from_numpy(y_train[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_scores = predict_proba(model, x_val, batch_size=max(int(batch_size), 4096))
        val_macro = macro_ap(y_val_uint, val_scores)
        history.append(
            {"epoch": float(epoch + 1), "loss": float(np.mean(losses)), "val_macro_ap": val_macro}
        )

        if val_macro > best_macro:
            best_macro = val_macro
            best_scores = val_scores
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            bad_epochs = 0
            if model_path is not None:
                save_torch_model(model, Path(model_path), hidden_dims, dropout)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    scores = best_scores if best_scores is not None else predict_proba(model, x_val)
    metrics = {
        "macro_ap": macro_ap(y_val_uint, scores),
        "micro_ap": micro_ap(y_val_uint, scores),
        "runtime_s": time.time() - start_time,
        "epochs": len(history),
        "history": history,
    }
    return model, scores, metrics


def _train_one_sklearn(
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
) -> tuple[MLPClassifier, np.ndarray, dict[str, Any]]:
    del dropout, patience
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.uint8)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.uint8)

    model = MLPClassifier(
        hidden_layer_sizes=tuple(int(dim) for dim in hidden_dims),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=min(int(batch_size), len(x_train)),
        learning_rate_init=float(lr),
        max_iter=int(epochs),
        random_state=int(seed),
        shuffle=True,
        early_stopping=False,
    )
    start_time = time.time()
    model.fit(x_train, y_train)
    scores = predict_proba(model, x_val)
    if model_path is not None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    metrics = {
        "macro_ap": macro_ap(y_val, scores),
        "micro_ap": micro_ap(y_val, scores),
        "runtime_s": time.time() - start_time,
        "epochs": int(getattr(model, "n_iter_", epochs)),
        "history": [],
    }
    return model, scores, metrics


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
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    if HAS_MLX:
        trainer = _train_one_mlx
    elif HAS_TORCH_CUDA:
        trainer = _train_one_torch
    else:
        trainer = _train_one_sklearn
    return trainer(
        x_train,
        y_train,
        x_val,
        y_val,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seed=seed,
        model_path=model_path,
    )


def iter_grid(grid: dict[str, Iterable[Any]] | None = None) -> list[dict[str, Any]]:
    source = grid or config.MLP_GRID
    return [
        {"hidden_dims": list(hidden_dims), "dropout": float(dropout), "lr": float(lr)}
        for hidden_dims, dropout, lr in itertools.product(
            source["hidden_dims"], source["dropout"], source["lr"]
        )
    ]


def backend_name() -> str:
    if HAS_MLX:
        return "mlx"
    if HAS_TORCH_CUDA:
        return "torch_cuda"
    return "sklearn"


def default_model_path() -> Path:
    if HAS_MLX:
        return config.MLP_BEST_MODEL
    if HAS_TORCH_CUDA:
        return config.RESULTS_DIR / "mlp_best_torch_cuda.pt"
    return config.RESULTS_DIR / "mlp_best_sklearn.joblib"


def save_torch_model(model: Any, path: Path, hidden_dims: Iterable[int], dropout: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dims": list(hidden_dims),
            "dropout": float(dropout),
            "backend": "torch_cuda",
        },
        path,
    )


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_MLX and isinstance(model, MLP):
        model.save_weights(str(path))
    elif HAS_TORCH_CUDA and TorchMLP is not None and isinstance(model, TorchMLP):
        torch.save({"state_dict": model.state_dict(), "backend": "torch_cuda"}, path)
    else:
        joblib.dump(model, path)


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
    x = np.asarray(
        dataset.get("features_context", dataset.get("features_scaled")), dtype=np.float32
    )
    y = np.asarray(dataset["labels"], dtype=np.uint8)
    class_names = dataset.get("class_names", np.asarray(config.CLASS_NAMES, dtype=object))
    train_idx = np.asarray(dataset["train_idx"], dtype=np.int64)
    val_idx = np.asarray(dataset["val_idx"], dtype=np.int64)
    test_idx = np.asarray(dataset["test_idx"], dtype=np.int64)

    rows: list[dict[str, Any]] = []
    best_macro = -np.inf
    best_model: Any | None = None
    best_val_scores: np.ndarray | None = None

    model_output = Path(model_path) if model_path is not None else default_model_path()
    grid_rows = iter_grid(grid)
    print(f"mlp backend={backend_name()}")
    for run_idx, params in enumerate(grid_rows, start=1):
        print(
            f"mlp {run_idx}/{len(grid_rows)} "
            f"hidden_dims={params['hidden_dims']} dropout={params['dropout']} lr={params['lr']}"
        )
        candidate_path = model_output if len(grid_rows) == 1 else None
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
        row.update(
            {f"ap_{name}": float(value) for name, value in zip(class_names, ap, strict=True)}
        )
        rows.append(row)
        if metrics["macro_ap"] > best_macro:
            best_macro = metrics["macro_ap"]
            best_model = model
            best_val_scores = val_scores
            save_model(model, model_output)

    frame = pd.DataFrame(rows).sort_values("macro_ap", ascending=False).reset_index(drop=True)
    output = Path(sweep_path) if sweep_path is not None else config.MLP_SWEEP_CSV
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    if best_model is not None and best_val_scores is not None:
        test_scores = predict_proba(best_model, x[test_idx])
        pred_output = (
            Path(predictions_path) if predictions_path is not None else config.PREDICTIONS_TEST
        )
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
