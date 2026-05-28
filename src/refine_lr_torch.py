from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .final_eval import run_final_evaluation
from .metrics import macro_ap, micro_ap, per_class_ap
from .train_lr import load_preprocessed
from .train_mlp import positive_class_weights

try:
    import torch
    import torch.nn as torch_nn
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch installed
    torch = None
    torch_nn = None


CANDIDATES: list[dict[str, Any]] = [
    {"feature_key": "features_context", "lr": 3.0e-3, "weight_decay": 1.0e-4, "pos_weight_max": 20.0},
    {"feature_key": "features_context", "lr": 1.0e-3, "weight_decay": 1.0e-4, "pos_weight_max": 20.0},
    {"feature_key": "features_context", "lr": 7.0e-4, "weight_decay": 1.0e-4, "pos_weight_max": 20.0},
    {"feature_key": "features_context", "lr": 3.0e-3, "weight_decay": 1.0e-3, "pos_weight_max": 20.0},
    {"feature_key": "features_context", "lr": 1.0e-3, "weight_decay": 1.0e-3, "pos_weight_max": 20.0},
    {"feature_key": "features_context", "lr": 3.0e-3, "weight_decay": 1.0e-2, "pos_weight_max": 20.0},
    {"feature_key": "features_context", "lr": 1.0e-3, "weight_decay": 1.0e-2, "pos_weight_max": 20.0},
    {"feature_key": "features_context", "lr": 3.0e-3, "weight_decay": 1.0e-4, "pos_weight_max": 10.0},
    {"feature_key": "features_context", "lr": 1.0e-3, "weight_decay": 1.0e-4, "pos_weight_max": 10.0},
    {"feature_key": "features_context", "lr": 3.0e-3, "weight_decay": 1.0e-4, "pos_weight_max": None},
    {"feature_key": "features_context", "lr": 6.0e-3, "weight_decay": 1.0e-4, "pos_weight_max": 20.0},
    {"feature_key": "features_context", "lr": 1.3e-3, "weight_decay": 1.0e-4, "pos_weight_max": 20.0},
    {"feature_key": "features_context", "lr": 3.0e-4, "weight_decay": 1.0e-4, "pos_weight_max": 20.0},
    {"feature_key": "features_scaled", "lr": 3.0e-3, "weight_decay": 1.0e-4, "pos_weight_max": 20.0},
    {"feature_key": "features_scaled", "lr": 1.0e-3, "weight_decay": 1.0e-4, "pos_weight_max": 20.0},
    {"feature_key": "features_scaled", "lr": 3.0e-3, "weight_decay": 1.0e-4, "pos_weight_max": None},
]


class TorchLogisticRegression(torch_nn.Module if torch_nn is not None else object):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = torch_nn.Linear(int(in_dim), int(out_dim))

    def forward(self, x: Any) -> Any:
        return self.linear(x)


def _incumbent_macro(path: Path = config.LR_SWEEP_CSV) -> float:
    if not path.exists():
        return -np.inf
    frame = pd.read_csv(path)
    if frame.empty or "macro_ap" not in frame:
        return -np.inf
    return float(frame["macro_ap"].max())


def _predict_proba(model: TorchLogisticRegression, features: np.ndarray, batch_size: int = 16384) -> np.ndarray:
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


def _criterion(y_train: np.ndarray, pos_weight_max: float | None) -> Any:
    if pos_weight_max is None:
        return torch_nn.BCEWithLogitsLoss()
    weights = positive_class_weights(y_train, max_weight=float(pos_weight_max))
    return torch_nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights).to("cuda"))


def _batch_indices(n_rows: int, batch_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    return [order[start : start + batch_size] for start in range(0, n_rows, batch_size)]


def _train_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    model_path: Path,
    epochs: int = 80,
    batch_size: int = 8192,
    patience: int = 8,
    seed: int = config.SEED,
) -> tuple[TorchLogisticRegression, np.ndarray, dict[str, Any]]:
    if torch is None or torch_nn is None or not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is required for refine_lr_torch")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val_uint = np.asarray(y_val, dtype=np.uint8)

    model = TorchLogisticRegression(x_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    criterion = _criterion(y_train, params["pos_weight_max"])

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

        val_scores = _predict_proba(model, x_val, batch_size=max(int(batch_size), 16384))
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
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_key": params["feature_key"],
                    "lr": float(params["lr"]),
                    "weight_decay": float(params["weight_decay"]),
                    "pos_weight_max": params["pos_weight_max"],
                    "backend": "torch_cuda",
                },
                model_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = best_scores if best_scores is not None else _predict_proba(model, x_val)
    metrics = {
        "macro_ap": macro_ap(y_val_uint, scores),
        "micro_ap": micro_ap(y_val_uint, scores),
        "runtime_s": time.time() - start_time,
        "epochs": len(history),
        "history": history,
    }
    return model, scores, metrics


def _row(
    params: dict[str, Any],
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: np.ndarray,
) -> dict[str, Any]:
    row = {
        "backend": "torch_cuda_linear",
        "feature_key": params["feature_key"],
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
        "pos_weight_max": "None" if params["pos_weight_max"] is None else float(params["pos_weight_max"]),
        "macro_ap": float(metrics["macro_ap"]),
        "micro_ap": float(metrics["micro_ap"]),
        "runtime_s": float(metrics["runtime_s"]),
        "epochs": int(metrics["epochs"]),
    }
    row.update(
        {
            f"ap_{name}": float(value)
            for name, value in zip(class_names, per_class_ap(y_true, y_score), strict=True)
        }
    )
    return row


def _update_predictions(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    class_names: np.ndarray,
) -> None:
    existing: dict[str, np.ndarray] = {}
    if config.PREDICTIONS_TEST.exists():
        with np.load(config.PREDICTIONS_TEST, allow_pickle=True) as loaded:
            existing = {key: loaded[key] for key in loaded.files}
    existing.update(
        {
            "lr_val_scores": val_scores,
            "lr_test_scores": test_scores,
            "y_val": y_val,
            "y_test": y_test,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "class_names": np.asarray(class_names, dtype=object),
        }
    )
    np.savez_compressed(config.PREDICTIONS_TEST, **existing)


def _clear_cuda_cache() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def refine_lr_torch() -> pd.DataFrame:
    dataset = load_preprocessed()
    y = np.asarray(dataset["labels"], dtype=np.uint8)
    class_names = np.asarray(dataset.get("class_names", np.asarray(config.CLASS_NAMES, dtype=object)))
    train_idx = np.asarray(dataset["train_idx"], dtype=np.int64)
    val_idx = np.asarray(dataset["val_idx"], dtype=np.int64)
    test_idx = np.asarray(dataset["test_idx"], dtype=np.int64)

    incumbent = _incumbent_macro()
    best_macro = incumbent
    best_val_scores: np.ndarray | None = None
    best_test_scores: np.ndarray | None = None
    best_path: Path | None = None
    rows: list[dict[str, Any]] = []
    candidate_dir = config.RESULTS_DIR / "lr_refine_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print(f"incumbent_lr_val_macro={incumbent:.6f}")
    start = time.time()
    for run_idx, params in enumerate(CANDIDATES, start=1):
        feature_key = str(params["feature_key"])
        x = np.asarray(dataset[feature_key], dtype=np.float32)
        candidate_path = candidate_dir / (
            f"lr_{run_idx:02d}_{feature_key}_lr{params['lr']}"
            f"_wd{params['weight_decay']}_pw{params['pos_weight_max']}.pt"
        )
        print(
            f"lr_refine {run_idx}/{len(CANDIDATES)} feature_key={feature_key} "
            f"lr={params['lr']} weight_decay={params['weight_decay']} "
            f"pos_weight_max={params['pos_weight_max']}",
            flush=True,
        )
        model, val_scores, metrics = _train_one(
            x[train_idx],
            y[train_idx],
            x[val_idx],
            y[val_idx],
            params=params,
            model_path=candidate_path,
        )
        rows.append(_row(params, metrics, y[val_idx], val_scores, class_names))
        print(
            f"  val_macro={metrics['macro_ap']:.6f} "
            f"val_micro={metrics['micro_ap']:.6f} epochs={metrics['epochs']} "
            f"runtime_s={metrics['runtime_s']:.1f}",
            flush=True,
        )
        if metrics["macro_ap"] > best_macro:
            test_scores = _predict_proba(model, x[test_idx])
            best_macro = float(metrics["macro_ap"])
            best_val_scores = val_scores
            best_test_scores = test_scores
            best_path = candidate_path
            shutil.copyfile(candidate_path, config.RESULTS_DIR / "lr_best_torch_cuda.pt")
            print(f"  promoted new LR best val_macro={best_macro:.6f}", flush=True)
        del model
        _clear_cuda_cache()

    refine_frame = pd.DataFrame(rows).sort_values("macro_ap", ascending=False).reset_index(drop=True)
    refine_path = config.RESULTS_DIR / "lr_torch_sweep.csv"
    refine_frame.to_csv(refine_path, index=False)

    if best_val_scores is not None and best_test_scores is not None:
        _update_predictions(
            best_val_scores,
            best_test_scores,
            y[val_idx],
            y[test_idx],
            val_idx,
            test_idx,
            class_names,
        )
        final_table = run_final_evaluation()
        print("\nFinal table after LR promotion:")
        print(final_table[["model", "macro_ap", "micro_ap"]].to_string(index=False))
        print(f"Promoted LR model: {best_path}")
    else:
        print("No LR candidate beat incumbent; predictions_test.npz was left unchanged.")
    print(f"total_runtime_s={time.time() - start:.1f}")
    print("\nLR refine top 8:")
    print(
        refine_frame[
            [
                "feature_key",
                "lr",
                "weight_decay",
                "pos_weight_max",
                "macro_ap",
                "micro_ap",
                "epochs",
                "runtime_s",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )
    return refine_frame


if __name__ == "__main__":
    refine_lr_torch()
