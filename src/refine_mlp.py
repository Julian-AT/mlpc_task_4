"""Focused MLP refinement sweep used after the broad baseline sweep.

The broad, reusable MLP trainer lives in ``train_mlp.py``.  This module keeps the
small hand-selected refinement grid that was run for the submitted Task 4
results and promotes a candidate only when validation macro AP improves.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .final_eval import run_final_evaluation
from .metrics import per_class_ap
from .train_lr import load_preprocessed
from .train_mlp import predict_proba, train_one

CANDIDATES: list[dict[str, Any]] = [
    {"hidden_dims": [1024, 512], "dropout": 0.40, "lr": 1.0e-3, "seed": 42, "epochs": 45},
    {"hidden_dims": [1024, 512], "dropout": 0.35, "lr": 1.0e-3, "seed": 42, "epochs": 45},
    {"hidden_dims": [1024, 512], "dropout": 0.45, "lr": 1.0e-3, "seed": 42, "epochs": 45},
    {"hidden_dims": [1024, 512], "dropout": 0.50, "lr": 1.0e-3, "seed": 42, "epochs": 45},
    {"hidden_dims": [1024, 512], "dropout": 0.40, "lr": 7.0e-4, "seed": 42, "epochs": 45},
    {"hidden_dims": [1024, 512], "dropout": 0.40, "lr": 1.3e-3, "seed": 42, "epochs": 45},
    {"hidden_dims": [512, 256], "dropout": 0.40, "lr": 1.0e-3, "seed": 42, "epochs": 45},
    {"hidden_dims": [512, 256], "dropout": 0.45, "lr": 1.0e-3, "seed": 42, "epochs": 45},
    {"hidden_dims": [1024, 512], "dropout": 0.40, "lr": 1.0e-3, "seed": 7, "epochs": 45},
    {"hidden_dims": [1024, 512], "dropout": 0.40, "lr": 1.0e-3, "seed": 123, "epochs": 45},
    {"hidden_dims": [1536, 768], "dropout": 0.40, "lr": 1.0e-3, "seed": 42, "epochs": 35},
    {"hidden_dims": [1024, 512, 256], "dropout": 0.40, "lr": 1.0e-3, "seed": 42, "epochs": 35},
]


def _hidden_name(hidden_dims: list[int]) -> str:
    return "-".join(str(dim) for dim in hidden_dims)


def _incumbent_macro(path: Path = config.MLP_SWEEP_CSV) -> float:
    if not path.exists():
        return -np.inf
    frame = pd.read_csv(path)
    if frame.empty or "macro_ap" not in frame:
        return -np.inf
    return float(frame["macro_ap"].max())


def _row(
    params: dict[str, Any],
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: np.ndarray,
) -> dict[str, Any]:
    row = {
        "hidden_dims": _hidden_name(params["hidden_dims"]),
        "dropout": float(params["dropout"]),
        "lr": float(params["lr"]),
        "seed": int(params["seed"]),
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


def _merge_sweeps(refine_frame: pd.DataFrame) -> None:
    if config.MLP_SWEEP_CSV.exists():
        existing = pd.read_csv(config.MLP_SWEEP_CSV)
        combined = pd.concat([existing, refine_frame], ignore_index=True, sort=False)
    else:
        combined = refine_frame
    combined = combined.sort_values("macro_ap", ascending=False).reset_index(drop=True)
    combined.to_csv(config.MLP_SWEEP_CSV, index=False)


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
            "mlp_val_scores": val_scores,
            "mlp_test_scores": test_scores,
            "y_val": y_val,
            "y_test": y_test,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "class_names": np.asarray(class_names, dtype=object),
        }
    )
    np.savez_compressed(config.PREDICTIONS_TEST, **existing)


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        return


def refine_mlp() -> pd.DataFrame:
    dataset = load_preprocessed()
    x = np.asarray(dataset.get("features_context", dataset.get("features_scaled")), dtype=np.float32)
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
    candidate_dir = config.RESULTS_DIR / "mlp_refine_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    print(f"incumbent_val_macro={incumbent:.6f}")
    start = time.time()
    for run_idx, params in enumerate(CANDIDATES, start=1):
        candidate_path = candidate_dir / (
            f"mlp_{run_idx:02d}_{_hidden_name(params['hidden_dims'])}"
            f"_d{params['dropout']}_lr{params['lr']}_s{params['seed']}.pt"
        )
        print(
            f"refine {run_idx}/{len(CANDIDATES)} "
            f"hidden_dims={params['hidden_dims']} dropout={params['dropout']} "
            f"lr={params['lr']} seed={params['seed']} epochs={params['epochs']}",
            flush=True,
        )
        model, val_scores, metrics = train_one(
            x[train_idx],
            y[train_idx],
            x[val_idx],
            y[val_idx],
            hidden_dims=params["hidden_dims"],
            dropout=params["dropout"],
            lr=params["lr"],
            epochs=params["epochs"],
            batch_size=4096,
            patience=8,
            seed=params["seed"],
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
            test_scores = predict_proba(model, x[test_idx], batch_size=4096)
            best_macro = float(metrics["macro_ap"])
            best_val_scores = val_scores
            best_test_scores = test_scores
            best_path = candidate_path
            shutil.copyfile(candidate_path, config.RESULTS_DIR / "mlp_best_torch_cuda.pt")
            print(f"  promoted new best val_macro={best_macro:.6f}", flush=True)
        del model
        _clear_cuda_cache()

    refine_frame = pd.DataFrame(rows).sort_values("macro_ap", ascending=False).reset_index(drop=True)
    refine_path = config.RESULTS_DIR / "mlp_refine_sweep.csv"
    refine_frame.to_csv(refine_path, index=False)
    _merge_sweeps(refine_frame)

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
        print("\nFinal table after promotion:")
        print(final_table[["model", "macro_ap", "micro_ap"]].to_string(index=False))
        print(f"Promoted model: {best_path}")
    else:
        print("No candidate beat incumbent; predictions_test.npz was left unchanged.")
    print(f"total_runtime_s={time.time() - start:.1f}")
    print("\nRefine top 8:")
    print(
        refine_frame[
            ["hidden_dims", "dropout", "lr", "seed", "macro_ap", "micro_ap", "epochs", "runtime_s"]
        ]
        .head(8)
        .to_string(index=False)
    )
    return refine_frame


if __name__ == "__main__":
    refine_mlp()
