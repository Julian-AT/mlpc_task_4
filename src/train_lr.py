from __future__ import annotations

import itertools
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from . import config
from .metrics import macro_ap, micro_ap, per_class_ap


def load_preprocessed(path: Path | str | None = None) -> dict[str, np.ndarray]:
    source = Path(path) if path is not None else config.PREPROCESSED_CACHE
    with np.load(source, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _predict_proba(model: OneVsRestClassifier, x: np.ndarray) -> np.ndarray:
    scores = model.predict_proba(x)
    if isinstance(scores, list):
        scores = np.column_stack([score[:, 1] for score in scores])
    return np.asarray(scores, dtype=np.float32)


def fit_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    C: float,
    penalty: str,
    class_weight: str | None,
    max_iter: int = 2000,
    n_jobs: int = 1,
    parallel_backend_name: str | None = None,
) -> tuple[OneVsRestClassifier, np.ndarray, dict[str, float]]:
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    base = LogisticRegression(
        C=C,
        penalty=penalty,
        class_weight=class_weight,
        solver=solver,
        max_iter=max_iter,
    )
    model = OneVsRestClassifier(base, n_jobs=n_jobs)
    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="'penalty' was deprecated.*", category=FutureWarning
        )
        warnings.filterwarnings(
            "ignore", message="Inconsistent values: penalty=.*", category=UserWarning
        )
        if parallel_backend_name is None:
            model.fit(x_train, y_train)
        else:
            with parallel_backend(parallel_backend_name):
                model.fit(x_train, y_train)
    val_scores = _predict_proba(model, x_val)
    runtime = time.time() - start
    return (
        model,
        val_scores,
        {
            "macro_ap": macro_ap(y_val, val_scores),
            "micro_ap": micro_ap(y_val, val_scores),
            "runtime_s": runtime,
        },
    )


def iter_grid(grid: dict[str, Iterable[Any]] | None = None) -> list[dict[str, Any]]:
    source = grid or config.LR_GRID
    return [
        {"C": float(C), "penalty": str(penalty), "class_weight": class_weight}
        for C, penalty, class_weight in itertools.product(
            source["C"], source["penalty"], source["class_weight"]
        )
    ]


def _row(
    params: dict[str, Any],
    metrics: dict[str, float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str] | np.ndarray,
) -> dict[str, Any]:
    ap = per_class_ap(y_true, y_score)
    row = {
        "C": params["C"],
        "penalty": params["penalty"],
        "class_weight": "None" if params["class_weight"] is None else str(params["class_weight"]),
        "macro_ap": metrics["macro_ap"],
        "micro_ap": metrics["micro_ap"],
        "runtime_s": metrics["runtime_s"],
    }
    row.update({f"ap_{name}": float(value) for name, value in zip(class_names, ap, strict=True)})
    return row


def sweep_lr(
    data: dict[str, np.ndarray] | None = None,
    grid: dict[str, Iterable[Any]] | None = None,
    sweep_path: Path | str | None = None,
    model_path: Path | str | None = None,
    predictions_path: Path | str | None = None,
    max_iter: int = 2000,
    n_jobs: int = 1,
    parallel_backend_name: str | None = None,
) -> pd.DataFrame:
    dataset = data if data is not None else load_preprocessed()
    x = np.asarray(dataset.get("features_scaled", dataset.get("features")), dtype=np.float32)
    y = np.asarray(dataset["labels"], dtype=np.uint8)
    class_names = dataset.get("class_names", np.asarray(config.CLASS_NAMES, dtype=object))
    train_idx = np.asarray(dataset["train_idx"], dtype=np.int64)
    val_idx = np.asarray(dataset["val_idx"], dtype=np.int64)
    test_idx = np.asarray(dataset["test_idx"], dtype=np.int64)

    rows: list[dict[str, Any]] = []
    best_model: OneVsRestClassifier | None = None
    best_score = -np.inf
    best_val_scores: np.ndarray | None = None
    best_test_scores: np.ndarray | None = None

    grid_rows = iter_grid(grid)
    for run_idx, params in enumerate(grid_rows, start=1):
        print(
            f"lr {run_idx}/{len(grid_rows)} "
            f"C={params['C']} penalty={params['penalty']} class_weight={params['class_weight']}"
        )
        model, val_scores, metrics = fit_one(
            x[train_idx],
            y[train_idx],
            x[val_idx],
            y[val_idx],
            C=params["C"],
            penalty=params["penalty"],
            class_weight=params["class_weight"],
            max_iter=max_iter,
            n_jobs=n_jobs,
            parallel_backend_name=parallel_backend_name,
        )
        rows.append(_row(params, metrics, y[val_idx], val_scores, class_names))
        if metrics["macro_ap"] > best_score:
            best_score = metrics["macro_ap"]
            best_model = model
            best_val_scores = val_scores
            best_test_scores = _predict_proba(model, x[test_idx])

    frame = pd.DataFrame(rows).sort_values("macro_ap", ascending=False).reset_index(drop=True)
    output = Path(sweep_path) if sweep_path is not None else config.LR_SWEEP_CSV
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    if best_model is not None:
        model_output = Path(model_path) if model_path is not None else config.LR_BEST_MODEL
        model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_output)

    if best_val_scores is not None and best_test_scores is not None:
        pred_output = (
            Path(predictions_path) if predictions_path is not None else config.PREDICTIONS_TEST
        )
        pred_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pred_output,
            lr_val_scores=best_val_scores,
            lr_test_scores=best_test_scores,
            y_val=y[val_idx],
            y_test=y[test_idx],
            val_idx=val_idx,
            test_idx=test_idx,
            class_names=np.asarray(class_names, dtype=object),
        )

    return frame


def plot_lr_sweep(
    sweep_csv: Path | str | None = None,
    output_path: Path | str | None = None,
) -> None:
    source = Path(sweep_csv) if sweep_csv is not None else config.LR_SWEEP_CSV
    output = (
        Path(output_path) if output_path is not None else config.FIG_DIR / "lr_sweep_heatmap.png"
    )
    frame = pd.read_csv(source)
    pivot = frame.pivot_table(index="C", columns="penalty", values="macro_ap", aggfunc="mean")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), [str(value) for value in pivot.index])
    ax.set_xlabel("Penalty")
    ax.set_ylabel("C")
    ax.set_title("LR validation macro AP")
    fig.colorbar(image, ax=ax, label="macro AP")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    sweep_lr()
    plot_lr_sweep()


if __name__ == "__main__":
    main()
