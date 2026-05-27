from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .metrics import macro_ap, micro_ap, per_class_ap, per_class_f1_at_optimal
from .splits import load_dataset_cache


def class_prior_baseline_scores(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    y = np.asarray(y_train, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError("y_train must have shape [N, C]")
    prevalence = y.mean(axis=0)
    return np.tile(prevalence, (int(n_rows), 1)).astype(np.float32, copy=False)


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray, class_names: list[str] | np.ndarray) -> dict[str, Any]:
    ap = per_class_ap(y_true, y_score)
    thresholds, f1s = per_class_f1_at_optimal(y_true, y_score)
    return {
        "macro_ap": macro_ap(y_true, y_score),
        "micro_ap": micro_ap(y_true, y_score),
        "per_class_ap": {str(name): float(value) for name, value in zip(class_names, ap, strict=True)},
        "thresholds": {str(name): float(value) for name, value in zip(class_names, thresholds, strict=True)},
        "per_class_f1": {str(name): float(value) for name, value in zip(class_names, f1s, strict=True)},
    }


def evaluate_baseline(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    class_names: list[str] | np.ndarray = config.CLASS_NAMES,
) -> dict[str, Any]:
    scores = class_prior_baseline_scores(y_train, len(y_eval))
    return evaluate_scores(y_eval, scores, class_names)


def run_baseline(
    cache_path: Path | str | None = None,
    splits_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    dataset = load_dataset_cache(cache_path)
    split_file = Path(splits_path) if splits_path is not None else config.SPLITS_PATH
    with np.load(split_file, allow_pickle=True) as loaded:
        train_idx = loaded["train_idx"]
        val_idx = loaded["val_idx"]
        test_idx = loaded["test_idx"]

    labels = dataset["labels"]
    class_names = dataset["class_names"]
    y_train = labels[train_idx]
    result = {
        "validation": evaluate_baseline(y_train, labels[val_idx], class_names),
        "test": evaluate_baseline(y_train, labels[test_idx], class_names),
    }
    output = Path(output_path) if output_path is not None else config.BASELINE_JSON
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run_baseline()
