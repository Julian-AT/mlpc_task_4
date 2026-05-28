from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from . import config
from .splits import load_dataset_cache


def fit_scaler(x_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.asarray(x_train, dtype=np.float32))
    return scaler


def apply_scaler(scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    return scaler.transform(np.asarray(x, dtype=np.float32)).astype(np.float32, copy=False)


def scale_by_splits(
    features: np.ndarray,
    splits: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, StandardScaler]:
    x = np.asarray(features, dtype=np.float32)
    scaler = fit_scaler(x[np.asarray(splits["train"], dtype=np.int64)])
    return apply_scaler(scaler, x), scaler


def add_temporal_context(
    features: np.ndarray,
    file_ids: np.ndarray,
    k: int = config.TEMPORAL_CONTEXT,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    files = np.asarray(file_ids)
    if x.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    if files.shape[0] != x.shape[0]:
        raise ValueError("file_ids length must match feature rows")
    if k < 0:
        raise ValueError("k must be non-negative")

    n_rows, dim = x.shape
    out = np.zeros((n_rows, dim * (2 * k + 1)), dtype=np.float32)
    for row in range(n_rows):
        pieces: list[np.ndarray] = []
        for offset in range(-k, k + 1):
            source = row + offset
            if 0 <= source < n_rows and files[source] == files[row]:
                pieces.append(x[source])
            else:
                pieces.append(np.zeros(dim, dtype=np.float32))
        out[row] = np.concatenate(pieces)
    return out


def per_file_per_class_iou(
    annotations: np.ndarray, threshold: float = config.ANNOT_BINARIZE_THRESH
) -> np.ndarray:
    ann = np.asarray(annotations, dtype=np.float32)
    if ann.ndim == 3:
        ann = ann[None, ...]
    if ann.ndim != 4:
        raise ValueError("annotations must have shape [F, T, C, A] or [T, C, A]")

    binary = np.nan_to_num(ann, nan=0.0) >= threshold
    n_files, _, n_classes, n_annotators = binary.shape
    out = np.full((n_files, n_classes), np.nan, dtype=np.float32)
    for file_idx in range(n_files):
        for class_idx in range(n_classes):
            scores: list[float] = []
            for left in range(n_annotators):
                for right in range(left + 1, n_annotators):
                    a = binary[file_idx, :, class_idx, left]
                    b = binary[file_idx, :, class_idx, right]
                    union = np.logical_or(a, b).sum()
                    if union == 0:
                        continue
                    scores.append(float(np.logical_and(a, b).sum() / union))
            if scores:
                out[file_idx, class_idx] = float(np.mean(scores))
    return out


def high_agreement_mask(
    file_ids: np.ndarray,
    class_idx: int,
    per_file_iou: Mapping[str, float | np.ndarray] | np.ndarray,
    file_order: list[str] | np.ndarray | None = None,
    threshold: float = config.HIGH_AGREEMENT_IOU,
) -> np.ndarray:
    files = np.asarray(file_ids)
    if isinstance(per_file_iou, Mapping):
        lookup = {
            str(file_id): float(value[class_idx] if np.ndim(value) else value)
            for file_id, value in per_file_iou.items()
        }
    else:
        if file_order is None:
            raise ValueError("file_order is required when per_file_iou is an array")
        values = np.asarray(per_file_iou, dtype=np.float32)
        lookup = {str(file_id): float(values[i, class_idx]) for i, file_id in enumerate(file_order)}

    return np.asarray(
        [
            np.isfinite(lookup.get(str(file_id), np.nan)) and lookup[str(file_id)] >= threshold
            for file_id in files
        ],
        dtype=bool,
    )


def build_preprocessed(
    cache_path: Path | str | None = None,
    splits_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> dict[str, np.ndarray]:
    dataset = load_dataset_cache(cache_path)
    split_file = Path(splits_path) if splits_path is not None else config.SPLITS_PATH
    with np.load(split_file, allow_pickle=True) as loaded:
        splits = {
            "train": loaded["train_idx"],
            "val": loaded["val_idx"],
            "test": loaded["test_idx"],
        }

    scaled, scaler = scale_by_splits(dataset["features"], splits)
    contextual = add_temporal_context(scaled, dataset["file_ids"])
    output = Path(output_path) if output_path is not None else config.PREPROCESSED_CACHE
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features_scaled=scaled,
        features_context=contextual,
        labels=dataset["labels"],
        file_ids=dataset["file_ids"],
        train_idx=splits["train"],
        val_idx=splits["val"],
        test_idx=splits["test"],
    )
    joblib.dump(scaler, config.SCALER_PATH)
    return {"features_scaled": scaled, "features_context": contextual, "labels": dataset["labels"]}


if __name__ == "__main__":
    build_preprocessed()
