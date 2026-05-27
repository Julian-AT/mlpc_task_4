from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import config


NON_FEATURE_KEYS = {
    "annotations",
    "annotation",
    "class_names",
    "annotator_ids",
    "annotators",
    "start_time",
    "start_times",
    "end_time",
    "end_times",
    "filename",
    "file_id",
    "collector_id",
}


def load_metadata(path: Path | str | None = None) -> pd.DataFrame:
    """Load the task metadata CSV from an explicit path or the configured dataset path."""
    return pd.read_csv(Path(path) if path is not None else config.METADATA_CSV)


def load_annotations(path: Path | str | None = None) -> pd.DataFrame:
    """Load the task annotations CSV from an explicit path or the configured dataset path."""
    return pd.read_csv(Path(path) if path is not None else config.ANNOTATIONS_CSV)


def iter_feature_files(features_dir: Path | str | None = None) -> list[Path]:
    """Return feature `.npz` files in deterministic order."""
    directory = Path(features_dir) if features_dir is not None else config.FEATURES_DIR
    return sorted(directory.glob("*.npz"))


def _is_numeric_feature_array(value: Any) -> bool:
    array = np.asarray(value)
    return array.ndim >= 1 and np.issubdtype(array.dtype, np.number)


def _as_feature_matrix(value: Any, key: str, segment_count: int | None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"Feature key {key!r} is scalar and cannot be concatenated")
    if segment_count is not None and array.shape[0] != segment_count:
        raise ValueError(
            f"Feature key {key!r} has segment count {array.shape[0]}, expected {segment_count}"
        )
    if array.ndim == 1:
        array = array[:, None]
    else:
        array = array.reshape(array.shape[0], -1)
    return array.astype(np.float32, copy=False)


def concat_features(npz_dict: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Concatenate compatible numeric feature arrays in stable key order.

    Non-feature metadata arrays are excluded explicitly. All selected arrays must share the
    same first dimension, which is interpreted as the segment count.
    """
    keys = sorted(
        key
        for key, value in npz_dict.items()
        if key not in NON_FEATURE_KEYS and _is_numeric_feature_array(value)
    )
    if not keys:
        raise ValueError("No compatible feature arrays found")

    segment_count = int(np.asarray(npz_dict[keys[0]]).shape[0])
    matrices = [_as_feature_matrix(npz_dict[key], key, segment_count) for key in keys]
    return np.concatenate(matrices, axis=1).astype(np.float32, copy=False), keys
