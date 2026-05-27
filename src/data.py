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


def aggregate_labels(annotations: np.ndarray) -> np.ndarray:
    """Aggregate `[T, C, A]` overlap annotations to `[T, C]` binary labels."""
    ann = np.asarray(annotations, dtype=np.float32)
    if ann.ndim != 3:
        raise ValueError("annotations must have shape [T, C, A]")

    finite = np.isfinite(ann)
    has_finite_values = finite.any(axis=(0, 1))
    has_nonzero_values = (np.nan_to_num(np.abs(ann), nan=0.0) > 0).any(axis=(0, 1))
    valid_annotators = has_finite_values & has_nonzero_values

    if not valid_annotators.any():
        valid_annotators = has_finite_values
    if not valid_annotators.any():
        raise ValueError("annotations contain no valid annotator slices")

    binary = np.nan_to_num(ann[:, :, valid_annotators], nan=0.0) >= config.ANNOT_BINARIZE_THRESH
    votes = binary.mean(axis=2)
    return (votes >= config.MAJORITY_THRESH).astype(np.uint8)


def _npz_class_names(npz_dict: Mapping[str, Any]) -> list[str]:
    if "class_names" not in npz_dict:
        return list(config.CLASS_NAMES)
    return [str(name) for name in np.asarray(npz_dict["class_names"]).tolist()]


def _time_array(npz_dict: Mapping[str, Any], keys: tuple[str, ...], segment_count: int) -> np.ndarray:
    for key in keys:
        if key in npz_dict:
            arr = np.asarray(npz_dict[key], dtype=np.float32)
            if arr.shape[0] != segment_count:
                raise ValueError(f"{key!r} has segment count {arr.shape[0]}, expected {segment_count}")
            return arr
    starts = np.arange(segment_count, dtype=np.float32) * 0.5
    if "end" in keys[0]:
        return starts + 1.0
    return starts


def _metadata_lookup(metadata: pd.DataFrame) -> dict[str, Any]:
    filename_col = next(
        (col for col in ["filename", "file", "file_name", "audio_filename"] if col in metadata.columns),
        None,
    )
    collector_col = next(
        (col for col in ["collector_id", "collector", "user_id"] if col in metadata.columns),
        None,
    )
    if filename_col is None or collector_col is None:
        return {}
    return {
        str(row[filename_col]): row[collector_col]
        for _, row in metadata[[filename_col, collector_col]].dropna(subset=[filename_col]).iterrows()
    }


def _collector_for(path: Path, collectors: Mapping[str, Any]) -> Any:
    candidates = [
        path.name,
        path.stem,
        f"{path.stem}.wav",
        f"{path.stem}.flac",
        f"{path.stem}.mp3",
    ]
    for candidate in candidates:
        if candidate in collectors:
            return collectors[candidate]
    return ""


def _write_log(summary: dict[str, Any], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    class_lines = "\n".join(
        f"  - {name}: {count} positives ({rate:.6f})"
        for name, count, rate in zip(
            summary["class_names"], summary["positive_counts"], summary["positive_rates"], strict=True
        )
    )
    text = (
        "# Results Log\n\n"
        "## Phase 1: Project Scaffold and Data Foundation\n\n"
        f"- Files processed: {summary['file_count']}\n"
        f"- Total segments: {summary['segment_count']}\n"
        f"- Feature dimensionality: {summary['feature_dim']}\n"
        f"- Feature keys: {', '.join(summary['feature_keys'])}\n"
        f"- Cache path: `{summary['cache_path']}`\n"
        f"- Class order: {', '.join(summary['class_names'])}\n"
        "- Positive rates:\n"
        f"{class_lines}\n"
    )
    log_path.write_text(text)


def build_dataset(cache_path: Path | str | None = None) -> dict[str, np.ndarray]:
    """Build and cache the segment-level dataset arrays."""
    cache = Path(cache_path) if cache_path is not None else config.DATASET_CACHE
    feature_files = iter_feature_files()
    if not feature_files:
        raise FileNotFoundError(f"No .npz feature files found in {config.FEATURES_DIR}")

    metadata = load_metadata()
    load_annotations()  # Validate that the expected CSV exists for reproducibility.
    collectors = _metadata_lookup(metadata)

    features_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    file_ids: list[str] = []
    collector_ids: list[Any] = []
    segment_indices: list[int] = []
    start_times: list[float] = []
    end_times: list[float] = []
    expected_feature_keys: list[str] | None = None
    expected_feature_dim: int | None = None

    for file_index, feature_file in enumerate(feature_files):
        with np.load(feature_file, allow_pickle=True) as loaded:
            npz_dict = {key: loaded[key] for key in loaded.files}

        class_names = _npz_class_names(npz_dict)
        if class_names != config.CLASS_NAMES:
            raise ValueError(f"Class names in {feature_file} do not match config.CLASS_NAMES")
        if "annotations" not in npz_dict:
            raise ValueError(f"{feature_file} does not contain an 'annotations' array")

        features, feature_keys = concat_features(npz_dict)
        labels = aggregate_labels(np.asarray(npz_dict["annotations"]))
        if labels.shape[0] != features.shape[0]:
            raise ValueError(f"{feature_file} has mismatched feature and label segment counts")

        if expected_feature_keys is None:
            expected_feature_keys = feature_keys
            expected_feature_dim = features.shape[1]
        elif feature_keys != expected_feature_keys or features.shape[1] != expected_feature_dim:
            raise ValueError(f"{feature_file} has an incompatible feature schema")

        segment_count = features.shape[0]
        starts = _time_array(npz_dict, ("start_time", "start_times"), segment_count)
        ends = _time_array(npz_dict, ("end_time", "end_times"), segment_count)
        collector = _collector_for(feature_file, collectors)

        features_parts.append(features)
        label_parts.append(labels)
        file_ids.extend([feature_file.name] * segment_count)
        collector_ids.extend([collector] * segment_count)
        segment_indices.extend(range(segment_count))
        start_times.extend(starts.tolist())
        end_times.extend(ends.tolist())

    features_all = np.concatenate(features_parts, axis=0).astype(np.float32, copy=False)
    labels_all = np.concatenate(label_parts, axis=0).astype(np.uint8, copy=False)
    class_names_array = np.asarray(config.CLASS_NAMES, dtype=object)
    feature_keys_array = np.asarray(expected_feature_keys or [], dtype=object)

    dataset = {
        "features": features_all,
        "labels": labels_all,
        "X": features_all,
        "Y": labels_all,
        "file_ids": np.asarray(file_ids, dtype=object),
        "collector_ids": np.asarray(collector_ids, dtype=object),
        "segment_indices": np.asarray(segment_indices, dtype=np.int64),
        "start_times": np.asarray(start_times, dtype=np.float32),
        "end_times": np.asarray(end_times, dtype=np.float32),
        "class_names": class_names_array,
        "feature_keys": feature_keys_array,
    }

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **dataset)

    positive_counts = labels_all.sum(axis=0).astype(int)
    positive_rates = labels_all.mean(axis=0)
    _write_log(
        {
            "file_count": len(feature_files),
            "segment_count": int(labels_all.shape[0]),
            "feature_dim": int(features_all.shape[1]),
            "feature_keys": list(expected_feature_keys or []),
            "cache_path": str(cache),
            "class_names": list(config.CLASS_NAMES),
            "positive_counts": positive_counts.tolist(),
            "positive_rates": positive_rates.tolist(),
        },
        config.RESULTS_LOG,
    )
    return dataset


if __name__ == "__main__":
    build_dataset()
