from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config


def _as_indices(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def load_dataset_cache(cache_path: Path | str | None = None) -> dict[str, np.ndarray]:
    path = Path(cache_path) if cache_path is not None else config.DATASET_CACHE
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def assert_disjoint_collectors(splits: Mapping[str, np.ndarray], collector_ids: np.ndarray) -> None:
    collectors = {
        name: set(np.asarray(collector_ids)[_as_indices(indices)].tolist())
        for name, indices in splits.items()
    }
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for left, right in pairs:
        overlap = collectors[left] & collectors[right]
        if overlap:
            raise ValueError(f"collector leakage between {left} and {right}: {sorted(overlap)!r}")


def make_splits(
    collector_ids: np.ndarray,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
    test_frac: float = config.TEST_FRAC,
    seed: int = config.SEED,
) -> dict[str, np.ndarray]:
    groups = np.asarray(collector_ids)
    if groups.ndim != 1:
        raise ValueError("collector_ids must be one-dimensional")
    if len(groups) == 0:
        raise ValueError("collector_ids cannot be empty")
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("split fractions must sum to 1.0")
    if len(np.unique(groups)) < 3:
        raise ValueError("at least three collectors are required for train/val/test splits")

    indices = np.arange(len(groups), dtype=np.int64)
    first = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, temp_idx = next(first.split(indices, groups=groups))

    temp_groups = groups[temp_idx]
    if len(np.unique(temp_groups)) < 2:
        raise ValueError("not enough held-out collectors to create validation and test splits")

    val_share = val_frac / (val_frac + test_frac)
    second = GroupShuffleSplit(n_splits=1, train_size=val_share, random_state=seed + 1)
    val_rel, test_rel = next(second.split(temp_idx, groups=temp_groups))

    splits = {
        "train": np.sort(train_idx.astype(np.int64)),
        "val": np.sort(temp_idx[val_rel].astype(np.int64)),
        "test": np.sort(temp_idx[test_rel].astype(np.int64)),
    }
    assert_disjoint_collectors(splits, groups)
    return splits


def class_distribution_table(
    labels: np.ndarray,
    splits: Mapping[str, np.ndarray],
    class_names: list[str] | np.ndarray = config.CLASS_NAMES,
) -> pd.DataFrame:
    y = np.asarray(labels)
    if y.ndim != 2:
        raise ValueError("labels must have shape [N, C]")

    rows: list[dict[str, object]] = []
    names = [str(name) for name in class_names]
    for class_idx, class_name in enumerate(names):
        row: dict[str, object] = {"class_name": class_name}
        for split_name in ("train", "val", "test"):
            idx = _as_indices(splits[split_name])
            positives = int(y[idx, class_idx].sum())
            total = int(len(idx))
            row[f"{split_name}_positives"] = positives
            row[f"{split_name}_total"] = total
            row[f"{split_name}_rate"] = float(positives / total) if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_splits(
    splits: Mapping[str, np.ndarray],
    path: Path | str | None = None,
) -> None:
    output = Path(path) if path is not None else config.SPLITS_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        train_idx=_as_indices(splits["train"]),
        val_idx=_as_indices(splits["val"]),
        test_idx=_as_indices(splits["test"]),
    )


def plot_class_distribution(table: pd.DataFrame, output_path: Path | str | None = None) -> None:
    path = (
        Path(output_path)
        if output_path is not None
        else config.FIG_DIR / "class_dist_across_splits.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(table))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 5))
    for offset, split_name in [(-width, "train"), (0.0, "val"), (width, "test")]:
        ax.bar(x + offset, table[f"{split_name}_rate"], width=width, label=split_name)
    ax.set_ylabel("Positive rate")
    ax.set_xticks(x)
    ax.set_xticklabels(table["class_name"], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_splits(
    cache_path: Path | str | None = None,
    split_path: Path | str | None = None,
    distribution_path: Path | str | None = None,
) -> dict[str, np.ndarray]:
    dataset = load_dataset_cache(cache_path)
    splits = make_splits(dataset["collector_ids"])
    save_splits(splits, split_path)

    table = class_distribution_table(dataset["labels"], splits, dataset["class_names"])
    csv_path = (
        Path(distribution_path) if distribution_path is not None else config.CLASS_DISTRIBUTION_CSV
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    plot_class_distribution(table)
    return splits


if __name__ == "__main__":
    build_splits()
