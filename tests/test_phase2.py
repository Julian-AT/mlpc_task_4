import json

import numpy as np
import pandas as pd

from src import config
from src.baseline import class_prior_baseline_scores, evaluate_baseline, run_baseline
from src.metrics import best_threshold_f1, macro_ap, micro_ap, per_class_ap, per_class_f1_at_optimal
from src.preprocess import (
    add_temporal_context,
    high_agreement_mask,
    per_file_per_class_iou,
    scale_by_splits,
)
from src.splits import (
    assert_disjoint_collectors,
    build_splits,
    class_distribution_table,
    make_splits,
)


def test_make_splits_keeps_collectors_disjoint_and_reproducible():
    collector_ids = np.repeat([f"collector_{idx}" for idx in range(12)], 5)

    splits_a = make_splits(collector_ids, seed=42)
    splits_b = make_splits(collector_ids, seed=42)

    assert set(splits_a) == {"train", "val", "test"}
    assert sum(len(indices) for indices in splits_a.values()) == len(collector_ids)
    assert_disjoint_collectors(splits_a, collector_ids)
    for name in splits_a:
        np.testing.assert_array_equal(splits_a[name], splits_b[name])


def test_class_distribution_table_counts_positive_rates():
    labels = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    splits = {
        "train": np.array([0, 1, 2]),
        "val": np.array([3]),
        "test": np.array([4, 5]),
    }

    table = class_distribution_table(labels, splits, ["a", "b"])

    assert table.loc[0, "train_positives"] == 2
    assert table.loc[0, "test_rate"] == 0.5
    assert table.loc[1, "val_rate"] == 0.0


def test_build_splits_writes_split_and_distribution_outputs(tmp_path, monkeypatch):
    cache_path = tmp_path / "dataset_cache.npz"
    split_path = tmp_path / "splits.npz"
    dist_path = tmp_path / "class_distribution.csv"
    fig_dir = tmp_path / "figures"
    collector_ids = np.repeat([f"collector_{idx}" for idx in range(10)], 3)
    labels = np.zeros((len(collector_ids), config.NUM_CLASSES), dtype=np.uint8)
    labels[::2, 0] = 1
    np.savez_compressed(
        cache_path,
        features=np.ones((len(collector_ids), 2), dtype=np.float32),
        labels=labels,
        collector_ids=collector_ids,
        class_names=np.asarray(config.CLASS_NAMES, dtype=object),
    )
    monkeypatch.setattr(config, "FIG_DIR", fig_dir)

    splits = build_splits(cache_path, split_path, dist_path)

    assert split_path.exists()
    assert dist_path.exists()
    assert (fig_dir / "class_dist_across_splits.png").exists()
    assert_disjoint_collectors(splits, collector_ids)
    loaded = np.load(split_path)
    assert "train_idx" in loaded.files
    assert pd.read_csv(dist_path).shape[0] == config.NUM_CLASSES


def test_scale_by_splits_fits_scaler_on_train_rows_only():
    features = np.array([[1.0], [3.0], [100.0], [200.0]], dtype=np.float32)
    splits = {"train": np.array([0, 1]), "val": np.array([2]), "test": np.array([3])}

    scaled, scaler = scale_by_splits(features, splits)

    np.testing.assert_allclose(scaler.mean_, np.array([2.0]))
    np.testing.assert_allclose(scaled[[0, 1], 0], np.array([-1.0, 1.0]), atol=1e-6)
    assert scaled[2, 0] > 90.0


def test_add_temporal_context_zero_pads_file_boundaries():
    features = np.array([[1.0], [2.0], [10.0]], dtype=np.float32)
    file_ids = np.array(["a", "a", "b"])

    contextual = add_temporal_context(features, file_ids, k=1)

    np.testing.assert_allclose(contextual[0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(contextual[1], np.array([1.0, 2.0, 0.0]))
    np.testing.assert_allclose(contextual[2], np.array([0.0, 10.0, 0.0]))


def test_per_file_iou_and_high_agreement_mask():
    annotations = np.array(
        [
            [
                [[1.0, 1.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ],
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
            ],
        ],
        dtype=np.float32,
    )

    iou = per_file_per_class_iou(annotations)
    mask = high_agreement_mask(
        np.array(["file_a", "file_b", "file_a"]),
        class_idx=0,
        per_file_iou=iou,
        file_order=["file_a", "file_b"],
        threshold=0.6,
    )

    np.testing.assert_allclose(iou[:, 0], np.array([1.0, 0.0]), atol=1e-6)
    assert np.isnan(iou[0, 1]) or iou[0, 1] < 0.6
    np.testing.assert_array_equal(mask, np.array([True, False, True]))


def test_metrics_compute_ap_and_optimal_f1():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.uint8)
    y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.4], [0.1, 0.2]], dtype=np.float32)

    ap = per_class_ap(y_true, y_score)
    thresholds, f1s = per_class_f1_at_optimal(y_true, y_score)

    np.testing.assert_allclose(ap, np.ones(2), atol=1e-6)
    assert macro_ap(y_true, y_score) == 1.0
    assert micro_ap(y_true, y_score) == 1.0
    assert thresholds.shape == (2,)
    assert np.all(f1s == 1.0)
    assert best_threshold_f1(y_true[:, 0], y_score[:, 0])[1] == 1.0


def test_class_prior_baseline_scores_and_constant_ap_property():
    y_train = np.array([[1, 0], [0, 0], [1, 1], [0, 0]], dtype=np.uint8)
    y_eval = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.uint8)

    scores = class_prior_baseline_scores(y_train, len(y_eval))
    result = evaluate_baseline(y_train, y_eval, ["a", "b"])

    np.testing.assert_allclose(scores[0], np.array([0.5, 0.25]))
    np.testing.assert_allclose(scores[-1], np.array([0.5, 0.25]))
    np.testing.assert_allclose(
        list(result["per_class_ap"].values()),
        y_eval.mean(axis=0),
        atol=1e-6,
    )


def test_run_baseline_writes_json(tmp_path):
    cache_path = tmp_path / "dataset_cache.npz"
    split_path = tmp_path / "splits.npz"
    output_path = tmp_path / "baseline.json"
    labels = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]], dtype=np.uint8)
    np.savez_compressed(
        cache_path,
        labels=labels,
        class_names=np.array(["a", "b"], dtype=object),
    )
    np.savez_compressed(
        split_path,
        train_idx=np.array([0, 1, 2]),
        val_idx=np.array([3]),
        test_idx=np.array([4, 5]),
    )

    result = run_baseline(cache_path, split_path, output_path)

    assert output_path.exists()
    assert "validation" in result
    assert json.loads(output_path.read_text())["test"]["macro_ap"] > 0
