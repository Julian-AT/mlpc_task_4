import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import aggregate_labels, build_dataset, concat_features, load_annotations, load_metadata


def test_config_class_names_are_alphabetical():
    assert config.NUM_CLASSES == 15
    assert config.CLASS_NAMES == sorted(config.CLASS_NAMES)


def test_load_metadata_and_annotations_from_explicit_paths(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    annotations_path = tmp_path / "annotations.csv"
    pd.DataFrame({"filename": ["a.wav"], "collector_id": ["c1"]}).to_csv(metadata_path, index=False)
    pd.DataFrame({"filename": ["a.wav"], "note": ["ok"]}).to_csv(annotations_path, index=False)

    metadata = load_metadata(metadata_path)
    annotations = load_annotations(annotations_path)

    assert metadata.loc[0, "filename"] == "a.wav"
    assert metadata.loc[0, "collector_id"] == "c1"
    assert annotations.loc[0, "note"] == "ok"


def test_concat_features_deterministic_order_and_excludes_metadata():
    npz = {
        "z_feature": np.array([[2.0], [3.0]], dtype=np.float32),
        "a_feature": np.array([[1.0, 4.0], [1.5, 4.5]], dtype=np.float32),
        "annotations": np.zeros((2, 15, 1), dtype=np.float32),
        "start_time": np.array([0.0, 0.5]),
        "class_names": np.array(config.CLASS_NAMES),
    }

    features, keys = concat_features(npz)

    assert keys == ["a_feature", "z_feature"]
    assert features.dtype == np.float32
    np.testing.assert_allclose(features, np.array([[1.0, 4.0, 2.0], [1.5, 4.5, 3.0]]))


def test_concat_features_flattens_trailing_dimensions():
    npz = {
        "mel": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        "energy": np.array([1.0, 2.0], dtype=np.float32),
    }

    features, keys = concat_features(npz)

    assert keys == ["energy", "mel"]
    assert features.shape == (2, 7)
    np.testing.assert_allclose(features[:, 0], np.array([1.0, 2.0]))


def test_concat_features_rejects_inconsistent_segment_count():
    npz = {
        "a": np.ones((2, 1), dtype=np.float32),
        "b": np.ones((3, 1), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="segment count"):
        concat_features(npz)


def test_concat_features_rejects_no_compatible_features():
    npz = {
        "annotations": np.zeros((2, 15, 1), dtype=np.float32),
        "class_names": np.array(config.CLASS_NAMES),
    }

    with pytest.raises(ValueError, match="No compatible feature"):
        concat_features(npz)


def test_aggregate_labels_single_annotator():
    annotations = np.array([[[0.2], [0.5]], [[0.9], [0.0]]], dtype=np.float32)

    labels = aggregate_labels(annotations)

    expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(labels, expected)


def test_aggregate_labels_majority_vote_and_tie_threshold():
    annotations = np.array([[[0.8, 0.1], [0.6, 0.7]]], dtype=np.float32)

    labels = aggregate_labels(annotations)

    expected = np.array([[1, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(labels, expected)


def test_aggregate_labels_masks_nan_annotator():
    annotations = np.array([[[0.8, np.nan]], [[0.1, np.nan]]], dtype=np.float32)

    labels = aggregate_labels(annotations)

    expected = np.array([[1], [0]], dtype=np.uint8)
    np.testing.assert_array_equal(labels, expected)


def test_aggregate_labels_masks_all_zero_inactive_annotator():
    annotations = np.array([[[0.8, 0.0]], [[0.6, 0.0]]], dtype=np.float32)

    labels = aggregate_labels(annotations)

    expected = np.array([[1], [1]], dtype=np.uint8)
    np.testing.assert_array_equal(labels, expected)


def test_aggregate_labels_rejects_invalid_shape():
    with pytest.raises(ValueError, match=r"\[T, C, A\]"):
        aggregate_labels(np.zeros((2, 15), dtype=np.float32))


def test_build_dataset_writes_expected_cache_schema(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    features_dir = data_dir / "audio_features"
    features_dir.mkdir(parents=True)
    cache_path = tmp_path / "dataset_cache.npz"
    metadata_path = data_dir / "metadata.csv"
    annotations_path = data_dir / "annotations.csv"
    log_path = tmp_path / "log.md"

    pd.DataFrame({"filename": ["file1.npz"], "collector_id": ["collector_a"]}).to_csv(
        metadata_path, index=False
    )
    pd.DataFrame({"filename": ["file1.npz"]}).to_csv(annotations_path, index=False)
    np.savez(
        features_dir / "file1.npz",
        acoustic=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        annotations=np.array(
            [
                [[0.7], [0.1], *[[0.0] for _ in range(13)]],
                [[0.2], [0.9], *[[0.0] for _ in range(13)]],
            ],
            dtype=np.float32,
        ),
        class_names=np.array(config.CLASS_NAMES),
        start_time=np.array([0.0, 0.5], dtype=np.float32),
        end_time=np.array([1.0, 1.5], dtype=np.float32),
    )

    monkeypatch.setattr(config, "METADATA_CSV", metadata_path)
    monkeypatch.setattr(config, "ANNOTATIONS_CSV", annotations_path)
    monkeypatch.setattr(config, "FEATURES_DIR", features_dir)
    monkeypatch.setattr(config, "RESULTS_LOG", log_path)

    dataset = build_dataset(cache_path)
    cached = np.load(cache_path, allow_pickle=True)

    for key in [
        "features",
        "labels",
        "file_ids",
        "collector_ids",
        "segment_indices",
        "start_times",
        "end_times",
        "class_names",
        "feature_keys",
    ]:
        assert key in cached.files
        assert key in dataset
    assert cached["features"].shape == (2, 2)
    assert cached["labels"].shape == (2, 15)
    assert cached["collector_ids"].tolist() == ["collector_a", "collector_a"]
    assert "Feature dimensionality: 2" in log_path.read_text()
