import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import concat_features, load_annotations, load_metadata


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
