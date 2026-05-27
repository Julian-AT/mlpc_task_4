import numpy as np

from src.train_mlp import MLP, positive_class_weights, predict_proba, sweep_mlp, train_one


def _synthetic_mlp_data():
    x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [1.0, 1.0],
            [1.1, 0.9],
            [0.0, 1.0],
            [0.2, 1.1],
            [1.0, 0.0],
            [0.9, 0.1],
            [0.3, 0.2],
            [1.2, 1.2],
            [0.2, 1.2],
            [1.2, 0.2],
        ],
        dtype=np.float32,
    )
    y = np.array(
        [
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1],
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
        ],
        dtype=np.uint8,
    )
    return {
        "features_context": x,
        "labels": y,
        "train_idx": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        "val_idx": np.array([8, 9]),
        "test_idx": np.array([10, 11]),
        "class_names": np.array(["a", "b"], dtype=object),
    }


def test_mlp_forward_and_predict_shape():
    model = MLP(in_dim=2, hidden_dims=[4], out_dim=2, dropout=0.0)
    scores = predict_proba(model, np.ones((3, 2), dtype=np.float32))

    assert scores.shape == (3, 2)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_positive_class_weights_are_finite_and_clipped():
    labels = np.array([[1, 0], [0, 0], [0, 0], [0, 1]], dtype=np.uint8)

    weights = positive_class_weights(labels, max_weight=3.0)

    np.testing.assert_allclose(weights, np.array([3.0, 3.0], dtype=np.float32))


def test_train_one_writes_weights_and_history(tmp_path):
    data = _synthetic_mlp_data()
    weight_path = tmp_path / "mlp_best.npz"

    model, scores, metrics = train_one(
        data["features_context"][data["train_idx"]],
        data["labels"][data["train_idx"]],
        data["features_context"][data["val_idx"]],
        data["labels"][data["val_idx"]],
        hidden_dims=[8],
        dropout=0.0,
        lr=1e-2,
        epochs=3,
        batch_size=4,
        patience=3,
        model_path=weight_path,
    )

    assert weight_path.exists()
    assert scores.shape == (2, 2)
    assert metrics["epochs"] >= 1
    assert metrics["history"]
    assert model is not None


def test_sweep_mlp_writes_csv_weights_and_predictions(tmp_path):
    sweep_path = tmp_path / "mlp_sweep.csv"
    model_path = tmp_path / "mlp_best.npz"
    predictions_path = tmp_path / "predictions_test.npz"
    grid = {"hidden_dims": [[4]], "dropout": [0.0], "lr": [1e-2]}

    frame = sweep_mlp(
        data=_synthetic_mlp_data(),
        grid=grid,
        sweep_path=sweep_path,
        model_path=model_path,
        predictions_path=predictions_path,
        epochs=2,
        batch_size=4,
        patience=2,
    )

    assert sweep_path.exists()
    assert model_path.exists()
    assert predictions_path.exists()
    assert len(frame) == 1
    loaded = np.load(predictions_path, allow_pickle=True)
    assert loaded["mlp_test_scores"].shape == (2, 2)
