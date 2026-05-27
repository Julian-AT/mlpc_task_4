import numpy as np
import pandas as pd

from src.train_lr import fit_one, plot_lr_sweep, sweep_lr


def _synthetic_lr_data():
    x = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [1.0, 1.0],
            [1.2, 1.1],
            [0.0, 1.0],
            [0.1, 1.2],
            [1.0, 0.0],
            [1.1, 0.2],
            [0.3, 0.2],
            [1.3, 1.2],
            [0.2, 1.3],
            [1.2, 0.1],
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
        "features_scaled": x,
        "labels": y,
        "train_idx": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        "val_idx": np.array([8, 9]),
        "test_idx": np.array([10, 11]),
        "class_names": np.array(["a", "b"], dtype=object),
    }


def test_fit_one_returns_probabilities_and_metrics():
    data = _synthetic_lr_data()
    model, scores, metrics = fit_one(
        data["features_scaled"][data["train_idx"]],
        data["labels"][data["train_idx"]],
        data["features_scaled"][data["val_idx"]],
        data["labels"][data["val_idx"]],
        C=1.0,
        penalty="l2",
        class_weight=None,
        max_iter=200,
    )

    assert scores.shape == (2, 2)
    assert 0.0 <= metrics["macro_ap"] <= 1.0
    assert hasattr(model, "estimators_")


def test_sweep_lr_writes_csv_model_and_predictions(tmp_path):
    sweep_path = tmp_path / "lr_sweep.csv"
    model_path = tmp_path / "lr_best.pkl"
    predictions_path = tmp_path / "predictions_test.npz"
    grid = {"C": [0.1, 1.0], "penalty": ["l2"], "class_weight": [None]}

    frame = sweep_lr(
        data=_synthetic_lr_data(),
        grid=grid,
        sweep_path=sweep_path,
        model_path=model_path,
        predictions_path=predictions_path,
        max_iter=200,
    )

    assert sweep_path.exists()
    assert model_path.exists()
    assert predictions_path.exists()
    assert len(frame) == 2
    assert "ap_a" in frame.columns
    loaded = np.load(predictions_path, allow_pickle=True)
    assert loaded["lr_test_scores"].shape == (2, 2)


def test_plot_lr_sweep_writes_heatmap(tmp_path):
    sweep_path = tmp_path / "lr_sweep.csv"
    output_path = tmp_path / "lr_sweep_heatmap.png"
    pd.DataFrame(
        {
            "C": [0.1, 0.1, 1.0, 1.0],
            "penalty": ["l1", "l2", "l1", "l2"],
            "macro_ap": [0.2, 0.3, 0.5, 0.4],
        }
    ).to_csv(sweep_path, index=False)

    plot_lr_sweep(sweep_path, output_path)

    assert output_path.exists()
