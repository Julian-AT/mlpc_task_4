import json

import numpy as np

from src.final_eval import build_final_table, plot_case_study, select_case_studies, write_case_study_notes


def test_build_final_table_compares_baseline_lr_and_mlp(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    predictions_path = tmp_path / "predictions_test.npz"
    output_path = tmp_path / "final_table.csv"
    baseline_path.write_text(
        json.dumps(
            {
                "test": {
                    "macro_ap": 0.25,
                    "micro_ap": 0.3,
                    "per_class_ap": {"a": 0.2, "b": 0.3},
                }
            }
        )
    )
    y_test = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.uint8)
    np.savez_compressed(
        predictions_path,
        y_test=y_test,
        lr_test_scores=np.array([[0.9, 0.1], [0.1, 0.8], [0.7, 0.6], [0.2, 0.3]], dtype=np.float32),
        mlp_test_scores=np.array([[0.8, 0.2], [0.2, 0.9], [0.6, 0.7], [0.3, 0.1]], dtype=np.float32),
        class_names=np.array(["a", "b"], dtype=object),
    )

    table = build_final_table(baseline_path, predictions_path, output_path)

    assert output_path.exists()
    assert table["model"].tolist() == ["class_prior", "logistic_regression", "mlx_mlp"]
    assert "ap_a" in table.columns


def test_select_case_studies_excludes_training_files():
    y_true = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 1],
            [0, 0],
        ],
        dtype=np.uint8,
    )
    y_score = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.9, 0.8],
            [0.1, 0.1],
        ],
        dtype=np.float32,
    )
    file_ids = np.array(["train", "train", "bad", "bad", "good", "good"])

    cases = select_case_studies(y_true, y_score, file_ids, train_file_ids={"train"})

    assert cases == {"failure": "bad", "success": "good"}


def test_case_study_plot_and_notes(tmp_path):
    y_true = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    y_score = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.6]], dtype=np.float32)
    file_ids = np.array(["file_a", "file_a", "file_a"])
    fig_path = tmp_path / "case.png"
    notes_path = tmp_path / "notes.md"

    returned_fig = plot_case_study(
        "file_a",
        y_true,
        y_score,
        file_ids,
        times=np.array([0.0, 0.5, 1.0]),
        class_names=["a", "b"],
        output_path=fig_path,
    )
    returned_notes = write_case_study_notes({"success": "file_a", "failure": "file_b"}, notes_path)

    assert returned_fig.exists()
    assert returned_notes.exists()
    assert "Success case" in notes_path.read_text()
