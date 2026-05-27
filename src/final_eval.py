from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .metrics import macro_ap, micro_ap, per_class_ap


def _metric_row(name: str, y_true: np.ndarray, y_score: np.ndarray, class_names: list[str] | np.ndarray) -> dict[str, Any]:
    ap = per_class_ap(y_true, y_score)
    row = {
        "model": name,
        "macro_ap": macro_ap(y_true, y_score),
        "micro_ap": micro_ap(y_true, y_score),
    }
    row.update({f"ap_{class_name}": float(value) for class_name, value in zip(class_names, ap, strict=True)})
    return row


def build_final_table(
    baseline_json: Path | str | None = None,
    predictions_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    baseline_path = Path(baseline_json) if baseline_json is not None else config.BASELINE_JSON
    pred_path = Path(predictions_path) if predictions_path is not None else config.PREDICTIONS_TEST
    output = Path(output_path) if output_path is not None else config.FINAL_TABLE_CSV

    baseline = json.loads(baseline_path.read_text())
    with np.load(pred_path, allow_pickle=True) as loaded:
        y_test = loaded["y_test"]
        class_names = loaded["class_names"]
        rows = [
            {
                "model": "class_prior",
                "macro_ap": baseline["test"]["macro_ap"],
                "micro_ap": baseline["test"]["micro_ap"],
                **{f"ap_{name}": value for name, value in baseline["test"]["per_class_ap"].items()},
            }
        ]
        if "lr_test_scores" in loaded.files:
            rows.append(_metric_row("logistic_regression", y_test, loaded["lr_test_scores"], class_names))
        if "mlp_test_scores" in loaded.files:
            rows.append(_metric_row("mlx_mlp", y_test, loaded["mlp_test_scores"], class_names))

    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def _file_f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    pred = y_score >= threshold
    tp = np.logical_and(pred, y_true == 1).sum()
    fp = np.logical_and(pred, y_true == 0).sum()
    fn = np.logical_and(~pred, y_true == 1).sum()
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom else 0.0


def select_case_studies(
    y_true: np.ndarray,
    y_score: np.ndarray,
    file_ids: np.ndarray,
    train_file_ids: set[str] | None = None,
) -> dict[str, str]:
    train_files = train_file_ids or set()
    files = [str(file_id) for file_id in np.unique(file_ids) if str(file_id) not in train_files]
    if len(files) < 2:
        raise ValueError("at least two non-training files are required for case studies")

    scored = []
    for file_id in files:
        mask = np.asarray(file_ids).astype(str) == file_id
        scored.append((file_id, _file_f1(y_true[mask], y_score[mask])))
    scored.sort(key=lambda item: item[1])
    return {"failure": scored[0][0], "success": scored[-1][0]}


def plot_case_study(
    file_id: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    file_ids: np.ndarray,
    times: np.ndarray | None = None,
    class_names: list[str] | np.ndarray = config.CLASS_NAMES,
    output_path: Path | str | None = None,
) -> Path:
    mask = np.asarray(file_ids).astype(str) == str(file_id)
    if not mask.any():
        raise ValueError(f"file_id {file_id!r} not found")

    truth = np.asarray(y_true)[mask].T
    score = np.asarray(y_score)[mask].T
    x = np.asarray(times)[mask] if times is not None else np.arange(score.shape[1])

    output = Path(output_path) if output_path is not None else config.FIG_DIR / f"case_{file_id}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].imshow(truth, aspect="auto", interpolation="nearest", cmap="Greys", origin="lower")
    axes[0].set_title(f"{file_id} ground truth")
    axes[0].set_yticks(np.arange(len(class_names)), class_names)
    image = axes[1].imshow(score, aspect="auto", interpolation="nearest", vmin=0, vmax=1, origin="lower")
    axes[1].set_title("Predicted probabilities")
    axes[1].set_yticks(np.arange(len(class_names)), class_names)
    axes[1].set_xlabel("segment" if times is None else "time (s)")
    if times is not None and len(x) > 1:
        tick_positions = np.linspace(0, len(x) - 1, min(6, len(x)), dtype=int)
        axes[1].set_xticks(tick_positions, [f"{x[pos]:.1f}" for pos in tick_positions])
    fig.colorbar(image, ax=axes, label="probability")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_case_study_notes(
    cases: dict[str, str],
    output_path: Path | str | None = None,
) -> Path:
    output = Path(output_path) if output_path is not None else config.CASE_STUDY_NOTES
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Case Study Notes\n\n"
        f"- Success case: `{cases['success']}`\n"
        f"- Failure case: `{cases['failure']}`\n\n"
        "Detailed qualitative interpretation should be filled after real model predictions and figures exist.\n"
    )
    return output


def run_final_evaluation() -> pd.DataFrame:
    return build_final_table()


if __name__ == "__main__":
    run_final_evaluation()
