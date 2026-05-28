from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .final_eval import _file_f1


REPORT_FIG_DIR = config.ROOT / "report" / "figures"


def _short_label(name: str) -> str:
    return (
        name.replace("_open_close", "")
        .replace("wardrobe_drawer", "wardrobe")
        .replace("_", " ")
    )


def copy_existing_figures() -> None:
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    source = config.FIG_DIR / "class_dist_across_splits.png"
    if source.exists():
        shutil.copyfile(source, REPORT_FIG_DIR / "class_dist_across_splits.png")


def make_split_figure() -> None:
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(config.CLASS_DISTRIBUTION_CSV)
    labels = [_short_label(name) for name in table["class_name"]]
    x = np.arange(len(table))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    for offset, split_name, color in [
        (-width, "train", "#4c78a8"),
        (0.0, "val", "#f58518"),
        (width, "test", "#54a24b"),
    ]:
        ax.bar(x + offset, table[f"{split_name}_rate"], width=width, label=split_name, color=color)
    ax.set_ylabel("positive rate")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(REPORT_FIG_DIR / "class_dist_across_splits.pdf")
    plt.close(fig)


def make_hyperparameter_figure() -> None:
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    lr = pd.read_csv(config.RESULTS_DIR / "lr_torch_sweep.csv")
    mlp = pd.read_csv(config.MLP_SWEEP_CSV)

    lr = lr.sort_values("macro_ap", ascending=False).head(8).copy()
    lr["label"] = lr.apply(
        lambda row: f"{row['feature_key'].replace('features_', '')}, "
        f"lr={row['lr']:.4g}, pw={row['pos_weight_max']}",
        axis=1,
    )

    mlp = mlp.sort_values("macro_ap", ascending=False).head(8).copy()
    mlp["label"] = mlp.apply(
        lambda row: f"{row['hidden_dims']}, d={row['dropout']:.2g}, "
        f"lr={row['lr']:.4g}",
        axis=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), sharex=False)
    for ax, frame, title, color in [
        (axes[0], lr.iloc[::-1], "Logistic regression variants", "#4c78a8"),
        (axes[1], mlp.iloc[::-1], "Single MLP variants", "#f58518"),
    ]:
        ax.barh(np.arange(len(frame)), frame["macro_ap"], color=color)
        ax.set_yticks(np.arange(len(frame)), frame["label"], fontsize=7)
        ax.set_xlabel("validation macro AP")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        left = max(0.0, float(frame["macro_ap"].min()) - 0.02)
        right = min(0.66, float(frame["macro_ap"].max()) + 0.02)
        ax.set_xlim(left, right)
    fig.tight_layout()
    fig.savefig(REPORT_FIG_DIR / "hyperparameter_summary.png", dpi=220)
    fig.savefig(REPORT_FIG_DIR / "hyperparameter_summary.pdf")
    plt.close(fig)


def make_per_class_ap_figure() -> None:
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(config.FINAL_TABLE_CSV)
    classes = [col[3:] for col in frame.columns if col.startswith("ap_")]
    mlp = frame[frame["model"] == "mlp"].iloc[0]
    order = sorted(classes, key=lambda name: float(mlp[f"ap_{name}"]))
    labels = [_short_label(name) for name in order]
    y = np.arange(len(order))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    for offset, model, color in [
        (-width / 2, "logistic_regression", "#4c78a8"),
        (width / 2, "mlp", "#f58518"),
    ]:
        row = frame[frame["model"] == model].iloc[0]
        values = [float(row[f"ap_{name}"]) for name in order]
        label = "logistic regression" if model == "logistic_regression" else "MLP ensemble"
        ax.barh(y + offset, values, height=width, label=label, color=color)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("test AP")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORT_FIG_DIR / "per_class_ap.pdf")
    fig.savefig(REPORT_FIG_DIR / "per_class_ap.png", dpi=220)
    plt.close(fig)


def _load_file_features(file_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = config.FEATURES_DIR / file_id
    with np.load(path, allow_pickle=True) as loaded:
        mel = np.asarray(loaded["melspect_mean"], dtype=np.float32)
        times = np.asarray(loaded["start_time"], dtype=np.float32)
    return mel, times


def _case_rows() -> list[dict[str, object]]:
    data = np.load(config.DATASET_CACHE, allow_pickle=True)
    pred = np.load(config.PREDICTIONS_TEST, allow_pickle=True)
    test_idx = np.asarray(pred["test_idx"], dtype=np.int64)
    y = np.asarray(pred["y_test"], dtype=np.uint8)
    scores = np.asarray(pred["mlp_test_scores"], dtype=np.float32)
    files = np.asarray(data["file_ids"])[test_idx].astype(str)
    class_names = np.asarray(pred["class_names"]).astype(str)

    rows: list[dict[str, object]] = []
    for file_id in sorted(np.unique(files)):
        mask = files == file_id
        active = np.where(y[mask].sum(axis=0) > 0)[0]
        if active.size == 0:
            continue
        rows.append(
            {
                "file_id": file_id,
                "f1": _file_f1(y[mask], scores[mask]),
                "positives": int(y[mask].sum()),
                "predicted_positives": int((scores[mask] >= 0.5).sum()),
                "segments": int(mask.sum()),
                "active_classes": [str(class_names[i]) for i in active],
            }
        )
    return rows


def _plot_case_column(
    axes: np.ndarray,
    file_id: str,
    title: str,
    y_file: np.ndarray,
    score_file: np.ndarray,
    class_names: np.ndarray,
) -> None:
    mel, times = _load_file_features(file_id)
    active = np.where((y_file.sum(axis=0) + (score_file >= 0.5).sum(axis=0)) > 0)[0]
    if active.size > 7:
        active = active[:7]
    labels = [_short_label(str(class_names[idx])) for idx in active]

    axes[0].imshow(mel.T, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title(title, fontsize=10)
    axes[0].set_ylabel("mel bin")
    axes[0].set_xticks([])

    axes[1].imshow(y_file[:, active].T, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=1)
    axes[1].set_yticks(np.arange(len(active)), labels, fontsize=7)
    axes[1].set_ylabel("truth")
    axes[1].set_xticks([])

    image = axes[2].imshow(
        score_file[:, active].T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[2].set_yticks(np.arange(len(active)), labels, fontsize=7)
    axes[2].set_ylabel("prob.")
    tick_positions = np.linspace(0, len(times) - 1, min(5, len(times)), dtype=int)
    axes[2].set_xticks(tick_positions, [f"{times[pos]:.0f}" for pos in tick_positions])
    axes[2].set_xlabel("time (s)")
    return image


def make_case_study_figure() -> dict[str, object]:
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(config.DATASET_CACHE, allow_pickle=True)
    pred = np.load(config.PREDICTIONS_TEST, allow_pickle=True)
    test_idx = np.asarray(pred["test_idx"], dtype=np.int64)
    y = np.asarray(pred["y_test"], dtype=np.uint8)
    scores = np.asarray(pred["mlp_test_scores"], dtype=np.float32)
    files = np.asarray(data["file_ids"])[test_idx].astype(str)
    class_names = np.asarray(pred["class_names"]).astype(str)

    rows = _case_rows()
    failure = min(rows, key=lambda row: float(row["f1"]))
    success_candidates = [row for row in rows if len(row["active_classes"]) >= 3]
    success = max(success_candidates, key=lambda row: float(row["f1"]))

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 5.1), constrained_layout=True)
    image = None
    for col, case in enumerate([success, failure]):
        file_id = str(case["file_id"])
        mask = files == file_id
        title = f"{file_id.replace('.npz', '')}: F1={float(case['f1']):.2f}"
        image = _plot_case_column(axes[:, col], file_id, title, y[mask], scores[mask], class_names)
    if image is not None:
        fig.colorbar(image, ax=axes[2, :], shrink=0.85, label="predicted probability")
    fig.savefig(REPORT_FIG_DIR / "case_studies.png", dpi=220)
    fig.savefig(REPORT_FIG_DIR / "case_studies.pdf")
    plt.close(fig)

    summary = {"success": success, "failure": failure}
    (config.ROOT / "report" / "case_study_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    copy_existing_figures()
    make_split_figure()
    make_hyperparameter_figure()
    make_per_class_ap_figure()
    summary = make_case_study_figure()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
