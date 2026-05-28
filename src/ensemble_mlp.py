"""Build a validation-selected ensemble from saved MLP checkpoints.

This is intentionally simple: score available checkpoints, average probability
outputs, and keep the ensemble only if validation macro AP improves over the best
single MLP.  The final report uses the resulting predictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .final_eval import run_final_evaluation
from .metrics import macro_ap, micro_ap
from .train_lr import load_preprocessed
from .train_mlp import TorchMLP, predict_proba

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch installed
    torch = None


@dataclass
class CandidateScores:
    path: Path
    hidden_dims: list[int]
    dropout: float
    val_scores: np.ndarray
    test_scores: np.ndarray
    val_macro: float
    val_micro: float


def _discover_checkpoints() -> list[Path]:
    paths: list[Path] = []
    for directory in [config.RESULTS_DIR / "mlp_refine_candidates"]:
        if directory.exists():
            paths.extend(sorted(directory.glob("*.pt")))
    best = config.RESULTS_DIR / "mlp_best_torch_cuda.pt"
    if best.exists():
        paths.append(best)
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve())] = path
    return list(unique.values())


def _load_candidate(path: Path, in_dim: int, out_dim: int) -> TorchMLP | None:
    if torch is None or TorchMLP is None or not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is required for ensemble_mlp")
    checkpoint = torch.load(path, map_location="cpu")
    hidden_dims = checkpoint.get("hidden_dims")
    dropout = checkpoint.get("dropout")
    state_dict = checkpoint.get("state_dict")
    if hidden_dims is None or dropout is None or state_dict is None:
        return None
    model = TorchMLP(in_dim, [int(dim) for dim in hidden_dims], out_dim, float(dropout))
    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()
    return model


def _candidate_info(path: Path) -> tuple[list[int], float]:
    checkpoint = torch.load(path, map_location="cpu")
    return [int(dim) for dim in checkpoint["hidden_dims"]], float(checkpoint["dropout"])


def _load_scores(
    checkpoints: list[Path],
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_val: np.ndarray,
) -> list[CandidateScores]:
    rows: list[CandidateScores] = []
    for index, path in enumerate(checkpoints, start=1):
        print(f"mlp_ensemble score {index}/{len(checkpoints)} {path.name}", flush=True)
        model = _load_candidate(path, x_val.shape[1], y_val.shape[1])
        if model is None:
            print("  skipped checkpoint without architecture metadata", flush=True)
            continue
        val_scores = predict_proba(model, x_val, batch_size=4096)
        test_scores = predict_proba(model, x_test, batch_size=4096)
        val_macro = macro_ap(y_val, val_scores)
        val_micro = micro_ap(y_val, val_scores)
        hidden_dims, dropout = _candidate_info(path)
        rows.append(
            CandidateScores(
                path=path,
                hidden_dims=hidden_dims,
                dropout=dropout,
                val_scores=val_scores,
                test_scores=test_scores,
                val_macro=val_macro,
                val_micro=val_micro,
            )
        )
        print(f"  val_macro={val_macro:.6f} val_micro={val_micro:.6f}", flush=True)
        del model
        torch.cuda.empty_cache()
    return sorted(rows, key=lambda item: item.val_macro, reverse=True)


def _score_average(candidates: list[CandidateScores], y_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    val_scores = np.mean([candidate.val_scores for candidate in candidates], axis=0).astype(np.float32)
    test_scores = np.mean([candidate.test_scores for candidate in candidates], axis=0).astype(np.float32)
    return val_scores, test_scores, macro_ap(y_val, val_scores), micro_ap(y_val, val_scores)


def _search_ensembles(
    candidates: list[CandidateScores],
    y_val: np.ndarray,
    max_pool: int = 14,
) -> tuple[list[dict[str, Any]], list[CandidateScores], np.ndarray, np.ndarray, float, float]:
    if not candidates:
        raise ValueError("at least one candidate is required")

    pool = candidates[: min(max_pool, len(candidates))]
    best_members = [pool[0]]
    best_val, best_test, best_macro, best_micro = _score_average(best_members, y_val)
    rows: list[dict[str, Any]] = [
        {
            "strategy": "single",
            "member_count": 1,
            "macro_ap": best_macro,
            "micro_ap": best_micro,
            "members": json.dumps([str(pool[0].path)]),
        }
    ]

    for count in range(2, len(pool) + 1):
        members = pool[:count]
        val_scores, test_scores, val_macro, val_micro = _score_average(members, y_val)
        rows.append(
            {
                "strategy": "top_k",
                "member_count": count,
                "macro_ap": val_macro,
                "micro_ap": val_micro,
                "members": json.dumps([str(candidate.path) for candidate in members]),
            }
        )
        if val_macro > best_macro:
            best_members = members
            best_val = val_scores
            best_test = test_scores
            best_macro = val_macro
            best_micro = val_micro

    selected = [pool[0]]
    remaining = pool[1:]
    while remaining:
        trial_rows: list[tuple[float, float, CandidateScores, np.ndarray, np.ndarray]] = []
        for candidate in remaining:
            members = [*selected, candidate]
            val_scores, test_scores, val_macro, val_micro = _score_average(members, y_val)
            trial_rows.append((val_macro, val_micro, candidate, val_scores, test_scores))
        val_macro, val_micro, candidate, val_scores, test_scores = max(
            trial_rows, key=lambda item: item[0]
        )
        if val_macro <= macro_ap(y_val, np.mean([member.val_scores for member in selected], axis=0)):
            break
        selected.append(candidate)
        remaining = [item for item in remaining if item.path != candidate.path]
        rows.append(
            {
                "strategy": "greedy",
                "member_count": len(selected),
                "macro_ap": val_macro,
                "micro_ap": val_micro,
                "members": json.dumps([str(member.path) for member in selected]),
            }
        )
        if val_macro > best_macro:
            best_members = selected.copy()
            best_val = val_scores
            best_test = test_scores
            best_macro = val_macro
            best_micro = val_micro

    anchor = pool[0]
    for candidate in pool[1:]:
        for anchor_weight in np.linspace(0.1, 0.9, 9):
            other_weight = 1.0 - float(anchor_weight)
            val_scores = (
                anchor.val_scores * float(anchor_weight) + candidate.val_scores * other_weight
            ).astype(np.float32)
            test_scores = (
                anchor.test_scores * float(anchor_weight) + candidate.test_scores * other_weight
            ).astype(np.float32)
            val_macro = macro_ap(y_val, val_scores)
            val_micro = micro_ap(y_val, val_scores)
            rows.append(
                {
                    "strategy": "pair_weighted",
                    "member_count": 2,
                    "macro_ap": val_macro,
                    "micro_ap": val_micro,
                    "anchor_weight": float(anchor_weight),
                    "members": json.dumps([str(anchor.path), str(candidate.path)]),
                }
            )
            if val_macro > best_macro:
                best_members = [anchor, candidate]
                best_val = val_scores
                best_test = test_scores
                best_macro = val_macro
                best_micro = val_micro

    return rows, best_members, best_val, best_test, best_macro, best_micro


def _incumbent_mlp_macro(y_val: np.ndarray) -> float:
    if not config.PREDICTIONS_TEST.exists():
        return -np.inf
    with np.load(config.PREDICTIONS_TEST, allow_pickle=True) as loaded:
        if "mlp_val_scores" not in loaded.files:
            return -np.inf
        return macro_ap(y_val, loaded["mlp_val_scores"])


def _update_predictions(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    class_names: np.ndarray,
) -> None:
    existing: dict[str, np.ndarray] = {}
    if config.PREDICTIONS_TEST.exists():
        with np.load(config.PREDICTIONS_TEST, allow_pickle=True) as loaded:
            existing = {key: loaded[key] for key in loaded.files}
    existing.update(
        {
            "mlp_val_scores": val_scores,
            "mlp_test_scores": test_scores,
            "y_val": y_val,
            "y_test": y_test,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "class_names": np.asarray(class_names, dtype=object),
        }
    )
    np.savez_compressed(config.PREDICTIONS_TEST, **existing)


def run_mlp_ensemble() -> pd.DataFrame:
    dataset = load_preprocessed()
    x = np.asarray(dataset.get("features_context", dataset.get("features_scaled")), dtype=np.float32)
    y = np.asarray(dataset["labels"], dtype=np.uint8)
    class_names = np.asarray(dataset.get("class_names", np.asarray(config.CLASS_NAMES, dtype=object)))
    val_idx = np.asarray(dataset["val_idx"], dtype=np.int64)
    test_idx = np.asarray(dataset["test_idx"], dtype=np.int64)
    y_val = y[val_idx]
    y_test = y[test_idx]

    checkpoints = _discover_checkpoints()
    print(f"mlp_ensemble checkpoints={len(checkpoints)}", flush=True)
    candidates = _load_scores(checkpoints, x[val_idx], x[test_idx], y_val)
    rows, members, val_scores, test_scores, val_macro, val_micro = _search_ensembles(
        candidates,
        y_val,
    )
    frame = pd.DataFrame(rows).sort_values("macro_ap", ascending=False).reset_index(drop=True)
    output = config.RESULTS_DIR / "mlp_ensemble_sweep.csv"
    frame.to_csv(output, index=False)

    incumbent = _incumbent_mlp_macro(y_val)
    print(f"incumbent_mlp_val_macro={incumbent:.6f}", flush=True)
    print(f"best_ensemble_val_macro={val_macro:.6f} val_micro={val_micro:.6f}", flush=True)
    if val_macro > incumbent:
        _update_predictions(val_scores, test_scores, y_val, y_test, val_idx, test_idx, class_names)
        manifest = {
            "macro_ap": val_macro,
            "micro_ap": val_micro,
            "members": [str(member.path) for member in members],
        }
        (config.RESULTS_DIR / "mlp_ensemble_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        final_table = run_final_evaluation()
        print("\nFinal table after MLP ensemble promotion:")
        print(final_table[["model", "macro_ap", "micro_ap"]].to_string(index=False))
    else:
        print("No MLP ensemble beat incumbent; predictions_test.npz was left unchanged.")

    print("\nMLP ensemble top 8:")
    print(
        frame[["strategy", "member_count", "macro_ap", "micro_ap"]]
        .head(8)
        .to_string(index=False)
    )
    return frame


if __name__ == "__main__":
    run_mlp_ensemble()
