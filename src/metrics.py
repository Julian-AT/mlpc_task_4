from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def _validate_targets(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true)
    score = np.asarray(y_score, dtype=np.float32)
    if true.shape != score.shape:
        raise ValueError("y_true and y_score must have matching shapes")
    if true.ndim != 2:
        raise ValueError("y_true and y_score must have shape [N, C]")
    return true.astype(np.uint8, copy=False), score


def per_class_ap(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    true, score = _validate_targets(y_true, y_score)
    values = []
    for class_idx in range(true.shape[1]):
        y_c = true[:, class_idx]
        if y_c.sum() == 0:
            values.append(0.0)
        else:
            values.append(float(average_precision_score(y_c, score[:, class_idx])))
    return np.asarray(values, dtype=np.float32)


def macro_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(per_class_ap(y_true, y_score).mean())


def micro_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    true, score = _validate_targets(y_true, y_score)
    if true.sum() == 0:
        return 0.0
    return float(average_precision_score(true.ravel(), score.ravel()))


def best_threshold_f1(y_true_c: np.ndarray, y_score_c: np.ndarray) -> tuple[float, float]:
    true = np.asarray(y_true_c, dtype=np.uint8)
    score = np.asarray(y_score_c, dtype=np.float32)
    if true.ndim != 1 or score.ndim != 1 or true.shape[0] != score.shape[0]:
        raise ValueError("class vectors must be one-dimensional and have matching length")

    candidates = np.unique(score)
    candidates = np.unique(np.concatenate([candidates, np.array([0.0, 1.0], dtype=np.float32)]))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        pred = (score >= threshold).astype(np.uint8)
        f1 = float(f1_score(true, pred, zero_division=0))
        if f1 > best_f1 or (np.isclose(f1, best_f1) and threshold > best_threshold):
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def per_class_f1_at_optimal(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true, score = _validate_targets(y_true, y_score)
    thresholds = np.zeros(true.shape[1], dtype=np.float32)
    f1s = np.zeros(true.shape[1], dtype=np.float32)
    for class_idx in range(true.shape[1]):
        thresholds[class_idx], f1s[class_idx] = best_threshold_f1(true[:, class_idx], score[:, class_idx])
    return thresholds, f1s
