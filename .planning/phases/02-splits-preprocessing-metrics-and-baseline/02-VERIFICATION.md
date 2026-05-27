---
phase: 02-splits-preprocessing-metrics-and-baseline
verified: 2026-05-27T16:25:00Z
status: gaps_found
score: code paths verified; real artifacts blocked
---

# Phase 02 Verification Report

## Goal Achievement

Phase 2 code paths are implemented and covered by synthetic tests. Real result artifacts cannot be generated because Phase 1 still lacks the licensed Task 4 dataset cache.

## Verified

| Check | Status | Evidence |
|-------|--------|----------|
| Collector-disjoint split generation | VERIFIED | `tests/test_phase2.py` checks reproducibility and pairwise-empty collector sets. |
| Class-distribution table and figure writer | VERIFIED SYNTHETICALLY | Temporary CSV/figure outputs are created from a synthetic cache. |
| Train-only scaling | VERIFIED | Test checks scaler mean comes only from train rows. |
| Temporal context boundary padding | VERIFIED | Test checks no context crosses file IDs. |
| High-agreement helpers | VERIFIED | Test checks pairwise IoU and high-agreement masks. |
| Metrics and baseline | VERIFIED | Tests cover AP, macro/micro AP, optimal F1, class-prior scores, and baseline JSON writing. |

## Blocked Real Artifacts

| Artifact | Status | Blocker |
|----------|--------|---------|
| `results/splits.npz` | BLOCKED | `results/dataset_cache.npz` missing. |
| `results/class_distribution.csv` | BLOCKED | `results/dataset_cache.npz` missing. |
| `results/figures/class_dist_across_splits.png` | BLOCKED | `results/dataset_cache.npz` missing. |
| `results/preprocessed.npz` | BLOCKED | `results/dataset_cache.npz` and splits missing. |
| `results/scaler.joblib` | BLOCKED | `results/dataset_cache.npz` and splits missing. |
| `results/baseline.json` | BLOCKED | `results/dataset_cache.npz` and splits missing. |

## Commands Run

```bash
python -m pip install -r requirements.txt
python -m pytest tests/test_phase2.py -q --tb=short
python -m pytest -q --tb=short
python -m compileall src
```

Results:

- `tests/test_phase2.py`: 9 passed.
- Full suite: 21 passed.
- `compileall`: passed.

## Next Action

If the real dataset becomes available, run:

```bash
python -m src.data
python -m src.splits
python -m src.preprocess
python -m src.baseline
python -m pytest -q --tb=short
```

Until then, downstream model phases can be implemented against these module contracts but cannot produce real model results.
