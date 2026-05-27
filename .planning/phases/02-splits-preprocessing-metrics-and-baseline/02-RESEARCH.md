---
phase: 02-splits-preprocessing-metrics-and-baseline
researched: 2026-05-27T16:07:00Z
status: complete
---

# Phase 02 Research

## Findings

Phase 2 can be implemented as four small script modules:

- `src/splits.py` for cache loading, collector-disjoint splits, class-distribution CSV, and split figure.
- `src/preprocess.py` for train-only scaling, temporal context, and high-agreement helpers.
- `src/metrics.py` for AP/F1 metrics.
- `src/baseline.py` for class-prior baseline evaluation and JSON output.

The real run still depends on Phase 1 producing `results/dataset_cache.npz`, but the behavior can be validated with synthetic fixtures.

## Key Risks

| Risk | Mitigation |
|------|------------|
| Collector leakage | Assert pairwise-empty collector intersections after every split. |
| Scaling leakage | Tests must prove scaler mean is fit from train rows only. |
| Temporal leakage | Context function must check adjacent rows share the same file ID. |
| Misleading generated artifacts | Do not create real results files unless the real cache exists. |
| Constant-score AP misunderstanding | Test and document that AP equals positive prevalence for the baseline. |

## Verification Strategy

- Unit-test split disjointness and reproducibility.
- Unit-test class distribution counts/rates.
- Unit-test train-only scaling.
- Unit-test temporal boundary padding.
- Unit-test per-file IoU and high-agreement masking.
- Unit-test AP, macro AP, micro AP, optimal F1 thresholds, and baseline JSON writing.

---
*Research complete: 2026-05-27*
