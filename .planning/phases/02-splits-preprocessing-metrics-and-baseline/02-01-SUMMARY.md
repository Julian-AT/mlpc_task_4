---
phase: 02-splits-preprocessing-metrics-and-baseline
plan: 02-01
status: implemented
requirements-implemented: [SPLIT-01, SPLIT-02, SPLIT-03]
requirements-blocked: [SPLIT-03]
---

# Plan 02-01 Summary

Implemented `src/splits.py` with dataset cache loading, two-stage `GroupShuffleSplit`, collector leakage assertions, split persistence, class-distribution table generation, and class-distribution plotting.

Synthetic tests verify disjoint collectors, deterministic splits, class distribution rates, and temporary output writing.

Real split/distribution artifacts remain blocked until `results/dataset_cache.npz` exists.
