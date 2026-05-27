---
phase: 02-splits-preprocessing-metrics-and-baseline
plan: 02-03
status: implemented
requirements-implemented: [EVAL-01, EVAL-02, EVAL-03]
requirements-blocked: [EVAL-02, EVAL-03, EVAL-04]
---

# Plan 02-03 Summary

Implemented `src/metrics.py` with per-class AP, macro AP, micro AP, optimal per-class F1 threshold search, and per-class F1 helpers.

Implemented `src/baseline.py` with class-prior constant-score baseline evaluation and JSON output writing.

Synthetic tests verify AP/F1 behavior, baseline score shape, and the constant-score AP equals evaluation positive prevalence.

Real baseline metrics remain blocked until the dataset cache and splits exist. Report prose for metric justification remains in the report phase.
