---
phase: 03-logistic-regression-sweep
researched: 2026-05-27T16:42:00Z
status: complete
---

# Phase 03 Research

## Findings

The LR sweep can be a single script module, `src/train_lr.py`, with:

- a preprocessed-cache loader;
- `fit_one` for one configuration;
- deterministic grid iteration from `config.LR_GRID`;
- `sweep_lr` to train all configurations, select by validation macro AP, save CSV/model/predictions;
- `plot_lr_sweep` to generate a report-ready heatmap.

Synthetic tests can verify the contract without running the real course dataset.

## Verification Strategy

- Fit one small multi-label model and check score shape/metrics.
- Run a tiny synthetic sweep and check CSV, model, and prediction archive are written.
- Generate a heatmap from a small sweep CSV.

---
*Research complete: 2026-05-27*
