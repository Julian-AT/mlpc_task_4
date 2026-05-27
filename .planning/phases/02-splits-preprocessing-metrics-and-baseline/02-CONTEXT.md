---
phase: 02-splits-preprocessing-metrics-and-baseline
gathered: 2026-05-27T16:05:00Z
status: ready-for-planning
source: PRD/roadmap express path; user instructed autonomous continuation
---

# Phase 02: Splits, Preprocessing, Metrics, and Baseline - Context

<domain>
## Phase Boundary

Phase 2 establishes the evaluation foundation used by all classifiers:

- collector-disjoint train/validation/test splits;
- class-distribution table and figure;
- train-only standardization;
- temporal-context features for MLP inputs;
- optional high-agreement helpers;
- reusable metrics;
- class-prior baseline metrics.

It does not train logistic regression or MLP models and does not write report prose.
</domain>

<decisions>
## Implementation Decisions

### Split Discipline
- Splits must be grouped by `collector_id`, never by row alone.
- Use 70/15/15 train/validation/test fractions with `random_state=42`.
- Split generation must assert pairwise-empty collector intersections.
- `results/splits.npz` must store segment indices for downstream reproducibility.

### Preprocessing Discipline
- `StandardScaler` must be fit on training rows only, then applied to validation/test rows.
- Temporal context concatenates frames `t-2` through `t+2`.
- Temporal context must zero-pad at file boundaries rather than crossing into another file.

### Metrics Discipline
- Macro AUPRC remains the primary selection metric.
- Metrics must include per-class AP, macro AP, micro AP, globally usable F1 threshold helpers, per-class optimal thresholds, and per-class F1.
- Constant-score class-prior baseline must be saved as JSON when the real split/cache exists.

### Dataset Blocker Handling
- The real dataset cache is not locally available yet.
- Implement and test Phase 2 code with synthetic fixtures now.
- Do not fabricate `results/splits.npz`, `results/class_distribution.csv`, or `results/baseline.json`; generate them only once `results/dataset_cache.npz` exists.
</decisions>

<canonical_refs>
## Canonical References

- `.planning/ROADMAP.md` - Phase 2 goal, success criteria, and plan list.
- `.planning/REQUIREMENTS.md` - SPLIT, PREP, and EVAL requirements.
- `.planning/phases/01-project-scaffold-and-data-foundation/01-VERIFICATION.md` - explains the missing real dataset cache blocker.
- `MLPC_Task4_PRD.md` - detailed split, preprocessing, metrics, and baseline contracts.
- `src/data.py` - dataset cache schema consumed by Phase 2.
</canonical_refs>

<specifics>
## Specific Ideas

- Use `sklearn.model_selection.GroupShuffleSplit` for grouped splitting.
- Use `sklearn.preprocessing.StandardScaler` for train-only scaling.
- Use `sklearn.metrics.average_precision_score` and `f1_score` for evaluation.
- Use synthetic tests to cover behavior while real course data is unavailable.
</specifics>

<deferred>
## Deferred Ideas

- Real split/distribution/baseline artifact generation is deferred until `results/dataset_cache.npz` exists.
- Report wording for metric choice is deferred to the report phase, but metric outputs must support it.
</deferred>

---
*Phase: 02-splits-preprocessing-metrics-and-baseline*
*Context gathered: 2026-05-27*
