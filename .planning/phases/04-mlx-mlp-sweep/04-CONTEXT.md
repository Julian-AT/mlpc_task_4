---
phase: 04-mlx-mlp-sweep
gathered: 2026-05-27T17:05:00Z
status: ready-for-planning
source: PRD/roadmap express path; autonomous continuation
---

# Phase 04: MLX MLP Sweep - Context

<domain>
## Phase Boundary

Phase 4 implements the nonlinear MLX MLP classifier family: configurable MLP, weighted BCE, AdamW training, early stopping, sweep CSV, best weights, and prediction archive.

It does not produce the final model comparison or case-study figures.
</domain>

<decisions>
## Implementation Decisions

- Use MLX for the MLP model and optimizer.
- Model outputs logits; metrics use sigmoid probabilities.
- Use weighted binary cross-entropy with clipped positive class weights.
- Select best state by validation macro AP.
- Save sweep rows to `results/mlp_sweep.csv`.
- Save best weights to `results/mlp_best.npz`.
- Append MLP validation/test scores to `results/predictions_test.npz`.
- Real training remains blocked until `results/preprocessed.npz` exists.
</decisions>

<canonical_refs>
## Canonical References

- `.planning/ROADMAP.md` - Phase 4 scope and success criteria.
- `.planning/REQUIREMENTS.md` - MLP-01 through MLP-06.
- `MLPC_Task4_PRD.md` - MLP training contract.
- `src/preprocess.py` - temporal-context input source.
- `src/metrics.py` - validation metric functions.
</canonical_refs>

---
*Phase: 04-mlx-mlp-sweep*
