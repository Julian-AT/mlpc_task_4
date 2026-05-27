# Phase 1: Project Scaffold and Data Foundation - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 1 creates the reproducible Python project skeleton and the first verified dataset cache. It covers repository structure, dependency documentation, central configuration, metadata/feature loading, feature concatenation, label aggregation, cache writing, and sanity-check logging. It does not create splits, preprocessing artifacts, baselines, models, figures, report prose, or slides.

</domain>

<decisions>
## Implementation Decisions

### Dataset and Task 3 Inputs
- **D-01:** Use the PRD and official assignment PDFs as the source of truth for Phase 1. No additional user discussion is needed before planning.
- **D-02:** The dataset path must remain configurable through `src/config.py`, defaulting to a local `data/` directory or symlink that is not committed.
- **D-03:** If Task 3 code or report artifacts are present locally, downstream agents may reuse them for label aggregation facts and sanity-check expectations. If they are not present, reimplement the Phase 1 loader and aggregation behavior directly from `MLPC_Task4_PRD.md`.
- **D-04:** Raw dataset files, raw audio, `.npz` feature files, large generated caches, and model artifacts must not be committed.

### Feature Concatenation Contract
- **D-05:** Implement feature concatenation in `src/data.py` as a deterministic, inspectable function. It should discover compatible numeric feature arrays from each `.npz`, but the selected key order must be stable and recorded.
- **D-06:** Validate that every processed file exposes the same final feature dimension and compatible class order. Fail loudly on incompatible feature schemas instead of silently dropping or reordering data.
- **D-07:** Record the final feature dimensionality in Phase 1 sanity-check output and in `results/log.md` so later phases can cite or verify it.

### Missing Annotator Masking
- **D-08:** Aggregate labels using the locked Task 3/PRD rule: binarize `annotations[t,c,a] >= 0.5`, mask annotators who did not annotate the file, then majority vote over valid annotators.
- **D-09:** Treat NaN-only annotator slices and all-zero inactive annotator slices as missing when the file-level evidence indicates the annotator did not annotate that file. Do not count missing annotators as negative votes.
- **D-10:** Single-annotator files should use that annotator's binarized labels directly.
- **D-11:** Assert or log class-name order against the expected 15-class alphabetical list from `src/config.py`.

### Cache and Verification Scope
- **D-12:** Phase 1 must produce `results/dataset_cache.npz` containing features, labels, file IDs, collector IDs, segment indices, start times, end times, and class names.
- **D-13:** Phase 1 must also create the planned directories, dependency file, `.gitignore`, `src/config.py`, `src/data.py`, and `results/log.md`.
- **D-14:** Sanity checks must log segment count, class order, positive rates, and feature dimensionality. Expected segment count is approximately 168,239 based on Task 3, but the implementation should report the actual loaded count.
- **D-15:** Keep Phase 1 focused on data foundation only. Class-distribution CSVs/figures across train/validation/test belong to Phase 2 after collector-disjoint splits exist.

### the agent's Discretion
- Downstream agents may choose small helper functions and CLI entrypoint names that match the simple script-based style in the PRD.
- Downstream agents may add focused unit or smoke tests when they reduce risk around label aggregation, feature consistency, or cache schema.
- Downstream agents should prefer clarity and reproducibility over generalized pipeline infrastructure.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Control
- `.planning/PROJECT.md` - Core value, constraints, locked project decisions, and current blockers.
- `.planning/REQUIREMENTS.md` - Phase 1 requirements `SETUP-01` through `SETUP-04` and `DATA-01` through `DATA-06`.
- `.planning/ROADMAP.md` - Phase 1 goal, success criteria, and planned plan breakdown.
- `.planning/STATE.md` - Current planning position and open blockers about dataset path, Task 3 artifacts, and deadline pressure.

### Task Specification and PRD
- `MLPC_Task4_PRD.md` - Primary implementation plan for project structure, config constants, loader behavior, label aggregation, and sanity checks.
- `MLPC_2026S___Data_Classification-3.pdf` - Official Task 4 assignment, report rubric, dataset framing, and deliverable constraints.
- `UE_MLPC_2026_Data_Classification_Task.pdf` - Official task/lecture context for the classification assignment.

### Research Summary
- `.planning/research/STACK.md` - Confirmed technology stack and validation gaps.
- `.planning/research/SUMMARY.md` - High-level project risks and phase rationale.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- No implementation files exist yet. The reusable assets are the PRD, roadmap, requirements, official PDFs, and research summaries.

### Established Patterns
- Use standalone Python scripts and small modules under `src/`.
- Use `results/` for generated caches, logs, tables, and later figures.
- Keep large/generated/course-restricted files out of git.
- Centralize paths, constants, seeds, class names, label thresholds, and future grids in `src/config.py`.

### Integration Points
- `src/config.py` will be the shared configuration source for all later phases.
- `src/data.py` will produce `results/dataset_cache.npz`, which Phase 2 splits/preprocessing and later training phases consume.
- `results/log.md` will accumulate one-line verification notes after implementation phases.

</code_context>

<specifics>
## Specific Ideas

No extra user-specific preferences were added during discussion. The user explicitly approved skipping discussion where the roadmap and PRD are clear.

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope.

</deferred>

---

*Phase: 1-Project Scaffold and Data Foundation*
*Context gathered: 2026-05-27*
