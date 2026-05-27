# Phase 1: Project Scaffold and Data Foundation - Research

**Researched:** 2026-05-27
**Status:** Complete

## Research Question

What does Phase 1 need in order to be planned well?

Phase 1 is a data-foundation phase, not a modeling phase. The planner should prioritize deterministic cache schema, reproducible configuration, and label aggregation correctness because every later split, metric, training run, report table, and case-study figure consumes this cache.

## Sources Read

- `.planning/PROJECT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `.planning/phases/01-project-scaffold-and-data-foundation/01-CONTEXT.md`
- `MLPC_Task4_PRD.md`
- `AGENTS.md`

The official assignment PDFs are canonical references, but local `pdftotext` extraction produced no readable text in this environment. Planning should still preserve them as required references for later report/slide work.

## Implementation Strategy

### Project Skeleton

Use the PRD structure directly:

- `src/` for Python modules.
- `tests/` for focused pytest coverage.
- `results/` and `results/figures/` for generated outputs.
- `report/`, `slides/`, and `notebooks/` as placeholders for later phases.
- `requirements.txt` for the scientific Python stack.
- `.gitignore` that excludes `data/`, raw audio, `.npz` caches/features, model binaries, generated PDFs, and local Python artifacts.

Keep this simple. The project is a coursework pipeline; a package framework or experiment platform would add risk without improving the rubric deliverables.

### Central Configuration

`src/config.py` should be the single source of truth for paths and constants:

- `ROOT`, `DATA_DIR`, `FEATURES_DIR`, `METADATA_CSV`, `ANNOTATIONS_CSV`, `RESULTS_DIR`, `FIG_DIR`
- `SEED = 42`
- `TRAIN_FRAC = 0.70`, `VAL_FRAC = 0.15`, `TEST_FRAC = 0.15`
- `ANNOT_BINARIZE_THRESH = 0.5`, `MAJORITY_THRESH = 0.5`
- `TEMPORAL_CONTEXT = 2`
- `HIGH_AGREEMENT_IOU = 0.6`
- `CLASS_NAMES` with the 15 expected classes in alphabetical order
- LR and MLP grids from the PRD, even though Phase 1 does not train models

The default data path should be `ROOT / "data"` so the dataset can be provided as an uncommitted directory or symlink.

### Metadata and Feature Loading

The loader should read:

- `metadata.csv`
- `annotations.csv`
- every `.npz` in `audio_features/`

Useful helper boundaries for `src/data.py`:

- `load_metadata(path: Path | None = None) -> pd.DataFrame`
- `load_annotations(path: Path | None = None) -> pd.DataFrame`
- `iter_feature_files(features_dir: Path | None = None) -> list[Path]`
- `concat_features(npz: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]`
- `aggregate_labels(annotations: np.ndarray) -> np.ndarray`
- `build_dataset(cache_path: Path | None = None) -> dict[str, np.ndarray]`

Feature concatenation should be deterministic. Discover compatible numeric arrays shaped with the segment dimension first, exclude known non-feature keys such as `annotations`, `class_names`, `annotator_ids`, `start_time`, and `end_time`, then sort keys and concatenate arrays after flattening any trailing dimensions. Record the feature key order and final dimensionality in the cache/log so later phases can verify they are using the same feature contract.

### Label Aggregation

The risky part is missing annotator handling. The PRD and context lock this rule:

1. Input `annotations` has shape `[T, C, A]`.
2. Binarize overlap values with `annotations >= 0.5`.
3. Identify valid annotators before voting.
4. Majority vote only over valid annotators.
5. Single-annotator files use that annotator's binarized labels directly.

Recommended implementation:

- Validate the input has three dimensions.
- Treat an annotator slice as invalid when all finite values are zero or all values are NaN. This matches the context decision while staying conservative for inactive annotator slices.
- If every annotator appears invalid, fall back to considering all non-NaN annotators valid and raise/log a warning. The executor should avoid silently producing all-negative labels for a file.
- Compute votes over valid annotators: `labels = (binary[:, :, valid].mean(axis=2) >= 0.5).astype(np.uint8)`.

Tests should cover single annotator, two annotators with ties, three annotators with majority, NaN missing slices, and all-zero inactive slices.

### Cache Schema

`results/dataset_cache.npz` should include at least:

- `X` or `features`: `[N, D]` feature matrix
- `Y` or `labels`: `[N, C]` uint8 label matrix
- `file_ids`: `[N]`
- `collector_ids`: `[N]`
- `segment_indices`: `[N]`
- `start_times`: `[N]`
- `end_times`: `[N]`
- `class_names`: `[C]`
- `feature_keys`: ordered feature key list

Using both long descriptive keys (`features`, `labels`) and common short aliases (`X`, `Y`) is acceptable if documented, but the planner should avoid requiring duplicate data unless there is a concrete downstream benefit.

### Sanity Checks

Phase 1 should log:

- Number of files processed.
- Total segment count.
- Class order.
- Feature dimensionality.
- Ordered feature keys.
- Per-class positive counts/rates.
- Cache path.

Expected segment count is approximately 168,239 from Task 3, but the code should treat this as a sanity note, not a hard-coded requirement. Hard failures should be reserved for shape mismatches, missing required files, class-order mismatch, no feature files, empty feature matrices, or invalid annotation dimensions.

## Validation Architecture

Use pytest for Phase 1 because the important behavior is deterministic Python data transformation.

Recommended test targets:

- `tests/test_data.py::test_aggregate_labels_single_annotator`
- `tests/test_data.py::test_aggregate_labels_majority_vote`
- `tests/test_data.py::test_aggregate_labels_masks_nan_annotator`
- `tests/test_data.py::test_aggregate_labels_masks_all_zero_inactive_annotator`
- `tests/test_data.py::test_concat_features_deterministic_order`
- `tests/test_data.py::test_concat_features_rejects_inconsistent_segment_count`

Recommended commands:

- Quick: `python -m pytest tests/test_data.py`
- Full Phase 1: `python -m pytest`
- Cache smoke test, once the dataset exists: `python -m src.data`

## Planning Implications

Plan 01 should scaffold the repository and test harness. Plan 02 should implement loader and feature concatenation with tests. Plan 03 should implement label aggregation, cache building, and sanity logging with tests and a dataset-dependent smoke command.

The executor should not block on Task 3 code being present. If Task 3 artifacts are absent, implement directly from the PRD. The dataset path remains a known runtime prerequisite for the final cache smoke test.

## Out of Scope for Phase 1

- Collector-disjoint train/validation/test splits.
- StandardScaler fitting.
- Temporal context features.
- Baseline metrics.
- Model training.
- Report figures.
- Slide or report writing.

## RESEARCH COMPLETE
