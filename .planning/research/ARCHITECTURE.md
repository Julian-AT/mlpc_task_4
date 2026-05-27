# Architecture Research: MLPC 2026 Task 4

**Project:** MLPC 2026 Task 4: Data Classification  
**Domain:** Coursework sound event detection classification  
**Researched:** 2026-05-27  
**Confidence:** HIGH

## Suggested Project Structure

The PRD's script-oriented structure is appropriate:

- `src/config.py`: paths, constants, class names, seeds, split ratios, and hyperparameter grids.
- `src/data.py`: metadata loading, feature concatenation, label aggregation, dataset cache.
- `src/splits.py`: collector-disjoint split creation and class-distribution outputs.
- `src/preprocess.py`: standardization, temporal context, optional high-agreement masks.
- `src/metrics.py`: AP, macro/micro AP, optimal F1 thresholds, confusion utilities.
- `src/baseline.py`: class-prior baseline.
- `src/train_lr.py`: logistic regression sweep and best model persistence.
- `src/mlp_model.py` and `src/train_mlp.py`: MLX MLP and sweep loop.
- `src/evaluate.py`: final test-set evaluation and comparison figures.
- `src/case_study.py`: case-study file selection and visualization.
- `src/viz.py`: shared plotting helpers.
- `report/` and `slides/`: LaTeX sources.
- `results/`: generated caches, CSVs, models, predictions, notes, and figures.

## Data Flow

1. Load feature files and metadata.
2. Aggregate labels and concatenate features into `results/dataset_cache.npz`.
3. Split by `collector_id` into `results/splits.npz`.
4. Fit preprocessing only on training rows, then transform validation/test.
5. Run baseline, LR sweep, and MLP sweep using validation macro AUPRC for selection.
6. Evaluate final selected models on test set.
7. Generate report and slide figures from saved predictions, tables, and notes.

## Build Order

Build in the same order as the data flow. Do not start report-only polishing until baseline/splits and at least one model path are producing real outputs, but start drafting static report sections while long sweeps run.
