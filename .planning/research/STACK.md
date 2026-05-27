# Stack Research: MLPC 2026 Task 4

**Project:** MLPC 2026 Task 4: Data Classification  
**Domain:** Coursework sound event detection classification  
**Researched:** 2026-05-27  
**Confidence:** HIGH for assignment constraints; MEDIUM for exact package versions until the environment is installed and verified.

## Recommendation

Use a small Python scientific stack centered on NumPy, pandas, scikit-learn, MLX, matplotlib, and seaborn. This matches the PRD, the official task framing, and the MacBook M-series hardware target. Keep scripts simple and report-driven: each script should produce concrete cached outputs, CSV tables, JSON summaries, and figures needed by the report or slides.

## Core Technologies

- Python: orchestration and scripts.
- NumPy: loading `.npz` feature files, array operations, label aggregation, and metric inputs.
- pandas: metadata joins, class distribution tables, sweep CSVs, and report tables.
- scikit-learn: `GroupShuffleSplit`, `StandardScaler`, logistic regression, baseline utilities, and metrics.
- MLX: MLP implementation optimized for Apple Silicon.
- matplotlib/seaborn: class distribution, hyperparameter, final comparison, and case-study figures.
- librosa/soundfile: optional raw-audio support for qualitative listening and diagnostics.
- joblib: scikit-learn model persistence.

## Scope Controls

- Prefer standalone scripts over a large framework.
- Cache expensive intermediate outputs under `results/`.
- Do not commit `data/`, raw audio, large `.npz` caches, model binaries, or generated PDFs unless explicitly needed.
- If runtime becomes tight, reduce MLP grid size before sacrificing report/case-study quality.

## Validation Needed

- Verify exact feature dimensionality from the first `.npz`.
- Verify MLX imports and trains on the local machine.
- Confirm the dataset path and Task 3 code/report availability before implementation.
