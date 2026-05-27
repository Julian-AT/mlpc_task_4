# Feature Research: MLPC 2026 Task 4

**Project:** MLPC 2026 Task 4: Data Classification  
**Domain:** Coursework sound event detection classification  
**Researched:** 2026-05-27  
**Confidence:** HIGH

## Must Have

- Dataset loader that reads metadata, annotations, and every `audio_features/*.npz` file.
- Label aggregation from `[T, C, A]` overlap arrays to `[T, C]` binary labels.
- Collector-disjoint train/validation/test split with explicit leakage assertions.
- Class-distribution table or visualization across splits.
- Train-only standardization and MLP temporal context features.
- Macro AUPRC, micro AUPRC, per-class AP, F1/threshold analysis, and a class-prior baseline.
- Logistic regression sweep with systematic hyperparameter variation and visualization.
- MLX MLP sweep with systematic hyperparameter variation and visualization.
- Final held-out comparison against baseline.
- Two non-training file case studies with spectrogram, ground truth, predictions, and written interpretation.
- Report and slide deck matching the official page, word, slide, and topic constraints.
- LLM and AI tool disclosure in the report.

## Should Have

- High-agreement filter ablation based on Task 3 per-file/class IoU.
- PCA ablation for logistic regression if time permits.
- Focal-loss or no-temporal-context MLP ablation only if time permits.
- `results/log.md` checkpoint log after each phase.

## Defer or Avoid

- More than two main model families unless all required deliverables are already safe.
- Production packaging, APIs, dashboards, or interactive tooling.
- Large manual notebooks as the main workflow; notebooks can be scratch only.
- Complex multi-label stratification if it conflicts with collector-disjoint leakage prevention.
