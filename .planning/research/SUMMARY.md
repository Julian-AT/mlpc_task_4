# Project Research Summary

**Project:** MLPC 2026 Task 4: Data Classification  
**Domain:** Coursework sound event detection classification  
**Researched:** 2026-05-27  
**Confidence:** HIGH

## Executive Summary

This is not a product build; it is a tightly scoped university classification assignment. The best approach is a script-based, reproducible Python pipeline that converts the provided aligned features and annotations into model-ready arrays, runs controlled classifier experiments, and produces exactly the evidence needed for the report and slide rubric.

The central risks are leakage, noisy labels, class imbalance, and deadline pressure. The roadmap should therefore put leakage-safe data preparation and metric/baseline work before model sweeps, then reserve enough time for the case-study section because it is both the assigned slide topic and a major report section.

## Key Findings

### Recommended Stack

Use Python with NumPy, pandas, scikit-learn, MLX, matplotlib, and seaborn. Keep code modular but simple, with each script writing concrete artifacts under `results/` for report reuse.

**Core technologies:**
- NumPy/pandas: dataset loading, metadata joins, caches, and tables.
- scikit-learn: splits, scaler, logistic regression, and metrics.
- MLX: Apple-Silicon MLP implementation.
- matplotlib/seaborn: required visualizations.

### Expected Features

**Must have:**
- Label aggregation, collector-disjoint splits, preprocessing, metrics, baseline, LR sweep, MLP sweep, final evaluation, case studies, report, and slides.

**Should have:**
- High-agreement filtering and PCA/MLP ablations only when they do not endanger required deliverables.

**Defer:**
- Extra model families, dashboards, APIs, or production packaging.

### Architecture Approach

Use a linear data pipeline: load/cache dataset, split, preprocess, evaluate baseline, train LR, train MLP, evaluate on test, generate case studies, then write report/slides. The project structure in `MLPC_Task4_PRD.md` is appropriate and should be followed closely.

### Critical Pitfalls

1. **Split leakage** - prevent with collector-disjoint `GroupShuffleSplit` and assertions.
2. **Missing annotator handling** - mask non-participating annotators before majority vote.
3. **Metric mismatch** - use macro AUPRC rather than accuracy.
4. **Runtime overrun** - reduce sweep sizes before sacrificing report/case-study quality.
5. **Formal submission errors** - enforce report/page/word limits, LLM disclosure, and slide topic.

## Implications for Roadmap

### Phase 1: Project Scaffold and Data Foundation
**Rationale:** All later work depends on reliable loading, label aggregation, and cache outputs.  
**Delivers:** Repository structure, config, dataset cache, and label sanity checks.  
**Avoids:** Incorrect label targets and hidden data assumptions.

### Phase 2: Splits, Preprocessing, and Metrics
**Rationale:** Evaluation correctness must be established before model selection.  
**Delivers:** Leakage-safe splits, class distribution evidence, scaler, temporal context, metrics, and baseline.

### Phase 3: Logistic Regression Sweep
**Rationale:** Provides a strong linear model and fast reportable comparison point.  
**Delivers:** Sweep CSV, best model, validation metrics, and LR figures.

### Phase 4: MLX MLP Sweep
**Rationale:** Provides the required different model class and tests nonlinear temporal-context modeling.  
**Delivers:** Sweep CSV, best MLP, curves, and MLP figures.

### Phase 5: Final Evaluation and Case Study
**Rationale:** Converts experiments into held-out evidence and qualitative insight.  
**Delivers:** Final tables, predictions, case-study figures, per-class analysis, and notes.

### Phase 6: Report
**Rationale:** Most grading points are awarded through the report.  
**Delivers:** Compiled report PDF within limits.

### Phase 7: Slides and Submission
**Rationale:** Slides must match the assigned topic and both PDFs must be submitted on time.  
**Delivers:** Case Study and Reflection slide deck plus final sanity checks.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Matches PRD and assignment context |
| Features | HIGH | Directly from official rubric and PRD |
| Architecture | HIGH | Simple pipeline maps to required outputs |
| Pitfalls | HIGH | Derived from official task and Task 3 decisions |

**Overall confidence:** HIGH

### Gaps to Address

- Dataset path: confirm local path or symlink before implementation.
- Task 3 artifacts: locate prior code/report for label aggregation and IoU facts.
- LaTeX template: add report and slide templates before writing final documents.

## Sources

### Primary

- `MLPC_2026S___Data_Classification-3.pdf` - official Task 4 assignment, report rubric, slide limits, dataset description, and LLM disclosure requirement.
- `UE_MLPC_2026_Data_Classification_Task.pdf` - lecture/task context.
- `MLPC_Task4_PRD.md` - project-specific implementation plan and strategic decisions.

---
*Research completed: 2026-05-27*
*Ready for roadmap: yes*
