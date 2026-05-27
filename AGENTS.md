<!-- GSD:project-start source:PROJECT.md -->
## Project

**MLPC 2026 Task 4: Data Classification**

This is a university classification project for the MLPC 2026 KIAL sound event detection dataset. The work must prepare segment-level labels, train and tune at least two classifier families, compare them against a simple baseline, and produce a short report plus slide deck that match the official Task 4 rubric.

The implementation should be clean, reproducible, and easy to explain in the report. It should follow the existing PRD and assignment PDFs closely, reusing Task 3 decisions where available instead of adding unnecessary research or product-style scope.

**Core Value:** Deliver a correct, well-evidenced Task 4 submission that answers every rubric question and can be reconstructed from the code, results, figures, report, and slides.

### Constraints

- **Deadline**: Submit report and slides before May 28, 2026, 23:59 - completeness beats optional experiments.
- **Page and word limits**: Report must stay within 6 pages and 2000 words - figures and prose must be selected tightly.
- **Rubric coverage**: Every question under Dataset Preparation, Evaluation, Experiments, and Case Study and Reflection must be answered.
- **Slides topic**: Slides must cover Case Study and Reflection because the group name starts with A.
- **Dataset license**: Do not commit, redistribute, or publish course dataset files or raw audio.
- **Split discipline**: Splits must avoid leakage by keeping collectors disjoint across train, validation, and test.
- **Metric discipline**: Macro AUPRC is the primary selection metric because the task is multi-label and imbalanced.
- **Scope discipline**: Use the PRD as the plan; avoid overengineering beyond clean scripts, reproducible outputs, and report-ready figures.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommendation
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
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
