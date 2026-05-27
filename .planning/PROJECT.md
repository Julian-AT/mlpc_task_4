# MLPC 2026 Task 4: Data Classification

## What This Is

This is a university classification project for the MLPC 2026 KIAL sound event detection dataset. The work must prepare segment-level labels, train and tune at least two classifier families, compare them against a simple baseline, and produce a short report plus slide deck that match the official Task 4 rubric.

The implementation should be clean, reproducible, and easy to explain in the report. It should follow the existing PRD and assignment PDFs closely, reusing Task 3 decisions where available instead of adding unnecessary research or product-style scope.

## Core Value

Deliver a correct, well-evidenced Task 4 submission that answers every rubric question and can be reconstructed from the code, results, figures, report, and slides.

## Requirements

### Validated

(None yet - ship to validate)

### Active

- [ ] Prepare the dataset for classification using majority-vote label aggregation over aligned annotation arrays.
- [ ] Create collector-disjoint train, validation, and test splits with leakage checks and class-distribution evidence.
- [ ] Implement train-only preprocessing, temporal context for the MLP, and scoped noise-reduction ablations.
- [ ] Define and compute macro AUPRC, baseline performance, per-class metrics, and threshold-based F1 analysis.
- [ ] Train and tune logistic regression and an MLX MLP with systematic hyperparameter sweeps.
- [ ] Compare final models to baseline on held-out test data and save tables/figures for the report.
- [ ] Build two qualitative case studies on non-training files with spectrograms, ground truth, predictions, and reflection.
- [ ] Produce a max 6 page, max 2000 word report that covers all four rubric sections and includes LLM disclosure.
- [ ] Produce a title slide plus 4 content slides on Case Study and Reflection for the team topic.
- [ ] Keep the project reproducible and course-compliant by excluding datasets, generated caches, large outputs, and licensed audio from git.

### Out of Scope

- Building a deployable SED product or API - this is a coursework analysis and submission project.
- Creating a large experiment platform - small scripts and cached outputs are enough.
- Exhaustive model search beyond the planned LR and MLP sweeps - cut scope if runtime threatens the deadline.
- Publishing or redistributing dataset contents - the course PDF restricts dataset use to the course.
- Re-justifying all Task 3 decisions from scratch - cite and reuse the Task 3 exploration where applicable.

## Context

- Official deadline: Thursday, May 28, 2026, 23:59.
- Team: Autonomous Pipelines, Julian Schmidt and Paul Breburda. The first letter "A" assigns the slide topic to Case Study and Reflection.
- Report deliverable: PDF, max 6 pages and max 2000 words, worth up to 37 points.
- Slide deliverable: PDF, max 4 content slides plus one title slide, worth up to 3 points.
- Dataset domain: domestic sound event detection for Kepler Intelligent Audio Labs (KIAL).
- Dataset files: `metadata.csv`, `annotations.csv`, and `audio_features/*.npz`.
- Feature files contain one-second windows with 0.5 second hop, acoustic feature aggregations, `annotations` shaped `[T, C, A]`, `start_time`, `end_time`, `class_names`, `annotator_ids`, and related metadata.
- Raw waveforms are optional and only needed for listening during case study; the case study can use `melspect_mean` from feature files for visualization.
- PRD file: `MLPC_Task4_PRD.md`.
- Official assignment PDFs: `MLPC_2026S___Data_Classification-3.pdf` and `UE_MLPC_2026_Data_Classification_Task.pdf`.
- Hardware target: MacBook Pro M-series with 24 GB unified memory and MLX.
- Planned stack: Python, NumPy, pandas, scikit-learn, MLX, matplotlib, seaborn, tqdm, librosa/soundfile for case study support, and pyarrow if useful.

## Constraints

- **Deadline**: Submit report and slides before May 28, 2026, 23:59 - completeness beats optional experiments.
- **Page and word limits**: Report must stay within 6 pages and 2000 words - figures and prose must be selected tightly.
- **Rubric coverage**: Every question under Dataset Preparation, Evaluation, Experiments, and Case Study and Reflection must be answered.
- **Slides topic**: Slides must cover Case Study and Reflection because the group name starts with A.
- **Dataset license**: Do not commit, redistribute, or publish course dataset files or raw audio.
- **Split discipline**: Splits must avoid leakage by keeping collectors disjoint across train, validation, and test.
- **Metric discipline**: Macro AUPRC is the primary selection metric because the task is multi-label and imbalanced.
- **Scope discipline**: Use the PRD as the plan; avoid overengineering beyond clean scripts, reproducible outputs, and report-ready figures.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use majority vote after 0.5 overlap binarization for labels | Matches Task 3 decision, is interpretable, and favors precision over noisy union labels | - Pending |
| Split by `collector_id` with 70/15/15 train/val/test | Prevents leakage through recording device, environment, and collector annotation style | - Pending |
| Use macro AUPRC as primary model-selection metric | Handles multi-label imbalance and gives rare classes equal weight | - Pending |
| Compare logistic regression and MLX MLP | Satisfies "different model classes" while keeping a clear linear-vs-nonlinear narrative | - Pending |
| Make Case Study and Reflection the strongest visual section | It is the assigned slide topic and worth 10 report points | - Pending |
| Keep implementation clean but not overbuilt | Coursework deadline and rubric require reliable results, not a production platform | - Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-27 after initialization*
