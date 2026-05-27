# Roadmap: MLPC 2026 Task 4

## Overview

The project moves from a reproducible data foundation to classifier experiments, then converts results into the exact report and slide evidence required by the official Task 4 rubric. The ordering prioritizes correctness first: leakage-safe splits and metrics are established before model sweeps, and the case-study phase is protected because it is both a report section and the assigned slide topic.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions, marked with INSERTED

- [ ] **Phase 1: Project Scaffold and Data Foundation** - Set up the repo structure, load the dataset, aggregate labels, and create the dataset cache. (implemented 2026-05-27; verification blocked by missing local dataset)
- [ ] **Phase 2: Splits, Preprocessing, Metrics, and Baseline** - Establish leakage-safe evaluation inputs and baseline performance.
- [ ] **Phase 3: Logistic Regression Sweep** - Train, tune, save, and visualize the linear model family.
- [ ] **Phase 4: MLX MLP Sweep** - Train, tune, save, and visualize the nonlinear temporal-context model family.
- [ ] **Phase 5: Final Evaluation and Case Study** - Compare final models on test data and generate qualitative case-study evidence.
- [ ] **Phase 6: Report** - Write and compile the 6-page, 2000-word report against the official rubric.
- [ ] **Phase 7: Slides and Submission** - Build the Case Study and Reflection slide deck and perform final submission checks.

## Phase Details

### Phase 1: Project Scaffold and Data Foundation

**Goal**: Create the reproducible project skeleton and produce a verified dataset cache with correct labels.
**Depends on**: Nothing (first phase)
**Requirements**: SETUP-01, SETUP-02, SETUP-03, SETUP-04, DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06
**Success Criteria** (what must be TRUE):

  1. The repository has the planned directories, dependency file, gitignore rules, and central config.
  2. The dataset loader reads metadata, annotations, and feature `.npz` files from a configurable path.
  3. Labels are aggregated with 0.5 binarization and majority vote over valid annotators.
  4. `results/dataset_cache.npz` exists with features, labels, IDs, times, class names, and collector IDs.
  5. Sanity checks for segment count, class order, positive rates, and feature dimensionality are logged.

**Plans**: 3 plans
Plans:
**Wave 1**

- [x] 01-01: Scaffold repository, dependencies, config, gitignore, and results log.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02: Implement metadata/feature loading and feature concatenation.

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 01-03: Implement label aggregation, dataset cache, and sanity checks.

### Phase 2: Splits, Preprocessing, Metrics, and Baseline

**Goal**: Build a leakage-safe evaluation foundation and simple baseline before training learned models.
**Depends on**: Phase 1
**Requirements**: SPLIT-01, SPLIT-02, SPLIT-03, PREP-01, PREP-02, PREP-03, EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):

  1. Train, validation, and test splits are collector-disjoint and cached.
  2. Class-distribution CSV and figure exist for report section 1b.
  3. Scaling is fit only on training rows and temporal-context features respect file boundaries.
  4. Metrics and baseline outputs are saved and can be cited in report section 2.
  5. Optional high-agreement filtering is either implemented or explicitly skipped with a reportable reason.

**Plans**: 3 plans

Plans:

- [ ] 02-01: Implement collector-disjoint splits and class-distribution outputs.
- [ ] 02-02: Implement preprocessing, temporal context, scaler persistence, and optional agreement masks.
- [ ] 02-03: Implement metrics and class-prior baseline outputs.

### Phase 3: Logistic Regression Sweep

**Goal**: Produce a tuned linear one-vs-rest classifier with saved results, model artifact, and report-ready visualizations.
**Depends on**: Phase 2
**Requirements**: LR-01, LR-02, LR-03, LR-04, LR-05
**Success Criteria** (what must be TRUE):

  1. LR training runs for the planned grid or a documented reduced grid if deadline pressure requires it.
  2. `results/lr_sweep.csv` contains one row per configuration with macro AP, micro AP, per-class AP, and runtime.
  3. Best LR model is selected by validation macro AP and saved.
  4. LR validation/test predictions are saved for final evaluation.
  5. At least one LR sweep figure exists for the report.

**Plans**: 2 plans

Plans:

- [ ] 03-01: Implement and run the LR sweep with best-model persistence.
- [ ] 03-02: Generate LR sweep figures and optional ablation notes.

### Phase 4: MLX MLP Sweep

**Goal**: Produce a tuned nonlinear MLP with temporal context, saved results, model state, and report-ready visualizations.
**Depends on**: Phase 2
**Requirements**: MLP-01, MLP-02, MLP-03, MLP-04, MLP-05, MLP-06
**Success Criteria** (what must be TRUE):

  1. MLX imports and the MLP trains with weighted BCE, AdamW, early stopping, and validation macro AP selection.
  2. `results/mlp_sweep.csv` contains one row per configuration run.
  3. Best MLP state and predictions are saved.
  4. MLP architecture/dropout/LR figures and best-model training curves are generated.
  5. Any reduced grid or skipped ablation is documented for transparent reporting.

**Plans**: 3 plans

Plans:

- [ ] 04-01: Implement MLX model, loss, training loop, and prediction utilities.
- [ ] 04-02: Run MLP sweep with early stopping and best-state persistence.
- [ ] 04-03: Generate MLP sweep figures, training curves, and ablation notes.

### Phase 5: Final Evaluation and Case Study

**Goal**: Convert trained models into final test evidence and qualitative analysis for the strongest report/slide section.
**Depends on**: Phases 3 and 4
**Requirements**: FINAL-01, FINAL-02, CASE-01, CASE-02, CASE-03, CASE-04
**Success Criteria** (what must be TRUE):

  1. Baseline, best LR, and best MLP are compared on held-out test data in `results/final_table.csv`.
  2. `results/predictions_test.npz` contains final scores, labels, file IDs, and timing data needed for case studies.
  3. Per-class AP comparison figure exists.
  4. Two non-training case-study files are selected with documented reasons.
  5. Case-study figures and notes explain correct predictions, failures, reliable classes, and difficult classes.

**Plans**: 3 plans

Plans:

- [ ] 05-01: Implement final evaluation, final table, and prediction archive.
- [ ] 05-02: Select success/failure case-study files and generate visualizations.
- [ ] 05-03: Generate per-class analysis and case-study notes for report/slides.

### Phase 6: Report

**Goal**: Produce a concise, complete report PDF that answers every official Task 4 rubric item.
**Depends on**: Phase 5
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04, DOC-05, DOC-06
**Success Criteria** (what must be TRUE):

  1. Report compiles from the required template and stays within 6 pages and 2000 words.
  2. Dataset Preparation covers aggregation, leakage-safe split, class distribution, and preprocessing.
  3. Evaluation covers macro AUPRC, baseline, and best-possible performance under annotator disagreement.
  4. Experiments covers both classifier families, hyperparameter trends, selected settings, and final comparison.
  5. Case Study and Reflection includes two qualitative examples and per-class failure/reliability discussion.
  6. Mandatory LLM/AI disclosure is included.

**Plans**: 3 plans

Plans:

- [ ] 06-01: Create report LaTeX structure and write sections 1-2.
- [ ] 06-02: Write experiment and case-study sections using generated results.
- [ ] 06-03: Compile, trim, proofread, and run rubric/word/page checks.

### Phase 7: Slides and Submission

**Goal**: Produce the assigned slide deck and complete final submission checks with margin before the deadline.
**Depends on**: Phase 6
**Requirements**: SLIDE-01, SLIDE-02, SUBMIT-01
**Success Criteria** (what must be TRUE):

  1. Slide deck has one title slide plus no more than four content slides.
  2. Content focuses on Case Study and Reflection.
  3. Slides include success case, failure case, per-class breakdown, and takeaways.
  4. Report PDF and slide PDF are checked against the rubric and ready for Moodle.
  5. Submission package excludes restricted dataset/audio files.

**Plans**: 2 plans

Plans:

- [ ] 07-01: Build and compile Case Study and Reflection slide deck.
- [ ] 07-02: Perform final sanity checks and prepare Moodle submission files.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Project Scaffold and Data Foundation | 3/3 | Blocked on local dataset | - |
| 2. Splits, Preprocessing, Metrics, and Baseline | 0/3 | Not started | - |
| 3. Logistic Regression Sweep | 0/2 | Not started | - |
| 4. MLX MLP Sweep | 0/3 | Not started | - |
| 5. Final Evaluation and Case Study | 0/3 | Not started | - |
| 6. Report | 0/3 | Not started | - |
| 7. Slides and Submission | 0/2 | Not started | - |
