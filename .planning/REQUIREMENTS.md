# Requirements: MLPC 2026 Task 4

**Defined:** 2026-05-27  
**Core Value:** Deliver a correct, well-evidenced Task 4 submission that answers every rubric question and can be reconstructed from the code, results, figures, report, and slides.

## v1 Requirements

### Setup

- [x] **SETUP-01**: Repository contains the planned project structure (`src/`, `results/`, `report/`, `slides/`, and support files) without committing course dataset files or generated large artifacts.
- [x] **SETUP-02**: Python environment dependencies are documented in `requirements.txt` and cover NumPy, pandas, scikit-learn, matplotlib, seaborn, tqdm, MLX, librosa/soundfile, pyarrow if used, and joblib if used.
- [x] **SETUP-03**: `src/config.py` defines paths, seeds, split fractions, label thresholds, class names, and hyperparameter grids in one place.
- [x] **SETUP-04**: `results/log.md` records one-line verification notes after each implementation phase.

### Data Preparation

- [x] **DATA-01**: Dataset loader reads `metadata.csv`, `annotations.csv`, and all `audio_features/*.npz` files from a configurable data path or symlink.
- [x] **DATA-02**: Feature concatenation combines all available base feature aggregations consistently and records the final feature dimensionality.
- [x] **DATA-03**: Label aggregation converts each `.npz` `annotations` array from `[T, C, A]` overlap values to `[T, C]` binary labels using 0.5 binarization and majority voting over valid annotators.
- [x] **DATA-04**: Label aggregation correctly handles single-annotator files and masks annotators that did not annotate a file instead of counting them as negative votes.
- [ ] **DATA-05**: Dataset cache stores features, labels, file IDs, collector IDs, segment indices, start times, end times, and class names under `results/dataset_cache.npz`.
- [ ] **DATA-06**: Data preparation sanity checks print or log segment count, class names/order, positive rates, and feature dimensionality.

### Splits and Preprocessing

- [ ] **SPLIT-01**: Train/validation/test splits use `collector_id` groups with 70/15/15 proportions and `random_state=42`.
- [ ] **SPLIT-02**: Split generation asserts pairwise-empty collector intersections across train, validation, and test.
- [ ] **SPLIT-03**: Class-distribution table and visualization compare positive counts/rates across train, validation, and test.
- [ ] **PREP-01**: `StandardScaler` is fit on training rows only and then applied to validation and test rows.
- [ ] **PREP-02**: Temporal-context features for MLP concatenate frames `t-2` through `t+2` while zero-padding at file boundaries.
- [ ] **PREP-03**: Optional per-file/per-class IoU and high-agreement masks are implemented or explicitly skipped with a reportable reason.

### Evaluation

- [ ] **EVAL-01**: `src/metrics.py` computes per-class AP, macro AP, micro AP, globally selected F1 threshold, per-class optimal thresholds, and per-class F1.
- [ ] **EVAL-02**: Class-prior baseline scores each class by its training prevalence and saves validation/test baseline metrics to `results/baseline.json`.
- [ ] **EVAL-03**: Evaluation outputs include per-class AP and threshold/confusion data needed for the report and case-study reflection.
- [ ] **EVAL-04**: Metric choice is documented for the report: macro AUPRC is primary because the task is multi-label and imbalanced.

### Logistic Regression

- [ ] **LR-01**: Logistic regression is implemented as one-vs-rest multi-label classification with appropriate solvers for L1 and L2 penalties.
- [ ] **LR-02**: LR sweep evaluates the planned grid over `C`, `penalty`, and `class_weight`, saving one row per configuration to `results/lr_sweep.csv`.
- [ ] **LR-03**: Best LR model is selected by validation macro AP, saved to `results/lr_best.pkl`, and used to produce validation/test scores.
- [ ] **LR-04**: LR sweep visualization is generated for the report.
- [ ] **LR-05**: Optional LR PCA and high-agreement-filter ablations are implemented only if time permits, otherwise clearly skipped.

### MLP

- [ ] **MLP-01**: MLX MLP model supports configurable hidden dimensions and dropout with 15 sigmoid outputs.
- [ ] **MLP-02**: MLP training uses weighted binary cross-entropy with softened positive class weights, AdamW, early stopping on validation macro AP, and learning-rate warmup/cosine decay.
- [ ] **MLP-03**: MLP sweep evaluates the planned grid or a time-reduced grid if needed, saving one row per configuration to `results/mlp_sweep.csv`.
- [ ] **MLP-04**: Best MLP state is selected by validation macro AP, saved, and used to produce validation/test scores.
- [ ] **MLP-05**: MLP figures show architecture/dropout/learning-rate effects and training curves for the best model.
- [ ] **MLP-06**: Optional focal-loss or no-temporal-context ablations are implemented only if time permits, otherwise clearly skipped.

### Final Evaluation and Case Study

- [ ] **FINAL-01**: Final evaluation compares baseline, best LR, and best MLP on the held-out test split using macro AP, micro AP, and per-class AP.
- [ ] **FINAL-02**: `results/final_table.csv`, `results/predictions_test.npz`, and `results/figures/per_class_ap_comparison.png` are generated.
- [ ] **CASE-01**: Case-study selection picks two non-training test files: one informative success case and one informative failure case.
- [ ] **CASE-02**: Each case-study figure shows spectrogram, ground-truth tracks, predicted probabilities, and thresholds or equivalent label visualization.
- [ ] **CASE-03**: Per-class analysis identifies reliable classes, difficult classes, and systematic confusions such as door/wardrobe and bell/phone where supported by results.
- [ ] **CASE-04**: `results/case_study_notes.md` summarizes correct predictions, errors, and links to Task 3 agreement/noise findings for report prose.

### Report and Slides

- [ ] **DOC-01**: Report PDF compiles from the Moodle/Task 3 LaTeX template and stays within 6 pages and 2000 words.
- [ ] **DOC-02**: Report section 1 answers label aggregation, data split/leakage/class-distribution, and preprocessing questions with evidence.
- [ ] **DOC-03**: Report section 2 justifies macro AUPRC, reports baseline performance, and discusses best-possible performance under annotator disagreement.
- [ ] **DOC-04**: Report section 3 explains both classifier families, hyperparameter sweeps, selected settings, visualizations, and final comparison to baseline.
- [ ] **DOC-05**: Report section 4 presents two case studies and reflects on failure modes and reliable/unreliable classes.
- [ ] **DOC-06**: Report includes the mandatory Disclosure of LLM and AI Tool Use.
- [ ] **SLIDE-01**: Slide deck contains one title slide plus at most four content slides.
- [ ] **SLIDE-02**: Slide deck covers Case Study and Reflection, matching the group-name topic assignment.
- [ ] **SUBMIT-01**: Final report PDF and slide PDF are checked against the rubric and ready for Moodle submission before the deadline.

## v2 Requirements

No v2 scope is planned. Future challenge-task work should be tracked separately after Task 4 submission.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Production SED API or app | Not required by the university task and would distract from grading criteria |
| More than two main classifier families | The assignment requires at least two; extra models are lower priority than report completeness |
| Full multi-label stratified group split implementation | Collector-disjoint leakage prevention is higher priority; distribution is verified post-hoc |
| Redistribution of dataset/audio | Course dataset license restricts use to the course |
| Large committed caches/models | Keep repository lightweight and avoid committing generated artifacts unless explicitly needed |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SETUP-01 | Phase 1 | Complete |
| SETUP-02 | Phase 1 | Complete |
| SETUP-03 | Phase 1 | Complete |
| SETUP-04 | Phase 1 | Complete |
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Blocked: local dataset required |
| DATA-06 | Phase 1 | Blocked: local dataset required |
| SPLIT-01 | Phase 2 | Implemented; real run blocked by DATA-05 |
| SPLIT-02 | Phase 2 | Implemented; real run blocked by DATA-05 |
| SPLIT-03 | Phase 2 | Implemented; real outputs blocked by DATA-05 |
| PREP-01 | Phase 2 | Implemented; real output blocked by DATA-05 |
| PREP-02 | Phase 2 | Implemented; real output blocked by DATA-05 |
| PREP-03 | Phase 2 | Implemented helper; real use blocked by DATA-05 |
| EVAL-01 | Phase 2 | Implemented |
| EVAL-02 | Phase 2 | Implemented; real output blocked by DATA-05 |
| EVAL-03 | Phase 2 | Implemented; real output blocked by DATA-05 |
| EVAL-04 | Phase 2 | Deferred to report prose |
| LR-01 | Phase 3 | Implemented |
| LR-02 | Phase 3 | Implemented; real sweep blocked by DATA-05 |
| LR-03 | Phase 3 | Implemented; real model blocked by DATA-05 |
| LR-04 | Phase 3 | Implemented; real figure blocked by DATA-05 |
| LR-05 | Phase 3 | Deferred optional ablation |
| MLP-01 | Phase 4 | Implemented |
| MLP-02 | Phase 4 | Implemented; real training blocked by DATA-05 |
| MLP-03 | Phase 4 | Implemented; real sweep blocked by DATA-05 |
| MLP-04 | Phase 4 | Implemented; real state blocked by DATA-05 |
| MLP-05 | Phase 4 | Deferred until real sweep exists |
| MLP-06 | Phase 4 | Deferred optional ablation |
| FINAL-01 | Phase 5 | Pending |
| FINAL-02 | Phase 5 | Pending |
| CASE-01 | Phase 5 | Pending |
| CASE-02 | Phase 5 | Pending |
| CASE-03 | Phase 5 | Pending |
| CASE-04 | Phase 5 | Pending |
| DOC-01 | Phase 6 | Pending |
| DOC-02 | Phase 6 | Pending |
| DOC-03 | Phase 6 | Pending |
| DOC-04 | Phase 6 | Pending |
| DOC-05 | Phase 6 | Pending |
| DOC-06 | Phase 6 | Pending |
| SLIDE-01 | Phase 7 | Pending |
| SLIDE-02 | Phase 7 | Pending |
| SUBMIT-01 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 46 total
- Mapped to phases: 46
- Unmapped: 0

## User Stories

- As a student team, we can run a reproducible pipeline that produces the tables and figures needed for the Task 4 report.
- As a grader, I can reconstruct the experiment choices from the report, code, and saved outputs.
- As presenters, we can explain two concrete non-training case studies and use them to reflect on model strengths and weaknesses.

## Acceptance Criteria

- All official report and slide sub-questions are answered.
- Both classifiers outperform the class-prior baseline or any failure to do so is honestly analyzed.
- Split leakage checks pass.
- Final deliverables compile and stay within formal limits.

## Definition of Done

- `results/log.md` has a verification note for every implementation phase.
- Report PDF and slide PDF compile without missing figures.
- Rubric checklist is complete.
- Large/generated/course-restricted files are not committed.

---
*Requirements defined: 2026-05-27*
*Last updated: 2026-05-27 after initialization*
