---
status: testing
phase: 05-final-evaluation-and-case-study
source:
  - 05-01-SUMMARY.md
  - 05-02-SUMMARY.md
  - 05-03-SUMMARY.md
started: 2026-05-27T19:41:26Z
updated: 2026-05-27T19:41:26Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

number: 1
name: Final Comparison Table
expected: |
  Running the final evaluation with real baseline, LR, and MLP prediction artifacts writes `results/final_table.csv` with baseline, best LR, and best MLP rows and the macro AUPRC/micro AUPRC values needed for the report.
awaiting: user response

## Tests

### 1. Final Comparison Table
expected: Running the final evaluation with real baseline, LR, and MLP prediction artifacts writes `results/final_table.csv` with baseline, best LR, and best MLP rows and the macro AUPRC/micro AUPRC values needed for the report.
result: [pending]

### 2. Prediction Archive
expected: Running the final evaluation writes `results/predictions_test.npz` containing final scores, labels, file IDs, timing data, and class names needed for the case-study scripts.
result: [pending]

### 3. Case-Study Selection and Figures
expected: Running the case-study workflow selects two non-training files, documents why they were chosen, and writes success/failure visualization PNGs for report and slides.
result: [pending]

### 4. Per-Class Analysis and Case Notes
expected: Running the analysis workflow writes a per-class AP comparison figure and `results/case_study_notes.md` with reliable classes, difficult classes, correct predictions, and failure explanations.
result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps

[none yet]
