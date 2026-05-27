# Pitfalls Research: MLPC 2026 Task 4

**Project:** MLPC 2026 Task 4: Data Classification  
**Domain:** Coursework sound event detection classification  
**Researched:** 2026-05-27  
**Confidence:** HIGH

## Critical Pitfalls

1. **Leakage across splits**
   - Warning sign: segments or recordings from the same collector appear in train and validation/test.
   - Prevention: split with `GroupShuffleSplit` using `collector_id`, and assert pairwise collector intersections are empty.

2. **Counting missing annotators as negative votes**
   - Warning sign: files with fewer annotators become artificially negative.
   - Prevention: mask annotator slices that are all missing/NaN before majority vote.

3. **Fitting preprocessing on validation/test data**
   - Warning sign: `StandardScaler` is fit on the full dataset.
   - Prevention: fit only on train rows and persist the scaler.

4. **Optimizing for accuracy or AUROC instead of imbalance-aware metrics**
   - Warning sign: impressive results from all-negative behavior on rare classes.
   - Prevention: use macro AUPRC as the primary metric and report per-class AP.

5. **Letting optional experiments consume report time**
   - Warning sign: MLP sweep or ablations continue while report/case study are unfinished.
   - Prevention: cut the grid if needed and prioritize rubric completeness.

6. **Weak case study section**
   - Warning sign: case studies are generic or only show aggregate metrics.
   - Prevention: select one success and one failure file from the test set, visualize both, and explain specific class-level behavior.

7. **Forgetting formal assignment constraints**
   - Warning sign: report exceeds page/word limits or slides cover the wrong topic.
   - Prevention: keep report at 6 pages/2000 words, include LLM disclosure, and make slides about Case Study and Reflection.
