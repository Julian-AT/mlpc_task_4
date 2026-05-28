# MLPC 2026 Task 4: Data Classification

This repository contains the code, report, slides, and tracked result summaries for
the MLPC 2026 Task 4 sound event classification assignment.

## Contents

- `src/` - data preparation, splits, preprocessing, training, evaluation, and report figures
- `tests/` - unit tests for the core data and model utilities
- `results/` - tracked CSV/JSON summaries used in the report
- `report/` - LaTeX report source and figures
- `slides/` - Beamer slide source and figures
- `resources/` - report and slide templates used for the submission

The course dataset, model checkpoints, logs, and large caches are intentionally
ignored by Git.

## Results

Final held-out test performance:

| Model | Macro AP | Micro AP |
| --- | ---: | ---: |
| Class prior baseline | 0.0636 | 0.1207 |
| Logistic regression | 0.5287 | 0.6637 |
| MLP ensemble | 0.6381 | 0.7544 |

The final report is `report/main.tex`. The slide deck is `slides/slides.tex`.
Both expect the course/NeurIPS style file `neurips_2023.sty` to be available in
the Overleaf project.

## Reproducing the Pipeline

Use Python 3.11 or newer and install the requirements:

```bash
python -m pip install -r requirements.txt
```

Place the course dataset under:

```text
data/
  metadata.csv
  annotations.csv
  audio_features/
    000001.npz
    ...
```

Run the standard pipeline from the repository root:

```bash
python -m src.data
python -m src.splits
python -m src.preprocess
python -m src.baseline
python -m src.train_lr
python -m src.train_mlp
python -m src.final_eval
```

The additional scripts `src.refine_lr_torch`, `src.refine_mlp`, and
`src.ensemble_mlp` reproduce the stronger CUDA runs and ensemble selection used
for the submitted results. `src.report_assets` regenerates the figures in the
report.

## Validation

```bash
python -m pytest
```
