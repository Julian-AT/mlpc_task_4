# MLPC 2026 Task 4: Data Classification

This repository contains a reproducible segment-level classification pipeline for the
MLPC 2026 KIAL sound event detection dataset. It builds collector-disjoint splits,
aggregates annotator labels, trains a baseline, logistic regression models, and a
small MLP, then writes comparison tables and figures for the report.

## Repository Layout

```text
src/                 Data preparation, training, metrics, and evaluation code
tests/               Unit tests for data handling, splits, metrics, and training
results/             Small tracked summaries and generated report tables
results/figures/     Report figures
resources/           Report template files
report/              Report workspace
slides/              Slide workspace
```

The course dataset and large generated artifacts are intentionally not tracked.

## Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

For NVIDIA GPU acceleration on Windows, install a CUDA-enabled PyTorch build in the
same environment:

```powershell
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If that build is not compatible with the local driver, use the PyTorch install selector
and install the CUDA build it recommends.

## Data

Place the dataset under `data/`:

```text
data/
  metadata.csv
  annotations.csv
  audio_features/
    000001.npz
    ...
```

The `data/` directory is ignored because the dataset is course-provided material.

## Pipeline

Run the full preparation and evaluation pipeline from the repository root:

```bash
python -m src.data
python -m src.splits
python -m src.preprocess
python -m src.baseline
python -m src.train_lr
python -m src.train_mlp
python -m src.final_eval
```

The main outputs are:

```text
results/class_distribution.csv
results/baseline.json
results/lr_sweep.csv
results/mlp_sweep.csv
results/predictions_test.npz
results/final_table.csv
```

Large cache and model files are ignored by git.

## Faster Windows Runs

The default sweeps are intentionally broad. For a faster CPU/GPU run:

```powershell
py -c "from src.train_lr import sweep_lr, plot_lr_sweep; sweep_lr(grid={'C':[0.1,1.0,10.0], 'penalty':['l2'], 'class_weight':[None,'balanced']}, max_iter=500); plot_lr_sweep()"
py -c "from src.train_mlp import sweep_mlp; sweep_mlp(grid={'hidden_dims': [[128], [256], [256, 256]], 'dropout': [0.0], 'lr': [0.001]}, epochs=10, batch_size=2048, patience=3)"
py -m src.final_eval
```

## Validation

```bash
python -m pytest
```
