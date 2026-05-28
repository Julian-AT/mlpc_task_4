# MLPC 2026 Task 4 — Data Classification: PRD & Roadmap

**Team:** Autonomous Pipelines (Julian Schmidt, Paul Breburda)
**Slide Topic:** Case Study and Reflection (group name starts with A)
**Deadline:** Thursday, May 28, 2026, 23:59
**Hardware:** MacBook Pro M-series, 24 GB unified memory, MLX
**Stack:** Python, NumPy, scikit-learn, MLX, matplotlib, librosa (case study only)

---

## 0. Context & Scope

This is the classification phase of the KIAL SED project. The dataset, label-aggregation
strategy, agreement analysis, and split philosophy were all established in your Task 3
report. **Reuse those decisions and cite them.** Do not re-justify from scratch — the
report is graded for completeness, not redundancy.

Deliverables:

1. Report (PDF, max 6 pages, max 2000 words, LaTeX template from Moodle) — 37 pts
2. Slide deck (PDF, max 4 slides + 1 title) on **Case Study and Reflection** — 3 pts

Strategic principle: **section 4 (Case Study & Reflection) is your slide topic — make it
the strongest section.** It also happens to be the most concrete and visual, so a strong
section 4 carries the report.

---

## 1. Strategic Decisions (Locked)

These are pre-justified — Claude Code should not deviate. Each comes with a one-line
"report-ready" phrasing you can adapt.

### 1.1 Label Aggregation — Majority Vote with 0.5 Binarization

- **Procedure:** binarize `annotations[t,c,a] ≥ 0.5`, then majority vote over annotators:
  `y[t,c] = 1 iff mean_a(bin[t,c,a]) ≥ 0.5`
- **Single-annotator files (17%):** that annotator's binarized labels become the label directly
- **Threshold stability:** Task 3 swept {0.4, 0.5, 0.6} and saw IoU shift {+0.013, 0, −0.014}
- **Why majority not union:** union recovers polyphonic events but propagates single-annotator
  false positives. Majority vote is the precision-favoring choice that matches downstream
  precision-sensitive deployments (smart-home triggers).
- **Reuse Task 3 code if available** (mlpc-2026-task3 repo).

### 1.2 Data Split — Group-Disjoint by `collector_id`, 70/15/15

- `sklearn.model_selection.GroupShuffleSplit`, two-stage:
  1. Train (70%) vs Temp (30%) — groups = collector_ids
  2. Temp split 50/50 → Val (15%) and Test (15%) — groups = collector_ids
- `random_state=42` everywhere
- **Information leakage sources to discuss in report:**
  - Same segment in two splits → trivial; solved by splitting by file
  - Same recording, different segments → leaks acoustic conditions, mic response, ambient noise
  - Same collector, different recordings → leaks device, environment, annotation style
    (Task 3 showed own-vs-other annotators give +0.020 IoU, p=0.002, so collector identity
    is a confound)
- **Verification:** assert `set(train_collectors) ∩ set(val_collectors) == ∅` etc.
- **Class distribution table:** generate per-class positive rate for train/val/test, include in report
- **Stratification note:** true multi-label stratified group-shuffle is non-trivial
  (iterative-stratification); for this task, group-disjoint takes priority over stratification,
  and we verify post-hoc that no class is grossly imbalanced across splits.

### 1.3 Preprocessing — Z-score + Optional High-Agreement Filter

- **Standardization:** `StandardScaler` fit on train only, applied to val/test
- **Feature set:** all 14 base features × 4 aggregations (mean/std/min/max) = roughly 188 dims.
  (Confirm exact count by loading one .npz; some single-channel features may not have all
  aggregations.)
- **Optional: PCA experiment** — Task 3 found high redundancy (MFCC↔Mel r=0.82, ZCR↔Centroid
  r=0.99, Energy↔Flux r=0.93). Try PCA to 50 components as ablation for LR; report whether
  it helps or hurts.
- **Optional: high-agreement filter (per Task 3 recommendation):**
  - For each class c and each file f, look up the IoU_c(f) computed in Task 3
  - When training class c, drop training segments from files where IoU_c(f) < 0.6
  - Different filter per class (different files filtered for "light_switch" vs "vacuum_cleaner")
  - Implement once for LR; if it helps, apply to MLP too
- **Temporal context for MLP:** concatenate features from frames [t−2, t−1, t, t+1, t+2].
  Pad with zeros at recording boundaries. Yields ~940-dim input. Apply this **only** to
  the MLP variant — keep LR on single-frame to enable clean linear-vs-nonlinear narrative.

### 1.4 Evaluation Metric — Macro-Averaged AUPRC

- **Primary:** mean of per-class Average Precision (AUPRC)
- **Why this and not accuracy/F1/AUROC:**
  - Multi-label (sigmoid per class), not multi-class — no softmax
  - 24:1 class imbalance — accuracy and AUROC are misleading on rare classes
  - AUPRC is threshold-independent and better than AUROC under imbalance
  - Macro avg gives rare transients (light_switch) equal weight to dominant classes (footsteps),
    matching the deployment goal of detecting all event types
- **Also report:** micro AP, per-class AP table, macro F1 at globally optimal threshold,
  per-class F1 at per-class optimal threshold

### 1.5 Baseline — Class-Prior Random Predictor

- **Definition:** for each class c, predict a constant score equal to class prevalence
  `p_c` (positive rate in train). Equivalently, a random predictor with the right marginals.
- **Expected AP under this:** equals `p_c` for each class (well-known property of random
  predictors on imbalanced binary tasks)
- **Macro AP of baseline:** mean of class prevalences across the 15 classes — expected around
  0.05 to 0.07 given Task 3 frequencies
- **Best-possible discussion:** the average per-class IoU agreement between annotators
  is 0.640 (Task 3, Table 2). This is an effective performance ceiling — a classifier
  that perfectly matches "ground truth" can't exceed inter-annotator consistency. Per-class
  ceilings range from 0.179 (light_switch) to 0.870 (vacuum_cleaner).

### 1.6 Classifiers — Logistic Regression vs Small MLP (MLX)

**Clean narrative:** linear vs nonlinear, identical preprocessing and metric, same train/val/test.

**Logistic Regression** (`sklearn.linear_model.LogisticRegression` wrapped in
`OneVsRestClassifier`):

- Hyperparameter grid (20 configs):
  - `C` ∈ {0.01, 0.1, 1.0, 10.0, 100.0}
  - `penalty` ∈ {'l1', 'l2'}
  - `class_weight` ∈ {None, 'balanced'}
- Solver: `'liblinear'` for L1, `'lbfgs'` for L2 (sklearn requirement). Or use `'saga'`
  for both with `max_iter=2000`
- One model per class (OvR), fit independently, sigmoid scores via `predict_proba`

**Small MLP (MLX):**

- Architecture sweep:
  - `hidden_dims` ∈ {[128], [256], [512], [256, 256], [512, 256, 128]}
  - `dropout` ∈ {0.0, 0.2, 0.4}
  - `learning_rate` ∈ {1e-3, 3e-4}
- Total: 30 configs. Train each for 30 epochs with early stopping (patience 5 on val macro AP)
- Input: 940-dim (with ±2 temporal context)
- Output: 15-dim sigmoid (multi-label)
- Loss: class-weighted binary cross-entropy. Use `pos_weight = sqrt(n_neg / n_pos)` per class
  (square root softens the extreme reweighting that pure inverse-frequency gives, which tends
  to over-trigger on rare classes)
- Optimizer: AdamW, weight decay 1e-4
- Schedule: cosine LR with warmup (3 epochs warmup)
- Batch size: 1024 (fits easily in 24 GB)
- **Optional ablation if time permits:** focal loss (γ=2) on the best architecture

---

## 2. Project Structure

```
mlpc-task4/
├── data/                          # symlink to dataset
│   ├── metadata.csv
│   ├── annotations.csv
│   └── audio_features/
├── src/
│   ├── __init__.py
│   ├── config.py                  # all constants (seeds, paths, sweep grids)
│   ├── data.py                    # loading + label aggregation
│   ├── splits.py                  # collector-disjoint splits
│   ├── preprocess.py              # scaling, temporal context, high-agreement filter
│   ├── metrics.py                 # AP, macro AP, per-class, F1@optimal
│   ├── baseline.py                # class-prior baseline
│   ├── train_lr.py                # LR sweep entry point
│   ├── train_mlp.py               # MLX MLP sweep entry point
│   ├── mlp_model.py               # MLX model class
│   ├── evaluate.py                # final test-set evaluation, comparison table
│   ├── case_study.py              # spectrograms + predictions on test files
│   └── viz.py                     # all matplotlib helpers
├── results/
│   ├── dataset_cache.npz          # cached aggregated features + labels
│   ├── splits.npz                 # cached indices
│   ├── scaler.pkl                 # fitted StandardScaler
│   ├── lr_sweep.csv               # one row per config
│   ├── lr_best.pkl
│   ├── mlp_sweep.csv
│   ├── mlp_best.npz               # MLX weights of best model
│   ├── predictions_test.npz       # final scores for case study
│   └── figures/                   # all report + slide figures
├── report/                        # LaTeX (use Task 3 template)
│   └── main.tex
├── slides/
│   └── slides.tex
├── notebooks/                     # scratch only — do not commit experiments here
├── requirements.txt
└── README.md
```

---

## 3. Execution Phases

### Phase 0 — Setup (30 min)

```bash
mkdir -p mlpc-task4/{src,results/figures,report,slides,notebooks}
cd mlpc-task4
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn tqdm mlx librosa soundfile pyarrow
ln -s /path/to/MLPC2026_dataset_development data
```

`requirements.txt`:
```
numpy>=1.26
pandas>=2.1
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
tqdm>=4.66
mlx>=0.18
librosa>=0.10
soundfile>=0.12
pyarrow>=15
```

**Verify MLX works:**
```python
import mlx.core as mx
print(mx.array([1, 2, 3]) * 2)  # should print mx array
```

**`src/config.py`** — single source of truth for constants:

```python
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "audio_features"
METADATA_CSV = DATA_DIR / "metadata.csv"
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# Seeds and splits
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15  # implied

# Label aggregation
ANNOT_BINARIZE_THRESH = 0.5
MAJORITY_THRESH = 0.5

# Features
TEMPORAL_CONTEXT = 2  # ±2 frames for MLP

# High-agreement filter
HIGH_AGREEMENT_IOU = 0.6

# Class names — alphabetical, must match dataset
CLASS_NAMES = [
    "bell_ringing", "coffee_machine", "cutlery_dishes", "door_open_close",
    "footsteps", "keyboard_typing", "keychain", "light_switch",
    "microwave", "phone_ringing", "running_water", "toilet_flushing",
    "vacuum_cleaner", "wardrobe_drawer_open_close", "window_open_close",
]
NUM_CLASSES = len(CLASS_NAMES)

# Sweeps
LR_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "penalty": ["l1", "l2"],
    "class_weight": [None, "balanced"],
}

MLP_GRID = {
    "hidden_dims": [[128], [256], [512], [256, 256], [512, 256, 128]],
    "dropout": [0.0, 0.2, 0.4],
    "lr": [1e-3, 3e-4],
}
MLP_EPOCHS = 30
MLP_BATCH = 1024
MLP_PATIENCE = 5
```

---

### Phase 1 — Data Loading & Label Aggregation (1h)

**`src/data.py`**

Key functions:

```python
def load_metadata() -> pd.DataFrame: ...
def load_annotations() -> pd.DataFrame: ...

def aggregate_labels(ann: np.ndarray) -> np.ndarray:
    """
    ann: [T, C, A] float in [0, 1]
    returns: [T, C] uint8
    Procedure:
      1. binarize: bin = (ann >= 0.5)
      2. mask annotators that didn't annotate this file (all-zero or NaN slice along T)
      3. majority vote: y = (bin.mean(axis=2) >= 0.5).astype(uint8)
    Single-annotator files: mean of single annotator equals their binarized label.
    """

def concat_features(npz_dict) -> np.ndarray:
    """Concatenate all 14 feature groups × 4 aggregations → [T, D]."""

def build_dataset(cache_path=None) -> dict:
    """
    Iterate all .npz files. For each:
      - load features, aggregate to single matrix [T_i, D]
      - aggregate labels [T_i, C]
      - record collector_id (from metadata), file_id, segment start times
    Concatenate across files:
      - X: [N_total, D]
      - Y: [N_total, C]
      - file_ids: [N_total] (int)
      - collector_ids: [N_total] (int)
      - file_segment_idx: [N_total] (int) — index within file, for temporal context
    Cache to results/dataset_cache.npz.
    """
```

**Important implementation notes:**

- The `annotations` array in `.npz` has shape `[T, C, A]`. Some annotators didn't annotate
  some files — those slices may be all zeros or NaN. Mask them out before voting (don't
  count them as "voted negative").
- Verify class index order matches the alphabetical list in `config.CLASS_NAMES` — the
  task spec says it's alphabetical, but assert it.
- `collector_id` comes from `metadata.csv` keyed by `filename`. Map to integer for storage.

**Sanity checks at end of phase 1:**

- Total segments ≈ 168,239 (Task 3 number)
- Positive rates per class match Task 3 Figure 1 (footsteps top, light_switch bottom)
- Feature dimensionality printed and recorded in README

---

### Phase 2 — Splits (45 min)

**`src/splits.py`**

```python
from sklearn.model_selection import GroupShuffleSplit

def make_splits(collector_ids: np.ndarray, seed: int = 42) -> dict:
    """
    Two-stage GroupShuffleSplit.
    Returns dict with keys 'train', 'val', 'test', each an int array of segment indices.
    Asserts pairwise empty intersection of collector sets.
    """

def class_distribution_table(Y, splits) -> pd.DataFrame:
    """For each class, positive count and rate per split. Save as CSV."""
```

**Deliverables for the report:**

- `results/splits.npz` (cached)
- `results/class_distribution.csv` (Table for section 1b)
- `figures/class_dist_across_splits.png` (grouped bar chart, 15 classes × 3 splits)

**Verification (must pass before continuing):**

```python
assert len(set(train_collectors) & set(val_collectors)) == 0
assert len(set(train_collectors) & set(test_collectors)) == 0
assert len(set(val_collectors) & set(test_collectors)) == 0
print(f"Train {len(train_idx):,} | Val {len(val_idx):,} | Test {len(test_idx):,}")
```

---

### Phase 3 — Preprocessing (1h)

**`src/preprocess.py`**

```python
def fit_scaler(X_train) -> StandardScaler: ...
def apply_scaler(scaler, X) -> np.ndarray: ...

def add_temporal_context(X, file_segment_idx, file_ids, k=2) -> np.ndarray:
    """
    For each segment, concatenate features from [t-k, ..., t+k].
    At file boundaries, pad with zeros.
    Output shape: [N, D * (2k+1)]
    """

def high_agreement_mask(file_ids, class_idx, per_file_iou) -> np.ndarray:
    """
    Returns bool mask over segments: True where IoU of this class on this file >= 0.6.
    per_file_iou: dict[file_id] -> array of length C
    Used to subset training data per class.
    """
```

**For computing per-file per-class IoU**, reuse Task 3 code if available; otherwise:

```python
def per_file_per_class_iou(annotations_4d) -> np.ndarray:
    """
    For each file, for each class, mean pairwise IoU across annotators
    on binarized labels at threshold 0.5. Mask pairs where union is empty.
    Returns: [num_files, num_classes]
    """
```

**Cache to `results/per_file_iou.npz`.**

---

### Phase 4 — Baseline & Metrics (45 min)

**`src/metrics.py`**

```python
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score

def per_class_ap(y_true, y_score) -> np.ndarray:
    """Returns [C] array of per-class AP."""

def macro_ap(y_true, y_score) -> float:
    return per_class_ap(y_true, y_score).mean()

def micro_ap(y_true, y_score) -> float:
    return average_precision_score(y_true.ravel(), y_score.ravel())

def best_threshold_f1(y_true_c, y_score_c) -> tuple[float, float]:
    """Per-class: returns (best_threshold, best_f1)."""

def per_class_f1_at_optimal(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    """Returns (thresholds, f1s) of length C."""
```

**`src/baseline.py`**

```python
def class_prior_baseline_scores(y_train, n_val) -> np.ndarray:
    """Score = class prevalence, repeated for all val/test rows. Shape [n, C]."""

def evaluate_baseline(y_train, y_val):
    scores = class_prior_baseline_scores(y_train, len(y_val))
    print({
        "macro_ap": macro_ap(y_val, scores),
        "per_class_ap": per_class_ap(y_val, scores),
    })
```

**Expected output:** macro AP around 0.05–0.07, per-class AP = class prevalence
(prove this analytically in report: a constant-score predictor's AP equals the positive rate).

**`results/baseline.json`** — save numbers for the report.

---

### Phase 5 — Logistic Regression Sweep (~2h runtime)

**`src/train_lr.py`**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import itertools, joblib, time

def fit_one(X_tr, Y_tr, X_val, Y_val, C, penalty, class_weight):
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    base = LogisticRegression(
        C=C, penalty=penalty, class_weight=class_weight,
        solver=solver, max_iter=2000, n_jobs=-1,
    )
    model = OneVsRestClassifier(base, n_jobs=-1)
    t0 = time.time()
    model.fit(X_tr, Y_tr)
    scores = model.predict_proba(X_val)
    return model, scores, time.time() - t0

def sweep():
    rows = []
    grid = itertools.product(LR_GRID["C"], LR_GRID["penalty"], LR_GRID["class_weight"])
    for C, penalty, cw in grid:
        model, val_scores, runtime = fit_one(...)
        rows.append({
            "C": C, "penalty": penalty, "class_weight": str(cw),
            "macro_ap": macro_ap(Y_val, val_scores),
            "micro_ap": micro_ap(Y_val, val_scores),
            **{f"ap_{cn}": v for cn, v in zip(CLASS_NAMES, per_class_ap(Y_val, val_scores))},
            "runtime_s": runtime,
        })
        # save best by macro_ap incrementally
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "lr_sweep.csv", index=False)
```

**After sweep:**

- Save best model (`joblib.dump(best_model, "lr_best.pkl")`)
- Save best model's val + test predictions to `predictions_test.npz` (key: `lr_test_scores`)
- Generate sweep heatmap: `figures/lr_sweep_heatmap.png` (C × penalty, mean over class_weight)

**Optional sub-experiment (1 extra config):** Best HP + PCA(50). Report whether PCA hurts/helps.

**Optional sub-experiment (1 extra config):** Best HP + high-agreement filter. Report delta.

---

### Phase 6 — MLP Sweep in MLX (3–4h runtime)

**`src/mlp_model.py`**

```python
import mlx.core as mx
import mlx.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.2):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def __call__(self, x):
        return self.net(x)  # logits, sigmoid applied in loss
```

**`src/train_mlp.py`**

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.losses import binary_cross_entropy

def pos_weight_per_class(Y_train):
    n_pos = Y_train.sum(axis=0)
    n_neg = len(Y_train) - n_pos
    return mx.array(np.sqrt(n_neg / np.maximum(n_pos, 1)).astype(np.float32))

def weighted_bce(logits, targets, pos_weight):
    # logits, targets: [B, C], pos_weight: [C]
    # equivalent to BCEWithLogits with pos_weight
    # bce = -[pw*t*log(sigmoid(x)) + (1-t)*log(1-sigmoid(x))]
    log_sigmoid = -nn.softplus(-logits)        # log σ(x)
    log_one_minus = -nn.softplus(logits)        # log(1 - σ(x))
    loss = -(pos_weight * targets * log_sigmoid + (1 - targets) * log_one_minus)
    return loss.mean()

def train_one(X_tr, Y_tr, X_val, Y_val, hidden_dims, dropout, lr,
              epochs=30, batch=1024, patience=5):
    model = MLP(X_tr.shape[1], hidden_dims, NUM_CLASSES, dropout)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    pw = pos_weight_per_class(Y_tr)

    def loss_fn(model, x, y):
        return weighted_bce(model(x), y, pw)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    best_val_ap, best_epoch, best_state = -1, -1, None
    X_tr_mx, Y_tr_mx = mx.array(X_tr.astype(np.float32)), mx.array(Y_tr.astype(np.float32))

    for epoch in range(epochs):
        # cosine schedule with 3-epoch linear warmup
        if epoch < 3:
            cur_lr = lr * (epoch + 1) / 3
        else:
            cur_lr = lr * 0.5 * (1 + np.cos(np.pi * (epoch - 3) / (epochs - 3)))
        optimizer.learning_rate = cur_lr

        # shuffled mini-batch loop
        perm = mx.array(np.random.permutation(len(X_tr)))
        for i in range(0, len(X_tr), batch):
            idx = perm[i:i+batch]
            x, y = X_tr_mx[idx], Y_tr_mx[idx]
            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        # validation
        val_logits = model(mx.array(X_val.astype(np.float32)))
        val_scores = mx.sigmoid(val_logits)
        mx.eval(val_scores)
        ap = macro_ap(Y_val, np.array(val_scores))
        if ap > best_val_ap:
            best_val_ap, best_epoch = ap, epoch
            best_state = tree_flatten(model.parameters())  # snapshot
        elif epoch - best_epoch >= patience:
            break

    return best_val_ap, best_state, best_epoch
```

**Implementation notes:**

- MLX uses lazy evaluation — `mx.eval()` at end of each step keeps memory bounded
- For 24 GB, the full train set in fp32 at 940 dims fits easily (~700 MB)
- If running out of memory, switch to fp16: `mx.array(..., dtype=mx.float16)` for inputs;
  keep model in fp32
- `tree_flatten` / `tree_unflatten` from `mlx.utils` for state save/load

**Sweep loop runs 30 configs.** Save CSV row per config to `mlp_sweep.csv`.

**Figures for report:**

- `figures/mlp_arch_vs_ap.png` — hidden architecture vs val macro AP
- `figures/mlp_dropout_vs_ap.png` — dropout vs val macro AP (with error bars across other HPs)
- `figures/mlp_lr_vs_ap.png` — learning rate vs val macro AP
- `figures/mlp_training_curves.png` — best model's train/val loss + val AP over epochs

**Optional ablations (only if time permits):**

- Focal loss (γ=2) on best architecture — 1 extra run
- No temporal context (single-frame input) — 1 extra run, demonstrates value of context

---

### Phase 7 — Final Evaluation on Test Set (45 min)

**`src/evaluate.py`**

- Load best LR and best MLP from their sweep CSVs (argmax val macro AP)
- Apply to test set (LR: single-frame, scaled; MLP: temporal context, scaled)
- Compute on test:
  - macro AP, micro AP, per-class AP for both models and baseline
  - macro F1 at globally optimal threshold (chosen on val), applied to test
  - Per-class confusion: TP/FP/FN counts at per-class optimal threshold
- Save final comparison table: `results/final_table.csv`
- Save `predictions_test.npz` with `{lr_scores, mlp_scores, y_true, file_ids, start_times}`
  for the case study phase
- Generate figure: `figures/per_class_ap_comparison.png` — grouped bar chart, 3 series
  (baseline, LR, MLP), 15 classes

---

### Phase 8 — Case Study & Reflection (2h — **your slide topic, highest priority**)

**`src/case_study.py`**

**Strategy: pick 2 files in test set, one success and one failure.**

#### Step 1 — Select files

Iterate test files. For each file, compute:

- `f1_macro_file` = mean per-class F1 on that file (at the global optimal thresholds)
- Number of distinct active classes (polyphony)
- Confusion of pre-identified ambiguous pairs (door↔wardrobe, bell↔phone)

Pick:

- **Success file:** F1 in top quartile, has at least 3 classes, polyphonic, model handles
  overlapping events
- **Failure file:** has light_switch or keychain (the hardest classes per Task 3), OR
  has door/wardrobe or bell/phone confusion in predictions

Save filenames and reasons to `results/case_study_files.json`.

#### Step 2 — Generate visualizations per file

Three-panel figure per file:

```
┌─────────────────────────────────────────────────┐
│ Mel Spectrogram (time × mel bins, log power)    │
├─────────────────────────────────────────────────┤
│ Ground Truth event tracks (15 rows, 1 per class)│
│   active classes shown as filled bars           │
├─────────────────────────────────────────────────┤
│ Predicted probabilities (15 rows, line curves)  │
│   with optimal-threshold horizontal lines       │
└─────────────────────────────────────────────────┘
```

Reuse the mel spectrogram from `melspect_mean` (already in features) — no need to recompute
from raw audio. Raw audio is only needed for listening qualitatively.

```python
def plot_case_study(file_id, file_ids, y_true, y_score, melspect_mean,
                    optimal_thresholds, save_path):
    """3-panel figure as described above."""
```

#### Step 3 — Per-class confusion analysis

For each class:

- Precision, recall, F1 on test
- Top 3 confusion: which other classes does the model wrongly co-fire with?

For acoustically similar pairs identified in Task 3 (door/wardrobe, bell/phone), compute
mutual confusion rate: fraction of true-class-A frames where model predicts class-B.

Save `results/per_class_analysis.csv`. This drives section 4b of the report and slide 4.

#### Step 4 — Notes for writing

Create `results/case_study_notes.md` with bullet points: what was right, what was wrong,
why (with reference to Task 3 findings — IoU 0.179 for light_switch, etc.). This is your
draft for section 4 prose.

---

### Phase 9 — Report Writing (3–4h)

Use the LaTeX template from Task 3. Section budget (target 2000 words total):

| Section | Words | Points |
|---|---|---|
| Intro/abstract | 60 | — |
| 1. Dataset Preparation | 700 | 10 |
| ↳ 1a Label Aggregation | 250 | |
| ↳ 1b Data Split | 250 | |
| ↳ 1c Preprocessing | 200 | |
| 2. Evaluation | 280 | 5 |
| 3. Experiments | 600 | 12 |
| ↳ 3a HP sweeps | 380 | |
| ↳ 3b Final comparison | 220 | |
| 4. Case Study & Reflection | 360 | 10 |
| Disclosure | 50 | — |
| **Total** | **2050** | **37** |

Slight overflow allowed (figure captions and the disclosure don't count toward word limit
in most interpretations — check the Task 3 template).

See **Section 6** of this PRD for section-by-section guidance with report-ready phrasings.

---

### Phase 10 — Slides (1h)

Topic: **Case Study and Reflection**. 4 slides + 1 title.

| # | Slide | Content |
|---|---|---|
| 0 | Title | "Case Study and Reflection — Autonomous Pipelines", names |
| 1 | Success case | Three-panel figure (spec + GT + preds) of success file. One sentence: what the model gets right. |
| 2 | Failure case | Three-panel figure of failure file. One sentence: what fails and why (specific class confusion or transient miss). |
| 3 | Per-class breakdown | Bar chart: per-class AP for LR vs MLP vs baseline. Annotate the easy classes (vacuum, water) and hard classes (light_switch, keychain). Link to Task 3 IoU finding. |
| 4 | Takeaways | 3 bullets: (a) sustained sounds easy, transients hard, matches IoU agreement; (b) acoustically similar pairs (door/wardrobe, bell/phone) systematically confused; (c) collector-disjoint split is essential and reduces apparent performance vs naive split. |

---

## 4. Time Budget (24h plan)

Working backwards from May 28 23:59 submission. Assume effective time tomorrow ~10h.

### Tonight (~5h, 18:00–23:00)
- **18:00–18:30** Phase 0 setup (PRD in hand, repo created)
- **18:30–19:30** Phase 1 data loading + label aggregation (reuse Task 3 code)
- **19:30–20:15** Phase 2 splits + class distribution table
- **20:15–21:00** Phase 3 preprocessing + temporal context + high-agreement filter
- **21:00–21:30** Phase 4 baseline + metrics (small)
- **21:30–22:30** Phase 5 launch LR sweep — runs while you sleep / write
- **22:30–23:00** Start raw audio download (for case study tomorrow)

### Tomorrow morning (~5h, 08:00–13:00)
- **08:00–08:30** Check LR sweep results, pick best, generate sweep figures
- **08:30–12:00** Phase 6 MLP sweep (mostly hands-off, runs ~3h on M-series)
- **08:30–11:30 in parallel** Start drafting report sections 1 and 2 (data prep + evaluation —
  these don't depend on results)
- **12:00–13:00** Phase 7 final evaluation on test set, comparison table

### Tomorrow afternoon (~5h, 13:00–18:00)
- **13:00–15:00** Phase 8 case study (your slide topic — give it real care)
- **15:00–17:00** Phase 9 finish report — sections 3, 4, polish
- **17:00–18:00** Phase 10 slides

### Tomorrow evening (~3h, 18:00–21:00) — BUFFER
- Polish, proofread, check word count, verify figures render in LaTeX
- Cross-check report against rubric line by line
- Submit by 22:00 to leave 2h margin

**Hard rule:** if you're not done with Phase 5 by tomorrow 09:00, cut MLP sweep to 12 configs.
If you're not done with Phase 6 by 13:00, take whatever the best so far is and move on.

---

## 5. Risk Register

| Risk | P(occur) | Mitigation |
|---|---|---|
| Dataset too big for memory | Low | Cache as npz, load in chunks if needed. 168k × 940 fp32 ≈ 630 MB — fine |
| MLX bug / install issue | Medium | Fallback: scikit-learn MLP. Less impressive but works |
| LR sweep doesn't finish overnight | Low | Reduce grid to {C ∈ [0.1, 1, 10], penalty ∈ [l1, l2], cw ∈ [None, balanced]} = 12 configs |
| MLP overfits on temporal-context features | Medium | Dropout in sweep already covers this; weight decay helps |
| Splits leak | Low (we're explicit) | Hard assertion in `make_splits()`; verify integer collector_id mapping |
| Class distribution catastrophically uneven across splits | Low–medium | Verify in Phase 2; if light_switch has < 30 positives in val/test, re-seed |
| Macro AP looks flat because class_weight='balanced' over-triggers rare classes | Medium | Compare with/without; pos_weight=sqrt(n_neg/n_pos) softens this for MLP |
| Case study files are boring | Low | Iterate file selection; criterion is "informative", not "best score" |
| Word count exceeds 2000 | High | Write tight, no filler. Use the prewritten phrasings in Section 8 below |
| LaTeX figures don't render in 6 pages | Medium | Reserve space early. Push exploratory figures to appendix |
| Last-minute Moodle outage | Low | Submit 2h early |

---

## 6. Report Outline (Section-by-Section Spec)

### Section 1 — Dataset Preparation (10 pts, ~700 words)

#### 1a Label Aggregation (~250 words)

Must answer:

1. Which aggregation strategy did you use? — describe the [T,C,A] → [T,C] procedure
2. Why suitable? Advantages + limitations

**Content:**

- Reference Task 3 majority-vote decision and threshold-stability sweep ({0.4, 0.5, 0.6}
  yields IoU shift {+0.013, 0, −0.014})
- Spell out the equation: `y[t,c] = 1 iff mean_a(ann[t,c,a] >= 0.5) >= 0.5`
- Discuss single-annotator file handling (17% of files); note this inflates trust in
  those annotators
- Advantages: precision-favoring (good for downstream trigger systems), interpretable,
  robust to single noisy annotator
- Limitations: under-counts polyphonic events (the Task 3 case-study file 002871 had 6
  events from one annotator and 2 from the other; majority vote keeps very few)
- Alternative considered: union — recovers polyphonic events but propagates false
  positives from any single annotator

#### 1b Data Split (~250 words)

Must answer:

1. Information leakage sources + prevention. Should segments from same recording / collector
   appear in multiple splits?
2. Class distribution consistency across splits

**Content:**

- Three leakage levels: segment-level (trivial), recording-level (acoustic conditions,
  mic response), collector-level (device, environment, annotation style — Task 3 showed
  +0.020 IoU advantage when annotator is also collector, p=0.002)
- Decision: split by `collector_id` using `GroupShuffleSplit` (70/15/15, seed 42)
- Verification: pairwise empty intersection of collector sets, no recording appears in
  multiple splits (transitively follows from collector-disjoint)
- Class distribution: should be similar across splits for variance reduction in evaluation,
  but stratification under multi-label + group constraints is non-trivial — we verify
  post-hoc rather than enforce
- **Include Table 1: per-class positive rate × {train, val, test}, plus class counts.
  Comment on which classes are sparsest (light_switch with ~150 positives in test set,
  noting this limits AP estimate precision).**

#### 1c Preprocessing (~200 words)

Must answer:

1. Which preprocessing applied and why

**Content:**

- Z-score normalization fit on train only (justify with Task 3 Table 3: feature scales
  span from [0,1] flatness to 0–11140 power)
- All 4 aggregations included (mean, std, min, max) → ~188 dims
- For MLP: ±2 frame temporal context appended → ~940 dims. Justification: SED events
  span multiple frames (Task 3 noted transients can take longer to characterize than the
  single-second window provides). Zero-padding at recording boundaries.
- **High-agreement filter ablation:** for each class, train only on segments from files
  where that class's annotator IoU ≥ 0.6. Reduces label noise on hard classes (light_switch)
  at the cost of training set size. Reported in section 3 ablation.

### Section 2 — Evaluation (5 pts, ~280 words)

Must answer:

1. Evaluation criterion + why
2. Baseline + best-possible performance

**Content:**

- Macro AUPRC: multi-label (no softmax), threshold-independent (avoids picking thresholds
  during HP sweeps), robust to class imbalance (Task 3 measured 24:1 imbalance), macro
  giving equal weight to all classes matches deployment goal of equal-quality detection
  across event types
- Why not accuracy: trivially high under imbalance (constant-negative on light_switch
  scores 99.4% accuracy on its segment frequency)
- Why not AUROC: AUPRC is more discriminating under imbalance
- Also reported: micro AP, per-class AP, macro F1 at globally optimal threshold
- **Baseline:** class-prior random predictor (sample from Bernoulli with `p_c` = class
  prevalence). Expected per-class AP = `p_c`. Compute and report macro AP across the
  15 classes.
- **Best-possible:** the average per-class IoU agreement between annotators (Task 3 Table 2:
  overall mean 0.640, ranging from 0.179 light_switch to 0.870 vacuum_cleaner). A perfect
  classifier cannot exceed inter-annotator consistency on noisy labels.

### Section 3 — Experiments (12 pts, ~600 words)

Must answer:

1. Two classifiers from different model classes. Vary HPs, visualize, explain selection.
2. Compare final estimates to baseline.

**Content:**

#### 3a Hyperparameter exploration (~380 words)

- **Logistic Regression:**
  - Model class: linear, one-vs-rest. Hyperparameters: regularization strength `C`,
    penalty type (L1 induces sparsity / feature selection, L2 standard ridge),
    class_weight to address imbalance.
  - Grid: 5×2×2 = 20. Figure: heatmap of val macro AP across C × penalty (averaged over
    class_weight). Comment on trends (e.g., L1 likely competitive given feature redundancy
    Task 3 identified; balanced class_weight likely helps rare classes).
  - Selection: best macro AP on val.

- **MLP (MLX):**
  - Model class: nonlinear, capacity controlled by depth × width × dropout. HPs:
    `hidden_dims`, `dropout`, `learning_rate`.
  - Grid: 5×3×2 = 30. Figures: (1) hidden dim × val AP, (2) dropout × val AP, (3) training
    curves of best model.
  - Selection: best macro AP on val with early stopping.
  - **Ablations:** (a) without temporal context, (b) with high-agreement filter, (c) focal
    loss (if time).

#### 3b Final comparison (~220 words)

- Test-set numbers: baseline / LR / MLP, macro AP and micro AP. Expect LR ~0.40–0.55,
  MLP ~0.50–0.65 (rough guess given Task 3 IoU ceiling 0.640).
- Per-class AP comparison figure.
- Strengths/weaknesses: LR is fast, interpretable, surprisingly strong baseline given
  good features. MLP captures nonlinear feature interactions and benefits from temporal
  context. Both vastly outperform the class-prior baseline.
- Acoustic intuition: sustained sounds (vacuum, water) are nearly trivial; transients
  (light_switch, keychain) approach the agreement ceiling, indicating data limits rather
  than model limits.

### Section 4 — Case Study & Reflection (10 pts, ~360 words) — **YOUR SLIDE TOPIC**

Must answer:

1. Pick 2 audio files not in training. Visualize predictions vs ground truth. How well?
2. Reflect: failure cases, which classes reliable/unreliable?

**Content:**

- **Two case studies:** one success file, one failure file (Phase 8 outputs)
- Each gets a paragraph: what the model gets right, what it gets wrong, and the acoustic
  reason (refer to Task 3 findings)
- **Per-class reliability summary:** point to per-class AP table. Reliable classes:
  vacuum_cleaner, running_water, microwave (sustained, broadband). Difficult: light_switch
  (transient, IoU 0.179 in Task 3), keychain (similar profile).
- **Systematic confusions:** door/wardrobe and bell/phone — quantify mutual confusion from
  the analysis, link to Task 3's acoustic-similarity observation.
- **Connection to dataset biases (Task 3 section 4.1):** classes well-represented in kitchen
  recordings tend to be easier (data quantity + acoustic separability both help).

### Disclosure (~50 words, doesn't count)

State LLM usage transparently. Models used, tasks (code, analysis, writing), verification
procedure. **Required.**

---

## 7. Slides Outline (Case Study and Reflection)

(Already shown in Phase 10 above.)

Build in Beamer using your Task 3 template (the one that produced the Padlet-style deck
or a clean academic Beamer). Time budget: 1h. **The figures are already done in Phase 8;
slides are just layout.**

---

## 8. Report-Ready Phrasings (Adapt; Don't Quote Verbatim)

These are starter phrasings — your team should adjust voice, expand reasoning, and adapt
to fit the LaTeX template. Treat as scaffolding.

> **Label aggregation:** We aggregate the [T, C, A] overlap array into binary targets via
> per-annotator binarization at 0.5 followed by majority voting across annotators. The 0.5
> binarization threshold was justified in Task 3 by a stability sweep over {0.4, 0.5, 0.6},
> which shifted overall IoU by only {+0.013, 0, −0.014}. For the 17% of files with a single
> annotator, that annotator's binarized labels become the target directly. This conservative
> scheme prioritizes precision over recall: a union strategy would recover polyphonic events
> missed by under-annotating reviewers but would also propagate single-annotator false
> positives, biasing the classifier toward over-triggering.

> **Information leakage:** Three sources of leakage are present in the dataset. Splitting
> by segment alone leaks acoustic context within a recording; splitting by recording leaks
> the collector's device, environment, and annotation style. Task 3 demonstrated that
> annotator-collector identity yields a small but statistically significant +0.020 IoU
> advantage (p=0.002, bootstrap CI [0.006, 0.033]), confirming collector identity as a
> confound. We therefore partition by collector_id using `GroupShuffleSplit` into a
> 70/15/15 train/validation/test split, and verify empirically that the three collector
> sets are pairwise disjoint.

> **Evaluation metric:** We use macro-averaged AUPRC as the primary metric. AUPRC is
> threshold-independent and well-suited to the heavy class imbalance documented in Task 3
> (24:1 ratio between footsteps and light_switch). Macro averaging treats all 15 classes
> equally, matching the deployment goal of detecting all event types at comparable quality.
> Accuracy is uninformative under this imbalance — a constant-negative predictor scores
> 99.4% on light_switch. We additionally report per-class AP, micro AP, and macro F1 at
> the globally optimal threshold.

> **Baseline and ceiling:** A class-prior random predictor that scores each class at its
> training-set prevalence achieves an expected per-class AP equal to that prevalence; the
> resulting macro AP is approximately the mean class prevalence (~0.06). The annotator
> agreement ceiling from Task 3 (mean per-class IoU 0.640, ranging from 0.179 for
> light_switch to 0.870 for vacuum_cleaner) is an effective upper bound on what any
> classifier can achieve on these labels.

---

## 9. What Claude Code Needs from You

When you hand this PRD to Claude Code, also provide:

1. **The dataset path** (where you extracted `MLPC2026_dataset_development.zip`)
2. **Raw audio path** once downloaded (for case study spectrogram listening, not strictly
   required since `melspect_mean` is already in features)
3. **The Task 3 code if available** — saves time on label aggregation and per-file IoU
   computation
4. **The LaTeX template** in `report/` directory
5. **A `.cursorignore` / `.gitignore`** that excludes `data/` and large `.npz` files

When something is ambiguous, Claude Code should default to: smaller scope, run the
experiment, see what happens. Don't over-engineer.

---

## 10. Checkpoints — Do Not Skip

After each phase, verify and write a one-line note in `results/log.md`:

- ☐ Phase 1: dataset cache exists, N segments printed and matches Task 3 (168,239)
- ☐ Phase 2: splits cached, collector intersections empty, class distribution table generated
- ☐ Phase 3: scaler fit, X scaled, temporal context shape correct, high-agreement masks computed
- ☐ Phase 4: baseline macro AP recorded (~0.05–0.07)
- ☐ Phase 5: LR sweep CSV has 20 rows, best macro AP printed
- ☐ Phase 6: MLP sweep CSV has 30 rows, training curves figure generated
- ☐ Phase 7: final_table.csv exists, test predictions saved
- ☐ Phase 8: 2 case study figures generated, per_class_analysis.csv saved, case_study_notes.md written
- ☐ Phase 9: report compiles, < 2000 words, < 6 pages, disclosure included
- ☐ Phase 10: slides compile, 5 slides total

---

## 11. Final Sanity Checks Before Submission

- [ ] Report PDF compiles cleanly, fits in 6 pages
- [ ] Word count ≤ 2000 (Overleaf: File → Word Count)
- [ ] All four sections (Dataset Prep, Evaluation, Experiments, Case Study & Reflection)
      addressed with all sub-questions
- [ ] Class distribution table present in section 1b
- [ ] HP sweep figures present in section 3
- [ ] Two case study figures present in section 4
- [ ] Per-class confusion matrix or per-class AP table present in section 4
- [ ] Disclosure of LLM and AI Tool Use included
- [ ] Slide deck: title + 4 content slides, on "Case Study and Reflection"
- [ ] At least one team member available June 1 to present
- [ ] Submit to Moodle, both PDFs separately, well before 23:59

---

**End of PRD.** Hand this whole document to Claude Code along with your dataset and Task 3
repo. Execute phases in order; do not skip checkpoints.
