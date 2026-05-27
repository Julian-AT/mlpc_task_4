from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Paths
DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "audio_features"
METADATA_CSV = DATA_DIR / "metadata.csv"
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
DATASET_CACHE = RESULTS_DIR / "dataset_cache.npz"
SPLITS_PATH = RESULTS_DIR / "splits.npz"
CLASS_DISTRIBUTION_CSV = RESULTS_DIR / "class_distribution.csv"
PREPROCESSED_CACHE = RESULTS_DIR / "preprocessed.npz"
SCALER_PATH = RESULTS_DIR / "scaler.joblib"
PER_FILE_IOU_PATH = RESULTS_DIR / "per_file_iou.npz"
BASELINE_JSON = RESULTS_DIR / "baseline.json"
LR_SWEEP_CSV = RESULTS_DIR / "lr_sweep.csv"
LR_BEST_MODEL = RESULTS_DIR / "lr_best.pkl"
PREDICTIONS_TEST = RESULTS_DIR / "predictions_test.npz"
MLP_SWEEP_CSV = RESULTS_DIR / "mlp_sweep.csv"
MLP_BEST_MODEL = RESULTS_DIR / "mlp_best.npz"
RESULTS_LOG = RESULTS_DIR / "log.md"

# Seeds and splits
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Label aggregation
ANNOT_BINARIZE_THRESH = 0.5
MAJORITY_THRESH = 0.5

# Features
TEMPORAL_CONTEXT = 2

# High-agreement filter
HIGH_AGREEMENT_IOU = 0.6

# Class names are alphabetical and must match the dataset.
CLASS_NAMES = [
    "bell_ringing",
    "coffee_machine",
    "cutlery_dishes",
    "door_open_close",
    "footsteps",
    "keyboard_typing",
    "keychain",
    "light_switch",
    "microwave",
    "phone_ringing",
    "running_water",
    "toilet_flushing",
    "vacuum_cleaner",
    "wardrobe_drawer_open_close",
    "window_open_close",
]
NUM_CLASSES = len(CLASS_NAMES)

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
