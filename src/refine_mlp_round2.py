from __future__ import annotations

from . import refine_mlp


refine_mlp.CANDIDATES = [
    {"hidden_dims": [512, 256], "dropout": 0.45, "lr": 1.0e-3, "seed": 42, "epochs": 70},
    {"hidden_dims": [512, 256], "dropout": 0.43, "lr": 1.0e-3, "seed": 42, "epochs": 60},
    {"hidden_dims": [512, 256], "dropout": 0.47, "lr": 1.0e-3, "seed": 42, "epochs": 60},
    {"hidden_dims": [512, 256], "dropout": 0.45, "lr": 7.0e-4, "seed": 42, "epochs": 60},
    {"hidden_dims": [512, 256], "dropout": 0.45, "lr": 1.3e-3, "seed": 42, "epochs": 60},
    {"hidden_dims": [512, 256], "dropout": 0.50, "lr": 1.0e-3, "seed": 42, "epochs": 60},
    {"hidden_dims": [512, 256], "dropout": 0.45, "lr": 1.0e-3, "seed": 7, "epochs": 60},
    {"hidden_dims": [512, 256], "dropout": 0.45, "lr": 1.0e-3, "seed": 123, "epochs": 60},
    {"hidden_dims": [768, 384], "dropout": 0.45, "lr": 1.0e-3, "seed": 42, "epochs": 55},
    {"hidden_dims": [768, 384], "dropout": 0.40, "lr": 1.0e-3, "seed": 42, "epochs": 55},
]


if __name__ == "__main__":
    refine_mlp.refine_mlp()
