from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .final_eval import run_final_evaluation
from .train_lr import plot_lr_sweep, sweep_lr
from .train_mlp import sweep_mlp

MLP_GRID = {
    "hidden_dims": [
        [128],
        [256],
        [512],
        [256, 128],
        [512, 256],
        [1024, 512],
    ],
    "dropout": [0.0, 0.2, 0.4],
    "lr": [1e-3],
}

LR_GRID = {
    "C": [0.1, 0.3, 1.0, 3.0, 10.0],
    "penalty": ["l2"],
    "class_weight": [None],
}


def _print_top(name: str, frame: pd.DataFrame, count: int = 5) -> None:
    columns = [col for col in ["macro_ap", "micro_ap", "runtime_s", "epochs"] if col in frame]
    params = [col for col in frame.columns if col not in columns and not col.startswith("ap_")]
    print(f"\n{name} top {min(count, len(frame))}:")
    print(frame[params + columns].head(count).to_string(index=False))


def run_mlp() -> pd.DataFrame:
    frame = sweep_mlp(grid=MLP_GRID, epochs=15, batch_size=4096, patience=4)
    _print_top("MLP", frame)
    return frame


def run_lr(parallel: bool = False) -> pd.DataFrame:
    frame = sweep_lr(
        grid=LR_GRID,
        predictions_path="results/predictions_lr_sweep.npz",
        max_iter=1000,
        n_jobs=-1 if parallel else 1,
    )
    plot_lr_sweep()
    _print_top("LR", frame)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["mlp", "lr", "final", "all"])
    parser.add_argument(
        "--parallel-lr",
        action="store_true",
        help="Use sklearn process parallelism for LR. This requires normal Windows pipe access.",
    )
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    if args.target in {"mlp", "all"}:
        run_mlp()
    if args.target in {"lr", "all"}:
        run_lr(parallel=args.parallel_lr)
    if args.target in {"final", "all"}:
        table = run_final_evaluation()
        print("\nFinal table:")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
