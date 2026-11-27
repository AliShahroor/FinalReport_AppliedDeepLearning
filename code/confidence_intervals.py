#!/usr/bin/env python3
"""Bootstrap confidence intervals for grouped WattBot validation folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Score import row_bits


def row_scores(gold: pd.DataFrame, pred: pd.DataFrame) -> np.ndarray:
    shared = [row_id for row_id in gold["id"] if row_id in set(pred["id"])]
    g = gold.set_index("id")
    p = pred.set_index("id")
    values = []
    for row_id in shared:
        bits = row_bits(g.loc[row_id], p.loc[row_id])
        values.append(0.75 * float(bits["val"]) + 0.20 * float(bits["ref"]) + 0.05 * float(bits["na"]))
    return np.asarray(values, dtype=float)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()

    rng = np.random.default_rng(34060)
    fold_means = []
    fold_details = []
    for fold in range(args.folds):
        gold = pd.read_csv(args.gold_dir / f"fold{fold}_val_gold.csv", dtype=str).fillna("")
        pred = pd.read_csv(args.pred_dir / f"fold{fold}_predictions.csv", dtype=str).fillna("")
        values = row_scores(gold, pred)
        low, high = bootstrap_mean(values, rng, args.draws)
        mean = float(values.mean())
        fold_means.append(mean)
        fold_details.append({"fold": fold, "n": len(values), "mean": mean, "ci95": [low, high]})

    fold_array = np.asarray(fold_means)
    fold_low, fold_high = bootstrap_mean(fold_array, rng, args.draws)
    result = {
        "folds": fold_details,
        "macro_mean": float(fold_array.mean()),
        "macro_sd": float(fold_array.std(ddof=1)),
        "macro_bootstrap_ci95": [fold_low, fold_high],
        "seed": 34060,
        "draws": args.draws,
        "note": "Intervals are bootstrap diagnostics over saved predictions; they do not account for additional model or data-selection uncertainty.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

