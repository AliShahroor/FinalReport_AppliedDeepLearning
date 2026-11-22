#!/usr/bin/env python3
"""Normalize the public WattBot 2025 files for the WattBot 2026 runner."""

import argparse
import json
from pathlib import Path

import pandas as pd


def read_csv(path):
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metadata = read_csv(args.data / "metadata.csv")
    metadata.to_csv(args.out / "metadata.csv", index=False)

    train = read_csv(args.data / "train_QA.csv")
    # Some public copies contain a UTF-8 BOM rendered as literal characters.
    train = train.rename(columns={c: "id" if c.lstrip("ï»¿") == "id" else c for c in train.columns})
    train[["id", "question", "answer_unit", "ref_id"]].to_csv(args.out / "questions.csv", index=False)
    train.to_csv(args.out / "labels.csv", index=False)
    print(json.dumps({"questions": len(train), "documents": len(metadata), "out": str(args.out)}))


if __name__ == "__main__":
    main()
