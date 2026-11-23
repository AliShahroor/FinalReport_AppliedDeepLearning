#!/usr/bin/env python3
"""Create a blinded, reproducible sample for a small citation audit."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", action="append", required=True, metavar="NAME=CSV")
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-per-condition", type=int, default=25)
    args = ap.parse_args()

    question = pd.read_csv(args.questions, dtype=str).fillna("").set_index("id")
    conditions = {}
    for spec in args.predictions:
        name, path = spec.split("=", 1)
        conditions[name] = pd.read_csv(path, dtype=str).fillna("").set_index("id")

    common = set(question.index)
    for frame in conditions.values():
        common &= set(frame.index)
    if not common:
        raise ValueError("no shared question IDs")

    rows = []
    for name, frame in conditions.items():
        ordered = sorted(common, key=lambda row_id: hashlib.sha256(f"{name}:{row_id}".encode()).hexdigest())
        for row_id in ordered[: args.n_per_condition]:
            row = frame.loc[row_id]
            rows.append(
                {
                    "audit_id": f"{name}:{row_id}",
                    "condition": name,
                    "question_id": row_id,
                    "question": question.loc[row_id, "question"],
                    "predicted_answer": row.get("answer", ""),
                    "answer_value": row.get("answer_value", ""),
                    "predicted_refs": row.get("ref_id", ""),
                    "supporting_materials": row.get("supporting_materials", ""),
                    "annotator_a_value": "",
                    "annotator_a_citation": "",
                    "annotator_a_unit": "",
                    "annotator_a_abstain": "",
                    "annotator_b_value": "",
                    "annotator_b_citation": "",
                    "annotator_b_unit": "",
                    "annotator_b_abstain": "",
                    "adjudicated": "",
                    "notes": "",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} blinded audit rows to {args.output}")


if __name__ == "__main__":
    main()
