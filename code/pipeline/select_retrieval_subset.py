#!/usr/bin/env python3
"""Keep retrieval records whose IDs occur in a held-out question file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument(
        "--chunks",
        type=Path,
        help="Chunk JSONL used to materialize BM25 best_chunks as candidates.",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    questions = pd.read_csv(args.questions, dtype=str).fillna("")
    question_map = questions.set_index("id").to_dict("index")
    ids = set(question_map)

    chunks = {}
    if args.chunks:
        for line in args.chunks.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                chunks[str(item["chunk_id"])] = item

    rows = [json.loads(line) for line in args.retrieval.read_text().splitlines() if line.strip()]
    selected = []
    for row in rows:
        row_id = str(row["id"])
        if row_id not in ids:
            continue
        question = question_map[row_id]
        row["question"] = question["question"]
        row["answer_unit"] = question["answer_unit"]

        # BM25 retrieval stores chunk IDs as best_chunks, while the generator
        # consumes the same candidate objects produced by dense retrieval.
        if "candidates" not in row:
            if not chunks:
                raise ValueError("retrieval records lack candidates; pass --chunks")
            row["candidates"] = [
                chunks[str(chunk_id)]
                for chunk_id in row.get("best_chunks", [])
                if str(chunk_id) in chunks
            ]
        if not row["candidates"]:
            raise ValueError(f"{row_id}: no candidate chunks available")
        selected.append(row)
    if len(selected) != len(ids):
        raise ValueError(f"selected {len(selected)} records for {len(ids)} question IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    print(f"wrote {len(selected)} records to {args.output}")


if __name__ == "__main__":
    main()
