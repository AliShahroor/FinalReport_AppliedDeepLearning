#!/usr/bin/env python3
"""Convert SciFact into WattBot's retrieval input format."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus = {}
    with open(args.data / "corpus.jsonl") as f:
        for line in f:
            row = json.loads(line)
            doc_id = str(row["doc_id"])
            corpus[doc_id] = row

    chunks = []
    for doc_id, row in corpus.items():
        for page, paragraph in enumerate(row.get("abstract", []), start=1):
            chunks.append({
                "doc_id": doc_id,
                "title": row.get("title", ""),
                "page": page,
                "text": paragraph,
            })
    (args.out_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(x) for x in chunks) + "\n"
    )

    questions = []
    with open(args.data / "claims_dev.jsonl") as f:
        for line in f:
            row = json.loads(line)
            gold = sorted(str(x) for x, evidence in row.get("evidence", {}).items() if evidence)
            if not gold:
                continue
            questions.append({
                "id": f"scifact-{row['id']}",
                "question": row["claim"],
                "answer_unit": "claim verification",
                "ref_id": json.dumps(gold),
            })
    pd.DataFrame(questions).to_csv(args.out_dir / "questions.csv", index=False)
    print(f"wrote {len(chunks)} chunks and {len(questions)} questions to {args.out_dir}")


if __name__ == "__main__":
    main()
