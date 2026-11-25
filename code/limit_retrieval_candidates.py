#!/usr/bin/env python3
"""Keep only the first k ranked evidence candidates in each retrieval record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main_with_paths(input_path: Path, output_path: Path, k: int) -> None:
    rows = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["candidates"] = row["candidates"][:k]
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} records with top-{k} candidates to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    main_with_paths(args.input, args.output, args.k)


if __name__ == "__main__":
    main()
