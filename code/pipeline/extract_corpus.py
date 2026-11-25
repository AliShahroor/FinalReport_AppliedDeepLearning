#!/usr/bin/env python3
"""Extract page-aware text chunks from the cached WattBot corpus."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz
import pandas as pd


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_windows(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, len(words), step) if words[start:]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/metadata.csv"))
    parser.add_argument("--documents", type=Path, default=Path("data/documents"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--chunk-words", type=int, default=450)
    parser.add_argument("--overlap-words", type=int, default=100)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata, dtype=str).fillna("").set_index("id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for doc_id, row in metadata.iterrows():
        matches = sorted(args.documents.glob(f"{doc_id}.*"))
        matches = [path for path in matches if path.suffix.lower() == ".pdf"]
        if not matches:
            failures.append({"id": doc_id, "error": "no cached PDF"})
            continue
        path = matches[0]
        try:
            document = fitz.open(path)
            for page_index, page in enumerate(document):
                text = clean_text(page.get_text("text", sort=True))
                for window_index, window in enumerate(
                    word_windows(text, args.chunk_words, args.overlap_words)
                ):
                    chunks.append({
                        "chunk_id": f"{doc_id}:p{page_index + 1}:c{window_index}",
                        "doc_id": doc_id,
                        "title": row["title"],
                        "year": row["year"],
                        "url": row["url"],
                        "page": page_index + 1,
                        "page_count": document.page_count,
                        "text": window,
                    })
            document.close()
            print(f"{doc_id}: extracted {path.name}", flush=True)
        except Exception as exc:
            failures.append({"id": doc_id, "error": f"{type(exc).__name__}: {exc}"})

    with args.output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    report = {
        "documents_in_metadata": len(metadata),
        "documents_extracted": len(set(chunk["doc_id"] for chunk in chunks)),
        "chunks": len(chunks),
        "failures": failures,
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 2 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())

