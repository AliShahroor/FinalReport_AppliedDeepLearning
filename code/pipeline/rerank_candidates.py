#!/usr/bin/env python3
"""Rerank hybrid candidates using Qwen3-Reranker-8B."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--keep-k", type=int, default=12)
    parser.add_argument("--max-chunks-per-doc", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    model = CrossEncoder(
        args.model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        trust_remote_code=True,
        device="cuda",
        max_length=4096,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recalls: dict[int, list[float]] = defaultdict(list)

    with args.output.open("w", encoding="utf-8") as handle:
        for number, record in enumerate(records, start=1):
            pairs = [
                (
                    record["question"],
                    f"Title: {item['title']}\nPage: {item['page']}\n{item['text']}",
                )
                for item in record["candidates"]
            ]
            scores = np.asarray(model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)).reshape(-1)
            ranked = np.argsort(scores)[::-1]
            selected = []
            per_doc: dict[str, int] = defaultdict(int)
            for index in ranked:
                candidate = record["candidates"][int(index)]
                doc_id = candidate["doc_id"]
                if per_doc[doc_id] >= args.max_chunks_per_doc:
                    continue
                candidate["reranker_score"] = float(scores[int(index)])
                selected.append(candidate)
                per_doc[doc_id] += 1
                if len(selected) >= args.keep_k:
                    break

            gold = set(record.get("gold_refs", []))
            ranked_docs = list(dict.fromkeys(item["doc_id"].lower() for item in selected))
            if gold:
                for k in (1, 3, 5, 10):
                    recalls[k].append(len(gold & set(ranked_docs[:k])) / len(gold))
            record["candidates"] = selected
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{number:03d}/{len(records)}] reranked {record['id']}", flush=True)

    if recalls:
        for k in (1, 3, 5, 10):
            print(f"reranked gold citation recall@{k}: {np.mean(recalls[k]):.4f}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
