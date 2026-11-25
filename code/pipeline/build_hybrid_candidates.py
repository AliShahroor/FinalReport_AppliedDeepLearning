#!/usr/bin/env python3
"""Hybrid BM25 + Qwen3 dense retrieval with train citation-recall diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%/+:-]*")
QUERY_INSTRUCTION = (
    "Given a question about the environmental impacts of AI, retrieve passages "
    "that directly support the exact answer and its source citation."
)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def parse_refs(value: object) -> set[str]:
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "is_blank", "na"}:
        return set()
    try:
        parsed = json.loads(raw.replace("'", '"'))
        if isinstance(parsed, list):
            return {str(item).strip().lower() for item in parsed}
    except json.JSONDecodeError:
        pass
    return {raw.lower()}


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    if k == len(scores):
        return np.argsort(scores)[::-1]
    indices = np.argpartition(scores, -k)[-k:]
    return indices[np.argsort(scores[indices])[::-1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("indexes/qwen3_embedding_8b"))
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--retrieve-k", type=int, default=100)
    parser.add_argument("--keep-k", type=int, default=40)
    parser.add_argument("--max-chunks-per-doc", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    raw_lines = args.chunks.read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(line) for line in raw_lines]
    questions = pd.read_csv(args.questions, dtype=str).fillna("")
    passages = [f"Title: {item['title']}\nPage: {item['page']}\n{item['text']}" for item in chunks]
    bm25 = BM25Okapi([tokenize(text) for text in passages])

    model = SentenceTransformer(
        args.model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        trust_remote_code=True,
        device="cuda",
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256("\n".join(raw_lines).encode()).hexdigest()[:16]
    embedding_path = args.cache_dir / f"corpus-{digest}.npy"
    if embedding_path.exists():
        corpus_embeddings = np.load(embedding_path)
        print(f"loaded corpus embeddings: {embedding_path}", flush=True)
    else:
        corpus_embeddings = model.encode(
            passages,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        np.save(embedding_path, corpus_embeddings.astype(np.float32))
        print(f"saved corpus embeddings: {embedding_path}", flush=True)

    query_texts = [
        f"Instruct: {QUERY_INSTRUCTION}\nQuery: {row.question}\nExpected unit: {row.answer_unit}"
        for row in questions.itertuples(index=False)
    ]
    query_embeddings = model.encode(
        query_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    del model
    torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    recalls: dict[int, list[float]] = defaultdict(list)
    with args.output.open("w", encoding="utf-8") as handle:
        for question_index, row in questions.iterrows():
            query = query_texts[question_index]
            dense_scores = corpus_embeddings @ query_embeddings[question_index]
            lexical_scores = bm25.get_scores(tokenize(query))
            dense_rank = top_indices(dense_scores, args.retrieve_k)
            lexical_rank = top_indices(lexical_scores, args.retrieve_k)

            rrf: dict[int, float] = defaultdict(float)
            for rank, index in enumerate(dense_rank):
                rrf[int(index)] += 1.0 / (60 + rank + 1)
            for rank, index in enumerate(lexical_rank):
                rrf[int(index)] += 1.0 / (60 + rank + 1)

            selected: list[dict[str, object]] = []
            per_doc: dict[str, int] = defaultdict(int)
            for index, score in sorted(rrf.items(), key=lambda item: item[1], reverse=True):
                chunk = chunks[index]
                if per_doc[chunk["doc_id"]] >= args.max_chunks_per_doc:
                    continue
                candidate = dict(chunk)
                candidate["rrf_score"] = score
                candidate["dense_score"] = float(dense_scores[index])
                candidate["bm25_score"] = float(lexical_scores[index])
                selected.append(candidate)
                per_doc[chunk["doc_id"]] += 1
                if len(selected) >= args.keep_k:
                    break

            gold = parse_refs(row.get("ref_id", ""))
            ranked_docs = list(dict.fromkeys(str(item["doc_id"]).lower() for item in selected))
            if gold:
                for k in (1, 3, 5, 10, 20):
                    recalls[k].append(len(gold & set(ranked_docs[:k])) / len(gold))

            handle.write(json.dumps({
                "id": row["id"],
                "question": row["question"],
                "answer_unit": row["answer_unit"],
                "gold_refs": sorted(gold),
                "candidates": selected,
            }, ensure_ascii=False) + "\n")

    if recalls:
        for k in (1, 3, 5, 10, 20):
            print(f"hybrid gold citation recall@{k}: {np.mean(recalls[k]):.4f}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

