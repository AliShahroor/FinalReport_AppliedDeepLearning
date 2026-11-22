#!/usr/bin/env python3
"""Evaluate a simple BM25 evidence retriever on external scientific QA data.

This is deliberately retrieval-only: it tests whether WattBot's evidence
selection idea transfers before adding a model-specific answer generator.
"""

import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text):
    return TOKEN_RE.findall(str(text).lower())


def recall_at(ranked, gold, k):
    return bool(set(ranked[:k]) & set(gold))


def mean_recall(rows):
    if not rows:
        return {"n": 0, "r@1": None, "r@5": None, "r@10": None}
    return {
        "n": len(rows),
        "r@1": sum(recall_at(r, g, 1) for r, g in rows) / len(rows),
        "r@5": sum(recall_at(r, g, 5) for r, g in rows) / len(rows),
        "r@10": sum(recall_at(r, g, 10) for r, g in rows) / len(rows),
    }


def run_scifact(path):
    root = Path(path)
    corpus = {}
    with open(root / "corpus.jsonl") as f:
        for line in f:
            row = json.loads(line)
            abstract = " ".join(row.get("abstract", []))
            corpus[str(row["doc_id"])] = f"{row.get('title', '')} {abstract}"

    doc_ids = list(corpus)
    bm25 = BM25Okapi([tokens(corpus[i]) for i in doc_ids])
    rows = []
    with open(root / "claims_dev.jsonl") as f:
        for line in f:
            claim = json.loads(line)
            gold = set()
            for doc_id, evidence in claim.get("evidence", {}).items():
                if evidence:
                    gold.add(str(doc_id))
            if not gold:
                continue
            scores = bm25.get_scores(tokens(claim["claim"]))
            ranked = [doc_ids[i] for i in scores.argsort()[::-1][:10]]
            rows.append((ranked, gold))
    return {"task": "SciFact", "corpus_documents": len(doc_ids), **mean_recall(rows)}


def run_qasper(path):
    data = json.load(open(path))
    rows = []
    papers = 0
    for paper in data.values():
        papers += 1
        paragraphs = []
        for section in paper.get("full_text", []):
            section_name = section.get("section_name", "")
            for paragraph in section.get("paragraphs", []):
                paragraphs.append(f"{section_name} {paragraph}")
        if not paragraphs:
            continue
        bm25 = BM25Okapi([tokens(p) for p in paragraphs])
        for qa in paper.get("qas", []):
            answers = qa.get("answers", [])
            evidence_texts = []
            for annotation in answers:
                answer = annotation.get("answer", {})
                if answer.get("unanswerable"):
                    continue
                evidence_texts.extend(answer.get("evidence", []))
            if not evidence_texts:
                continue

            # QASPER stores highlighted evidence as text. Match it back to the
            # paragraph containing the most highlighted words.
            gold = set()
            for evidence in evidence_texts:
                ev_tokens = set(tokens(evidence))
                if not ev_tokens:
                    continue
                overlaps = [len(ev_tokens & set(tokens(p))) for p in paragraphs]
                best = max(overlaps, default=0)
                if best:
                    gold.update(i for i, score in enumerate(overlaps) if score == best)
            if not gold:
                continue
            scores = bm25.get_scores(tokens(qa["question"]))
            ranked = list(scores.argsort()[::-1][:10])
            rows.append((ranked, gold))
    result = {"task": "QASPER", "papers": papers, **mean_recall(rows)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scifact", type=Path, required=True)
    ap.add_argument("--qasper", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = {
        "note": "BM25 evidence-retrieval smoke test; not end-to-end answer accuracy.",
        "scifact": run_scifact(args.scifact),
        "qasper": run_qasper(args.qasper),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
