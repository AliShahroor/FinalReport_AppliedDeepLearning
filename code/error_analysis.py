#!/usr/bin/env python3
"""Categorize held-out EviRAG errors without an API call.

The retrieval category is intentionally conservative: it is assigned only when a
gold document is absent from the saved candidate manifest. Other wrong answers
remain generation/reasoning errors rather than being attributed to retrieval.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def refs(value) -> set[str]:
    if isinstance(value, list):
        return {str(x) for x in value}
    text = str(value or "").strip()
    if not text:
        return set()
    try:
        parsed = ast.literal_eval(text)
        return {str(x) for x in parsed} if isinstance(parsed, (list, tuple, set)) else {str(parsed)}
    except (SyntaxError, ValueError):
        return {text}


def load_manifest(path: Path) -> dict[str, set[str]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result[row["id"]] = {str(c.get("doc_id", "")) for c in row.get("candidates", [])}
    return result


def classify(gold, pred, candidate_docs: set[str]) -> tuple[str, str]:
    gold_blank = str(gold["answer_value"]).strip().lower() in {"", "is_blank"}
    pred_blank = str(pred["answer_value"]).strip().lower() in {"", "is_blank"}

    if gold_blank != pred_blank:
        return "abstention_failure", "gold and prediction disagree on abstention"

    # Import lazily so the script remains useful as a standalone report utility.
    from Score import row_bits

    bits = row_bits(gold, pred)
    if not bits["val"] and not gold_blank:
        gold_refs = refs(gold["ref_id"])
        if gold_refs and not gold_refs.issubset(candidate_docs):
            return "retrieval_risk", "at least one gold source is absent from saved candidates"
        structured = any(str(gold.get(field, "0")).strip() in {"1", "1.0", "True", "true"}
                         for field in ("Table", "Figure", "Math", "CrossPaper", "Reconcile"))
        if structured:
            return "structured_reasoning_failure", "wrong value on table, figure, math, cross-paper, or reconcile item"
        return "generation_failure", "gold evidence was available but value was incorrect"

    if bits["val"] and bits["ref"] < 1.0:
        return "citation_failure", "value matched but citation set was incomplete or unsupported"
    return "correct_or_partial", "no primary failure detected by the scorer"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    gold = pd.read_csv(args.gold).set_index("id")
    pred = pd.read_csv(args.predictions).set_index("id")
    manifest = load_manifest(args.manifest)
    shared = [row_id for row_id in gold.index if row_id in pred.index]
    rows = []
    for row_id in shared:
        category, reason = classify(gold.loc[row_id], pred.loc[row_id], manifest.get(row_id, set()))
        rows.append({"id": row_id, "category": category, "reason": reason})

    counts = Counter(row["category"] for row in rows)
    total = len(rows)
    summary = {
        "n": total,
        "categories": dict(counts),
        "protocol": "Primary categories are scorer- and manifest-based; retrieval_risk is conservative and not a causal proof.",
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Analyzed {total} held-out questions")
    for category, count in counts.most_common():
        print(f"{category}: {count} ({count / total:.1%})")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

