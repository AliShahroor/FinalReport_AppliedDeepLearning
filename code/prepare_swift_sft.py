#!/usr/bin/env python3
"""Build leakage-safe MS-SWIFT folds for WattBot answer generation."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from generate_answers import SYSTEM_PROMPT, build_user_prompt


def parse_refs(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"is_blank", "nan"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = [text]
    if not isinstance(parsed, list):
        parsed = [parsed]
    return [str(item).strip() for item in parsed if str(item).strip()]


def source_groups(questions: pd.DataFrame) -> list[int]:
    """Assign connected components so no cited source crosses a split.

    A question with multiple references connects those references.  Treating
    the connected component as its group is conservative: even indirect
    source overlap cannot leak from training into validation.
    """
    parent = list(range(len(questions)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    owner: dict[str, int] = {}
    for index, value in enumerate(questions["ref_id"]):
        for ref in parse_refs(value):
            if ref in owner:
                union(index, owner[ref])
            else:
                owner[ref] = index
    component_roots = sorted({find(i) for i in range(len(questions))})
    roots = {root: group for group, root in enumerate(component_roots)}
    return [roots[find(index)] for index in range(len(questions))]


def select_training_evidence(
    record: dict[str, object], row: pd.Series, max_candidates: int
) -> dict[str, object]:
    gold_refs = parse_refs(row["ref_id"])
    candidates = list(record["candidates"])
    selected: list[dict[str, object]] = []
    selected_chunks: set[str] = set()

    # Put genuine retrieved positives first, then mine the highest-ranked negatives.
    for ref_id in gold_refs:
        matches = [item for item in candidates if str(item["doc_id"]) == ref_id]
        for item in matches[:2]:
            chunk_id = str(item["chunk_id"])
            if chunk_id not in selected_chunks:
                selected.append(item)
                selected_chunks.add(chunk_id)

    # Retrieval misses are valid training positives, but never inject these into validation.
    present_refs = {str(item["doc_id"]) for item in selected}
    for ref_id in gold_refs:
        if ref_id not in present_refs:
            selected.append(
                {
                    "chunk_id": f"gold-support:{row['id']}:{ref_id}",
                    "doc_id": ref_id,
                    "title": "Released gold supporting material",
                    "page": "unknown",
                    "text": str(row["supporting_materials"]),
                }
            )

    for item in candidates:
        if len(selected) >= max_candidates:
            break
        chunk_id = str(item["chunk_id"])
        if chunk_id not in selected_chunks and str(item["doc_id"]) not in gold_refs:
            selected.append(item)
            selected_chunks.add(chunk_id)

    output = dict(record)
    output["candidates"] = selected[:max_candidates]
    return output


def assistant_target(row: pd.Series) -> str:
    refs = parse_refs(row["ref_id"])
    blank = str(row["answer_value"]).strip().lower() in {"", "is_blank", "nan"}
    payload = {
        "answer": str(row["answer"]),
        "answer_value": "is_blank" if blank else str(row["answer_value"]),
        "ref_ids": [] if blank else refs,
        "supporting_materials": "is_blank" if blank else str(row["supporting_materials"]),
        "explanation": str(row["explanation"]),
        "confidence": "high" if not blank else "low",
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def swift_record(record: dict[str, object], row: pd.Series, include_target: bool) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(record)},
    ]
    if include_target:
        messages.append({"role": "assistant", "content": assistant_target(row)})
    return {"messages": messages}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=34060)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument(
        "--group-by-source",
        action="store_true",
        help="keep questions sharing cited papers in the same fold",
    )
    args = parser.parse_args()

    questions = pd.read_csv(args.questions, dtype=str).fillna("")
    retrieval_rows = [
        json.loads(line)
        for line in args.retrieval.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retrieval = {str(row["id"]): row for row in retrieval_rows}
    missing = set(questions["id"]) - set(retrieval)
    if missing:
        raise ValueError(f"retrieval is missing {len(missing)} question ids")
    if not 0 <= args.fold < args.folds:
        raise ValueError(f"fold must be in [0, {args.folds - 1}]")

    if args.group_by_source:
        groups = source_groups(questions)
        splitter = GroupKFold(n_splits=args.folds)
        splits = list(splitter.split(questions, groups=groups))
    else:
        groups = None
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(questions))
    train_indices, val_indices = splits[args.fold]
    train_rows = questions.iloc[train_indices].reset_index(drop=True)
    val_rows = questions.iloc[val_indices].reset_index(drop=True)

    train_dataset = []
    for _, row in train_rows.iterrows():
        record = select_training_evidence(
            retrieval[str(row["id"])], row, args.max_candidates
        )
        train_dataset.append(swift_record(record, row, include_target=True))

    val_dataset = []
    val_infer_dataset = []
    val_manifest = []
    for _, row in val_rows.iterrows():
        record = dict(retrieval[str(row["id"])])
        record["candidates"] = list(record["candidates"])[: args.max_candidates]
        val_dataset.append(swift_record(record, row, include_target=True))
        val_infer_dataset.append(swift_record(record, row, include_target=False))
        val_manifest.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_dataset)
    write_jsonl(args.output_dir / "val.jsonl", val_dataset)
    write_jsonl(args.output_dir / "val_infer.jsonl", val_infer_dataset)
    write_jsonl(args.output_dir / "val_manifest.jsonl", val_manifest)
    val_rows.to_csv(args.output_dir / "val_gold.csv", index=False)
    (args.output_dir / "split.json").write_text(
        json.dumps(
            {
                "fold": args.fold,
                "folds": args.folds,
                "seed": args.seed,
                "group_by_source": args.group_by_source,
                "source_group_count": len(set(groups)) if groups is not None else None,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "train_ids": train_rows["id"].tolist(),
                "val_ids": val_rows["id"].tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote fold {args.fold}: train={len(train_rows)} val={len(val_rows)}")


if __name__ == "__main__":
    main()
