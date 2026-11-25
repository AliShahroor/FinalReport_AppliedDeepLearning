#!/usr/bin/env python3
"""Small no-API smoke test for the submitted EviRAG code package."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def load_score_module():
    spec = importlib.util.spec_from_file_location("wattbot_score", ROOT / "Score.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load code/Score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    score = load_score_module()
    rows = [
        {
            "id": "demo_numeric",
            "answer_value": "42",
            "answer_unit": "percent",
            "ref_id": '["demo2026"]',
            "ref_url": "https://example.org/demo2026",
            "supporting_materials": "demo evidence",
            "explanation": "demo explanation",
        },
        {
            "id": "demo_abstain",
            "answer_value": "is_blank",
            "answer_unit": "percent",
            "ref_id": "",
            "ref_url": "",
            "supporting_materials": "",
            "explanation": "No supporting evidence in the demo corpus.",
        },
    ]
    frame = pd.DataFrame(rows)
    value = float(score.score(frame, frame.copy(), "id"))
    if value != 1.0:
        raise AssertionError(f"scorer self-check expected 1.0, got {value}")

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "candidates.jsonl"
        target = Path(tmp) / "top3.jsonl"
        source.write_text(
            json.dumps(
                {"id": "demo", "candidates": [{"rank": i} for i in range(5)]}
            )
            + "\n",
            encoding="utf-8",
        )
        helper_path = ROOT / "limit_retrieval_candidates.py"
        helper_spec = importlib.util.spec_from_file_location("limit_candidates", helper_path)
        if helper_spec is None or helper_spec.loader is None:
            raise RuntimeError("Could not load retrieval-budget helper")
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        helper.main_with_paths(source, target, 3)
        result = json.loads(target.read_text(encoding="utf-8"))
        if len(result["candidates"]) != 3:
            raise AssertionError("retrieval-budget helper did not keep three candidates")

    expected = [
        ROOT / "results/evidence_budget_ablation_summary.csv",
        ROOT / "results/source_disjoint_five_fold_summary.csv",
        ROOT / "results/bm25_qasper_scifact_2026-11-23.json",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing archived result files: " + ", ".join(missing))
    print("EviRAG no-API reproducibility check: PASS")
    print("Scorer self-check: 1.000")
    print("Retrieval-budget helper: top-3 operation verified")
    print(f"Archived result files verified: {len(expected)}")


if __name__ == "__main__":
    main()

