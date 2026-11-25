#!/usr/bin/env python3
"""Generate grounded WattBot answers from reranked evidence with vLLM."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = """You are an evidence-grounded scientific question-answering system.
Answer only from the supplied corpus excerpts. Never use unsupported background knowledge.

Return exactly one JSON object with these string/array fields:
answer, answer_value, ref_ids, supporting_materials, explanation, confidence

Rules:
- answer_value is the scored field. Give only the normalized value, without its unit.
- For a name or other categorical answer, copy the category into BOTH answer and answer_value.
- Expected answer unit "is_blank" means the answer has no unit; it does NOT mean abstain.
- For True/False questions, answer_value must be "1" for true or "0" for false.
- If a source states a range and both endpoints are requested, use "(low,high)".
- If a chart/derivation permits any value in an interval, give one defensible numeric value.
- Never include <, >, about, approximately, commas, or unit text in a numeric answer_value.
- ref_ids must use the exact values shown after ref_id= (for example, "smith2025").
  Never put evidence labels such as "E1" or "E2" in ref_ids.
- ref_ids must contain every supplied document directly needed to support the answer, no extras.
- supporting_materials must quote or identify the exact supplied evidence. Do not invent quotations.
- For arithmetic, show the operands and calculation in explanation and cite every premise document.
- Match the exact entity asked about. Never transfer a number from a different model,
  company, experiment, or system merely because it has the requested unit.
- Keep the reasoning focused and proceed to the JSON as soon as the evidence is identified.
- Reconcile apparent conflicts by respecting year, hardware, system boundary, and service target.
- If the supplied evidence does not support an answer, set answer_value to "is_blank", ref_ids to [],
  supporting_materials to "is_blank", and clearly explain why you abstained.
- confidence must be one of: high, medium, low.
- Output JSON only, with no Markdown fence or commentary.
"""


def build_user_prompt(record: dict[str, object]) -> str:
    example_blocks = []
    for index, example in enumerate(record.get("few_shot_examples", []), start=1):
        example_blocks.append(
            f"[Solved example {index}]\n"
            f"Question: {example['question']}\n"
            f"Expected unit: {example['answer_unit']}\n"
            f"Normalized answer_value: {example['answer_value']}\n"
            f"Required ref_ids: {example['ref_id']}\n"
            f"Evidence/reasoning: {example['supporting_materials']}\n"
            f"{example['explanation']}"
        )
    examples = "\n\n".join(example_blocks)
    evidence_blocks = []
    for index, item in enumerate(record["candidates"], start=1):
        evidence_blocks.append(
            f"[E{index}] ref_id={item['doc_id']} | title={item['title']} | page={item['page']}\n"
            f"{item['text']}"
        )
    evidence = "\n\n".join(evidence_blocks)
    example_section = (
        "Solved examples demonstrate dataset conventions. Do not copy their factual "
        "answers; solve the target only from its retrieved evidence.\n\n"
        f"{examples}\n\n"
        if examples
        else ""
    )
    return (
        example_section
        +
        f"Question ID: {record['id']}\n"
        f"Question: {record['question']}\n"
        f"Expected answer unit: {record['answer_unit']}\n\n"
        f"Retrieved evidence:\n{evidence}\n\n"
        "Produce the required JSON object."
    )


def extract_json(text: str) -> dict[str, object] | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            result, _ = decoder.raw_decode(text[match.start() :])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    # A generation can begin with correct structured fields and later become
    # invalid when a long explanation contains an unescaped newline or is
    # truncated. Recover only independently valid JSON values from the three
    # fields needed for scoring; do not guess values from free-form prose.
    recovered: dict[str, object] = {}
    for key in ("answer", "answer_value", "ref_ids"):
        match = re.search(rf'"{key}"\s*:\s*', text)
        if match is None:
            return None
        try:
            recovered[key], _ = decoder.raw_decode(text[match.end() :])
        except json.JSONDecodeError:
            return None
    if not isinstance(recovered["ref_ids"], (str, list)):
        return None
    recovered["supporting_materials"] = "Evidence in the cited retrieved passages."
    recovered["explanation"] = "The normalized answer is supported by the cited retrieved evidence."
    recovered["confidence"] = "low"
    return recovered


def clean_answer_value(value: object) -> str:
    value = str(value if value is not None else "").strip()
    if value.lower() in {"", "na", "n/a", "none", "null", "is_blank"}:
        return "is_blank"
    # Keep commas that delimit ranges/lists, while still accepting scalar values
    # formatted with thousands separators (for example, "40,000,000").
    if not (
        len(value) >= 2
        and value[0] in "[({"
        and value[-1] in "])}"
    ):
        value = value.replace(",", "")
    return value


def normalize_result(
    result: dict[str, object] | None,
    record: dict[str, object],
    url_by_id: dict[str, str],
) -> dict[str, str]:
    if result is None:
        return {
            "answer": "Unable to answer with confidence based on the provided documents",
            "answer_value": "is_blank",
            "answer_unit": str(record["answer_unit"]),
            "ref_id": "is_blank",
            "ref_url": "is_blank",
            "supporting_materials": "is_blank",
            "explanation": "Generation did not return valid structured output, so the system abstained.",
        }

    answer_value = clean_answer_value(result.get("answer_value"))
    answer = str(result.get("answer") or "").strip()
    # Models sometimes mistake the unit marker "is_blank" for an instruction to
    # blank categorical answers. The scorer expects the entity name in answer_value.
    if (
        answer_value == "is_blank"
        and str(record["answer_unit"]).strip().lower() == "is_blank"
        and answer
        and answer.lower() not in {"is_blank", "na", "n/a", "none", "null"}
        and not answer.lower().startswith("unable to answer")
    ):
        answer_value = answer
    if answer_value == "is_blank":
        return {
            "answer": str(result.get("answer") or "Unable to answer with confidence based on the provided documents"),
            "answer_value": "is_blank",
            "answer_unit": str(record["answer_unit"]),
            "ref_id": "is_blank",
            "ref_url": "is_blank",
            "supporting_materials": "is_blank",
            "explanation": str(result.get("explanation") or "The retrieved evidence was insufficient."),
        }

    allowed_ids = {str(item["doc_id"]) for item in record["candidates"]}
    raw_ids = result.get("ref_ids", [])
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    ref_ids = []
    evidence_id_map = {
        f"E{index}": str(item["doc_id"])
        for index, item in enumerate(record["candidates"], start=1)
    }
    for ref_id in raw_ids if isinstance(raw_ids, list) else []:
        ref_id = str(ref_id).strip()
        ref_id = evidence_id_map.get(ref_id.upper(), ref_id)
        if ref_id in allowed_ids and ref_id not in ref_ids:
            ref_ids.append(ref_id)

    if not ref_ids:
        return {
            "answer": "Unable to answer with confidence based on the provided documents",
            "answer_value": "is_blank",
            "answer_unit": str(record["answer_unit"]),
            "ref_id": "is_blank",
            "ref_url": "is_blank",
            "supporting_materials": "is_blank",
            "explanation": "The generated answer did not identify a valid supporting document, so the system abstained.",
        }

    return {
        "answer": answer or answer_value,
        "answer_value": answer_value,
        "answer_unit": str(record["answer_unit"]),
        "ref_id": json.dumps(ref_ids),
        # Some transfer-corpus documents have no URL in their metadata. Keep
        # the citation rather than failing the entire evaluation at export.
        "ref_url": json.dumps([url_by_id.get(ref_id, "") for ref_id in ref_ids]),
        "supporting_materials": str(result.get("supporting_materials") or "Evidence in cited retrieved passage."),
        "explanation": str(result.get("explanation") or "The cited evidence directly supports the normalized answer."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-output-tokens", type=int, default=2200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--linear-backend", default=None)
    parser.add_argument("--moe-backend", default=None)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Render Qwen chat prompts in direct-response mode when supported.",
    )
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help="Regenerate cached rows whose previous output did not parse as JSON.",
    )
    args = parser.parse_args()

    retrieval = [json.loads(line) for line in args.retrieval.read_text(encoding="utf-8").splitlines()]
    questions = pd.read_csv(args.questions, dtype=str).fillna("")
    metadata = pd.read_csv(args.metadata, dtype=str).fillna("")
    url_by_id = dict(zip(metadata["id"], metadata["url"]))
    by_id = {record["id"]: record for record in retrieval}

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)

    completed: dict[str, dict[str, object]] = {}
    if args.raw_output.exists():
        for line in args.raw_output.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("parsed") is None:
                item["parsed"] = extract_json(str(item.get("raw", "")))
            if args.retry_invalid and item.get("parsed") is None:
                continue
            completed[item["id"]] = item

    pending = [by_id[row_id] for row_id in questions["id"] if row_id not in completed]
    if pending:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        llm_args = {
            "model": args.model,
            "tensor_parallel_size": 1,
            "trust_remote_code": True,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": 0.92,
            "max_num_seqs": args.batch_size,
        }
        if any(family in args.model for family in ("Qwen3.5", "Qwen3.6")):
            # This flag exists only in some vLLM releases. Keep the runner
            # compatible with Panther's installed version.
            try:
                import inspect
                from vllm.engine.arg_utils import EngineArgs
                if "language_model_only" in inspect.signature(EngineArgs).parameters:
                    llm_args["language_model_only"] = True
            except (ImportError, TypeError, ValueError):
                pass
        if args.linear_backend:
            llm_args["linear_backend"] = args.linear_backend
        if args.moe_backend:
            llm_args["moe_backend"] = args.moe_backend
        llm = LLM(**llm_args)
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            presence_penalty=args.presence_penalty,
            seed=args.seed,
            max_tokens=args.max_output_tokens,
        )
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(record)},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=not args.disable_thinking,
                )
                for record in batch
            ]
            outputs = llm.generate(prompts, sampling)
            with args.raw_output.open("a", encoding="utf-8") as handle:
                for record, output in zip(batch, outputs):
                    text = output.outputs[0].text
                    item = {"id": record["id"], "raw": text, "parsed": extract_json(text)}
                    completed[record["id"]] = item
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"generated {min(start + len(batch), len(pending))}/{len(pending)} pending rows", flush=True)

    rows = []
    for _, question in questions.iterrows():
        record = by_id[question["id"]]
        normalized = normalize_result(completed[question["id"]].get("parsed"), record, url_by_id)
        output_row = question.to_dict()
        output_row.update(normalized)
        rows.append(output_row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_columns = list(questions.columns)
    for column in (
        "answer",
        "answer_value",
        "answer_unit",
        "ref_id",
        "ref_url",
        "supporting_materials",
        "explanation",
    ):
        if column not in output_columns:
            output_columns.append(column)
    pd.DataFrame(rows, columns=output_columns).to_csv(args.output, index=False)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
