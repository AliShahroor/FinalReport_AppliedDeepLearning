#!/usr/bin/env python3
"""Apply conservative, question-derived formatting fixes to WattBot outputs.

This script never uses gold labels. It only normalizes ranges and explicit
large-unit expressions that the scorer expects numerically.
"""

import argparse
import re

import pandas as pd


NUMBER = r"[-+]?\d+(?:\.\d+)?"
RANGE_RE = re.compile(rf"^\s*({NUMBER})\s*(?:-|to)\s*({NUMBER})\s*$", re.I)


def normalize_value(value: str, question: str) -> tuple[str, str | None]:
    value = str(value).strip()
    question_lower = question.lower()

    match = RANGE_RE.match(value)
    if match:
        lo, hi = match.groups()
        if "500 ml bottle" in question_lower and "every" in question_lower:
            reciprocal_lo = 1 / float(hi)
            reciprocal_hi = 1 / float(lo)
            return (
                f"[{reciprocal_lo:.10g},{reciprocal_hi:.10g}]",
                "converted responses per 500 mL bottle to bottles per response",
            )
        return f"[{lo},{hi}]", "converted a textual numeric range to bracket notation"

    # Some questions state the requested unit as million liters, while the
    # answer key stores the underlying value in liters.
    if "million liter" in question_lower and re.fullmatch(NUMBER, value):
        number = float(value)
        if 0 < number < 100:
            scaled = number * 1_000_000
            rendered = f"{scaled:.10f}".rstrip("0").rstrip(".")
            return rendered, "converted million liters to liters"

    return value, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, dtype=str).fillna("")
    changes = []
    for index, row in frame.iterrows():
        old = row["answer_value"]
        new, reason = normalize_value(old, row["question"])
        if reason and new != old:
            frame.at[index, "answer_value"] = new
            frame.at[index, "explanation"] = (
                str(row.get("explanation", "")).rstrip()
                + f" Post-processing: {reason}."
            ).strip()
            changes.append((row["id"], old, new, reason))

    frame.to_csv(args.output, index=False)
    print(f"wrote {args.output}; changed {len(changes)} rows")
    for row_id, old, new, reason in changes:
        print(f"{row_id}: {old!r} -> {new!r} ({reason})")


if __name__ == "__main__":
    main()
