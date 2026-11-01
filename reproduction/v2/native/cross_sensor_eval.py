"""Unified parser and metrics for the cross-sensor optical/SAR pilot.

This module intentionally has no model-specific or condition-specific parser.
It scores closed-set yes/no/abstain answers from raw model outputs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.cross_sensor_path_guard import assert_sanitized_data_path
except ModuleNotFoundError:
    from cross_sensor_path_guard import assert_sanitized_data_path


VALID_ANSWERS = {"yes", "no", "abstain"}
ANSWER_RE = re.compile(r"\b(yes|no|abstain)\b", flags=re.IGNORECASE)
JSON_OBJECT_RE = re.compile(r"\{.*?\}", flags=re.DOTALL)


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str | None
    strict_json_valid: bool
    parser_status: str


def _normalize_answer(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in VALID_ANSWERS:
        return normalized
    return None


def _parse_json_object(raw: str) -> ParsedAnswer | None:
    stripped = raw.strip()
    candidates = [stripped]
    candidates.extend(match.group(0) for match in JSON_OBJECT_RE.finditer(stripped))

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        answer = _normalize_answer(obj.get("answer")) if isinstance(obj, dict) else None
        if answer is not None:
            return ParsedAnswer(answer=answer, strict_json_valid=True, parser_status="json")
    return None


def parse_answer(raw_output: str) -> ParsedAnswer:
    """Parse one raw output with the single frozen rule in PILOT_PROTOCOL_FREEZE."""
    json_answer = _parse_json_object(raw_output)
    if json_answer is not None:
        return json_answer

    tokens = [match.group(1).lower() for match in ANSWER_RE.finditer(raw_output)]
    unique_tokens = set(tokens)
    if not unique_tokens:
        return ParsedAnswer(answer=None, strict_json_valid=False, parser_status="missing")
    if len(unique_tokens) > 1:
        return ParsedAnswer(answer=None, strict_json_valid=False, parser_status="ambiguous")
    return ParsedAnswer(answer=tokens[0], strict_json_valid=False, parser_status="recovered")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _label(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    normalized = _normalize_answer(value)
    if normalized in {"yes", "no"}:
        return normalized
    return None


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        parsed = parse_answer(str(row.get("raw_output", "")))
        target_label = _label(row, "target_label")
        distractor_label = _label(row, "distractor_label")
        answer = parsed.answer

        plain_correct = answer == target_label if target_label and answer in {"yes", "no"} else False
        strict_json_correct = bool(parsed.strict_json_valid and plain_correct)
        false_positive = bool(target_label == "no" and answer == "yes")
        false_negative = bool(target_label == "yes" and answer == "no")
        conflict_valid = bool(target_label and distractor_label and target_label != distractor_label)
        wrong_sensor = bool(conflict_valid and answer == distractor_label and answer != target_label)

        output = dict(row)
        output.update(
            {
                "parsed_answer": answer,
                "strict_json_valid": parsed.strict_json_valid,
                "parser_status": parsed.parser_status,
                "plain_answer_em": plain_correct,
                "strict_json_em": strict_json_correct,
                "false_positive_presence": false_positive,
                "false_negative_miss": false_negative,
                "wrong_sensor_following": wrong_sensor,
                "abstention": answer == "abstain",
                "conflict_valid": conflict_valid,
            }
        )
        enriched.append(output)
    return enriched


def _risk_curve(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    scored = [row for row in rows if isinstance(row.get("risk_score"), (int, float))]
    if not scored:
        return {"aurc": None, "risk_at_80_coverage": None}
    scored.sort(key=lambda row: float(row["risk_score"]))
    errors = [0.0 if row["plain_answer_em"] else 1.0 for row in scored]
    cumulative_errors = 0.0
    risks: list[float] = []
    for index, error in enumerate(errors, start=1):
        cumulative_errors += error
        risks.append(cumulative_errors / index)
    target_index = max(1, int(0.8 * len(risks))) - 1
    return {"aurc": _mean(risks), "risk_at_80_coverage": risks[target_index]}


def summarize(enriched_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(enriched_rows)
    conflict_rows = [row for row in enriched_rows if row["conflict_valid"]]
    summary = {
        "n": total,
        "plain_answer_em": _mean([float(row["plain_answer_em"]) for row in enriched_rows]),
        "strict_json_em": _mean([float(row["strict_json_em"]) for row in enriched_rows]),
        "json_validity": _mean([float(row["strict_json_valid"]) for row in enriched_rows]),
        "false_positive_presence": _mean([float(row["false_positive_presence"]) for row in enriched_rows]),
        "false_negative_miss": _mean([float(row["false_negative_miss"]) for row in enriched_rows]),
        "wrong_sensor_following": _mean([float(row["wrong_sensor_following"]) for row in conflict_rows]),
        "abstention_rate": _mean([float(row["abstention"]) for row in enriched_rows]),
        "parser_status_counts": {},
    }
    counts: dict[str, int] = {}
    for row in enriched_rows:
        counts[row["parser_status"]] = counts.get(row["parser_status"], 0) + 1
    summary["parser_status_counts"] = counts
    summary.update(_risk_curve(enriched_rows))
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    assert_sanitized_data_path(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--parsed-output", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    enriched = enrich_rows(rows)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(summarize(enriched), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.parsed_output is not None:
        write_jsonl(args.parsed_output, enriched)


if __name__ == "__main__":
    main()
