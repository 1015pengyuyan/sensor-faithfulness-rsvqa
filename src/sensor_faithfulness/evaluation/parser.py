"""The single model- and condition-independent parser used in the paper."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


VALID_ANSWERS = {"yes", "no", "abstain"}
ANSWER_RE = re.compile(r"\b(yes|no|abstain)\b", flags=re.IGNORECASE)
JSON_OBJECT_RE = re.compile(r"\{.*?\}", flags=re.DOTALL)


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str | None
    strict_json_valid: bool
    parser_status: str


def _normalize(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in VALID_ANSWERS else None


def _parse_json(raw: str) -> ParsedAnswer | None:
    candidates = [raw.strip()]
    candidates.extend(match.group(0) for match in JSON_OBJECT_RE.finditer(raw.strip()))
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        answer = _normalize(obj.get("answer")) if isinstance(obj, dict) else None
        if answer is not None:
            return ParsedAnswer(answer, True, "json")
    return None


def parse_answer(raw_output: str) -> ParsedAnswer:
    parsed_json = _parse_json(raw_output)
    if parsed_json is not None:
        return parsed_json
    tokens = [match.group(1).lower() for match in ANSWER_RE.finditer(raw_output)]
    unique = set(tokens)
    if not unique:
        return ParsedAnswer(None, False, "missing")
    if len(unique) > 1:
        return ParsedAnswer(None, False, "ambiguous")
    return ParsedAnswer(tokens[0], False, "recovered")
