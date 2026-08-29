"""Deterministic A0/A2/A3 fixture construction from a validated source catalog.

The catalog must be prepared from dataset annotations, never model outputs. Each
source supplies binary labels for the same normalized questions and one optical
and one SAR image reference. Matching metadata are used only for deterministic
distractor ordering; they never change labels or the requested source.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from .prompt_protocol import SYSTEM_PROMPT, build_user_prompt


CONDITIONS = (
    ("A0_CONGRUENT", 0, "congruent"),
    ("A2_CONFLICT", 0, "opposite"),
    ("A3_CONFLICT_REPLICATE_1", 1, "opposite"),
    ("A3_CONFLICT_REPLICATE_2", 2, "opposite"),
)


def stable_key(seed: str, *parts: Any) -> str:
    payload = "||".join([seed, *(str(part) for part in parts)]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _answer(source: dict[str, Any], question_id: str) -> str:
    answer = str(source["answers"][question_id]).lower()
    if answer not in {"yes", "no"}:
        raise ValueError(f"non-binary answer for {source['source_id']}/{question_id}: {answer}")
    return answer


def _matching_cost(target: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, ...]:
    keys = ("complexity", "object_density", "object_size")
    cost = []
    for key in keys:
        left = target.get(key)
        right = candidate.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            cost.append(abs(float(left) - float(right)))
        else:
            cost.append(0.0 if left == right else 1.0)
    return tuple(cost)


def select_distractors(
    target: dict[str, Any],
    sources: Sequence[dict[str, Any]],
    question_id: str,
    seed: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_answer = _answer(target, question_id)
    candidates = [source for source in sources if str(source["source_id"]) != str(target["source_id"])]
    congruent = [source for source in candidates if _answer(source, question_id) == target_answer]
    opposite = [source for source in candidates if _answer(source, question_id) != target_answer]
    sort_key = lambda source: (
        _matching_cost(target, source),
        stable_key(seed, target["source_id"], question_id, source["source_id"]),
    )
    congruent.sort(key=sort_key)
    opposite.sort(key=sort_key)
    if not congruent or len(opposite) < 3:
        raise ValueError(f"insufficient distractors for {target['source_id']}/{question_id}")
    return congruent[0], opposite[:3]


def _image_ref(source: dict[str, Any], sensor: str, role: str, slot: int) -> dict[str, Any]:
    ref = dict(source["images"][sensor])
    ref.update({"source_id": str(source["source_id"]), "sensor": sensor, "role": role, "slot": slot})
    return ref


def construct_target_rows(
    target: dict[str, Any],
    sources: Sequence[dict[str, Any]],
    question_id: str,
    question_text: str,
    seed: str,
    image_order: str,
) -> list[dict[str, Any]]:
    congruent, opposites = select_distractors(target, sources, question_id, seed)
    target_answer = _answer(target, question_id)
    rows: list[dict[str, Any]] = []
    for requested in ("optical", "sar"):
        non_requested = "sar" if requested == "optical" else "optical"
        for condition, replicate_index, relation in CONDITIONS:
            distractor = congruent if relation == "congruent" else opposites[replicate_index]
            distractor_answer = _answer(distractor, question_id)
            if image_order == "requested_first":
                refs = [
                    _image_ref(target, requested, "target_requested", 1),
                    _image_ref(distractor, non_requested, f"nonrequested_{relation}", 2),
                ]
            elif image_order == "optical_first_then_sar":
                optical_source = target if requested == "optical" else distractor
                sar_source = target if requested == "sar" else distractor
                refs = [
                    _image_ref(optical_source, "optical", "target_requested" if requested == "optical" else f"nonrequested_{relation}", 1),
                    _image_ref(sar_source, "sar", "target_requested" if requested == "sar" else f"nonrequested_{relation}", 2),
                ]
            else:
                raise ValueError(image_order)
            rows.append({
                "fixture_id": f"{target['source_id']}_{question_id}_{requested}_{condition}",
                "example_id": str(target["source_id"]),
                "source_id": str(target["source_id"]),
                "question_id": question_id,
                "question": question_text,
                "requested_sensor": requested,
                "non_requested_sensor": non_requested,
                "condition_id": condition,
                "condition_relation": relation,
                "target_label": target_answer,
                "distractor_label": distractor_answer,
                "target_source_id": str(target["source_id"]),
                "distractor_source_id": str(distractor["source_id"]),
                "image_order": image_order,
                "image_refs": refs,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": build_user_prompt(question_text, requested, image_order),
                "bootstrap_unit": "target_source_cluster",
                "selection_seed": seed,
            })
    return rows
