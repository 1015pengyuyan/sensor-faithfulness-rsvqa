"""Frozen prompt text and image-order helpers.

M4-SAR used requested-evidence-first ordering. BigEarthNet.txt used the fixed
optical-first/SAR-second interface. Fixtures embed the exact prompts used; these
helpers reproduce either frozen construction without changing semantics.
"""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = (
    "You answer remote-sensing visual questions. Use only the requested sensor "
    "evidence named in the user instruction. Non-requested sensor evidence may "
    "be present as a distractor and must not determine the answer. Return exactly "
    "one JSON object: {\"answer\":\"yes|no|abstain\"}."
)
ANSWER_SCHEMA = '{"answer":"yes|no|abstain"}'


def ordered_image_refs(image_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(image_refs, key=lambda ref: int(ref["slot"]))
    if [int(ref["slot"]) for ref in ordered] != [1, 2]:
        raise ValueError("the frozen dual-image interface requires slots 1 and 2")
    return ordered


def build_user_prompt(
    question: str,
    requested_sensor: str,
    image_order: str = "optical_first_then_sar",
) -> str:
    requested = requested_sensor.lower()
    if requested not in {"optical", "sar"}:
        raise ValueError(f"unsupported requested sensor: {requested_sensor}")
    if image_order == "requested_first":
        non_requested = "sar" if requested == "optical" else "optical"
        first = f"Image 1 is the requested {requested.upper()} evidence."
        second = f"Image 2 is non-requested {non_requested.upper()} evidence."
        instruction = "Answer using only Image 1. Ignore Image 2 for the final answer, even if it appears relevant."
    elif image_order == "optical_first_then_sar":
        if requested == "optical":
            first = "Image 1 is the requested OPTICAL evidence."
            second = "Image 2 is non-requested SAR evidence."
            instruction = "Answer using only Image 1. Ignore Image 2 for the final answer, even if it appears relevant."
        else:
            first = "Image 1 is non-requested OPTICAL evidence."
            second = "Image 2 is the requested SAR evidence."
            instruction = "Answer using only Image 2. Ignore Image 1 for the final answer, even if it appears relevant."
    else:
        raise ValueError(f"unsupported image order: {image_order}")
    return (
        f"{first} {second} {instruction}\n"
        f"Question: {question}\n"
        "Candidate answers: yes, no, abstain.\n"
        f"Return exactly one JSON object matching this schema: {ANSWER_SCHEMA}"
    )
