"""Small, model-independent helper for preserving generated-token termination evidence."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import time
from typing import Any


def _as_sequence(output: Any) -> list[int]:
    if output is None:
        raise ValueError("generation output is unavailable")
    if isinstance(output, Mapping):
        if "sequences" not in output:
            raise ValueError("generation mapping has no sequences field")
        output = output["sequences"]
    elif hasattr(output, "sequences"):
        output = output.sequences

    if hasattr(output, "detach"):
        output = output.detach()
    if hasattr(output, "cpu"):
        output = output.cpu()
    if hasattr(output, "tolist"):
        output = output.tolist()
    if not isinstance(output, (list, tuple)) or not output:
        raise ValueError("generation output is not a non-empty token sequence")

    first = output[0]
    if isinstance(first, (list, tuple)):
        if len(output) != 1:
            raise ValueError("technical fixture expects batch size one")
        output = first
    if not isinstance(output, (list, tuple)):
        raise ValueError("generation output is not a token sequence")
    try:
        return [int(token) for token in output]
    except (TypeError, ValueError) as exc:
        raise ValueError("generation output contains a non-integer token") from exc


def _normalize_eos(eos_token_ids: Any) -> list[int]:
    if eos_token_ids is None:
        return []
    if hasattr(eos_token_ids, "tolist"):
        eos_token_ids = eos_token_ids.tolist()
    if isinstance(eos_token_ids, int):
        return [eos_token_ids]
    if isinstance(eos_token_ids, (list, tuple, set)):
        try:
            return sorted({int(token) for token in eos_token_ids})
        except (TypeError, ValueError) as exc:
            raise ValueError("EOS configuration contains a non-integer token") from exc
    raise ValueError("unsupported EOS token configuration")


def build_generation_trace(
    output: Any,
    *,
    prompt_token_count: int,
    max_new_tokens: int,
    eos_token_ids: Any,
) -> dict[str, Any]:
    """Return generated-only token IDs and explicit EOS/cap evidence.

    ``output`` is the exact object returned by the model's generation call.
    The caller supplies the input-prefix length and the model-native EOS IDs
    captured from the pinned tokenizer/generation configuration.
    """
    if not isinstance(prompt_token_count, int) or prompt_token_count < 0:
        raise ValueError("prompt_token_count must be a non-negative integer")
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    sequence = _as_sequence(output)
    if prompt_token_count > len(sequence):
        raise ValueError("prompt token count exceeds returned sequence length")
    generated = sequence[prompt_token_count:]
    eos_ids = _normalize_eos(eos_token_ids)
    eos_offsets = [index for index, token in enumerate(generated) if token in eos_ids]
    eos_emitted = bool(eos_offsets)
    reached_limit = len(generated) >= max_new_tokens
    first_eos_offset = eos_offsets[0] if eos_offsets else None
    if eos_emitted and first_eos_offset == max_new_tokens - 1:
        termination = "eos_at_limit"
    elif eos_emitted:
        termination = "eos"
    elif reached_limit:
        termination = "length_limit"
    else:
        termination = "other"
    return {
        "generated_token_ids": generated,
        "generated_token_count": len(generated),
        "native_eos_token_ids": eos_ids,
        "native_eos_emitted": eos_emitted,
        "first_eos_offset": first_eos_offset,
        "new_token_limit": max_new_tokens,
        "new_token_limit_reached": reached_limit,
        "cap_hit_without_eos": reached_limit and not eos_emitted,
        "generated_tokens_exceed_limit": len(generated) > max_new_tokens,
        "termination_classification": termination,
    }


def _prompt_length(kwargs: dict[str, Any], args: tuple[Any, ...]) -> int:
    input_ids = kwargs.get("input_ids")
    if input_ids is None:
        input_ids = kwargs.get("inputs_embeds")
    if input_ids is None:
        input_ids = kwargs.get("attention_mask")
    if input_ids is None and args:
        input_ids = args[0]
    if input_ids is None:
        raise RuntimeError("cannot capture generation: low-level input_ids were not supplied")
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        if len(shape) == 1:
            return int(shape[0])
        if len(shape) >= 2 and int(shape[0]) == 1:
            return int(shape[1])
        raise RuntimeError("technical token trace requires one prompt sequence")
    if isinstance(input_ids, (list, tuple)):
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            if len(input_ids) != 1:
                raise RuntimeError("technical token trace requires batch size one")
            return len(input_ids[0])
        return len(input_ids)
    raise RuntimeError("unsupported low-level input_ids representation")


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def install_generation_trace_capture(
    model: Any,
    *,
    counts: dict[str, int],
    required_max_new_tokens: int,
    tokenizer_eos_token_ids: Any = None,
    synchronize: Any = None,
) -> Any:
    """Instrument the actual ``model.generate`` boundary without altering output.

    The original generation return object is returned unchanged. A trace is
    attached to ``model.protocol_generation_trace`` only after a successful
    low-level return; call counts are updated at that same boundary.
    """
    original_generate = model.generate
    generation_config = getattr(model, "generation_config", None)

    class CaptureState:
        def __init__(self):
            self.counts = counts
            self.last_trace: dict[str, Any] = {}
            self.last_seconds = 0.0
            self.last_error: str | None = None

    state = CaptureState()

    def traced_generate(*args: Any, **kwargs: Any) -> Any:
        state.last_trace = {}
        state.last_error = None
        model.protocol_generation_trace = None
        actual_limit = kwargs.get("max_new_tokens")
        if actual_limit != required_max_new_tokens:
            raise RuntimeError(
                f"max_new_tokens at model.generate was {actual_limit!r}; "
                f"required frozen value is {required_max_new_tokens}"
            )
        if kwargs.get("do_sample") is not False:
            raise RuntimeError("do_sample at model.generate must be False")
        prompt_count = _prompt_length(kwargs, args)
        active_eos = kwargs.get("eos_token_id")
        if active_eos is None and generation_config is not None:
            active_eos = getattr(generation_config, "eos_token_id", None)

        if synchronize is not None:
            synchronize()
        started = time.perf_counter()
        counts["invocations"] += 1
        try:
            output = original_generate(*args, **kwargs)
        except Exception:
            counts["errors"] += 1
            state.last_error = "model.generate raised before returning"
            raise
        finally:
            if synchronize is not None:
                synchronize()
            state.last_seconds = time.perf_counter() - started

        counts["successful_returns"] += 1
        trace = build_generation_trace(
            output,
            prompt_token_count=prompt_count,
            max_new_tokens=required_max_new_tokens,
            eos_token_ids=active_eos,
        )
        trace.update({
            "prompt_token_count": prompt_count,
            "model_generate_max_new_tokens": actual_limit,
            "model_generate_do_sample": kwargs.get("do_sample"),
            "tokenizer_eos_token_ids": _normalize_eos(tokenizer_eos_token_ids),
            "generated_token_ids_sha256": _json_sha256(trace["generated_token_ids"]),
        })
        state.last_trace = trace
        state.last_error = None
        model.protocol_generation_trace = trace
        return output

    model.generate = traced_generate
    return state
