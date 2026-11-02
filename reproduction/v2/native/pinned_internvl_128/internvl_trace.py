"""InternVL-specific trace for input-ID and inputs-embeds generation paths.

The pinned InternVL chat implementation calls its language model with
``inputs_embeds`` only. Job 7765 recorded that this boundary returns only newly
generated token IDs. That path therefore preserves the full returned sequence;
it never subtracts the embedding context length. If input IDs are supplied,
the returned sequence must instead contain their exact prefix or tracing fails.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import time
from typing import Any


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _single_sequence(value: Any, name: str) -> list[int]:
    value = _plain(value)
    if isinstance(value, Mapping):
        if "sequences" not in value:
            raise ValueError(f"{name} mapping has no sequences field")
        value = _plain(value["sequences"])
    elif hasattr(value, "sequences"):
        value = _plain(value.sequences)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} is not a non-empty token sequence")
    if isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("technical InternVL trace requires batch size one")
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} is not a token sequence")
    try:
        return [int(token) for token in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a non-integer token") from exc


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported generation-input shape") from exc


def _eos_ids(value: Any) -> list[int]:
    value = _plain(value)
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple, set)):
        try:
            return sorted({int(item) for item in value})
        except (TypeError, ValueError) as exc:
            raise ValueError("EOS configuration contains a non-integer token") from exc
    raise ValueError("unsupported EOS token configuration")


def _input_schema(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, list[int] | None, list[int] | None, list[int] | None]:
    input_ids = kwargs.get("input_ids")
    inputs_embeds = kwargs.get("inputs_embeds")
    if input_ids is None and inputs_embeds is None and args:
        first_shape = _shape(args[0])
        if first_shape and len(first_shape) >= 3:
            inputs_embeds = args[0]
        else:
            input_ids = args[0]
    input_ids_shape = _shape(input_ids) if input_ids is not None else None
    embeds_shape = _shape(inputs_embeds) if inputs_embeds is not None else None
    ids: list[int] | None = None
    if input_ids is not None:
        ids = _single_sequence(input_ids, "input_ids")
        if input_ids_shape and (len(input_ids_shape) > 2 or (len(input_ids_shape) == 2 and input_ids_shape[0] != 1)):
            raise ValueError("technical InternVL trace requires one input-id sequence")
        return "input_ids_prefix", ids, input_ids_shape, embeds_shape
    if inputs_embeds is not None:
        if embeds_shape is None or len(embeds_shape) != 3 or embeds_shape[0] != 1:
            raise ValueError("inputs_embeds must have shape [1, sequence, hidden]")
        if embeds_shape[1] < 1:
            raise ValueError("inputs_embeds sequence length must be positive")
        return "inputs_embeds_generation_only", None, input_ids_shape, embeds_shape
    raise RuntimeError("cannot trace InternVL generate: neither input_ids nor inputs_embeds was supplied")


def build_internvl_generation_trace(
    output: Any,
    *,
    input_ids: Any,
    inputs_embeds: Any,
    max_new_tokens: int,
    eos_token_ids: Any,
    positional_input: Any = None,
) -> dict[str, Any]:
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    args = (positional_input,) if positional_input is not None else ()
    mode, ids, ids_shape, embeds_shape = _input_schema(args, {
        "input_ids": input_ids, "inputs_embeds": inputs_embeds,
    })
    returned = _single_sequence(output, "generation output")
    if ids is not None:
        if len(returned) < len(ids):
            raise ValueError("returned sequence is shorter than the input_ids prefix")
        if returned[:len(ids)] != ids:
            raise ValueError("returned sequence input_ids prefix does not match")
        prefix_length = len(ids)
        generated = returned[prefix_length:]
        context_length = len(ids)
    else:
        # The pinned InternVL35 implementation uses inputs_embeds-only at this
        # boundary and job 7765 captured generated-only returned sequences.
        # Do not crop by the embedding context length or infer a prefix.
        prefix_length = 0
        generated = returned
        context_length = int(embeds_shape[1])  # validated in _input_schema
        if len(generated) > max_new_tokens:
            raise ValueError(
                "inputs_embeds-only returned sequence exceeds max_new_tokens; "
                "the pinned generation-only return schema does not match"
            )
    if len(generated) > max_new_tokens:
        raise ValueError("generated sequence exceeds max_new_tokens")

    eos_ids = _eos_ids(eos_token_ids)
    eos_positions = [i for i, token in enumerate(generated) if token in eos_ids]
    eos_emitted = bool(eos_positions)
    limit_reached = len(generated) >= max_new_tokens
    first_eos = eos_positions[0] if eos_positions else None
    if eos_emitted and first_eos == max_new_tokens - 1:
        termination = "eos_at_limit"
    elif eos_emitted:
        termination = "eos"
    elif limit_reached:
        termination = "length_limit"
    elif not generated:
        termination = "empty_generation"
    else:
        termination = "other"

    ids_digest = lambda values: hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("ascii")).hexdigest()
    return {
        "return_sequence_mode": mode,
        "input_ids_shape": ids_shape,
        "input_ids_prefix_length": len(ids) if ids is not None else 0,
        "input_ids_prefix_sha256": ids_digest(ids) if ids is not None else None,
        "input_id_prefix_verified": True if ids is not None else None,
        "inputs_embeds_shape": embeds_shape,
        "model_context_token_count": context_length,
        "returned_sequence_ids": returned,
        "returned_sequence_ids_sha256": ids_digest(returned),
        "returned_sequence_token_count": len(returned),
        "generation_sequence_prefix_length": prefix_length,
        "generated_token_ids": generated,
        "generated_token_ids_sha256": ids_digest(generated),
        "generated_token_count": len(generated),
        "prompt_token_count": context_length,
        "native_eos_token_ids": eos_ids,
        "native_eos_emitted": eos_emitted,
        "first_eos_offset": first_eos,
        "new_token_limit": max_new_tokens,
        "new_token_limit_reached": limit_reached,
        "cap_hit_without_eos": limit_reached and not eos_emitted,
        "generated_tokens_exceed_limit": False,
        "termination_classification": termination,
    }


def install_internvl_generation_trace_capture(
    model: Any,
    *,
    counts: dict[str, int],
    required_max_new_tokens: int,
    tokenizer_eos_token_ids: Any = None,
    synchronize: Any = None,
) -> Any:
    """Wrap the exact model.generate boundary and preserve its return object."""
    original_generate = model.generate
    generation_config = getattr(model, "generation_config", None)

    class CaptureState:
        def __init__(self):
            self.counts = counts
            self.last_trace: dict[str, Any] = {}
            self.last_seconds = 0.0
            self.last_error: str | None = None
            self.trace_validation_errors = 0

    state = CaptureState()

    def traced_generate(*args: Any, **kwargs: Any) -> Any:
        state.last_trace = {}
        state.last_error = None
        model.protocol_generation_trace = None
        actual_limit = kwargs.get("max_new_tokens")
        if actual_limit != required_max_new_tokens:
            raise RuntimeError(f"max_new_tokens at model.generate was {actual_limit!r}; required {required_max_new_tokens}")
        if kwargs.get("do_sample") is not False:
            raise RuntimeError("do_sample at model.generate must be False")
        # Validate that the low-level call provides one of the two traceable
        # input schemas before counting an actual underlying generate call.
        mode, ids, ids_shape, embeds_shape = _input_schema(args, kwargs)
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
        try:
            trace = build_internvl_generation_trace(
                output,
                input_ids=ids,
                inputs_embeds=None if ids is not None else type("InputShape", (), {"shape": embeds_shape})(),
                positional_input=None,
                max_new_tokens=required_max_new_tokens,
                eos_token_ids=active_eos,
            )
            trace.update({
                "model_generate_max_new_tokens": actual_limit,
                "model_generate_do_sample": kwargs.get("do_sample"),
                "tokenizer_eos_token_ids": _eos_ids(tokenizer_eos_token_ids),
                "generation_output_type": type(output).__name__,
            })
        except Exception as exc:
            state.trace_validation_errors += 1
            state.last_error = f"generation return trace validation failed: {type(exc).__name__}: {exc}"
            raise
        state.last_trace = trace
        model.protocol_generation_trace = trace
        return output

    model.generate = traced_generate
    return state
