"""Shared, CPU-testable mechanics for isolated Transformers adapters.

This module does not create prompts or parse answers. It consumes the exact
system/user text supplied by a frozen matrix row and records only native
processor and generation-boundary evidence.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROTOCOL_ROOT = Path(__file__).resolve().parent
if str(PROTOCOL_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_ROOT))

from token_trace import build_generation_trace  # noqa: E402


class NativeAdapterError(RuntimeError):
    """The model-native interface did not preserve the requested input path."""


def image_sha256(image: Any) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(size) for size in shape]


def move_inputs(inputs: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def eos_ids_for(model: Any, processor: Any) -> Any:
    config = getattr(model, "generation_config", None)
    value = getattr(config, "eos_token_id", None)
    if value is None:
        value = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    return value


class NativeHFAdapter:
    """Common generation boundary; subclasses only implement native input packing."""

    model_id = ""
    visual_keys: frozenset[str] = frozenset()

    def __init__(self, processor: Any, model: Any, max_new_tokens: int = 128, torch_module: Any = None):
        if max_new_tokens != 128:
            raise ValueError("unified 128-token matrix protocol requires max_new_tokens=128")
        self.processor = processor
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.torch = torch_module
        if self.torch is None:
            import torch
            self.torch = torch
        self.counts = {"invocations": 0, "successful_returns": 0, "errors": 0}
        self.last_trace: dict[str, Any] = {}

    @classmethod
    def from_pretrained(cls, model_path: str, max_new_tokens: int = 128):
        raise NotImplementedError

    def prepare_inputs(self, row: Mapping[str, Any], images: Sequence[Any]):
        raise NotImplementedError

    def generate(self, row: Mapping[str, Any], images: Sequence[Any]):
        self.last_trace = {}
        inputs, trace = self.prepare_inputs(row, images)
        self.last_trace = trace
        parameters = getattr(self.model, "parameters", None)
        device = getattr(self.model, "device", "cpu")
        if callable(parameters):
            first_parameter = next(parameters(), None)
            if first_parameter is not None:
                device = getattr(first_parameter, "device", device)
        model_inputs = move_inputs(inputs, device)
        trace["model_input_keys"] = sorted(model_inputs)
        trace["model_input_visual_tensor_keys"] = sorted(
            key for key in self.visual_keys
            if key in model_inputs and model_inputs[key] is not None
        )
        trace["model_input_shapes"] = {
            key: tensor_shape(value) for key, value in model_inputs.items()
            if key == "input_ids" or key in self.visual_keys
        }
        if images:
            if not trace["model_input_visual_tensor_keys"]:
                raise NativeAdapterError("image input produced no visual model tensor")
            for key in trace["model_input_visual_tensor_keys"]:
                shape = trace["model_input_shapes"].get(key)
                if shape is not None and any(dimension == 0 for dimension in shape):
                    raise NativeAdapterError(f"image input produced an empty visual tensor: {key}")
        elif trace["model_input_visual_tensor_keys"]:
            raise NativeAdapterError("text-only route produced visual model tensors")

        input_ids = model_inputs.get("input_ids")
        if input_ids is None or tensor_shape(input_ids) is None or len(tensor_shape(input_ids)) < 2:
            raise NativeAdapterError("native processor did not return batched input_ids")
        prompt_length = tensor_shape(input_ids)[-1]
        trace["prompt_token_count"] = prompt_length
        self.torch.cuda.synchronize() if getattr(getattr(self.torch, "cuda", None), "is_available", lambda: False)() else None
        with self.torch.inference_mode():
            self.counts["invocations"] += 1
            try:
                generated = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    return_dict_in_generate=False,
                )
            except Exception:
                self.counts["errors"] += 1
                raise
            self.counts["successful_returns"] += 1
        token_trace = build_generation_trace(
            generated,
            prompt_token_count=prompt_length,
            max_new_tokens=self.max_new_tokens,
            eos_token_ids=eos_ids_for(self.model, self.processor),
        )
        token_trace["model_generate_max_new_tokens"] = self.max_new_tokens
        token_trace["model_generate_do_sample"] = False
        token_trace["model_generate_num_beams"] = 1
        trace["generation_token_trace"] = token_trace
        raw = self.processor.batch_decode(
            [token_trace["generated_token_ids"]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        trace["raw_output"] = raw
        return raw, trace

