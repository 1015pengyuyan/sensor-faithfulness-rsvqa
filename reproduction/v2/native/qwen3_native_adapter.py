"""Isolated native Hugging Face adapter for the pinned Qwen3-VL control.

This module consumes already-frozen prompts and image order. It does not
construct or alter scientific prompts, fixtures, parsers, or labels.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Sequence


VISUAL_KEYS = ("pixel_values", "image_grid_thw", "pixel_attention_mask")


class NativeInterfaceError(RuntimeError):
    """Raised when a processor output contradicts the requested image path."""


def build_messages(user_prompt: str, images: Sequence[Any]) -> list[dict[str, Any]]:
    """Create the official Qwen3-VL message content in the supplied image order."""
    if not isinstance(user_prompt, str) or not user_prompt:
        raise ValueError("user_prompt must be a non-empty string")
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": user_prompt})
    return [{"role": "user", "content": content}]


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return [int(dim) for dim in shape] if shape is not None else None


def _as_nested_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else list(value)


def _move_inputs(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, Mapping):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    raise TypeError("processor must return a mapping-like BatchEncoding")


def _image_sha256(image: Any) -> str:
    rgb = image.convert("RGB")
    return hashlib.sha256(rgb.tobytes()).hexdigest()


class Qwen3NativeAdapter:
    """Thin native-interface adapter; model loading is opt-in and local-only."""

    def __init__(self, processor: Any, model: Any, max_new_tokens: int):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.processor = processor
        self.model = model
        self.max_new_tokens = int(max_new_tokens)

    @classmethod
    def from_pretrained(cls, model_path: str, max_new_tokens: int = 16):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=False,
            local_files_only=True,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=False,
            local_files_only=True,
        )
        model.eval()
        return cls(processor=processor, model=model, max_new_tokens=max_new_tokens)

    def prepare_inputs(self, row: Mapping[str, Any], images: Sequence[Any]):
        prompt = row["user_prompt"]
        messages = build_messages(prompt, images)
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("apply_chat_template must return a mapping-like BatchEncoding")

        visual_keys = [key for key in VISUAL_KEYS if key in encoded and encoded[key] is not None]
        if images and not visual_keys:
            raise NativeInterfaceError("images were supplied but processor produced no visual tensors")
        if not images and visual_keys:
            raise NativeInterfaceError("text-only input unexpectedly produced visual tensors")

        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise NativeInterfaceError("processor produced no input_ids")
        return encoded, {
            "message_content_types": [item["type"] for item in messages[0]["content"]],
            "prompt_exactly_preserved": messages[0]["content"][-1]["text"] == prompt,
            "received_image_count": len(images),
            "image_pixel_sha256": [_image_sha256(image) for image in images],
            "image_sizes": [list(image.size) for image in images],
            "visual_tensor_keys": visual_keys,
            "visual_tensor_shapes": {key: _shape(encoded[key]) for key in visual_keys},
            "visual_input_present": bool(visual_keys),
            "prompt_input_shape": _shape(input_ids),
            "prompt_token_count": len(_as_nested_list(input_ids)[0]),
            "chat_template_tokenized_by_processor": True,
            "generation_prompt_added_by_processor": True,
        }

    def generate(self, row: Mapping[str, Any], images: Sequence[Any]):
        encoded, trace = self.prepare_inputs(row, images)
        model_inputs = _move_inputs(encoded, getattr(self.model, "device", "cpu"))
        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        sequences = _as_nested_list(generated)
        if not sequences:
            raise NativeInterfaceError("generate returned no sequences")
        prompt_token_count = trace["prompt_token_count"]
        full_sequence = [int(token) for token in sequences[0]]
        if len(full_sequence) < prompt_token_count:
            raise NativeInterfaceError("generated sequence is shorter than its input prompt")
        generated_ids = full_sequence[prompt_token_count:]
        decoded = self.processor.batch_decode(
            [generated_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw = decoded[0] if decoded else ""
        generation_config = getattr(self.model, "generation_config", None)
        eos_value = getattr(generation_config, "eos_token_id", None)
        if eos_value is None:
            eos_value = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None)
        eos_ids = set(eos_value if isinstance(eos_value, (list, tuple, set)) else ([eos_value] if eos_value is not None else []))
        ended_with_eos = bool(generated_ids and generated_ids[-1] in eos_ids)
        trace.update({
            "model_input_shapes": {
                key: _shape(value) for key, value in model_inputs.items()
                if key in ("input_ids", *VISUAL_KEYS) and value is not None
            },
            "generated_sequence_shape": _shape(generated),
            "generated_token_ids": generated_ids,
            "generated_token_count": len(generated_ids),
            "eos_token_ids": sorted(int(token) for token in eos_ids),
            "ended_with_eos": ended_with_eos,
            "hit_new_token_cap_without_eos": len(generated_ids) >= self.max_new_tokens and not ended_with_eos,
            "raw_output": raw,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        })
        return raw, trace
