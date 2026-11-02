"""128-token bridge that reuses the preserved Qwen3 native adapter."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

from qwen3_native_adapter import (  # type: ignore[import-not-found]
    NativeInterfaceError,
    Qwen3NativeAdapter,
    VISUAL_KEYS,
    _as_nested_list,
    _image_sha256,
    _shape,
)


class Qwen3Native128Adapter(Qwen3NativeAdapter):
    """Preserve frozen system/user roles while explicitly pinning 128 tokens."""

    def __init__(self, processor: Any, model: Any, max_new_tokens: int = 128):
        if max_new_tokens != 128:
            raise ValueError("model-chain protocol requires explicit max_new_tokens=128")
        super().__init__(processor=processor, model=model, max_new_tokens=128)

    @classmethod
    def from_pretrained(cls, model_path: str, max_new_tokens: int = 128):
        if max_new_tokens != 128:
            raise ValueError("model-chain protocol requires explicit max_new_tokens=128")
        # Reuse the legacy loader, but never its 16-token default.
        return super().from_pretrained(model_path, max_new_tokens=128)

    def prepare_inputs(self, row: Mapping[str, Any], images: Sequence[Any]):
        system_prompt = row.get("system_prompt")
        user_prompt = row.get("user_prompt")
        if not isinstance(system_prompt, str) or not system_prompt:
            raise ValueError("the frozen system_prompt is required")
        if not isinstance(user_prompt, str) or not user_prompt:
            raise ValueError("the frozen user_prompt is required")
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": user_prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
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
        prompt_ids = _as_nested_list(input_ids)
        if not prompt_ids or not prompt_ids[0]:
            raise NativeInterfaceError("processor returned empty prompt tokens")
        trace = {
            "message_roles": [message["role"] for message in messages],
            "system_prompt_exactly_preserved": messages[0]["content"][0]["text"] == system_prompt,
            "user_prompt_exactly_preserved": messages[1]["content"][-1]["text"] == user_prompt,
            "message_content_types": [item["type"] for item in content],
            "received_image_count": len(images),
            "processor_received_image_count": len(images),
            "image_pixel_sha256": [_image_sha256(image) for image in images],
            "image_sizes": [list(image.size) for image in images],
            "visual_tensor_keys": visual_keys,
            "visual_tensor_shapes": {key: _shape(encoded[key]) for key in visual_keys},
            "visual_input_present": bool(visual_keys),
            "processor_images_argument_omitted": not bool(images),
            "rendered_image_marker_count": len(images),
            "prompt_input_shape": _shape(input_ids),
            "prompt_token_count": len(prompt_ids[0]),
            "chat_template_tokenized_by_processor": True,
            "generation_prompt_added_by_processor": True,
            "native_template": getattr(self.processor, "chat_template", None),
            "max_new_tokens_configured": self.max_new_tokens,
        }
        return encoded, trace
