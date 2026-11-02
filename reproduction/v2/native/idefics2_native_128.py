"""Independent Idefics2 adapter revision for native typed chat content.

The only protocol-facing correction is the representation of the unchanged
frozen system instruction as the typed text segment required by the pinned
Idefics2 chat template. The fixed task strings and generation settings are
passed through without rewriting.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def _find_admission_root() -> Path:
    return Path(__file__).resolve().parent


ADMISSION_ROOT = _find_admission_root()
MODELS_ROOT = Path(__file__).resolve().parent
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from hf_native_runtime import (  # noqa: E402
    NativeAdapterError,
    NativeHFAdapter,
    image_sha256,
    tensor_shape,
    text_sha256,
)


class NativeIdefics2Adapter(NativeHFAdapter):
    model_id = "idefics2_8b"
    visual_keys = frozenset({"pixel_values", "pixel_attention_mask", "image_attention_mask"})

    @classmethod
    def from_pretrained(cls, model_path: str, max_new_tokens: int = 128):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForVision2Seq.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map="auto",
        ).eval()
        return cls(processor, model, max_new_tokens=max_new_tokens, torch_module=torch)

    @staticmethod
    def _messages(row: Mapping[str, Any], images: Sequence[Any]) -> list[dict[str, Any]]:
        system_text = row.get("system_prompt")
        user_text = row.get("user_prompt")
        if not system_text or not user_text:
            raise ValueError("the frozen system_prompt and user_prompt are required")
        user_content = [{"type": "image"} for _ in images]
        user_content.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ]

    def prepare_inputs(self, row: Mapping[str, Any], images: Sequence[Any]):
        messages = self._messages(row, images)
        if messages[0]["content"][0]["text"] != row["system_prompt"]:
            raise NativeAdapterError("typed system content differs from the frozen instruction")
        if messages[1]["content"][-1] != {"type": "text", "text": row["user_prompt"]}:
            raise NativeAdapterError("typed user content differs from the frozen instruction")

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        system_pos = prompt.find(row["system_prompt"])
        user_pos = prompt.find(row["user_prompt"])
        if system_pos < 0 or user_pos < 0:
            raise NativeAdapterError("Idefics2 native template omitted frozen instruction text")
        if system_pos >= user_pos:
            raise NativeAdapterError("Idefics2 rendered message order differs from system then user")

        image_token = (
            getattr(self.processor, "image_token", None)
            or getattr(getattr(self.processor, "tokenizer", None), "image_token", None)
            or "<image>"
        )
        image_marker_count = prompt.count(str(image_token))
        if image_marker_count != len(images):
            raise NativeAdapterError(
                f"Idefics2 native template rendered {image_marker_count} image markers for {len(images)} images"
            )

        processor_call: dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
        if images:
            processor_call["images"] = list(images)
        inputs = self.processor(**processor_call)
        visual_keys = sorted(
            key for key in self.visual_keys if key in inputs and inputs[key] is not None
        )
        if images and "pixel_values" not in visual_keys:
            raise NativeAdapterError("Idefics2 multi-image processor returned no pixel_values")
        if not images and visual_keys:
            raise NativeAdapterError("Idefics2 text-only processor returned visual tensors")
        visual_shapes = {key: tensor_shape(inputs[key]) for key in visual_keys}
        if images and any(
            shape is not None and any(dimension == 0 for dimension in shape)
            for shape in visual_shapes.values()
        ):
            raise NativeAdapterError("Idefics2 processor returned an empty visual tensor")

        ordered_image_hashes = [image_sha256(image) for image in images]
        template = getattr(self.processor, "chat_template", None)
        trace = {
            "native_interface": "AutoProcessor typed chat-template + AutoModelForVision2Seq",
            "message_roles": [message["role"] for message in messages],
            "message_content_types": [
                [segment["type"] for segment in message["content"]]
                for message in messages
            ],
            "system_text_sha256": text_sha256(row["system_prompt"]),
            "user_text_sha256": text_sha256(row["user_prompt"]),
            "system_text_present_in_rendered_prompt": True,
            "user_text_present_in_rendered_prompt": True,
            "rendered_system_before_user": True,
            "rendered_prompt_sha256": text_sha256(prompt),
            "processor_chat_template_sha256": text_sha256(template) if isinstance(template, str) else None,
            "image_marker": str(image_token),
            "image_marker_count": image_marker_count,
            "image_count": len(images),
            "image_sha256": ordered_image_hashes,
            "processor_image_order_sha256": ordered_image_hashes,
            "processor_input_image_order_sha256": ordered_image_hashes,
            "image_sizes": [list(image.size) for image in images],
            "processor_images_argument_omitted": "images" not in processor_call,
            "processor_images_argument_present": "images" in processor_call,
            "visual_tensor_keys": visual_keys,
            "visual_tensor_shapes": visual_shapes,
        }
        return inputs, trace
