"""Qwen2.5-VL-7B native Transformers adapter for the unified 128-token matrix."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

MODELS_ROOT = Path(__file__).resolve().parent
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from hf_native_runtime import NativeAdapterError, NativeHFAdapter, image_sha256, tensor_shape, text_sha256  # noqa: E402


class NativeQwen25VL7BAdapter(NativeHFAdapter):
    model_id = "qwen25_vl_7b"
    visual_keys = frozenset({"pixel_values", "image_grid_thw"})

    @classmethod
    def from_pretrained(cls, model_path: str, max_new_tokens: int = 128):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map="auto",
        ).eval()
        return cls(processor, model, max_new_tokens=max_new_tokens, torch_module=torch)

    def prepare_inputs(self, row: Mapping[str, Any], images: Sequence[Any]):
        if not row.get("system_prompt") or not row.get("user_prompt"):
            raise ValueError("the frozen system_prompt and user_prompt are required")
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": row["user_prompt"]})
        messages = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": content},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if row["system_prompt"] not in prompt or row["user_prompt"] not in prompt:
            raise NativeAdapterError("Qwen2.5-VL native template omitted frozen instruction text")
        call = {"text": [prompt], "return_tensors": "pt", "padding": True}
        if images:
            call["images"] = list(images)
        inputs = self.processor(**call)
        visual_keys = sorted(
            key for key in self.visual_keys if key in inputs and inputs[key] is not None
        )
        if images and not {"pixel_values", "image_grid_thw"}.issubset(visual_keys):
            raise NativeAdapterError("Qwen2.5-VL multi-image processor omitted visual payload or grid")
        if not images and visual_keys:
            raise NativeAdapterError("Qwen2.5-VL text-only processor returned visual tensors")
        trace = {
            "native_interface": "Qwen2_5_VLForConditionalGeneration + AutoProcessor chat template",
            "system_text_present_in_rendered_prompt": row["system_prompt"] in prompt,
            "user_text_present_in_rendered_prompt": row["user_prompt"] in prompt,
            "image_count": len(images),
            "image_sha256": [image_sha256(image) for image in images],
            "image_sizes": [list(image.size) for image in images],
            "processor_image_order_sha256": [image_sha256(image) for image in images],
            "processor_images_argument_omitted": not bool(images),
            "visual_tensor_keys": visual_keys,
            "visual_tensor_shapes": {key: tensor_shape(inputs[key]) for key in visual_keys},
            "processor_chat_template": getattr(self.processor, "chat_template", None),
            "rendered_prompt_sha256": text_sha256(prompt),
        }
        return inputs, trace

