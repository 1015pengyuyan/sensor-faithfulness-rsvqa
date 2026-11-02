"""Independent LLaVA-OneVision adapter with explicit native no-image trace.

The processor inputs, frozen task text, image ordering, decoding parameters,
and token budget match the prior fixed adapter. This revision only records
that the image argument is genuinely omitted on the text-only branch, which
the unchanged shared integrity gate requires as auditable evidence.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Any


def _find_protocol_root() -> Path:
    return Path(__file__).resolve().parent


PROTOCOL_ROOT = Path(__file__).resolve().parent
if str(PROTOCOL_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_ROOT))

from token_trace import build_generation_trace  # noqa: E402


VISUAL_KEYS = {"pixel_values", "image_sizes", "pixel_attention_mask", "image_grid_thw"}


class NativeInputError(RuntimeError):
    def __init__(self, message: str, trace: dict[str, Any]):
        super().__init__(message)
        self.trace = trace


def _shape_map(inputs: dict[str, Any]) -> dict[str, list[int]]:
    return {
        key: [int(dimension) for dimension in value.shape]
        for key, value in inputs.items()
        if hasattr(value, "shape")
    }


def prepare_native_inputs(processor: Any, row: dict[str, Any], images: list[Any]):
    if not row.get("system_prompt") or not row.get("user_prompt"):
        raise ValueError("the frozen system_prompt and user_prompt are required")
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": row["user_prompt"]})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": row["system_prompt"]}]},
        {"role": "user", "content": content},
    ]
    started = time.perf_counter()
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    template_seconds = time.perf_counter() - started

    processor_call: dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
    if images:
        processor_call["images"] = list(images)
    started = time.perf_counter()
    inputs = processor(**processor_call)
    process_seconds = time.perf_counter() - started
    shapes = _shape_map(inputs)
    visual_keys = sorted(VISUAL_KEYS.intersection(inputs))
    system_position = prompt.find(row["system_prompt"])
    user_position = prompt.find(row["user_prompt"])
    trace = {
        "images_argument_present": "images" in processor_call,
        "processor_images_argument_omitted": "images" not in processor_call,
        "processor_images_argument_present": "images" in processor_call,
        "received_image_count": len(images),
        "image_count": len(images),
        "image_marker_count": prompt.count("<image>"),
        "image_pixel_sha256": [
            hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() for image in images
        ],
        "processor_image_order_sha256": [
            hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() for image in images
        ],
        "image_sizes": [list(image.size) for image in images],
        "system_text_present_in_rendered_prompt": system_position >= 0,
        "user_text_present_in_rendered_prompt": user_position >= 0,
        "rendered_system_before_user": system_position >= 0 and user_position >= 0 and system_position < user_position,
        "message_roles": [message["role"] for message in messages],
        "message_content_types": [[segment["type"] for segment in message["content"]] for message in messages],
        "input_shapes": shapes,
        "visual_tensor_keys": visual_keys,
        "processor_output_visual_keys": visual_keys,
        "chat_template_seconds": template_seconds,
        "processor_seconds": process_seconds,
    }
    if not trace["system_text_present_in_rendered_prompt"] or not trace["user_text_present_in_rendered_prompt"]:
        raise NativeInputError("official template omitted supplied frozen instruction text", trace)
    if not trace["rendered_system_before_user"]:
        raise NativeInputError("native template rendered frozen user text before system text", trace)
    if images:
        if (
            trace["image_marker_count"] != len(images)
            or shapes.get("pixel_values", [None])[0] != len(images)
            or shapes.get("image_sizes", [None])[0] != len(images)
        ):
            raise NativeInputError("native multi-image tensor/count mismatch", trace)
        if not visual_keys:
            raise NativeInputError("native multi-image processor returned no visual tensors", trace)
    elif trace["image_marker_count"] or visual_keys or not trace["processor_images_argument_omitted"]:
        raise NativeInputError("native text-only input included an image argument or visual payload", trace)
    return inputs, trace


class NativeLlavaAdapter:
    def __init__(self, model_path: str, max_new_tokens: int = 128):
        if max_new_tokens != 128:
            raise ValueError("unified 128-token matrix protocol requires max_new_tokens=128")
        import torch
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16,
            device_map="auto", low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.counts = {"invocations": 0, "successful_returns": 0, "errors": 0}
        self.last_trace: dict[str, Any] = {}

    def generate(self, row: dict[str, Any], images: list[Any]):
        self.last_trace = {}
        inputs, trace = prepare_native_inputs(self.processor, row, images)
        self.last_trace = trace
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        trace["model_input_keys"] = sorted(inputs)
        trace["model_input_visual_keys"] = sorted(VISUAL_KEYS.intersection(inputs))
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            self.counts["invocations"] += 1
            try:
                generated = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, return_dict_in_generate=False,
                )
            except Exception:
                self.counts["errors"] += 1
                raise
            self.counts["successful_returns"] += 1
        self.torch.cuda.synchronize()
        trace["generate_seconds"] = time.perf_counter() - started
        prompt_len = inputs["input_ids"].shape[1]
        generation_config = getattr(self.model, "generation_config", None)
        active_eos = getattr(generation_config, "eos_token_id", None)
        tokenizer_eos = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None)
        trace["generation_token_trace"] = build_generation_trace(
            generated,
            prompt_token_count=int(prompt_len),
            max_new_tokens=self.max_new_tokens,
            eos_token_ids=active_eos,
        )
        trace["generation_token_trace"].update({
            "model_generate_max_new_tokens": self.max_new_tokens,
            "model_generate_do_sample": False,
            "generation_config_eos_token_id": active_eos,
            "tokenizer_eos_token_id": tokenizer_eos,
        })
        return self.processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0].strip(), trace
