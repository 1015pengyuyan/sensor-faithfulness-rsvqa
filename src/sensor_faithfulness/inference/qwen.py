from __future__ import annotations

from typing import Any

from .common import chat_messages


class QwenRunner:
    """Frozen deterministic Qwen2.5-VL wrapper (general VLM control)."""

    model_role = "general VLM control"

    def __init__(self, model_id_or_path: str, revision: str, max_new_tokens: int = 16) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id_or_path, revision=revision, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, row: dict[str, Any], images: list[Any]) -> str:
        text = self.processor.apply_chat_template(chat_messages(row, len(images)), tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        prompt_length = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0].strip()
