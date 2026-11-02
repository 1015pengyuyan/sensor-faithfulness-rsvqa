"""Mantis native chat bridge; checkpoint use remains subject to its license."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Sequence

PROTOCOL_ROOT = Path(__file__).resolve().parent
if str(PROTOCOL_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_ROOT))

from token_trace import install_generation_trace_capture  # noqa: E402


class NativeMantisAdapter:
    """Use the official Mantis chat helper with the frozen 128-token limit.

    The official chat interface represents image slots with ordered ``<image>``
    markers and does not expose a system role; system and task text are therefore
    carried together as the same user content while preserving both strings.
    This difference is recorded and is not a claim of input-template identity.
    """

    def __init__(self, processor: Any, model: Any, max_new_tokens: int = 128,
                 torch_module: Any = None, chat_mllava: Any = None):
        if max_new_tokens != 128:
            raise ValueError("unified 128-token matrix protocol requires max_new_tokens=128")
        if torch_module is None:
            import torch as torch_module
        if chat_mllava is None:
            from mantis.models.mllava import chat_mllava
        self.processor = processor
        self.model = model
        self.torch = torch_module
        self.chat_mllava = chat_mllava
        self.max_new_tokens = max_new_tokens
        self.counts = {"invocations": 0, "successful_returns": 0, "errors": 0}
        self.capture = install_generation_trace_capture(
            model,
            counts=self.counts,
            required_max_new_tokens=128,
            tokenizer_eos_token_ids=getattr(getattr(processor, "tokenizer", None), "eos_token_id", None),
        )
        self.last_trace: dict[str, Any] = {}

    @classmethod
    def from_pretrained(cls, model_path: str, max_new_tokens: int = 128):
        import torch
        from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration, chat_mllava

        processor = MLlavaProcessor.from_pretrained(str(model_path), local_files_only=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation=None,
        ).eval()
        return cls(processor, model, max_new_tokens=max_new_tokens,
                   torch_module=torch, chat_mllava=chat_mllava)

    def generate(self, row: dict[str, Any], images: Sequence[Any]):
        self.last_trace = {}
        if not row.get("system_prompt") or not row.get("user_prompt"):
            raise ValueError("the frozen system_prompt and user_prompt are required")
        ordered_images = list(images)
        marker_prefix = "".join("<image>\n" for _ in ordered_images)
        native_user_text = marker_prefix + row["system_prompt"] + "\n\n" + row["user_prompt"]
        before = dict(self.counts)
        with self.torch.inference_mode():
            result = self.chat_mllava(
                native_user_text,
                ordered_images if ordered_images else None,
                self.model,
                self.processor,
                history=None,
                max_new_tokens=128,
                num_beams=1,
                do_sample=False,
            )
        if not isinstance(result, (tuple, list)) or len(result) < 1:
            raise RuntimeError("official Mantis chat helper returned an unsupported result")
        raw = str(result[0]).strip()
        token_trace = dict(self.capture.last_trace)
        if not token_trace:
            raise RuntimeError("Mantis chat returned without an actual model.generate token trace")
        trace = {
            "native_interface": "official Mantis chat_mllava",
            "native_system_role_available": False,
            "system_and_task_text_preserved_in_user_content": (
                row["system_prompt"] in native_user_text and row["user_prompt"] in native_user_text
            ),
            "image_count": len(ordered_images),
            "image_sha256": [
                hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
                for image in ordered_images
            ],
            "image_sizes": [list(image.size) for image in ordered_images],
            "image_marker_count": native_user_text.count("<image>"),
            "native_prompt_sha256": hashlib.sha256(native_user_text.encode("utf-8")).hexdigest(),
            "processor_images_argument": "ordered_list" if ordered_images else "None",
            "generation_token_trace": token_trace,
            "raw_output": raw,
            "model_generate_invocations_for_attempt": self.counts["invocations"] - before["invocations"],
            "successful_model_generate_returns_for_attempt": self.counts["successful_returns"] - before["successful_returns"],
            "model_generate_errors_for_attempt": self.counts["errors"] - before["errors"],
        }
        self.last_trace = trace
        return raw, trace

