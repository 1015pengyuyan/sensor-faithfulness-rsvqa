"""Technical-smoke-only adapter for native text-only Qwen calls.

The frozen project runner is intentionally left unchanged. Image-bearing
calls delegate to it verbatim; only empty-image calls use the processor's
native text-only path (no ``images`` argument).
"""
from __future__ import annotations

from typing import Any


VISUAL_INPUT_KEYS = {
    "pixel_values",
    "image_grid_thw",
    "pixel_attention_mask",
    "vision_features",
}


class QwenTechnicalAdapter:
    def __init__(self, frozen_runner: Any) -> None:
        self._frozen_runner = frozen_runner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._frozen_runner, name)

    def generate(self, row: dict[str, Any], image_paths: list[Any]) -> dict[str, Any]:
        if image_paths:
            return self._frozen_runner.generate(row, image_paths)

        content = [{"type": "text", "text": row["user_prompt"]}]
        text = self.processor.apply_chat_template(
            [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Native text-only call: omitting the `images` keyword prevents
        # processors from manufacturing empty visual tensors for images=[].
        inputs = self.processor(text=[text], return_tensors="pt")
        visual_keys = sorted(VISUAL_INPUT_KEYS.intersection(inputs.keys()))
        trace = getattr(self.processor, "last_trace", None)
        if isinstance(trace, dict):
            trace["model_input_keys"] = sorted(str(key) for key in inputs)
            trace["model_input_visual_tensor_keys"] = visual_keys
            trace["model_input_images_argument_omitted"] = True
        if visual_keys:
            raise RuntimeError(
                "text_only_model_inputs_contain_visual_tensors: " + ",".join(visual_keys)
            )

        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated.sequences[:, prompt_len:]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        risk_score = None
        token_logprob_mean = None
        if generated.scores:
            logps = []
            for step_index, logits in enumerate(generated.scores):
                if step_index >= new_tokens.shape[1]:
                    break
                token_id = int(new_tokens[0, step_index].item())
                logp = self.torch.log_softmax(logits[0].float(), dim=-1)[token_id].item()
                logps.append(float(logp))
            if logps:
                token_logprob_mean = sum(logps) / len(logps)
                risk_score = -token_logprob_mean
        return {
            "raw_output": decoded,
            "risk_score": risk_score,
            "token_logprob_mean": token_logprob_mean,
        }
