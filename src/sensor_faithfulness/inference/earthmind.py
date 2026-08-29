from __future__ import annotations

from typing import Any


class EarthMindRunner:
    """Frozen dual-sensor EarthMind wrapper (multi-sensor EO model)."""

    model_role = "multi-sensor EO model"

    def __init__(self, model_id_or_path: str, revision: str, max_new_tokens: int = 16) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, revision=revision, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if hasattr(self.model, "preparing_for_generation"):
            self.model.preparing_for_generation(tokenizer=self.tokenizer, max_new_tokens=max_new_tokens, torch_dtype=dtype)
            if hasattr(self.model, "gen_config"):
                self.model.gen_config.do_sample = False
                self.model.gen_config.max_new_tokens = max_new_tokens

    def generate(self, row: dict[str, Any], images: list[Any]) -> str:
        refs = sorted(row["image_refs"], key=lambda ref: int(ref["slot"]))
        by_sensor = {ref["sensor"]: image for ref, image in zip(refs, images)}
        if set(by_sensor) != {"optical", "sar"}:
            raise RuntimeError("EarthMind requires one optical and one SAR image")
        prompt = row["user_prompt"]
        if "<image>" not in prompt:
            prompt = "<image>\n" + prompt
        with self.torch.inference_mode():
            output = self.model.predict_forward_multi(
                image=by_sensor["sar"],
                rgb_image=by_sensor["optical"],
                text=prompt,
                tokenizer=self.tokenizer,
            )
        prediction = output["prediction"] if isinstance(output, dict) else output
        return str(prediction).strip()
