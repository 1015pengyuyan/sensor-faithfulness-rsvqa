"""Isolated 128-token adapter for EarthMind's named SAR/RGB interface.

EarthMind is not assigned a generic image-slot position condition. Its native
dual route is keyed by ``image`` (SAR) and ``rgb_image`` (optical/RGB).
Single-sensor routes are rejected rather than approximated with copied or
blank images. This file does not alter the frozen prompt, fixture, or parser.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping


class NativeEarthMindError(RuntimeError):
    """The native named-branch route or its generation trace is invalid."""


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _flat_ints(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        flattened: list[int] = []
        for item in value:
            flattened.extend(_flat_ints(item))
        return flattened
    return [int(value)]


def _sequence_ids(output: Any) -> list[int]:
    if isinstance(output, Mapping):
        output = output.get("sequences")
    else:
        output = getattr(output, "sequences", output)
    if hasattr(output, "detach"):
        output = output.detach()
    if hasattr(output, "cpu"):
        output = output.cpu()
    if hasattr(output, "tolist"):
        output = output.tolist()
    if not isinstance(output, (tuple, list)) or not output:
        raise NativeEarthMindError("language model returned no token sequence")
    if isinstance(output[0], (tuple, list)):
        if len(output) != 1:
            raise NativeEarthMindError("EarthMind adapter expects one generated sequence")
        output = output[0]
    try:
        return [int(token) for token in output]
    except (TypeError, ValueError) as exc:
        raise NativeEarthMindError("language-model sequence contains non-integer IDs") from exc


def _sha256_image(image: Any) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


class NativeEarthMindAdapter:
    """Thin adapter around the pinned model's ``predict_forward_multi`` API."""

    model_id = "earthmind_4b"

    def __init__(self, model: Any, tokenizer: Any, max_new_tokens: int = 128,
                 torch_module: Any = None):
        if max_new_tokens != 128:
            raise ValueError("unified 128-token matrix protocol requires max_new_tokens=128")
        if not callable(getattr(model, "predict_forward_multi", None)):
            raise NativeEarthMindError("model lacks native predict_forward_multi interface")
        language_model = getattr(model, "language_model", None)
        if language_model is None or not callable(getattr(language_model, "generate", None)):
            raise NativeEarthMindError("model lacks the traceable language_model.generate boundary")
        if torch_module is None:
            import torch as torch_module
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.max_new_tokens = max_new_tokens
        self.counts = {"invocations": 0, "successful_returns": 0, "errors": 0}
        self.last_trace: dict[str, Any] = {}
        self._last_generate_multi: dict[str, Any] = {}
        self._active_feature_capture = False
        self._feature_extractions: list[dict[str, Any]] = []
        self._current_visual_mode: str | None = None
        self._capture_model_routes()

    @classmethod
    def from_pretrained(cls, model_path: str, max_new_tokens: int = 128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True,
            torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None,
        ).eval()
        if not callable(getattr(model, "preparing_for_generation", None)):
            raise NativeEarthMindError("pinned model code lacks preparing_for_generation")
        model.preparing_for_generation(
            tokenizer=tokenizer, max_new_tokens=128, torch_dtype=dtype
        )
        if not hasattr(model, "gen_config"):
            raise NativeEarthMindError("native preparation did not create gen_config")
        model.gen_config.do_sample = False
        model.gen_config.max_new_tokens = 128
        return cls(model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens, torch_module=torch)

    def _capture_model_routes(self) -> None:
        original_multi = self.model.generate_multi
        original_extract = getattr(self.model, "extract_feature", None)
        if not callable(original_extract):
            raise NativeEarthMindError("pinned model lacks the traceable extract_feature boundary")

        def traced_extract(*args: Any, **kwargs: Any):
            output = original_extract(*args, **kwargs)
            if self._active_feature_capture:
                input_tensor = args[0] if args else kwargs.get("pixel_values")
                self._feature_extractions.append({
                    "input_shape": _shape(input_tensor),
                    "output_shape": _shape(output),
                })
            return output

        self.model.extract_feature = traced_extract

        def traced_multi(*args: Any, **kwargs: Any):
            pixel = kwargs.get("pixel_values")
            rgb_pixel = kwargs.get("rgb_pixel_values")
            input_ids = kwargs.get("input_ids")
            flat_ids = _flat_ints(input_ids) if input_ids is not None else []
            native_placeholder_count = (
                int(getattr(self, "_active_native_text", "").count("<image>"))
                if self._current_visual_mode == "named_dual" else 0
            )
            self._last_generate_multi = {
                "pixel_values_present": pixel is not None,
                "rgb_pixel_values_present": rgb_pixel is not None,
                "pixel_values_shape": _shape(pixel),
                "rgb_pixel_values_shape": _shape(rgb_pixel),
                "input_ids_flat": flat_ids,
                "native_placeholder_count": native_placeholder_count,
            }
            self._feature_extractions = []
            self._active_feature_capture = self._current_visual_mode == "named_dual"
            try:
                return original_multi(*args, **kwargs)
            finally:
                self._active_feature_capture = False

        self.model.generate_multi = traced_multi

        language_model = self.model.language_model
        original_generate = language_model.generate

        def traced_generate(*args: Any, **kwargs: Any):
            config = kwargs.get("generation_config")
            max_new_tokens = getattr(config, "max_new_tokens", None)
            do_sample = getattr(config, "do_sample", None)
            eos_ids = getattr(config, "eos_token_id", None)
            self.counts["invocations"] += 1
            if max_new_tokens != 128:
                self.counts["errors"] += 1
                raise NativeEarthMindError(
                    f"native generation config max_new_tokens={max_new_tokens!r}; required 128"
                )
            if do_sample is not False:
                self.counts["errors"] += 1
                raise NativeEarthMindError("native generation config must set do_sample=False")
            try:
                if self._current_visual_mode == "named_dual":
                    self._verify_native_context_contract(kwargs)
            except Exception:
                self.counts["errors"] += 1
                raise
            try:
                output = original_generate(*args, **kwargs)
            except Exception:
                self.counts["errors"] += 1
                raise
            self.counts["successful_returns"] += 1
            ids = _sequence_ids(output)
            eos_values = eos_ids if isinstance(eos_ids, (list, tuple, set)) else [eos_ids]
            normalized_eos = sorted({int(value) for value in eos_values if value is not None})
            eos_offsets = [index for index, token in enumerate(ids) if token in normalized_eos]
            self._token_trace = {
                "generation_return_mode": "native_inputs_embeds_returned_sequence_decoded_as_output",
                "generated_token_ids": ids,
                "generated_token_count": len(ids),
                "native_eos_token_ids": normalized_eos,
                "native_eos_emitted": bool(eos_offsets),
                "first_eos_offset": eos_offsets[0] if eos_offsets else None,
                "max_new_tokens": 128,
                "new_token_limit_reached": len(ids) >= 128,
                "cap_hit_without_eos": len(ids) >= 128 and not eos_offsets,
                "termination_classification": (
                    "eos_at_limit" if eos_offsets and eos_offsets[0] == 127 else
                    "eos" if eos_offsets else "length_limit" if len(ids) >= 128 else "other"
                ),
                "eos_source": "native generation_config passed by EarthMind generate_multi",
                "raw_return_sequence_sha256": hashlib.sha256(
                    ",".join(str(token) for token in ids).encode("ascii")
                ).hexdigest(),
            }
            return output

        language_model.generate = traced_generate

    def _verify_native_context_contract(self, generate_kwargs: Mapping[str, Any]) -> None:
        """Verify the pinned image expansion maps to the actual SAR-fused feature sequence."""
        input_ids = self._last_generate_multi.get("input_ids_flat")
        if not isinstance(input_ids, list) or not input_ids:
            raise NativeEarthMindError("native generate_multi did not expose rendered input_ids")
        token_ids = {
            token: self.tokenizer.convert_tokens_to_ids(token)
            for token in ("<img>", "<IMG_CONTEXT>", "</img>")
        }
        if any(value is None for value in token_ids.values()) or len(set(token_ids.values())) != 3:
            raise NativeEarthMindError("native image marker token IDs are missing or ambiguous")
        start_positions = [i for i, token in enumerate(input_ids) if token == token_ids["<img>"]]
        context_positions = [i for i, token in enumerate(input_ids) if token == token_ids["<IMG_CONTEXT>"]]
        end_positions = [i for i, token in enumerate(input_ids) if token == token_ids["</img>"]]
        if self._last_generate_multi.get("native_placeholder_count") != 1:
            raise NativeEarthMindError("dual named input must contain exactly one native image anchor")
        if len(start_positions) != 1 or len(end_positions) != 1:
            raise NativeEarthMindError("rendered input_ids lack exactly one native <img>/</img> pair")
        if not context_positions:
            raise NativeEarthMindError("rendered input_ids contain no <IMG_CONTEXT> tokens")
        if context_positions != list(range(start_positions[0] + 1, end_positions[0])):
            raise NativeEarthMindError("native image-context tokens are not contiguous inside the image markers")

        image_shape = self._last_generate_multi.get("pixel_values_shape")
        rgb_shape = self._last_generate_multi.get("rgb_pixel_values_shape")
        patch_token = getattr(self.model, "patch_token", None)
        if not image_shape or len(image_shape) != 4 or not rgb_shape or len(rgb_shape) != 4:
            raise NativeEarthMindError("named SAR/RGB inputs did not reach generate_multi as batched image tensors")
        if not isinstance(patch_token, int) or patch_token <= 0:
            raise NativeEarthMindError("pinned model patch_token is missing or invalid")
        expected_from_native_preprocessing = image_shape[0] * patch_token
        if len(context_positions) != expected_from_native_preprocessing:
            raise NativeEarthMindError(
                "rendered context-token count does not match the native SAR patch-token expansion"
            )

        if len(self._feature_extractions) != 2:
            raise NativeEarthMindError(
                f"pinned dual route expected 2 visual feature extractions, observed {len(self._feature_extractions)}"
            )
        sar_shape = self._feature_extractions[0].get("output_shape")
        optical_shape = self._feature_extractions[1].get("output_shape")
        if not sar_shape or len(sar_shape) < 2 or not optical_shape or len(optical_shape) < 2:
            raise NativeEarthMindError("native SAR/RGB feature extraction did not return shaped tensors")
        sar_feature_tokens = math.prod(sar_shape[:-1])
        if len(context_positions) != sar_feature_tokens:
            raise NativeEarthMindError(
                "context-token count does not match the SAR fused feature sequence"
            )
        if sar_shape[-2] != optical_shape[-2] or sar_shape[-1] != optical_shape[-1]:
            raise NativeEarthMindError("native SAR/RGB feature token width or embedding width differs")

        inputs_embeds_shape = _shape(generate_kwargs.get("inputs_embeds"))
        if not inputs_embeds_shape or len(inputs_embeds_shape) < 3:
            raise NativeEarthMindError("native generate did not receive shaped inputs_embeds")
        if inputs_embeds_shape[-2] != len(input_ids):
            raise NativeEarthMindError("native inputs_embeds sequence length differs from rendered input_ids")

        self._last_generate_multi["native_context_contract"] = {
            "status": "verified",
            "rendered_input_token_count": len(input_ids),
            "image_start_token_count": len(start_positions),
            "context_token_count": len(context_positions),
            "image_end_token_count": len(end_positions),
            "native_placeholder_count": 1,
            "native_patch_token": patch_token,
            "sar_pixel_crop_count": image_shape[0],
            "optical_pixel_crop_count": rgb_shape[0],
            "sar_feature_shape": sar_shape,
            "optical_feature_shape": optical_shape,
            "sar_feature_token_count": sar_feature_tokens,
            "optical_feature_token_width": optical_shape[-2],
            "feature_call_count": len(self._feature_extractions),
            "feature_branch_order": ["SAR: pixel_values", "optical: rgb_pixel_values"],
            "context_tokens_match_native_sar_patch_expansion": True,
            "context_tokens_match_sar_fused_feature_sequence": True,
            "inputs_embeds_sequence_matches_rendered_input_ids": True,
            "inputs_embeds_shape": inputs_embeds_shape,
        }

    def generate_named(self, row: Mapping[str, Any], images_by_sensor: Mapping[str, Any]):
        self.last_trace = {}
        prompt = row.get("user_prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("frozen user_prompt is required")
        row_family = row.get("row_family")
        if row_family == "dual_named" and row.get("input_mode") == "earthmind_named_dual_sensor_branches":
            if set(images_by_sensor) != {"optical", "sar"}:
                raise NativeEarthMindError("EarthMind named dual route requires exactly optical and SAR images")
            if "<image>" in prompt:
                raise NativeEarthMindError(
                    "frozen task text contains the reserved native <image> placeholder"
                )
            branches = row.get("native_sensor_branches")
            if not isinstance(branches, Mapping) or set(branches) != {"rgb_image", "sar_image"}:
                raise NativeEarthMindError("frozen row lacks the canonical RGB/SAR branch map")
            optical = images_by_sensor["optical"]
            sar = images_by_sensor["sar"]
            expected = {"rgb_image": branches["rgb_image"], "sar_image": branches["sar_image"]}
            observed = {"rgb_image": _sha256_image(optical), "sar_image": _sha256_image(sar)}
            for key, ref in expected.items():
                sensor = str(ref.get("sensor", "")).lower()
                wanted_sensor = "optical" if key == "rgb_image" else "sar"
                if sensor != wanted_sensor:
                    raise NativeEarthMindError(f"branch {key} has sensor label {sensor!r}")
                if observed[key] != str(ref.get("sha256", "")).lower():
                    raise NativeEarthMindError(f"branch image hash mismatch for {key}")
            sar_image, rgb_image = sar, optical
            visual_mode = "named_dual"
            # EarthMind's pinned predict_forward_multi implementation only
            # expands image features at a literal <image> anchor. Keep this
            # interface marker separate from the frozen natural-language task.
            native_text = "<image>\n" + prompt
        elif row_family == "text_only" and row.get("input_mode") == "text_only":
            if images_by_sensor:
                raise NativeEarthMindError("text-only row must not receive image assets")
            sar_image = rgb_image = None
            observed = {"rgb_image": None, "sar_image": None}
            visual_mode = "text_only"
            native_text = prompt
        else:
            raise NotImplementedError(
                "EarthMind's fixed named branch does not support a single-sensor route in this matrix"
            )

        self._last_generate_multi = {}
        self._token_trace = {}
        self.counts_before = dict(self.counts)
        self._current_visual_mode = visual_mode
        self._active_native_text = native_text
        with self.torch.inference_mode():
            result = self.model.predict_forward_multi(
                image=sar_image,
                rgb_image=rgb_image,
                text=native_text,
                tokenizer=self.tokenizer,
            )
        if not isinstance(result, Mapping) or "prediction" not in result:
            raise NativeEarthMindError("EarthMind native route returned no prediction field")
        if not self._token_trace:
            raise NativeEarthMindError("EarthMind route returned without traced language_model.generate output")
        raw = str(result["prediction"]).strip()
        decoded = self.tokenizer.decode(
            self._token_trace["generated_token_ids"], skip_special_tokens=False
        ).strip()
        if raw != decoded:
            raise NativeEarthMindError("native raw prediction does not match the captured generated token IDs")
        trace = {
            "native_interface": "pinned EarthMind predict_forward_multi(image=SAR, rgb_image=optical)",
            "input_mode": visual_mode,
            "position_control_applicability": "not_applicable_named_sensor_branches",
            "input_construction": {
                "task_text_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "native_text_sha256": hashlib.sha256(native_text.encode("utf-8")).hexdigest(),
                "native_placeholder": "<image>" if visual_mode == "named_dual" else None,
                "native_placeholder_count": native_text.count("<image>"),
                "native_template": getattr(getattr(self.model, "config", None), "template", None),
                "template_source": "pinned model predict_forward_multi / PROMPT_TEMPLATE",
                "task_text_preserved_verbatim_after_marker": (
                    native_text == "<image>\n" + prompt
                    if visual_mode == "named_dual" else native_text == prompt
                ),
            },
            "native_branch_image_sha256": observed,
            "visual_route": dict(self._last_generate_multi),
            "native_context_contract": self._last_generate_multi.get(
                "native_context_contract",
                {"status": "not_applicable"} if visual_mode == "text_only" else {"status": "missing"},
            ),
            "visual_input_present": bool(self._last_generate_multi.get("pixel_values_present") and self._last_generate_multi.get("rgb_pixel_values_present")),
            "generation_token_trace": dict(self._token_trace),
            "raw_output": raw,
            "model_generate_invocations_for_attempt": self.counts["invocations"] - self.counts_before["invocations"],
            "model_generate_returns_for_attempt": self.counts["successful_returns"] - self.counts_before["successful_returns"],
            "model_generate_errors_for_attempt": self.counts["errors"] - self.counts_before["errors"],
        }
        if visual_mode == "named_dual" and not trace["visual_input_present"]:
            raise NativeEarthMindError("EarthMind named dual row did not reach both visual branches")
        if visual_mode == "text_only" and trace["visual_input_present"]:
            raise NativeEarthMindError("EarthMind text-only row unexpectedly contained visual tensors")
        self.last_trace = trace
        return raw, trace
