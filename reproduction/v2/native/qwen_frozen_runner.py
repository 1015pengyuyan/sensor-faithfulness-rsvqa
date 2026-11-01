"""Run fixed cross-sensor VQA inference on a frozen condition fixture.

The script is deliberately protocol-bound:

* it refuses fixture hash mismatches;
* it hard-fails on prohibited split path components;
* it verifies image hashes before model use;
* it uses one output schema for all backends and conditions;
* it supports a CPU dry run that does not read images or load models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.cross_sensor_eval import parse_answer
    from scripts.cross_sensor_path_guard import assert_sanitized_data_path
except ModuleNotFoundError:
    from cross_sensor_eval import parse_answer
    from cross_sensor_path_guard import assert_sanitized_data_path


VALID_BACKENDS = {"qwen", "earthmind", "dry_run_echo"}
VALID_ANSWERS = {"yes", "no", "abstain"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        for image_ref in row.get("image_refs", []):
            assert_sanitized_data_path(image_ref["relative_path"])
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + ("\n" if text else ""))


def append_jsonl_line(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def verify_image(data_root: Path, image_ref: dict[str, Any], cache: dict[str, Path]) -> Path:
    rel = image_ref["relative_path"]
    assert_sanitized_data_path(rel)
    path = data_root / rel
    assert_sanitized_data_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing image for sanitized relative path: {rel}")
    expected = image_ref["sha256"]
    if rel not in cache:
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"image hash mismatch for sanitized relative path: {rel}")
        cache[rel] = path
    return cache[rel]


class QwenRunner:
    def __init__(self, model_path: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, row: dict[str, Any], image_paths: list[Path]) -> dict[str, Any]:
        images = [load_image(path) for path in image_paths]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": row["user_prompt"]}]
        text = self.processor.apply_chat_template(
            [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
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


class EarthMindRunner:
    def __init__(self, model_path: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
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

    def generate(self, row: dict[str, Any], image_paths: list[Path]) -> dict[str, Any]:
        prompt = row["user_prompt"]
        if "<image>" not in prompt:
            prompt = "<image>\n" + prompt
        images_by_sensor = {
            ref["sensor"]: load_image(path)
            for ref, path in zip(row["image_refs"], image_paths)
        }
        with self.torch.inference_mode():
            if len(image_paths) == 1:
                out = self.model.predict_forward(
                    image=load_image(image_paths[0]),
                    text=prompt,
                    tokenizer=self.tokenizer,
                )
                interface_mode = "single_image_predict_forward"
            else:
                # EarthMind exposes a fixed SAR/RGB fusion interface rather
                # than a generic ordered multi-image chat interface.  We bind
                # SAR to `image` and optical to `rgb_image` for all two-sensor
                # conditions and do not interpret this as slot-specific routing.
                if "sar" not in images_by_sensor or "optical" not in images_by_sensor:
                    raise RuntimeError("EarthMind two-image condition requires one SAR and one optical image")
                out = self.model.predict_forward_multi(
                    image=images_by_sensor["sar"],
                    rgb_image=images_by_sensor["optical"],
                    text=prompt,
                    tokenizer=self.tokenizer,
                )
                interface_mode = "sar_image_plus_rgb_image_predict_forward_multi"
        prediction = out["prediction"] if isinstance(out, dict) else str(out)
        return {
            "raw_output": str(prediction).strip(),
            "risk_score": None,
            "token_logprob_mean": None,
            "earthmind_interface_mode": interface_mode,
        }


def make_runner(backend: str, model_path: str, max_new_tokens: int) -> Any:
    if backend == "qwen":
        return QwenRunner(model_path, max_new_tokens)
    if backend == "earthmind":
        return EarthMindRunner(model_path, max_new_tokens)
    if backend == "dry_run_echo":
        return None
    raise ValueError(f"unknown backend: {backend}")


def infer_row(
    row: dict[str, Any],
    backend: str,
    runner: Any,
    data_root: Path,
    hash_cache: dict[str, Path],
) -> dict[str, Any]:
    image_paths: list[Path] = []
    if backend != "dry_run_echo":
        for image_ref in row["image_refs"]:
            image_paths.append(verify_image(data_root, image_ref, hash_cache))

    start = time.time()
    if backend == "dry_run_echo":
        raw_output = json.dumps({"answer": row["expected_answer"]}, separators=(",", ":"))
        generation = {"raw_output": raw_output, "risk_score": 0.0, "token_logprob_mean": 0.0}
    else:
        generation = runner.generate(row, image_paths)
    elapsed = time.time() - start
    parsed = parse_answer(str(generation["raw_output"]))

    output = {
        "backend": backend,
        "fixture_id": row["fixture_id"],
        "example_id": row["example_id"],
        "split": row["split"],
        "condition_id": row["condition_id"],
        "requested_sensor": row["requested_sensor"],
        "class_id": row["class_id"],
        "class_name": row["class_name"],
        "target_label": row["target_label"],
        "distractor_label": row["distractor_label"],
        "expected_answer": row["expected_answer"],
        "raw_output": generation["raw_output"],
        "parsed_answer": parsed.answer,
        "strict_json_valid": parsed.strict_json_valid,
        "parser_status": parsed.parser_status,
        "risk_score": generation.get("risk_score"),
        "token_logprob_mean": generation.get("token_logprob_mean"),
        "elapsed_seconds": elapsed,
        "image_refs": row["image_refs"],
    }
    if "earthmind_interface_mode" in generation:
        output["earthmind_interface_mode"] = generation["earthmind_interface_mode"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(VALID_BACKENDS), required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--condition-id", action="append")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    fixture_sha = sha256_file(args.fixture)
    if fixture_sha != args.expected_fixture_sha256:
        raise RuntimeError(f"fixture sha256 mismatch: observed={fixture_sha} expected={args.expected_fixture_sha256}")
    if args.backend != "dry_run_echo" and not args.model_path:
        raise ValueError("--model-path is required unless backend=dry_run_echo")
    assert_sanitized_data_path(args.fixture)
    assert_sanitized_data_path(args.data_root)

    rows = read_jsonl(args.fixture)
    if args.condition_id:
        wanted = set(args.condition_id)
        rows = [row for row in rows if row["condition_id"] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]

    runner = make_runner(args.backend, args.model_path, args.max_new_tokens)
    hash_cache: dict[str, Path] = {}
    predictions: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows, start=1):
            try:
                prediction = infer_row(row, args.backend, runner, args.data_root, hash_cache)
            except Exception:
                print(
                    json.dumps(
                        {
                            "status": "INFERENCE_ROW_FAILED",
                            "backend": args.backend,
                            "index": index,
                            "fixture_id": row.get("fixture_id"),
                            "condition_id": row.get("condition_id"),
                            "example_id": row.get("example_id"),
                            "image_refs": row.get("image_refs"),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc()
                raise
            predictions.append(prediction)
            append_jsonl_line(handle, prediction)
            if args.progress_every and index % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "status": "INFERENCE_PROGRESS",
                            "backend": args.backend,
                            "completed": index,
                            "total": len(rows),
                            "output": str(args.output),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    summary = {
        "status": "INFERENCE_COMPLETED" if args.backend != "dry_run_echo" else "DRY_RUN_COMPLETED_NO_MODEL_USE",
        "backend": args.backend,
        "model_path": args.model_path,
        "fixture": str(args.fixture),
        "fixture_sha256": fixture_sha,
        "data_root": str(args.data_root),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "n_rows": len(predictions),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "image_files_hash_verified": len(hash_cache),
        "official_evaluation_split_used": False,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
