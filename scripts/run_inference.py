#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sensor_faithfulness.evaluation.parser import parse_answer
from sensor_faithfulness.inference.common import verify_and_load_images
from sensor_faithfulness.inference.earthmind import EarthMindRunner
from sensor_faithfulness.inference.llava_onevision import LlavaOneVisionRunner
from sensor_faithfulness.inference.qwen import QwenRunner
from sensor_faithfulness.utils.io import read_jsonl, sha256_file, write_jsonl


RUNNERS = {"qwen": QwenRunner, "earthmind": EarthMindRunner, "llava_onevision": LlavaOneVisionRunner}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(RUNNERS), required=True)
    parser.add_argument("--model", required=True, help="Public model ID or local checkpoint directory")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    fixture_hash = sha256_file(args.fixture)
    if fixture_hash != args.expected_fixture_sha256:
        raise SystemExit(f"fixture hash mismatch: {fixture_hash}")
    rows = read_jsonl(args.fixture)
    if any("/test/" in json.dumps(row).replace("\\", "/").lower() for row in rows):
        raise SystemExit("prohibited test reference in fixture")

    runner = RUNNERS[args.backend](args.model, args.revision, args.max_new_tokens)
    predictions = []
    failures = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        try:
            images, _ = verify_and_load_images(row, args.data_root)
            raw_output = runner.generate(row, images)
            parsed = parse_answer(raw_output)
            interface_error = None
            n_images = len(images)
        except Exception as exc:  # preserve every fixture row
            raw_output = "abstain"
            parsed = parse_answer(raw_output)
            interface_error = f"{type(exc).__name__}: {exc}"
            n_images = 0
            failures.append({"fixture_id": row.get("fixture_id"), "error": interface_error})
        predictions.append({
            "backend": args.backend,
            "model_role": runner.model_role,
            "fixture_id": row["fixture_id"],
            "example_id": row["example_id"],
            "requested_sensor": row["requested_sensor"],
            "condition_id": row["condition_id"],
            "target_label": row["target_label"],
            "distractor_label": row["distractor_label"],
            "raw_output": raw_output,
            "parsed_answer": parsed.answer,
            "strict_json_valid": parsed.strict_json_valid,
            "parser_status": "interface_error" if interface_error else parsed.parser_status,
            "interface_error": interface_error,
            "n_input_images": n_images,
        })
    write_jsonl(args.output, predictions)
    manifest = {
        "backend": args.backend,
        "model": args.model,
        "revision": args.revision,
        "model_role": runner.model_role,
        "fixture_sha256": fixture_hash,
        "rows": len(predictions),
        "failures": len(failures),
        "output_sha256": sha256_file(args.output),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "official_test_used": False,
        "elapsed_seconds": time.time() - started,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
