"""CPU audit of immutable eleven-model normalized evidence; no new draws/inference.

The original v2 estimand implementation is reused without changing historical
files. This checks point estimates and their denominators, not CI regeneration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path, PurePosixPath
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reproduction/v2"))
sys.path.insert(0, str(ROOT / "src"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_manifest(data):
    data = Path(data).resolve()
    manifest = json.loads((data / "MANIFEST.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    if not files:
        raise ValueError("Empty evidence manifest")
    names = set()
    for entry in files:
        name = entry["path"]
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or ":" in name or name in names):
            raise ValueError("Evidence paths must be unique safe relative paths")
        names.add(name)
        path = (data / name).resolve()
        if not path.is_relative_to(data):
            raise ValueError("Evidence path must remain relative to the data directory")
        if (not path.is_file() or path.stat().st_size != entry["bytes"]
                or sha256(path) != entry["sha256"]):
            raise ValueError("File integrity mismatch: " + name)
    return names


def same_number(actual, expected):
    wanted = float(expected) if expected not in (None, "", "null") else float("nan")
    return ((math.isnan(actual) and math.isnan(wanted))
            or (math.isfinite(actual) and math.isfinite(wanted)
                and abs(actual - wanted) < 1e-12))


def verify_current(data):
    data = Path(data)
    names = verify_manifest(data)
    lock = json.loads((ROOT / "configs/v3/evidence.json").read_text(encoding="utf-8"))
    for name, identity in lock["files"].items():
        path = data / name
        if (name not in names or not path.is_file() or path.stat().st_size != identity["bytes"]
                or sha256(path) != identity["sha256"]):
            raise ValueError("Current evidence lock mismatch: " + name)
    return lock, names


def validate_rows(rows):
    from sensor_faithfulness.evaluation.parser import parse_answer
    identities = set()
    for row in rows:
        identity = (row["model"], row["fixture_id"])
        if identity in identities:
            raise ValueError("Duplicate scientific model/fixture input")
        identities.add(identity)
        if row.get("call_id", "main:").startswith("replay:") or row.get("call_type") == "replay":
            raise ValueError("Replay record in scientific population")
        if row["gold"] not in ("yes", "no") or row["answer"] not in ("yes", "no", "abstain", None):
            raise ValueError("Unknown scientific answer/gold")
        expected = {"correct": int(row["answer"] == row["gold"]),
                    "abstain": int(row["answer"] == "abstain"),
                    "parse_failure": int(row["answer"] is None)}
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError("Saved score disagrees with answer/gold")
        if row["delivery_missing"] != 0:
            raise ValueError("Current sealed population must have zero missing deliveries")
        if "raw_output" in row:
            parsed = parse_answer(row["raw_output"])
            if (parsed.answer, parsed.parser_status, parsed.strict_json_valid) != (
                    row["answer"], row["parser_status"], row["strict_json"]):
                raise ValueError("Frozen parser disagrees with saved raw response annotation")


def write_csv(path, rows):
    if not rows:
        raise ValueError("No reconstructed results")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct(rows, endpoints, cohorts, excluded):
    from frozen_metrics import build_book
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["dataset"]].append(row)
    archived = {}
    for row in endpoints:
        key = (row["model"], row["dataset"], row["cohort"], row["metric"])
        if key in archived:
            raise ValueError("Duplicate archived endpoint")
        archived[key] = row
    points = []
    flow = []
    for cohort_row in cohorts:
        model, dataset, cohort = (cohort_row[k] for k in ("model", "dataset", "cohort"))
        selected = [r for r in groups[model, dataset]
                    if cohort != "source_exclusion" or r["source"] not in excluded]
        sources = sorted({r["source"] for r in selected})
        observed = dict(science=len(selected), questions=len({r["target"] for r in selected}),
                        sources=len(sources), missing=sum(r["delivery_missing"] for r in selected))
        if any(value != int(cohort_row[name]) for name, value in observed.items()):
            raise ValueError("Cohort accounting mismatch: " + str((model, dataset, cohort)))
        flow.append(dict(model=model, dataset=dataset, cohort=cohort, **observed))
        book = build_book(selected, sources, model == "earthmind_4b")
        for metric, definition in book.metrics.items():
            key = (model, dataset, cohort, metric)
            if key not in archived:
                raise ValueError("Missing archived endpoint: " + str(key))
            original = archived[key]
            value = 0.0
            for numerator, denominator, weight in definition["parts"]:
                divisor = denominator.sum()
                value += (numerator.sum() / divisor if divisor else float("nan")) * weight
            if not same_number(value, original["estimate"]):
                raise ValueError("Reconstructed point-estimate mismatch: " + str(key))
            for name in ("eligible_units", "eligible_physical_sources", "eligible_target_questions",
                         "eligible_source_directions"):
                if name in definition["meta"] and definition["meta"][name] != int(original[name]):
                    raise ValueError("Eligibility denominator mismatch: " + str(key) + "/" + name)
            points.append(dict(model=model, dataset=dataset, cohort=cohort, metric=metric,
                               estimate=value, eligible_units=original["eligible_units"],
                               eligible_physical_sources=original["eligible_physical_sources"]))
    if len(points) != len(archived):
        raise ValueError("Archived/reconstructed endpoint coverage differs")
    return points, flow


def run(data, out):
    data, out = Path(data), Path(out)
    lock, _ = verify_current(data)
    if out.exists():
        raise FileExistsError("Choose a new output directory; existing evidence is never overwritten")
    with (data / "NORMALIZED_SELECTED.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    validate_rows(rows)
    if len(rows) != lock["scientific_records"]:
        raise ValueError("Scientific record count differs")
    model_ids = {row[0] for row in json.loads((data / "MODELS.json").read_text(encoding="utf-8"))}
    if {row["model"] for row in rows} != model_ids or len(model_ids) != 11:
        raise ValueError("Current eleven-model population differs")
    excluded = set(json.loads((data / "SOURCE_EXCLUSION.json").read_text(encoding="utf-8")))
    points, flow = reconstruct(rows, csv_rows(data / "ENDPOINTS.csv"),
                               csv_rows(data / "COHORTS.csv"), excluded)
    # The sealed completion audit describes replays; they are not scientific rows.
    completion = json.loads((data / "COMPLETION_AUDIT.json").read_text(encoding="utf-8"))
    closed = {}
    for model in ("luna", "gemini"):
        stages = [stage for stage in completion["stages"] if stage["model"] == model]
        planned = sum(stage["planned"] for stage in stages)
        available = sum(stage["available"] for stage in stages)
        exact = sum(stage["replay_answer_exact"] for stage in stages)
        if (len(stages), planned, available, exact) != (2, 18894, 18894, 43):
            raise ValueError("Closed-model completion/replay accounting differs")
        closed[model] = dict(planned=planned, available=available,
                             parsed_replay_exact=exact, replay_pairs=52)
    receipt = dict(version="3.0.0", status="PASS", scientific_records=len(rows),
                   main_point_estimates_verified=len(points), cohorts_verified=len(flow),
                   closed_model_completion=closed,
                   intervals="Archived endpoint bytes verified; intervals not regenerated",
                   parser="Available raw responses reparsed; first five groups retain normalized annotations only",
                   raw_responses_reparsed=sum("raw_output" in row for row in rows),
                   scientific_replay_records=0, inference_calls=0, new_random_draws=0,
                   runtime=dict(python=platform.python_version(), numpy=version("numpy")),
                   resolved_configuration=dict(evidence_version=lock["version"],
                       numeric_tolerance=1e-12, seed="not applicable: deterministic replay without draws"),
                   input_manifest_sha256=sha256(data / "MANIFEST.json"),
                   implementation_sha256={str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                       for path in (Path(__file__).resolve(), ROOT / "reproduction/v2/frozen_metrics.py",
                           ROOT / "src/sensor_faithfulness/evaluation/parser.py", ROOT / "configs/v3/models.json")})
    out.mkdir(parents=True)
    write_csv(out / "RECOMPUTED_MAIN_POINTS.csv", points)
    write_csv(out / "RECOMPUTED_COHORTS.csv", flow)
    receipt["output_sha256"] = {name: sha256(out / name) for name in
        ("RECOMPUTED_MAIN_POINTS.csv", "RECOMPUTED_COHORTS.csv")}
    (out / "REPRODUCTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "all"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reproduced/v3"))
    args = parser.parse_args()
    result = ({"verified_files": len(verify_current(args.data)[1])} if args.command == "verify"
              else run(args.data, args.out))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
