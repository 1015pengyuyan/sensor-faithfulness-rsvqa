import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from sensor_faithfulness.data.fixture_validation import validate_fixture
from sensor_faithfulness.data.prompt_protocol import build_user_prompt, ordered_image_refs
from sensor_faithfulness.data.render_bigearthnet import render_s1_arrays, render_s2_arrays
from sensor_faithfulness.evaluation.metrics import bootstrap_metrics, compute_metrics, robust_flip_rate
from sensor_faithfulness.evaluation.parser import parse_answer
from sensor_faithfulness.paper.figures import reproduce_figure2
from sensor_faithfulness.paper.reproduce import load_frozen_summary, table3_rows


ROOT = Path(__file__).resolve().parents[1]
REPS = ("A2", "A3-1", "A3-2")


def metric_row(source, sensor, condition, target, answer):
    return {
        "source_id": source,
        "requested_sensor": sensor,
        "condition_id": condition,
        "target_label": target,
        "distractor_label": "no" if target == "yes" else "yes",
        "parsed_answer": answer,
    }


def synthetic_rows():
    rows = [metric_row("s1", "optical", "A0", "no", "no")]
    rows.extend(metric_row("s1", "optical", c, "no", a) for c, a in zip(REPS, ("yes", "yes", "no")))
    rows.append(metric_row("s1", "sar", "A0", "no", "no"))
    rows.extend(metric_row("s1", "sar", c, "no", "yes") for c in REPS)
    rows.append(metric_row("s2", "optical", "A0", "yes", "yes"))
    rows.extend(metric_row("s2", "optical", c, "yes", "no") for c in REPS)
    rows.append(metric_row("s2", "sar", "A0", "yes", "no"))
    rows.extend(metric_row("s2", "sar", c, "yes", "no") for c in REPS)
    return rows


class CoreTests(unittest.TestCase):
    def test_frozen_parser(self):
        parsed = parse_answer('{"answer":"yes"}')
        self.assertEqual(parsed.answer, "yes")
        self.assertTrue(parsed.strict_json_valid)
        self.assertEqual(parse_answer("The answer is NO.").answer, "no")
        self.assertEqual(parse_answer("yes or no").parser_status, "ambiguous")

    def test_a0_conditioned_source_level_metrics(self):
        result = compute_metrics(synthetic_rows(), "A0", REPS)
        self.assertEqual(result["eligible_n"], 3)
        self.assertAlmostEqual(result["cifr"], 8 / 9)
        self.assertAlmostEqual(result["rsr"], 1 / 9)
        self.assertAlmostEqual(result["optical_cifr"], 5 / 6)
        self.assertAlmostEqual(result["sar_cifr"], 1.0)
        # ASYM is paired by source; only s1 is A0-correct in both directions.
        self.assertAlmostEqual(result["asym"], 1 / 3)

    def test_robust_flip_and_source_bootstrap(self):
        robust = robust_flip_rate(synthetic_rows(), "A0", REPS)
        self.assertEqual(robust["eligible_n"], 3)
        self.assertAlmostEqual(robust["robust_flip_rate"], 1.0)
        first = bootstrap_metrics(synthetic_rows(), "A0", REPS, 100, 20260828)
        second = bootstrap_metrics(synthetic_rows(), "A0", REPS, 100, 20260828)
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_unit"], "target_source")

    def test_fixture_release_checks(self):
        row = {
            "fixture_id": "f0", "example_id": "s0", "requested_sensor": "optical",
            "condition_id": "A2_CONFLICT", "target_label": "yes", "distractor_label": "no",
            "image_refs": [
                {"source_id": "s0", "sensor": "optical", "role": "target_requested", "slot": 1},
                {"source_id": "d0", "sensor": "sar", "role": "distractor_non_requested", "slot": 2},
            ],
        }
        report = validate_fixture([row])
        for key in (
            "duplicate_fixture_ids", "target_distractor_overlap", "label_violations",
            "missing_requested_evidence", "prohibited_test_references",
        ):
            self.assertEqual(report[key], 0, key)

    def test_prompt_and_frozen_rendering(self):
        ordered = ordered_image_refs([{"sensor": "sar", "slot": 2}, {"sensor": "optical", "slot": 1}])
        self.assertEqual([row["sensor"] for row in ordered], ["optical", "sar"])
        prompt = build_user_prompt("Is there a bridge?", "sar")
        self.assertIn("Image 1 is non-requested OPTICAL evidence", prompt)
        self.assertIn("Image 2 is the requested SAR evidence", prompt)
        s2 = render_s2_arrays(
            np.array([[0, 3000]], np.float32),
            np.array([[1500, 3000]], np.float32),
            np.array([[3000, 0]], np.float32),
        )
        self.assertEqual(s2.tolist(), [[[255, 128, 0], [0, 255, 255]]])
        s1 = render_s1_arrays(np.array([[-25, 0]], np.float32), np.array([[-32, -5]], np.float32))
        self.assertEqual(s1.tolist(), [[[0, 0, 0], [255, 255, 255]]])

    def test_frozen_summary_and_table_values(self):
        path = ROOT / "results" / "paper_summary" / "cross_dataset_summary.csv"
        frame = load_frozen_summary(path)
        self.assertEqual(len(frame), 6)
        rows = table3_rows(path)
        qwen = next(row for row in rows if row["dataset"] == "M4-SAR" and row["model"] == "Qwen2.5-VL-3B")
        self.assertEqual(qwen["overall_cifr"], 0.12024756852343059)
        self.assertEqual(qwen["asym"], 0.18518518518518517)


    def test_pdf_reproduction_is_byte_deterministic(self):
        summary = ROOT / "results" / "paper_summary" / "cross_dataset_summary.csv"
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            reproduce_figure2(summary, Path(first))
            time.sleep(1.1)
            reproduce_figure2(summary, Path(second))
            self.assertEqual(
                (Path(first) / "figure2_cross_dataset_cifr.pdf").read_bytes(),
                (Path(second) / "figure2_cross_dataset_cifr.pdf").read_bytes(),
            )

if __name__ == "__main__":
    unittest.main()
