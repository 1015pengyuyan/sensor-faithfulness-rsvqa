"""Failure-sensitive checks for the current normalized-evidence replay."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "reproduction/v3/reproduce.py"


class CurrentReplayTests(unittest.TestCase):
    def load(self):
        self.assertTrue(SCRIPT.is_file(), "The current replay entry point is missing")
        spec = importlib.util.spec_from_file_location("current_replay", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_primary_scoring_counts_abstention_as_wrong(self):
        m = self.load()
        rows = [dict(model="x", fixture_id=str(i), answer=a, gold=g,
                     correct=int(a == g), abstain=int(a == "abstain"),
                     parse_failure=int(a is None), delivery_missing=0,
                     strict_json=True) for i, (a, g) in enumerate([
                         ("yes", "yes"), ("abstain", "yes"), ("no", "no")])]
        m.validate_rows(rows)
        self.assertAlmostEqual(sum(r["correct"] for r in rows) / len(rows), 2 / 3)
        rows[1]["correct"] = 1
        with self.assertRaisesRegex(ValueError, "score"):
            m.validate_rows(rows)

    def test_replay_and_duplicate_scientific_inputs_are_rejected(self):
        m = self.load()
        row = dict(model="x", fixture_id="a", call_id="replay:a", answer="yes",
                   gold="yes", correct=1, abstain=0, parse_failure=0,
                   delivery_missing=0, strict_json=True)
        with self.assertRaisesRegex(ValueError, "Replay"):
            m.validate_rows([row])
        row["call_id"] = "main:a"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            m.validate_rows([row, row])

    def test_undefined_ratio_is_not_zero(self):
        m = self.load()
        self.assertTrue(m.same_number(float("nan"), "nan"))
        self.assertFalse(m.same_number(float("nan"), "0"))

    def test_available_raw_response_is_reparsed_without_repair(self):
        m = self.load()
        row = dict(model="x", fixture_id="a", answer="yes", gold="yes",
                   correct=1, abstain=0, parse_failure=0, delivery_missing=0,
                   strict_json=True, parser_status="json", raw_output='{"answer":"no"}')
        with self.assertRaisesRegex(ValueError, "parser"):
            m.validate_rows([row])

    def test_manifest_rejects_escape_and_modified_bytes(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.txt").write_text("modified", encoding="utf-8")
            (root / "MANIFEST.json").write_text(json.dumps({"files": [dict(
                path="../outside", bytes=0, sha256="0" * 64)]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relative"):
                m.verify_manifest(root)
            (root / "MANIFEST.json").write_text(json.dumps({"files": [dict(
                path="data.txt", bytes=8, sha256="0" * 64)]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity"):
                m.verify_manifest(root)

    def test_missing_evidence_fails_without_creating_results(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "output"
            completed = subprocess.run([sys.executable, str(SCRIPT), "all", "--data",
                str(Path(directory) / "missing"), "--out", str(out)], capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(out.exists())

    def test_self_consistent_manifest_does_not_replace_sealed_evidence(self):
        m = self.load()
        self.assertTrue(hasattr(m, "verify_current"), "Sealed evidence verification is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"substitute"
            (root / "substitute.txt").write_bytes(payload)
            (root / "MANIFEST.json").write_text(json.dumps({"files": [dict(
                path="substitute.txt", bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest())]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lock"):
                m.verify_current(root)


if __name__ == "__main__":
    unittest.main()
