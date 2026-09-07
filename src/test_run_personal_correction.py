import json
import tempfile
import unittest
from pathlib import Path

from config import SearchConfig
from data_contracts import EXAMPLE_PATHS, validate_file, validate_payload
from run_personal_correction import run_personal_corrections, write_correction_results
from threshold_adapter import ThresholdAdapterError


class PersonalCorrectionRunnerTests(unittest.TestCase):
    def _run(self, **overrides):
        arguments = {
            "participant_id": "P_SYNTHETIC_01",
            "model": "D288",
            "stroke_width_px": 3,
            "original_min": 10.0,
            "development_mode": True,
            "search_config": SearchConfig(step=0.02, max_shift=0.30),
        }
        arguments.update(overrides)
        return run_personal_corrections(
            EXAMPLE_PATHS["validation_pairs"],
            EXAMPLE_PATHS["participant_thresholds"],
            **arguments,
        )

    def test_runs_unchanged_and_corrected_paths(self) -> None:
        payload = self._run()
        validate_payload("correction_results", payload)
        self.assertEqual(payload["summary"]["pair_count"], 2)
        self.assertEqual(payload["summary"]["unchanged_count"], 1)
        self.assertEqual(payload["summary"]["corrected_count"], 1)
        corrected = next(row for row in payload["results"] if row["status"] == "corrected")
        self.assertTrue(corrected["target_met"])
        self.assertGreater(corrected["total_color_change"], 0)

    def test_default_mode_rejects_synthetic_threshold(self) -> None:
        with self.assertRaises(ThresholdAdapterError):
            self._run(development_mode=False)

    def test_ci_high_policy_is_recorded_and_used(self) -> None:
        payload = self._run(target_policy="ci_high")
        self.assertEqual(payload["threshold"]["selected_target"], 3.4)
        self.assertEqual(payload["threshold"]["target_policy"], "ci_high")

    def test_writes_only_valid_utf8_json(self) -> None:
        payload = self._run()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "correction_results.json"
            write_correction_results(output, payload)
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded, payload)
            validate_file("correction_results", output)


if __name__ == "__main__":
    unittest.main()
