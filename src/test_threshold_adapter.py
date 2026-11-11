import copy
import json
import tempfile
import unittest
from pathlib import Path

from data_contracts import EXAMPLE_PATHS
from threshold_adapter import (
    ThresholdAdapterError,
    correction_thresholds,
    load_threshold_estimate,
)
from test_human_contracts import _human_thresholds


class ThresholdAdapterTests(unittest.TestCase):
    @staticmethod
    def _write_human(folder: str, payload: dict) -> Path:
        path = Path(folder) / "participant_thresholds.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_research_mode_rejects_synthetic_draft(self) -> None:
        with self.assertRaisesRegex(ThresholdAdapterError, "synthetic"):
            load_threshold_estimate(
                EXAMPLE_PATHS["participant_thresholds"],
                participant_id="P_SYNTHETIC_01",
                model="D384",
                stroke_width_px=3,
            )

    def test_development_mode_loads_exact_mock_threshold(self) -> None:
        estimate = load_threshold_estimate(
            EXAMPLE_PATHS["participant_thresholds"],
            participant_id="P_SYNTHETIC_01",
            model="D384",
            stroke_width_px=3,
            development_mode=True,
        )
        self.assertEqual(estimate.c80_sim, 2.8)
        self.assertEqual(estimate.value_type, "predicted")
        self.assertFalse(estimate.confirmed)

    def test_missing_width_is_not_interpolated(self) -> None:
        with self.assertRaisesRegex(ThresholdAdapterError, "보간"):
            load_threshold_estimate(
                EXAMPLE_PATHS["participant_thresholds"],
                participant_id="P_SYNTHETIC_01",
                model="D384",
                stroke_width_px=4,
                development_mode=True,
            )

    def test_missing_participant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ThresholdAdapterError, "참가자"):
            load_threshold_estimate(
                EXAMPLE_PATHS["participant_thresholds"],
                participant_id="P404",
                model="D384",
                stroke_width_px=3,
                development_mode=True,
            )

    def test_point_policy_builds_baseline_thresholds(self) -> None:
        estimate = load_threshold_estimate(
            EXAMPLE_PATHS["participant_thresholds"],
            participant_id="P_SYNTHETIC_01",
            model="D384",
            stroke_width_px=3,
            development_mode=True,
        )
        thresholds = correction_thresholds(
            estimate,
            original_min=10,
            development_mode=True,
        )
        self.assertEqual(thresholds.simulated_max, 2.8)
        self.assertEqual(thresholds.target, 2.8)
        self.assertEqual(thresholds.original_min, 10)

    def test_ci_high_policy_is_separate_sensitivity_analysis(self) -> None:
        estimate = load_threshold_estimate(
            EXAMPLE_PATHS["participant_thresholds"],
            participant_id="P_SYNTHETIC_01",
            model="D384",
            stroke_width_px=3,
            development_mode=True,
        )
        thresholds = correction_thresholds(
            estimate,
            original_min=10,
            target_policy="ci_high",
            development_mode=True,
        )
        self.assertEqual(thresholds.target, 3.4)

    def test_original_min_remains_a_separate_required_constraint(self) -> None:
        estimate = load_threshold_estimate(
            EXAMPLE_PATHS["participant_thresholds"],
            participant_id="P_SYNTHETIC_01",
            model="D384",
            stroke_width_px=3,
            development_mode=True,
        )
        with self.assertRaisesRegex(ThresholdAdapterError, "original_min"):
            correction_thresholds(
                estimate,
                original_min=1,
                development_mode=True,
            )

    def test_unconfirmed_estimate_needs_development_mode_twice(self) -> None:
        estimate = load_threshold_estimate(
            EXAMPLE_PATHS["participant_thresholds"],
            participant_id="P_SYNTHETIC_01",
            model="D384",
            stroke_width_px=3,
            development_mode=True,
        )
        with self.assertRaisesRegex(ThresholdAdapterError, "미확정"):
            correction_thresholds(estimate, original_min=10)

    def test_research_mode_accepts_confirmed_stable_human_1_0(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_human(folder, _human_thresholds())
            estimate = load_threshold_estimate(
                path,
                participant_id="P_TEST_01",
                model="D384",
                stroke_width_px=4,
            )
        self.assertEqual(estimate.data_origin, "human")
        self.assertEqual(estimate.schema_version, "1.0")
        self.assertTrue(estimate.confirmed)
        self.assertEqual(estimate.c80_sim, 2.8)

    def test_human_result_with_stale_numerical_dependency_is_rejected(self) -> None:
        payload = _human_thresholds()
        payload["implementation_hashes"]["src/validation_psychometrics.py"] = "0" * 64
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_human(folder, payload)
            with self.assertRaisesRegex(
                ThresholdAdapterError, "changed=src/validation_psychometrics.py"
            ):
                load_threshold_estimate(
                    path,
                    participant_id="P_TEST_01",
                    model="D384",
                    stroke_width_px=4,
                )

    def test_human_calibration_width_slope_claim_flag_is_checked_fail_closed(self) -> None:
        payload = _human_thresholds()
        sensitivity = payload["participants"][0][
            "calibration_width_slope_sensitivity"
        ]
        sensitivity["d384_beta_interpretation_eligible"] = False
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_human(folder, payload)
            with self.assertRaisesRegex(ThresholdAdapterError, "D384 beta"):
                load_threshold_estimate(
                    path,
                    participant_id="P_TEST_01",
                    model="D384",
                    stroke_width_px=4,
                )

    def test_confirmed_human_with_unstable_validation_is_rejected(self) -> None:
        payload = _human_thresholds()
        validation = payload["participants"][0]["validation"]
        validation.update(
            stable=False,
            pooled_c80=None,
            pooled_ci_low=None,
            pooled_ci_high=None,
        )
        comparison = payload["participants"][0]["model_comparisons"]
        comparison.update(
            observed_pooled_c80=None,
            observed_ci_low=None,
            observed_ci_high=None,
            observed_ci_half_width=None,
            resolution_check=False,
            primary_outcome="validation_unstable",
        )
        comparison["primary_delta_loss"].update(
            point_estimate=None,
            ci_low=None,
            ci_high=None,
            successful_combined_replicates=0,
            calibration_pairing_id=None,
            calibration_prediction_uncertainty_included=False,
            validation_observation_uncertainty_included=False,
            status="unresolvable",
            reason="validation_unstable",
        )
        for pair in (comparison["primary"], comparison["secondary"]):
            pair["first_abs_error"] = None
            pair["second_abs_error"] = None
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_human(folder, payload)
            with self.assertRaisesRegex(ThresholdAdapterError, "불안정"):
                load_threshold_estimate(
                    path,
                    participant_id="P_TEST_01",
                    model="D384",
                    stroke_width_px=4,
                )

    def test_confirmed_human_with_unresolvable_delta_loss_is_rejected(self) -> None:
        payload = _human_thresholds()
        comparison = payload["participants"][0]["model_comparisons"]
        comparison.update(
            resolution_check=False,
            primary_outcome="comparison_unresolvable",
        )
        comparison["primary_delta_loss"].update(
            ci_low=None,
            ci_high=None,
            successful_combined_replicates=0,
            calibration_pairing_id=None,
            calibration_prediction_uncertainty_included=False,
            validation_observation_uncertainty_included=False,
            status="unresolvable",
            reason="paired_calibration_bootstrap_unavailable",
        )
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_human(folder, payload)
            with self.assertRaisesRegex(ThresholdAdapterError, "결합"):
                load_threshold_estimate(
                    path,
                    participant_id="P_TEST_01",
                    model="D384",
                    stroke_width_px=4,
                )

    def test_confirmed_human_prediction_bootstrap_counts_must_be_consistent(self) -> None:
        payload = _human_thresholds()
        prediction = payload["participants"][0]["predictions"][0]
        prediction["bootstrap_successful_replicates"] = 999
        prediction["bootstrap_c80_samples"] = prediction["bootstrap_c80_samples"][:999]
        prediction["bootstrap_convergence_rate"] = 1.0
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_human(folder, payload)
            with self.assertRaisesRegex(ThresholdAdapterError, "bootstrap"):
                load_threshold_estimate(
                    path,
                    participant_id="P_TEST_01",
                    model="D384",
                    stroke_width_px=4,
                )

    def test_confirmed_human_validation_bootstrap_counts_must_be_consistent(self) -> None:
        payload = _human_thresholds()
        block = payload["participants"][0]["validation"]["blocks"][0]
        block["bootstrap_successful_replicates"] = 979
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_human(folder, payload)
            with self.assertRaisesRegex(ThresholdAdapterError, "bootstrap"):
                load_threshold_estimate(
                    path,
                    participant_id="P_TEST_01",
                    model="D384",
                    stroke_width_px=4,
                )

    def test_validation_range_flag_must_equal_all_numeric_block_flags(self) -> None:
        payload = _human_thresholds()
        validation = payload["participants"][0]["validation"]
        block = validation["blocks"][0]
        block["presented_delta_e00_min"] = 2.6
        block["range_bracketed"] = False
        # A forged top-level true must not hide an unbracketed validation block.
        validation["range_bracketed"] = True
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_human(folder, payload)
            with self.assertRaisesRegex(ThresholdAdapterError, "range_bracketed"):
                load_threshold_estimate(
                    path,
                    participant_id="P_TEST_01",
                    model="D384",
                    stroke_width_px=4,
                )

    def test_duplicate_human_model_width_is_rejected(self) -> None:
        payload = _human_thresholds()
        predictions = payload["participants"][0]["predictions"]
        predictions.append(copy.deepcopy(predictions[0]))
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_human(folder, payload)
            with self.assertRaisesRegex(ThresholdAdapterError, "중복"):
                load_threshold_estimate(
                    path,
                    participant_id="P_TEST_01",
                    model="D384",
                    stroke_width_px=4,
                )


if __name__ == "__main__":
    unittest.main()
