import unittest

from data_contracts import EXAMPLE_PATHS
from threshold_adapter import (
    ThresholdAdapterError,
    correction_thresholds,
    load_threshold_estimate,
)


class ThresholdAdapterTests(unittest.TestCase):
    def test_research_mode_rejects_synthetic_draft(self) -> None:
        with self.assertRaisesRegex(ThresholdAdapterError, "synthetic"):
            load_threshold_estimate(
                EXAMPLE_PATHS["participant_thresholds"],
                participant_id="P_SYNTHETIC_01",
                model="D288",
                stroke_width_px=3,
            )

    def test_development_mode_loads_exact_mock_threshold(self) -> None:
        estimate = load_threshold_estimate(
            EXAMPLE_PATHS["participant_thresholds"],
            participant_id="P_SYNTHETIC_01",
            model="D288",
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
                model="D288",
                stroke_width_px=4,
                development_mode=True,
            )

    def test_missing_participant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ThresholdAdapterError, "참가자"):
            load_threshold_estimate(
                EXAMPLE_PATHS["participant_thresholds"],
                participant_id="P404",
                model="D288",
                stroke_width_px=3,
                development_mode=True,
            )

    def test_point_policy_builds_baseline_thresholds(self) -> None:
        estimate = load_threshold_estimate(
            EXAMPLE_PATHS["participant_thresholds"],
            participant_id="P_SYNTHETIC_01",
            model="D288",
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
            model="D288",
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
            model="D288",
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
            model="D288",
            stroke_width_px=3,
            development_mode=True,
        )
        with self.assertRaisesRegex(ThresholdAdapterError, "미확정"):
            correction_thresholds(estimate, original_min=10)


if __name__ == "__main__":
    unittest.main()
