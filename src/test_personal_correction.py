import unittest

from baseline import run_baseline
from config import SearchConfig
from data_contracts import EXAMPLE_PATHS
from threshold_adapter import correction_thresholds, load_threshold_estimate


class PersonalCorrectionIntegrationTests(unittest.TestCase):
    def test_mock_c80_flows_into_existing_minimum_correction(self) -> None:
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
        result = run_baseline(
            (32, 96, 224),
            (160, 32, 224),
            estimate.cvd_type,
            estimate.severity,
            thresholds,
            search_config=SearchConfig(step=0.02, max_shift=0.30),
        )

        self.assertEqual(result["status"], "corrected")
        self.assertLess(result["before"]["simulated_distance"], estimate.c80_sim)
        self.assertGreaterEqual(
            result["after"]["simulated_distance"], estimate.c80_sim
        )
        self.assertTrue(result["after"]["target_met"])


if __name__ == "__main__":
    unittest.main()
