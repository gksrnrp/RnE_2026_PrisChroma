import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from implementation_provenance import implementation_hashes
from parameter_recovery import (
    RECOVERY_CRITERIA,
    main,
    run_parameter_recovery,
    simulate_trials,
)


class SyntheticTrialTests(unittest.TestCase):
    def test_same_seed_reproduces_every_trial(self) -> None:
        arguments = {
            "widths": (0.06, 0.18, 0.30),
            "trials_per_width": 24,
            "c80_at_width": lambda width: 2.0 * (width / 0.18) ** -0.4,
            "w": 0.3,
            "lapse": 0.02,
            "seed": 77,
        }
        first = simulate_trials(**arguments)
        second = simulate_trials(**arguments)
        for first_array, second_array in zip(first, second):
            np.testing.assert_array_equal(first_array, second_array)

    def test_invalid_trial_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            simulate_trials(
                widths=(0.06, 0.18),
                trials_per_width=11,
                c80_at_width=lambda _width: 2.0,
                w=0.3,
                lapse=0.02,
                seed=1,
            )

    def test_fixed_absolute_levels_do_not_follow_unknown_truth(self) -> None:
        levels = (0.5, 1.0, 2.0, 4.0)
        x, _, widths = simulate_trials(
            widths=(0.06, 0.18),
            trials_per_width=12,
            c80_at_width=lambda width: 1.0 if width < 0.1 else 7.0,
            w=0.3,
            lapse=0.02,
            seed=3,
            delta_e00_levels=levels,
        )
        for width in (0.06, 0.18):
            self.assertEqual(set(x[np.isclose(widths, width)]), set(levels))

    def test_invalid_fixed_levels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            simulate_trials(
                widths=(0.06, 0.18), trials_per_width=12,
                c80_at_width=lambda _width: 2.0, w=0.3, lapse=0.02,
                seed=1, delta_e00_levels=(0.5, 2.0, 1.0),
            )


class ParameterRecoveryTests(unittest.TestCase):
    def test_small_recovery_is_diagnostic_but_cannot_pass_human_gate(self) -> None:
        result = run_parameter_recovery(
            seed=20260914,
            trials_per_width=96,
            bootstrap_replicates=4,
            simulation_replicates=10,
        )
        self.assertEqual(result["data_origin"], "synthetic")
        self.assertFalse(result["research_conclusion_allowed"])
        expected_hashes = implementation_hashes("parameter_recovery")
        self.assertEqual(result["implementation_hashes"], expected_hashes)
        self.assertEqual(result["implementation_sha256"], expected_hashes)
        self.assertTrue(
            all(len(value) == 64 for value in result["implementation_sha256"].values())
        )
        self.assertTrue(result["all_fits_converged"])
        self.assertFalse(result["passed"])
        self.assertFalse(result["recovery_integrity"]["minimum_simulation_replicates_met"])
        self.assertFalse(result["recovery_integrity"]["minimum_bootstrap_replicates_met"])
        self.assertTrue(
            any("simulation_replicates 10 < 200" in reason for reason in result["reasons"])
        )
        self.assertTrue(
            any("bootstrap_replicates 4 < 200" in reason for reason in result["reasons"])
        )
        self.assertEqual(result["recovery_criteria"], RECOVERY_CRITERIA)
        self.assertEqual(result["monte_carlo"]["simulation_replicates"], 10)
        self.assertEqual(
            result["monte_carlo"]["sampling_design"],
            "fixed_absolute_delta_e00_levels",
        )
        self.assertTrue(result["sampling_plan"]["must_rerun_with_frozen_human_config"])
        profiles = result["monte_carlo"]["profiles"]
        self.assertEqual(
            [profile["name"] for profile in profiles],
            ["low-steep", "central", "high-shallow"],
        )
        self.assertLess(profiles[0]["c80_ref"], profiles[1]["c80_ref"])
        self.assertLess(profiles[1]["c80_ref"], profiles[2]["c80_ref"])
        for profile in profiles:
            self.assertGreaterEqual(
                profile["c80_ref"], profile["feasible_reference_c80_low"]
            )
            self.assertLessEqual(
                profile["c80_ref"], profile["feasible_reference_c80_high"]
            )
        self.assertEqual(
            set(result["monte_carlo"]["profile_results"]),
            {"low-steep", "central", "high-shallow"},
        )
        self.assertEqual(
            set(result["representative_bootstrap"]["models"]),
            {"B96", "B288", "C96", "D288"},
        )
        self.assertTrue(
            result["representative_bootstrap"]["paired_reference_integrity"]["passed"]
        )

        d_fit = result["log_width_scenario"]["fits"]["D288"]
        d_truth = result["log_width_scenario"]["truth"]
        self.assertAlmostEqual(
            d_fit["parameters"]["c80_ref"], d_truth["c80_ref"], delta=0.8
        )
        self.assertAlmostEqual(
            d_fit["parameters"]["beta"], d_truth["beta"], delta=0.35
        )
        self.assertEqual(
            result["log_width_scenario"]["bootstrap"]["requested_replicates"], 4
        )

        c_fit = result["szafir_relative_scenario"]["fits"]["C96"]
        c_truth = result["szafir_relative_scenario"]["truth"]
        self.assertAlmostEqual(
            c_fit["parameters"]["c80_ref"], c_truth["c80_ref"], delta=0.8
        )

    def test_config_hash_marks_recovery_as_bound_without_becoming_human_data(self) -> None:
        result = run_parameter_recovery(
            bootstrap_replicates=2,
            simulation_replicates=6,
            experiment_config_sha256="a" * 64,
            fixed_lapse=0.07,
        )
        self.assertEqual(result["config_sha256"], "a" * 64)
        self.assertEqual(result["experiment_config_sha256"], "a" * 64)
        self.assertEqual(result["bootstrap_replicates"], 2)
        self.assertEqual(result["fixed_lapse"], 0.07)
        self.assertEqual(result["sampling_plan"]["status"], "frozen_human_config_bound")
        self.assertFalse(result["sampling_plan"]["must_rerun_with_frozen_human_config"])
        self.assertFalse(result["research_conclusion_allowed"])

    def test_cli_writes_machine_readable_synthetic_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "recovery-1.json"
            arguments = [
                "--seed",
                "99",
                "--trials-per-width",
                "96",
                "--bootstrap-replicates",
                "2",
                "--simulation-replicates",
                "10",
            ]
            exit_code = main(
                [
                    *arguments,
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 2)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact_type"], "synthetic_parameter_recovery")
            self.assertEqual(payload["seed"], 99)
            self.assertFalse(payload["research_conclusion_allowed"])
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["simulation_replicates"], 10)

            second_output = Path(temporary_directory) / "recovery-2.json"
            self.assertEqual(
                main([*arguments, "--output", str(second_output)]), 2
            )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                second_output.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
