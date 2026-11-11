import unittest
from unittest.mock import patch

from path_correction import find_path_constrained_correction
from stimulus_design import generate_symmetric_pair


def _candidate_record(
    rgb_a: tuple[int, int, int],
    rgb_b: tuple[int, int, int],
    simulated_delta_e00: float,
    scale: float,
) -> dict:
    lab_a = tuple(float(value) for value in rgb_a)
    lab_b = tuple(float(value) for value in rgb_b)
    return {
        "scale": scale,
        "rgb_a": rgb_a,
        "rgb_b": rgb_b,
        "simulated_delta_e00": simulated_delta_e00,
        "continuous_lab_a": lab_a,
        "continuous_lab_b": lab_b,
        "displayed_lab_a": lab_a,
        "displayed_lab_b": lab_b,
        "simulated_rgb_a": lab_a,
        "simulated_rgb_b": lab_b,
        "simulated_lab_a": lab_a,
        "simulated_lab_b": lab_b,
    }


class PathConstrainedCorrectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.common = {
            "midpoint_lab": (60.0, 8.0, 8.0),
            "direction_lab": (0.0, 1.0, 0.35),
            "cvd_type": "deutan",
            "severity": 0.8,
            "grid_steps": 512,
        }
        self.source = generate_symmetric_pair(
            **self.common,
            target_simulated_delta_e00=1.5,
            max_target_error=0.5,
        )

    def _correct(self, target=4.0):
        return find_path_constrained_correction(
            **self.common,
            source_rgb_a=self.source["rgb_a"],
            source_rgb_b=self.source["rgb_b"],
            target_simulated_delta_e00=target,
        )

    def test_corrects_within_same_path_and_meets_target(self) -> None:
        result = self._correct()
        self.assertEqual(result["status"], "corrected")
        self.assertGreaterEqual(result["achieved_simulated_delta_e00"], 4.0)
        self.assertGreater(result["total_color_change_oklab"], 0.0)
        self.assertEqual(
            result["search"]["constraint"],
            "simulated_delta_e00 >= target_simulated_delta_e00",
        )
        self.assertEqual(
            result["search"]["constraint_boundary_tolerance_delta_e00"], 0.0
        )
        self.assertNotIn("maximum_overshoot_delta_e00", result["search"])
        self.assertEqual(
            result["behavioral_scope"],
            "same_lab_midpoint_direction_as_c80_measurement",
        )

    def test_returns_source_when_it_already_meets_target(self) -> None:
        target = self.source["simulated_delta_e00"] * 0.9
        result = self._correct(target)
        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(result["corrected_rgb_a"], self.source["rgb_a"])
        self.assertEqual(result["corrected_rgb_b"], self.source["rgb_b"])
        self.assertEqual(result["total_color_change_oklab"], 0.0)

    def test_unchanged_source_is_not_rejected_for_large_target_overshoot(self) -> None:
        result = find_path_constrained_correction(
            **self.common,
            source_rgb_a=self.source["rgb_a"],
            source_rgb_b=self.source["rgb_b"],
            target_simulated_delta_e00=self.source["simulated_delta_e00"] * 0.1,
        )
        self.assertEqual(result["status"], "unchanged")
        self.assertGreater(result["target_overshoot_delta_e00"], 0.01)
        self.assertEqual(result["total_color_change_oklab"], 0.0)

    def test_all_target_feasible_candidates_compete_even_with_large_overshoot(self) -> None:
        source = ((10, 20, 30), (40, 50, 60))
        lower_cost = ((11, 21, 31), (41, 51, 61))
        lower_overshoot = ((180, 190, 200), (210, 220, 230))

        def fake_candidate(_midpoint, _direction, scale, _cvd_type, _severity):
            if scale < 0.34:
                return _candidate_record(*source, 1.0, scale)
            if scale < 0.67:
                return _candidate_record(*lower_cost, 6.0, scale)
            return _candidate_record(*lower_overshoot, 5.25, scale)

        def fake_movement(_source_a, _source_b, candidate_a, _candidate_b):
            if candidate_a == lower_cost[0]:
                return 0.1, 0.1, 0.2
            return 1.0, 1.0, 2.0

        with patch(
            "path_correction._maximum_symmetric_gamut_scale",
            return_value=(1.0, True),
        ), patch("path_correction._candidate", side_effect=fake_candidate), patch(
            "path_correction._movement", side_effect=fake_movement
        ):
            result = find_path_constrained_correction(
                (50.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                *source,
                5.0,
                "deutan",
                0.8,
                maximum_scale=1.0,
                grid_steps=32,
            )

        self.assertEqual(result["corrected_rgb_a"], lower_cost[0])
        self.assertEqual(result["corrected_rgb_b"], lower_cost[1])
        self.assertEqual(result["target_overshoot_delta_e00"], 1.0)
        self.assertEqual(result["search"]["feasible_pair_count"], 2)

    def test_reachable_target_never_fails_only_because_overshoot_is_large(self) -> None:
        source = ((10, 20, 30), (40, 50, 60))
        destination = ((11, 21, 31), (41, 51, 61))

        def fake_candidate(_midpoint, _direction, scale, _cvd_type, _severity):
            if scale < 0.5:
                return _candidate_record(*source, 1.0, scale)
            return _candidate_record(*destination, 7.0, scale)

        with patch(
            "path_correction._maximum_symmetric_gamut_scale",
            return_value=(1.0, True),
        ), patch("path_correction._candidate", side_effect=fake_candidate):
            result = find_path_constrained_correction(
                (50.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                *source,
                5.0,
                "deutan",
                0.8,
                maximum_scale=1.0,
                grid_steps=32,
            )

        self.assertEqual(result["corrected_rgb_a"], destination[0])
        self.assertEqual(result["corrected_rgb_b"], destination[1])
        self.assertEqual(result["target_overshoot_delta_e00"], 2.0)

    def test_rejects_arbitrary_rgb_not_from_frozen_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "대칭 자극 쌍"):
            find_path_constrained_correction(
                **self.common,
                source_rgb_a=(10, 20, 30),
                source_rgb_b=(40, 50, 60),
                target_simulated_delta_e00=4.0,
            )

    def test_rejects_noninteger_display_rgb(self) -> None:
        with self.assertRaisesRegex(ValueError, "8-bit 정수"):
            find_path_constrained_correction(
                **self.common,
                source_rgb_a=(10.5, 20, 30),
                source_rgb_b=(40, 50, 60),
                target_simulated_delta_e00=4.0,
            )

    def test_rejects_unreachable_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "도달할 수 없습니다"):
            self._correct(100.0)


if __name__ == "__main__":
    unittest.main()
