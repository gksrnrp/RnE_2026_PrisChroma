import unittest

from color_spaces import is_lab_in_srgb_gamut
from stimulus_design import generate_symmetric_pair


class StimulusDesignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arguments = {
            "midpoint_lab": (60.0, 8.0, 8.0),
            "direction_lab": (0.0, 1.0, 0.35),
            "target_simulated_delta_e00": 4.0,
            "cvd_type": "deutan",
            "severity": 0.8,
            "max_target_error": 0.35,
            "grid_steps": 1024,
        }

    def test_generation_is_deterministic_and_hits_target_with_reported_error(self) -> None:
        first = generate_symmetric_pair(**self.arguments)
        second = generate_symmetric_pair(**self.arguments)
        self.assertEqual(first, second)
        self.assertLessEqual(first["target_error"], self.arguments["max_target_error"])
        self.assertAlmostEqual(
            first["target_error"],
            abs(first["simulated_delta_e00"] - self.arguments["target_simulated_delta_e00"]),
            places=12,
        )

    def test_continuous_coordinates_are_symmetric_and_in_gamut(self) -> None:
        result = generate_symmetric_pair(**self.arguments)
        first = result["continuous_lab_a"]
        second = result["continuous_lab_b"]
        for index, expected in enumerate(self.arguments["midpoint_lab"]):
            self.assertAlmostEqual((first[index] + second[index]) / 2.0, expected, places=12)
        self.assertTrue(is_lab_in_srgb_gamut(first))
        self.assertTrue(is_lab_in_srgb_gamut(second))
        self.assertEqual(result["metadata"]["continuous_symmetry_max_abs_error"], 0.0)

    def test_metrics_are_for_quantized_display_rgb(self) -> None:
        result = generate_symmetric_pair(**self.arguments)
        self.assertTrue(all(isinstance(channel, int) for channel in result["rgb_a"]))
        self.assertTrue(all(isinstance(channel, int) for channel in result["rgb_b"]))
        self.assertIn("rounding_error_a", result["metadata"]["quantization"])
        self.assertNotEqual(result["rgb_a"], result["rgb_b"])

    def test_rejects_out_of_gamut_midpoint_and_unreachable_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "midpoint_lab"):
            generate_symmetric_pair((50.0, 200.0, 200.0), (0.0, 1.0, 0.0), 2.0, "protan", 1.0)
        with self.assertRaisesRegex(ValueError, "허용 오차"):
            generate_symmetric_pair(
                (60.0, 8.0, 8.0),
                (0.0, 1.0, 0.35),
                100.0,
                "deutan",
                0.8,
                max_target_error=0.01,
                grid_steps=256,
            )

    def test_rejects_zero_direction_and_nonpositive_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "영벡터"):
            generate_symmetric_pair((60, 0, 0), (0, 0, 0), 2, "deutan", 1)
        with self.assertRaisesRegex(ValueError, "0보다 큰"):
            generate_symmetric_pair((60, 0, 0), (0, 1, 0), 0, "deutan", 1)


if __name__ == "__main__":
    unittest.main()
