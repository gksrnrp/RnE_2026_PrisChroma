import math
import unittest

from experiment_design import (
    geometric_validation_widths,
    pixel_pitch_mm,
    pixels_to_visual_angle_deg,
    szafir_line_axis_nd,
    szafir_line_direction_threshold,
    szafir_relative_scale,
    visual_angle_deg_to_pixels,
)


class VisualAngleTests(unittest.TestCase):
    DISPLAY = {
        "display_width_mm": 530.0,
        "horizontal_resolution_px": 1920,
        "viewing_distance_mm": 600.0,
    }

    def test_pixel_pitch(self) -> None:
        self.assertAlmostEqual(pixel_pitch_mm(530, 1920), 530 / 1920)

    def test_visual_angle_round_trip(self) -> None:
        for width_px in (1, 4, 16, 100):
            with self.subTest(width_px=width_px):
                angle = pixels_to_visual_angle_deg(width_px, **self.DISPLAY)
                reconstructed = visual_angle_deg_to_pixels(angle, **self.DISPLAY)
                self.assertAlmostEqual(reconstructed, width_px, places=10)

    def test_rejects_nonphysical_values(self) -> None:
        with self.assertRaises(ValueError):
            pixels_to_visual_angle_deg(0, **self.DISPLAY)
        with self.assertRaises(ValueError):
            visual_angle_deg_to_pixels(180, **self.DISPLAY)


class SzafirLineModelTests(unittest.TestCase):
    def test_axis_equation_matches_published_coefficients(self) -> None:
        expected = 0.5 / (0.0742 - 0.0023 / 0.25)
        self.assertAlmostEqual(szafir_line_axis_nd("L", 0.5, 0.25), expected)

    def test_pure_axis_direction_returns_that_axis_threshold(self) -> None:
        self.assertAlmostEqual(
            szafir_line_direction_threshold((0, 7, 0), 0.5, 0.25),
            szafir_line_axis_nd("a", 0.5, 0.25),
        )

    def test_thicker_line_has_smaller_required_difference(self) -> None:
        thin = szafir_line_direction_threshold((0, 1, 1), 0.5, 0.05)
        thick = szafir_line_direction_threshold((0, 1, 1), 0.5, 0.35)
        self.assertGreater(thin, thick)
        self.assertLess(szafir_relative_scale((0, 1, 1), 0.5, 0.35, 0.05), 1)

    def test_source_range_is_enforced_unless_explicit(self) -> None:
        with self.assertRaises(ValueError):
            szafir_line_axis_nd("b", 0.5, 0.40)
        self.assertTrue(
            math.isfinite(
                szafir_line_axis_nd("b", 0.5, 0.40, allow_extrapolation=True)
            )
        )

    def test_rejects_invalid_probability_and_direction(self) -> None:
        with self.assertRaises(ValueError):
            szafir_line_axis_nd("a", 1.0, 0.20)
        with self.assertRaises(ValueError):
            szafir_line_direction_threshold((0, 0, 0), 0.5, 0.20)


class ValidationWidthTests(unittest.TestCase):
    def test_geometric_candidates_are_inside_intervals(self) -> None:
        low, high = geometric_validation_widths((2, 8, 16))
        self.assertAlmostEqual(low, 4.0)
        self.assertAlmostEqual(high, math.sqrt(128))
        self.assertTrue(2 < low < 8 < high < 16)

    def test_requires_three_increasing_widths(self) -> None:
        with self.assertRaises(ValueError):
            geometric_validation_widths((2, 8))
        with self.assertRaises(ValueError):
            geometric_validation_widths((2, 2, 16))


if __name__ == "__main__":
    unittest.main()
