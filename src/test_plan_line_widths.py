import unittest

from plan_line_widths import build_width_report


class BuildWidthReportTests(unittest.TestCase):
    def test_reports_calibration_and_validation_widths(self) -> None:
        report = build_width_report(
            [2, 8, 16],
            display_width_mm=530,
            horizontal_resolution_px=1920,
            viewing_distance_mm=600,
        )
        self.assertEqual(len(report["calibration_widths"]), 3)
        self.assertEqual(len(report["validation_candidates"]), 2)
        self.assertAlmostEqual(
            report["validation_candidates"][0]["width_px"], 4.0
        )
        self.assertTrue(report["warnings"])

    def test_requires_three_increasing_widths(self) -> None:
        with self.assertRaises(ValueError):
            build_width_report(
                [2, 8],
                display_width_mm=530,
                horizontal_resolution_px=1920,
                viewing_distance_mm=600,
            )
        with self.assertRaises(ValueError):
            build_width_report(
                [8, 2, 16],
                display_width_mm=530,
                horizontal_resolution_px=1920,
                viewing_distance_mm=600,
            )


if __name__ == "__main__":
    unittest.main()
