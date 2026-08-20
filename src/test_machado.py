"""machado.py의 1단계 핵심 검증."""

import unittest

from machado import get_machado_matrix, simulate_color


class MachadoSimulationTests(unittest.TestCase):
    def test_zero_severity_returns_original(self) -> None:
        colors = [(0, 0, 0), (255, 255, 255), (220, 50, 50), (12, 127, 241)]
        for cvd_type in ("protan", "deutan"):
            for color in colors:
                with self.subTest(cvd_type=cvd_type, color=color):
                    self.assertEqual(simulate_color(color, cvd_type, 0.0), color)

    def test_severity_halfway_interpolates_adjacent_matrices(self) -> None:
        matrix_01 = get_machado_matrix("protan", 0.1)
        matrix_015 = get_machado_matrix("protan", 0.15)
        matrix_02 = get_machado_matrix("protan", 0.2)
        for row in range(3):
            for column in range(3):
                expected = (matrix_01[row][column] + matrix_02[row][column]) / 2
                self.assertAlmostEqual(matrix_015[row][column], expected, places=12)

    def test_full_severity_uses_published_matrix(self) -> None:
        self.assertEqual(
            get_machado_matrix("deutan", 1.0),
            (
                (0.367322, 0.860646, -0.227968),
                (0.280085, 0.672501, 0.047413),
                (-0.011820, 0.042940, 0.968881),
            ),
        )

    def test_protan_and_deutan_are_different(self) -> None:
        color = (220, 50, 50)
        self.assertNotEqual(
            simulate_color(color, "protan", 0.6),
            simulate_color(color, "deutan", 0.6),
        )

    def test_gray_is_preserved(self) -> None:
        for cvd_type in ("protan", "deutan"):
            for severity in (0.1, 0.5, 1.0):
                with self.subTest(cvd_type=cvd_type, severity=severity):
                    self.assertEqual(simulate_color((128, 128, 128), cvd_type, severity), (128, 128, 128))

    def test_aliases_are_supported(self) -> None:
        self.assertEqual(
            simulate_color((220, 50, 50), "deuteranomaly", 0.6),
            simulate_color((220, 50, 50), "deutan", 0.6),
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            simulate_color((220, 50, 50), "unknown", 0.5)
        with self.assertRaises(ValueError):
            simulate_color((220, 50, 50), "deutan", 1.1)
        with self.assertRaises(ValueError):
            simulate_color((256, 50, 50), "deutan", 0.5)


if __name__ == "__main__":
    unittest.main()
