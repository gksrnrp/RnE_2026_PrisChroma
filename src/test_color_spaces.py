import unittest

from color_spaces import (
    is_oklab_in_srgb_gamut,
    oklab_to_srgb,
    srgb_to_lab,
    srgb_to_oklab,
)


class ColorSpaceTests(unittest.TestCase):
    def test_cielab_reference_colors(self) -> None:
        white = srgb_to_lab((255, 255, 255))
        black = srgb_to_lab((0, 0, 0))
        red = srgb_to_lab((255, 0, 0))
        self.assertAlmostEqual(white[0], 100.0, places=4)
        self.assertAlmostEqual(white[1], 0.0, places=3)
        self.assertAlmostEqual(white[2], 0.0, places=3)
        self.assertAlmostEqual(black[0], 0.0, places=12)
        self.assertAlmostEqual(red[0], 53.2408, places=3)
        self.assertAlmostEqual(red[1], 80.0925, places=3)
        self.assertAlmostEqual(red[2], 67.2032, places=3)

    def test_oklab_round_trip(self) -> None:
        for rgb in ((0, 0, 0), (255, 255, 255), (12, 127, 241), (192, 32, 32)):
            with self.subTest(rgb=rgb):
                restored = oklab_to_srgb(srgb_to_oklab(rgb))
                for actual, expected in zip(restored, rgb):
                    self.assertAlmostEqual(actual, expected, places=3)

    def test_gamut_check(self) -> None:
        self.assertTrue(is_oklab_in_srgb_gamut(srgb_to_oklab((30, 120, 240))))
        self.assertFalse(is_oklab_in_srgb_gamut((0.8, 1.0, 1.0)))


if __name__ == "__main__":
    unittest.main()
