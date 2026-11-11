"""correction.py의 numpy 재구현이 color_spaces.py 기준 구현과 같은지 확인한다.

correction.py는 속도를 위해 sRGB->OKLab, sRGB->CIELAB, Machado 시뮬레이션을
벡터 형태로 다시 구현한다. 두 구현이 조용히 어긋나면 탐색은 정상으로 보이지만
결과 색만 틀리게 되므로, 무작위 색으로 두 경로를 대조한다.
"""

import unittest

import numpy as np

from color_spaces import srgb_to_lab, srgb_to_oklab
from correction import (
    _simulate_array_to_lab,
    _srgb_array_to_lab,
    _srgb_array_to_oklab,
)
from machado import simulate_color_float


class VectorConsistencyTests(unittest.TestCase):
    TOLERANCE = 1e-9

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.samples = rng.integers(0, 256, size=(512, 3)).astype(float)

    def _max_difference(self, vectorized: np.ndarray, scalar_rows) -> float:
        return float(np.max(np.abs(vectorized - np.asarray(list(scalar_rows)))))

    def test_oklab_matches_reference(self) -> None:
        difference = self._max_difference(
            _srgb_array_to_oklab(self.samples),
            (srgb_to_oklab(sample) for sample in self.samples),
        )
        self.assertLess(difference, self.TOLERANCE)

    def test_lab_matches_reference(self) -> None:
        difference = self._max_difference(
            _srgb_array_to_lab(self.samples),
            (srgb_to_lab(sample) for sample in self.samples),
        )
        self.assertLess(difference, self.TOLERANCE)

    def test_simulated_lab_matches_reference(self) -> None:
        for cvd_type, severity in (("deutan", 0.8), ("protan", 1.0)):
            with self.subTest(cvd_type=cvd_type, severity=severity):
                difference = self._max_difference(
                    _simulate_array_to_lab(self.samples, cvd_type, severity),
                    (
                        srgb_to_lab(simulate_color_float(sample, cvd_type, severity))
                        for sample in self.samples
                    ),
                )
                self.assertLess(difference, self.TOLERANCE)


if __name__ == "__main__":
    unittest.main()
