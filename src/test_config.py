import unittest
import math

from config import ConfusionThresholds, SearchConfig


class ConfusionThresholdsTests(unittest.TestCase):
    def test_valid_combination_is_accepted(self) -> None:
        thresholds = ConfusionThresholds(20.0, 5.0, 10.0)
        self.assertEqual(thresholds.original_min, 20.0)

    def test_original_min_below_simulated_max_is_rejected(self) -> None:
        # 이 조합을 허용하면 원래부터 비슷한 색쌍이 "색각이상 때문에 혼동되는
        # 쌍"으로 잘못 분류된다. README 3절의 주의사항이 코드로 강제되는지 본다.
        with self.assertRaises(ValueError):
            ConfusionThresholds(original_min=1.0, simulated_max=20.0, target=25.0)

    def test_target_below_simulated_max_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ConfusionThresholds(original_min=20.0, simulated_max=10.0, target=5.0)

    def test_equal_bounds_are_allowed(self) -> None:
        thresholds = ConfusionThresholds(10.0, 10.0, 10.0)
        self.assertEqual(thresholds.target, 10.0)


class SearchConfigTests(unittest.TestCase):
    def test_default_step_is_fine_enough_for_reported_minimum(self) -> None:
        # 후속 RGB 정밀 탐색이 있어도 첫 grid가 지나치게 거칠어지지 않도록
        # 보고서에서 사용하는 보수적인 기본값을 고정한다.
        self.assertLessEqual(SearchConfig.step, 0.005)

    def test_step_larger_than_max_shift_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SearchConfig(step=0.5, max_shift=0.1)

    def test_non_finite_search_distances_are_rejected(self) -> None:
        for kwargs in (
            {"step": math.nan},
            {"step": math.inf},
            {"max_shift": math.nan},
            {"max_shift": math.inf},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SearchConfig(**kwargs)

    def test_loop_settings_must_be_positive_integers(self) -> None:
        for kwargs in (
            {"direction_grid_radius": 1.5},
            {"direction_grid_radius": True},
            {"pair_chunk_size": 1.5},
            {"pair_chunk_size": False},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                SearchConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
