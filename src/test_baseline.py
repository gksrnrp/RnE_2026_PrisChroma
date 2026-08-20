import unittest

from baseline import run_baseline
from config import ConfusionThresholds, SearchConfig


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        # 아래 값은 기능 검증용 실험값이며 사용자 임계값의 정답이 아니다.
        self.thresholds = ConfusionThresholds(20.0, 5.0, 10.0)

    def test_same_red_is_preserved_without_confusing_partner(self) -> None:
        result = run_baseline(
            (192, 32, 32),
            (32, 32, 192),
            "deutan",
            1.0,
            self.thresholds,
        )
        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(result["corrected_A"], (192, 32, 32))
        self.assertEqual(result["corrected_B"], (32, 32, 192))

    def test_same_red_is_corrected_with_confusing_green_partner(self) -> None:
        result = run_baseline(
            (192, 32, 32),
            (64, 128, 0),
            "deutan",
            1.0,
            self.thresholds,
            search_config=SearchConfig(step=0.02, max_shift=0.30),
        )
        self.assertEqual(result["status"], "corrected")
        self.assertTrue(result["before"]["confusing"])
        self.assertTrue(result["after"]["target_met"])
        self.assertGreaterEqual(result["after"]["simulated_distance"], 10.0)
        self.assertNotEqual(
            (result["corrected_A"], result["corrected_B"]),
            ((192, 32, 32), (64, 128, 0)),
        )


if __name__ == "__main__":
    unittest.main()
