import unittest

from confusion import is_confusing


class ConfusionTests(unittest.TestCase):
    def test_red_green_pair_is_confusing_for_demo_parameters(self) -> None:
        result = is_confusing(
            (192, 32, 32),
            (64, 128, 0),
            "deutan",
            1.0,
            original_threshold=20.0,
            confusion_threshold=5.0,
        )
        self.assertGreater(result["original_distance"], 20.0)
        self.assertLess(result["simulated_distance"], 5.0)
        self.assertTrue(result["confusing"])

    def test_same_red_with_blue_is_not_confusing(self) -> None:
        result = is_confusing(
            (192, 32, 32),
            (32, 32, 192),
            "deutan",
            1.0,
            original_threshold=20.0,
            confusion_threshold=5.0,
        )
        self.assertGreater(result["simulated_distance"], 5.0)
        self.assertFalse(result["confusing"])

    def test_originally_similar_pair_is_not_cvd_caused_confusion(self) -> None:
        result = is_confusing(
            (100, 100, 100),
            (101, 101, 101),
            "deutan",
            1.0,
            original_threshold=20.0,
            confusion_threshold=5.0,
        )
        self.assertFalse(result["originally_distinct"])
        self.assertFalse(result["confusing"])


if __name__ == "__main__":
    unittest.main()
