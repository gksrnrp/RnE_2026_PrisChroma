import json
import unittest

from benchmark_search import _relative_spread_percent


class BenchmarkSearchTests(unittest.TestCase):
    def test_zero_denominator_uses_json_null_instead_of_infinity(self) -> None:
        spread = _relative_spread_percent([0.0, 1.0])
        self.assertIsNone(spread)
        encoded = json.dumps({"spread": spread}, allow_nan=False)
        self.assertEqual(encoded, '{"spread": null}')

    def test_relative_spread_definition(self) -> None:
        self.assertEqual(_relative_spread_percent([0.0, 0.0]), 0.0)
        self.assertAlmostEqual(_relative_spread_percent([1.0, 2.0]), 100.0)


if __name__ == "__main__":
    unittest.main()
