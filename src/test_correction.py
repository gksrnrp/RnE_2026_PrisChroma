import unittest

from config import SearchConfig
from correction import find_minimum_correction
from distance import color_distance
from machado import simulate_color_float


class CorrectionTests(unittest.TestCase):
    def test_compares_all_three_strategies_and_meets_target(self) -> None:
        result = find_minimum_correction(
            (192, 32, 32),
            (64, 128, 0),
            "deutan",
            1.0,
            10.0,
            search_config=SearchConfig(step=0.02, max_shift=0.30),
        )
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["simulated_distance"], 10.0)
        self.assertIn(result["strategy"], {"A_only", "B_only", "both"})
        self.assertTrue(result["in_srgb_gamut"])

        successful_alternatives = [
            alternative
            for alternative in result["alternatives"].values()
            if alternative is not None
        ]
        self.assertEqual(len(successful_alternatives), 3)
        self.assertAlmostEqual(
            result["total_color_change"],
            min(item["total_color_change"] for item in successful_alternatives),
        )

    def test_already_separated_pair_is_unchanged(self) -> None:
        result = find_minimum_correction(
            (192, 32, 32),
            (32, 32, 192),
            "deutan",
            1.0,
            10.0,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["strategy"], "unchanged")
        self.assertEqual(result["corrected_A"], (192, 32, 32))
        self.assertEqual(result["corrected_B"], (32, 32, 192))
        self.assertEqual(result["total_color_change"], 0.0)
        search = result["search"]
        self.assertTrue(search["search_skipped"])
        self.assertFalse(search["refinement_performed"])
        self.assertFalse(search["refinement_improved"])
        self.assertEqual(
            search["refinement_gain"],
            {"A_only": 0.0, "B_only": 0.0, "both": 0.0},
        )

    def test_result_reports_measured_gamut_and_hue_shift(self) -> None:
        result = find_minimum_correction(
            (192, 32, 32),
            (64, 128, 0),
            "deutan",
            1.0,
            10.0,
            search_config=SearchConfig(step=0.02, max_shift=0.30),
        )
        # in_srgb_gamut은 하드코딩이 아니라 반환 색을 실제로 검사한 결과여야 한다.
        for channel in result["corrected_A"] + result["corrected_B"]:
            self.assertIsInstance(channel, int)
            self.assertGreaterEqual(channel, 0)
            self.assertLessEqual(channel, 255)
        self.assertTrue(result["in_srgb_gamut"])
        self.assertEqual(
            result["max_hue_shift_degrees"],
            max(
                abs(result["hue_shift_A_degrees"]),
                abs(result["hue_shift_B_degrees"]),
            ),
        )

    def test_local_refinement_recovers_known_smaller_b_only_solution(self) -> None:
        # radial grid만 사용하던 버전은 총 이동량 0.050789의 양쪽 이동 해를
        # 선택했다. 정수 RGB 전체를 대조해 확인한 더 작은 B-only 해가
        # 정밀 탐색 변경 뒤에도 유지되는지 검사한다.
        result = find_minimum_correction(
            (192, 32, 32),
            (64, 128, 0),
            "deutan",
            1.0,
            10.0,
        )
        self.assertEqual(result["strategy"], "B_only")
        self.assertEqual(result["corrected_A"], (192, 32, 32))
        self.assertEqual(result["corrected_B"], (22, 130, 69))
        self.assertLess(result["total_color_change"], 0.0497)
        self.assertGreaterEqual(result["simulated_distance"], 10.0)
        self.assertEqual(
            result["search"]["method"],
            "radial_grid_with_multistart_rgb_refinement",
        )


class SearchStepStabilityTests(unittest.TestCase):
    """과거 단일 시작점 탐색에서 step별로 진동했던 색쌍의 회귀 테스트."""

    COLOR_A = (142, 160, 45)
    COLOR_B = (36, 169, 113)
    CVD_TYPE = "deutan"
    SEVERITY = 0.8
    TARGET = 15.0

    # 같은 색쌍을 여러 step으로 반복 탐색하므로 결과를 클래스 단위로 캐시한다.
    _cache: dict[float, dict[str, object]] = {}

    def _result(self, step: float) -> dict[str, object]:
        if step not in self._cache:
            result = find_minimum_correction(
                self.COLOR_A,
                self.COLOR_B,
                self.CVD_TYPE,
                self.SEVERITY,
                self.TARGET,
                search_config=SearchConfig(step=step, max_shift=0.30),
            )
            self.assertTrue(result["success"])
            self._cache[step] = result
        return self._cache[step]

    def test_multistart_removes_known_step_oscillation(self) -> None:
        results = {
            step: self._result(step)
            for step in (0.04, 0.02, 0.01, 0.005)
        }
        totals = [float(result["total_color_change"]) for result in results.values()]
        for result in results.values():
            self.assertEqual(result["strategy"], "B_only")
            self.assertEqual(result["corrected_A"], self.COLOR_A)
            self.assertEqual(result["corrected_B"], (35, 169, 114))
            self.assertGreaterEqual(result["simulated_distance"], self.TARGET)
        self.assertLess(max(totals) - min(totals), 1e-12)


class PairRefinementConvergenceTests(unittest.TestCase):
    """한 번의 pair cube로는 경계에서 멈췄던 benchmark 사례의 회귀 테스트."""

    def test_repeated_pair_refinement_removes_remaining_nonmonotonic_case(self) -> None:
        color_a = (81, 226, 75)
        color_b = (226, 217, 80)
        results = [
            find_minimum_correction(
                color_a,
                color_b,
                "deutan",
                0.8,
                15.0,
                search_config=SearchConfig(step=step, max_shift=0.30),
            )
            for step in (0.02, 0.005)
        ]
        for result in results:
            self.assertTrue(result["success"])
            self.assertEqual(result["strategy"], "both")
            self.assertEqual(result["corrected_A"], (17, 221, 135))
            self.assertEqual(result["corrected_B"], (226, 217, 79))
            self.assertGreaterEqual(result["simulated_distance"], 15.0)
            self.assertGreater(result["search"]["pair_refinement_passes"], 1)
        self.assertAlmostEqual(
            results[0]["total_color_change"],
            results[1]["total_color_change"],
            places=12,
        )

    def test_pair_refinement_uses_single_solution_as_a_nearby_center(self) -> None:
        color_a = (96, 214, 134)
        color_b = (172, 196, 18)
        results = [
            find_minimum_correction(
                color_a,
                color_b,
                "deutan",
                0.8,
                15.0,
                search_config=SearchConfig(step=step, max_shift=0.30),
            )
            for step in (0.02, 0.005)
        ]
        for result in results:
            self.assertEqual(result["strategy"], "both")
            self.assertEqual(result["corrected_A"], (92, 214, 138))
            self.assertEqual(result["corrected_B"], (172, 196, 17))
        self.assertAlmostEqual(
            results[0]["total_color_change"],
            results[1]["total_color_change"],
            places=12,
        )


class RefinementMetadataTests(unittest.TestCase):
    def test_running_refinement_is_reported_separately_from_improving_it(self) -> None:
        result = find_minimum_correction(
            (220, 50, 50),
            (60, 160, 60),
            "deutan",
            0.8,
            15.0,
            search_config=SearchConfig(step=0.02, max_shift=0.30),
        )
        search = result["search"]
        # 후보 개수는 정밀 탐색을 돌리기만 하면 항상 0보다 크므로,
        # 그것만으로 "개선됐다"고 읽으면 안 된다.
        self.assertTrue(search["refinement_performed"])
        self.assertTrue(all(count > 1 for count in search["refinement_start_counts"].values()))
        self.assertTrue(all(count > 0 for count in search["refinement_candidate_counts"].values()))
        self.assertEqual(
            search["refinement_improved"],
            any(gain > 0.0 for gain in search["refinement_gain"].values()),
        )
        for gain in search["refinement_gain"].values():
            self.assertGreaterEqual(gain, 0.0)  # 정밀 탐색은 결과를 악화시키지 않는다.


class ReconfusionTests(unittest.TestCase):
    """보정 결과가 주변 색과 새로 혼동되지 않는지 검사한다."""

    CVD_TYPE = "deutan"
    SEVERITY = 0.8
    TARGET = 15.0
    COLOR_A = (220, 50, 50)
    COLOR_B = (60, 160, 60)
    THIRD = (20, 175, 100)

    def _simulated_distance(self, first, second) -> float:
        return color_distance(
            simulate_color_float(first, self.CVD_TYPE, self.SEVERITY),
            simulate_color_float(second, self.CVD_TYPE, self.SEVERITY),
            method="ciede2000",
        )

    def _correct(self, palette):
        return find_minimum_correction(
            self.COLOR_A,
            self.COLOR_B,
            self.CVD_TYPE,
            self.SEVERITY,
            self.TARGET,
            search_config=SearchConfig(step=0.02, max_shift=0.30),
            palette=palette,
        )

    def test_without_palette_correction_can_collide_with_a_third_color(self) -> None:
        # 재혼동 제약이 실제로 필요한 상황임을 고정해 두는 회귀 테스트다.
        result = self._correct(None)
        moved = [
            corrected
            for corrected, original in (
                (result["corrected_A"], self.COLOR_A),
                (result["corrected_B"], self.COLOR_B),
            )
            if tuple(corrected) != original
        ]
        closest = min(self._simulated_distance(c, self.THIRD) for c in moved)
        self.assertLess(closest, self.TARGET)
        self.assertFalse(result["search"]["reconfusion_checked"])

    def test_palette_keeps_every_moved_color_away_from_third_color(self) -> None:
        result = self._correct([self.THIRD])
        self.assertTrue(result["success"])
        self.assertTrue(result["search"]["reconfusion_checked"])
        self.assertEqual(result["search"]["palette_size"], 1)
        for corrected, original in (
            (result["corrected_A"], self.COLOR_A),
            (result["corrected_B"], self.COLOR_B),
        ):
            if tuple(corrected) == original:
                continue  # 움직이지 않은 색은 보정이 만든 문제가 아니다.
            self.assertGreaterEqual(
                self._simulated_distance(corrected, self.THIRD), self.TARGET
            )

    def test_palette_ignores_the_pair_being_corrected(self) -> None:
        # A, B 자신이 팔레트에 있어도 "원본에서 멀어져라"로 해석되면 안 된다.
        with_pair = self._correct([self.COLOR_A, self.COLOR_B])
        without = self._correct(None)
        self.assertEqual(with_pair["search"]["palette_size"], 0)
        self.assertEqual(with_pair["corrected_A"], without["corrected_A"])
        self.assertEqual(with_pair["corrected_B"], without["corrected_B"])


if __name__ == "__main__":
    unittest.main()
