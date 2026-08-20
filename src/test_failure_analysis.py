"""failure_analysis 단위 테스트.

각 실패 유형을 손으로 만든 라벨 지도로 재현해 판정 순서를 고정한다.
"""

from __future__ import annotations

import unittest

import numpy as np

from failure_analysis import classify_elements
from slic_extract import Region


def _region(region_id, rgb, area):
    return Region(
        region_id=region_id,
        rgb=rgb,
        area_px=area,
        centroid_xy=(0.0, 0.0),
        bbox=(0, 0, 1, 1),
        stroke_width_px=1.0,
        color_spread=0.0,
        touches_border=False,
    )


def _element(rgb=(200, 30, 30)):
    return {
        "element_id": 1,
        "kind": "line",
        "nominal_stroke_width_px": 4.0,
        "rgb": list(rgb),
    }


class ClassifyElementsTests(unittest.TestCase):
    def setUp(self):
        self.id_map = np.zeros((10, 10), dtype=np.int32)
        self.id_map[4:6, :] = 1  # 폭 2, 길이 10인 가로선

    def test_detected(self):
        labels = np.zeros((10, 10), dtype=np.int64)
        labels[4:6, :] = 1
        regions = [_region(0, (255, 255, 255), 80), _region(1, (200, 30, 30), 20)]
        [record] = classify_elements(
            labels, regions, [_element()], self.id_map
        )
        self.assertEqual(record["verdict"], "detected")
        self.assertEqual(record["overlap_region_count"], 1)
        self.assertLess(record["delta_e_to_truth"], 1.0)

    def test_merged_with_background(self):
        # 선을 덮은 영역이 선 면적(20)의 4배를 넘는다.
        labels = np.zeros((10, 10), dtype=np.int64)
        regions = [_region(0, (255, 255, 255), 100)]
        [record] = classify_elements(
            labels, regions, [_element()], self.id_map
        )
        self.assertEqual(record["verdict"], "merged_with_background")

    def test_fragmented(self):
        # 선을 열 조각으로 쪼갠다. 90%를 덮는 데 9개가 필요하므로
        # 8개 기준을 넘긴다.
        labels = np.zeros((10, 10), dtype=np.int64)
        for column in range(10):
            labels[4:6, column] = column + 1
        regions = [_region(0, (255, 255, 255), 80)]
        regions += [_region(index + 1, (200, 30, 30), 2) for index in range(10)]
        [record] = classify_elements(
            labels, regions, [_element()], self.id_map
        )
        self.assertEqual(record["verdict"], "fragmented")
        self.assertEqual(record["overlap_region_count_any"], 10)

    def test_color_shifted(self):
        labels = np.zeros((10, 10), dtype=np.int64)
        labels[4:6, :] = 1
        # 안티앨리어싱으로 배경과 섞인 색을 흉내낸다.
        regions = [_region(0, (255, 255, 255), 80), _region(1, (240, 190, 190), 20)]
        [record] = classify_elements(
            labels, regions, [_element()], self.id_map
        )
        self.assertEqual(record["verdict"], "color_shifted")
        self.assertGreater(record["delta_e_to_truth"], 10.0)

    def test_not_covered_when_regions_were_dropped(self):
        # min_area_px에 걸려 사라진 영역은 regions 목록에 없다.
        labels = np.zeros((10, 10), dtype=np.int64)
        labels[4:6, :] = 7
        [record] = classify_elements(
            labels, [_region(0, (255, 255, 255), 80)], [_element()], self.id_map
        )
        self.assertEqual(record["verdict"], "not_covered")
        self.assertEqual(record["dominant_region_id"], -1)

    def test_not_covered_when_element_has_no_pixels(self):
        [record] = classify_elements(
            np.zeros((10, 10), dtype=np.int64),
            [_region(0, (255, 255, 255), 100)],
            [{"element_id": 99, "kind": "line",
              "nominal_stroke_width_px": 1.0, "rgb": [0, 0, 0]}],
            self.id_map,
        )
        self.assertEqual(record["verdict"], "not_covered")

    def test_background_check_comes_before_color_check(self):
        # 배경에 흡수된 요소는 색도 어긋나지만 흡수로만 센다.
        labels = np.zeros((10, 10), dtype=np.int64)
        regions = [_region(0, (255, 255, 255), 100)]
        [record] = classify_elements(
            labels, regions, [_element()], self.id_map
        )
        self.assertEqual(record["verdict"], "merged_with_background")
        self.assertGreater(record["delta_e_to_truth"], 10.0)

    def test_color_error_is_weighted_by_overlap(self):
        # 큰 조각은 색이 맞고 작은 조각만 어긋나면 detected로 남는다.
        labels = np.zeros((10, 10), dtype=np.int64)
        labels[4:6, :] = 1
        labels[4:6, 9] = 2
        regions = [
            _region(0, (255, 255, 255), 80),
            _region(1, (200, 30, 30), 18),
            _region(2, (255, 255, 255), 2),
        ]
        [record] = classify_elements(
            labels, regions, [_element()], self.id_map
        )
        self.assertEqual(record["verdict"], "detected")


if __name__ == "__main__":
    unittest.main()
