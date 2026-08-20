"""slic_extract 단위 테스트."""

from __future__ import annotations

import math
import unittest

import numpy as np

from slic_extract import (
    Region,
    build_adjacency,
    describe_regions,
    estimate_background,
    px_to_arcmin,
)


class BuildAdjacencyTests(unittest.TestCase):
    def test_shares_border_length(self):
        labels = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 1, 1],
            ]
        )
        adjacency = build_adjacency(labels)
        # 0과 1은 세로 경계 2칸, 0과 2는 가로 경계 2칸을 공유한다.
        self.assertEqual(adjacency[(0, 1)], 2)
        self.assertEqual(adjacency[(0, 2)], 2)
        self.assertEqual(adjacency[(1, 2)], 1)

    def test_single_label_has_no_edges(self):
        self.assertEqual(build_adjacency(np.zeros((4, 4), dtype=int)), {})

    def test_keys_are_ordered(self):
        labels = np.array([[3, 1], [3, 1]])
        for left, right in build_adjacency(labels):
            self.assertLess(left, right)


class DescribeRegionsTests(unittest.TestCase):
    def setUp(self):
        # 왼쪽 절반은 빨강, 오른쪽 절반은 흰색인 8x8 이미지.
        self.image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.image[:, :4] = (200, 20, 20)
        self.image[:, 4:] = (255, 255, 255)
        self.labels = np.zeros((8, 8), dtype=np.int64)
        self.labels[:, 4:] = 1

    def test_representative_color_is_median(self):
        # 이상치 한 픽셀이 있어도 중앙값은 흔들리지 않는다.
        self.image[0, 0] = (0, 255, 0)
        regions = describe_regions(self.image, self.labels)
        red = next(region for region in regions if region.region_id == 0)
        self.assertEqual(red.rgb, (200, 20, 20))

    def test_area_and_bbox(self):
        regions = describe_regions(self.image, self.labels)
        red = next(region for region in regions if region.region_id == 0)
        self.assertEqual(red.area_px, 32)
        self.assertEqual(red.bbox, (0, 0, 7, 3))
        self.assertTrue(red.touches_border)

    def test_stroke_width_of_a_line(self):
        # 폭 2픽셀 세로선은 두께 추정값이 2 근처여야 한다.
        labels = np.zeros((10, 10), dtype=np.int64)
        labels[:, 4:6] = 1
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        regions = describe_regions(image, labels)
        line = next(region for region in regions if region.region_id == 1)
        self.assertAlmostEqual(line.stroke_width_px, 2.0, places=6)

    def test_small_regions_are_dropped(self):
        labels = np.zeros((8, 8), dtype=np.int64)
        labels[0, 0] = 1
        regions = describe_regions(self.image, labels, min_area_px=4)
        self.assertEqual([region.region_id for region in regions], [0])

    def test_color_spread_flags_mixed_regions(self):
        mixed = np.zeros((8, 8, 3), dtype=np.uint8)
        mixed[:4] = (0, 0, 0)
        mixed[4:] = (255, 255, 255)
        labels = np.zeros((8, 8), dtype=np.int64)
        uniform = describe_regions(self.image, labels)[0]
        blended = describe_regions(mixed, labels)[0]
        self.assertGreater(blended.color_spread, uniform.color_spread)


class BackgroundTests(unittest.TestCase):
    def _region(self, region_id, area, touches):
        return Region(
            region_id=region_id,
            rgb=(0, 0, 0),
            area_px=area,
            centroid_xy=(0.0, 0.0),
            bbox=(0, 0, 1, 1),
            stroke_width_px=1.0,
            color_spread=0.0,
            touches_border=touches,
        )

    def test_picks_largest_border_region(self):
        regions = [
            self._region(0, 500, False),
            self._region(1, 100, True),
            self._region(2, 300, True),
        ]
        self.assertEqual(estimate_background(regions), 2)

    def test_none_when_nothing_touches_border(self):
        self.assertIsNone(estimate_background([self._region(0, 10, False)]))


class ArcminTests(unittest.TestCase):
    scale = {
        "screen_diagonal_inch": 15.6,
        "screen_resolution": (1920, 1080),
        "viewing_distance_cm": 50.0,
    }

    def test_known_value(self):
        # 15.6인치 1920x1080에서 1 px은 약 0.018 cm, 50 cm 거리에서
        # 약 1.24 arcmin이다.
        self.assertAlmostEqual(px_to_arcmin(1.0, **self.scale), 1.236, places=2)

    def test_proportional_for_small_angles(self):
        single = px_to_arcmin(1.0, **self.scale)
        self.assertAlmostEqual(
            px_to_arcmin(10.0, **self.scale) / single, 10.0, places=3
        )

    def test_zero_size_is_zero_angle(self):
        self.assertEqual(px_to_arcmin(0.0, **self.scale), 0.0)

    def test_rejects_zero_distance(self):
        broken = dict(self.scale, viewing_distance_cm=0.0)
        with self.assertRaises(ValueError):
            px_to_arcmin(1.0, **broken)

    def test_farther_screen_gives_smaller_angle(self):
        near = px_to_arcmin(10.0, **self.scale)
        far = px_to_arcmin(
            10.0, **dict(self.scale, viewing_distance_cm=100.0)
        )
        self.assertLess(far, near)
        self.assertTrue(math.isfinite(far))


if __name__ == "__main__":
    unittest.main()
