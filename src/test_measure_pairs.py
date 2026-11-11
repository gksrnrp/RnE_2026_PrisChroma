"""measure_pairs 단위 테스트."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from measure_pairs import (
    _bbox_gap,
    _condition_key,
    _validate_capture_meta,
    load_capture_meta_csv,
    measure_directory,
    measure_image,
)


def _write_test_image(path: Path) -> None:
    """빨강과 초록 사각형이 흰 배경 위에 떨어져 있는 작은 이미지."""
    from PIL import Image

    image = np.full((60, 60, 3), 255, dtype=np.uint8)
    image[10:25, 10:25] = (214, 39, 40)
    image[35:50, 35:50] = (44, 160, 44)
    Image.fromarray(image).save(path)


class BboxGapTests(unittest.TestCase):
    def test_separated_boxes(self):
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 0.0, 30.0, 10.0],
            ]
        )
        gap = _bbox_gap(boxes, np.array([0]), np.array([1]))
        # 좌표 11~19의 빈 픽셀 9개가 사이에 있다.
        self.assertAlmostEqual(float(gap[0]), 9.0)

    def test_boxes_that_touch_on_next_pixel_have_zero_gap(self):
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [11.0, 0.0, 20.0, 10.0],
            ]
        )
        gap = _bbox_gap(boxes, np.array([0]), np.array([1]))
        self.assertEqual(float(gap[0]), 0.0)

    def test_overlapping_boxes_have_zero_gap(self):
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [5.0, 5.0, 15.0, 15.0],
            ]
        )
        gap = _bbox_gap(boxes, np.array([0]), np.array([1]))
        self.assertEqual(float(gap[0]), 0.0)

    def test_diagonal_gap_uses_both_axes(self):
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [13.0, 14.0, 20.0, 20.0],
            ]
        )
        gap = _bbox_gap(boxes, np.array([0]), np.array([1]))
        # 빈 행 2개, 빈 열 3개의 대각 거리다.
        self.assertAlmostEqual(float(gap[0]), math.sqrt(13.0))


class ConditionKeyTests(unittest.TestCase):
    def test_severity_becomes_three_digits(self):
        self.assertEqual(_condition_key("deutan", 1.0), "deutan_100")
        self.assertEqual(_condition_key("protan", 0.6), "protan_060")
        self.assertEqual(_condition_key("protan", 0.0), "protan_000")


class CaptureMetaTests(unittest.TestCase):
    def test_none_passes_through(self):
        self.assertIsNone(_validate_capture_meta(None))

    def test_missing_key_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_capture_meta({"screen_diagonal_inch": 15.6})

    def test_complete_meta_is_copied(self):
        meta = {
            "screen_diagonal_inch": 15.6,
            "screen_resolution": [1920, 1080],
            "viewing_distance_cm": 50,
        }
        result = _validate_capture_meta(meta)
        self.assertEqual(result, meta)
        self.assertIsNot(result, meta)

    def test_csv_rows_without_screen_values_are_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sources.csv"
            path.write_text(
                "filename,screen_diagonal_inch,screen_width_px,"
                "screen_height_px,viewing_distance_cm\n"
                "good.png,15.6,1920,1080,50\n"
                "bad.png,,,,\n",
                encoding="utf-8",
            )
            table = load_capture_meta_csv(path)
        self.assertIn("good.png", table)
        self.assertNotIn("bad.png", table)
        self.assertEqual(table["good.png"]["screen_resolution"], [1920, 1080])


class MeasureImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._folder = tempfile.TemporaryDirectory()
        cls.image_dir = Path(cls._folder.name) / "images"
        cls.image_dir.mkdir()
        _write_test_image(cls.image_dir / "two_squares.png")
        cls.measurement = measure_image(
            cls.image_dir / "two_squares.png",
            n_segments=60,
            merge_thresh=2.0,
        )

    @classmethod
    def tearDownClass(cls):
        cls._folder.cleanup()

    def test_finds_both_squares_and_background(self):
        colors = {
            (record["r"], record["g"], record["b"])
            for record in self.measurement["regions"]
        }
        self.assertIn((214, 39, 40), colors)
        self.assertIn((44, 160, 44), colors)
        self.assertIn((255, 255, 255), colors)

    def test_does_not_judge_confusion(self):
        # ②단계는 수치만 남긴다. 0/1 판정 열이 생기면 안 된다.
        forbidden = {"is_confused", "confusing", "confused", "verdict"}
        for pair in self.measurement["pairs"]:
            self.assertEqual(forbidden & set(pair), set())

    def test_records_both_default_cvd_conditions(self):
        pair = self.measurement["pairs"][0]
        self.assertIn("delta_e_sim_protan_100", pair)
        self.assertIn("delta_e_sim_deutan_100", pair)

    def test_red_green_pair_collapses_under_deutan(self):
        pair = self._red_green_pair()
        self.assertGreater(pair["delta_e_original"], 50.0)
        self.assertLess(pair["delta_e_sim_deutan_100"], 15.0)

    def test_separated_squares_are_not_adjacent(self):
        self.assertEqual(self._red_green_pair()["adjacent"], 0)
        self.assertGreater(self._red_green_pair()["bbox_gap_px"], 0.0)

    def test_arcmin_columns_only_with_capture_meta(self):
        self.assertNotIn(
            "stroke_min_arcmin", self.measurement["pairs"][0]
        )
        with_meta = measure_image(
            self.image_dir / "two_squares.png",
            n_segments=60,
            merge_thresh=2.0,
            capture_meta={
                "screen_diagonal_inch": 15.6,
                "screen_resolution": [1920, 1080],
                "viewing_distance_cm": 50,
            },
        )
        self.assertIn("stroke_min_arcmin", with_meta["pairs"][0])

    def test_distance_limit_drops_far_pairs(self):
        near_only = measure_image(
            self.image_dir / "two_squares.png",
            n_segments=60,
            merge_thresh=2.0,
            d_max=1.0,
        )
        for pair in near_only["pairs"]:
            self.assertTrue(
                pair["adjacent"] or pair["centroid_distance_px"] <= 1.0
            )

    def test_max_pairs_records_what_was_dropped(self):
        capped = measure_image(
            self.image_dir / "two_squares.png",
            n_segments=60,
            merge_thresh=2.0,
            max_pairs=1,
        )
        self.assertEqual(len(capped["pairs"]), 1)
        self.assertGreater(capped["pairs_dropped"], 0)

    def _red_green_pair(self):
        index = {
            record["region_id"]: (record["r"], record["g"], record["b"])
            for record in self.measurement["regions"]
        }
        for pair in self.measurement["pairs"]:
            colors = {index[pair["region_a"]], index[pair["region_b"]]}
            if colors == {(214, 39, 40), (44, 160, 44)}:
                return pair
        self.fail("빨강-초록 쌍을 찾지 못했습니다.")


class MeasureDirectoryTests(unittest.TestCase):
    def test_writes_json_and_reads_per_image_meta(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images"
            images.mkdir()
            _write_test_image(images / "two_squares.png")
            sources = root / "sources.csv"
            sources.write_text(
                "filename,screen_diagonal_inch,screen_width_px,"
                "screen_height_px,viewing_distance_cm\n"
                "two_squares.png,15.6,1920,1080,50\n",
                encoding="utf-8",
            )
            out_json = root / "measurements.json"
            measure_directory(
                images,
                out_json,
                sources_csv=sources,
                n_segments=60,
                merge_thresh=2.0,
                verbose=False,
            )
            payload = json.loads(out_json.read_text(encoding="utf-8"))

        self.assertEqual(payload["image_count"], 1)
        measurement = payload["measurements"][0]
        self.assertEqual(
            measurement["capture_meta"]["screen_diagonal_inch"], 15.6
        )
        self.assertIn("stroke_min_arcmin", measurement["pairs"][0])

    def test_empty_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                measure_directory(folder, Path(folder) / "out.json")


if __name__ == "__main__":
    unittest.main()
