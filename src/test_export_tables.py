"""export_tables 단위 테스트."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from export_tables import (
    MEASURED_OPTIMAL_THRESHOLD,
    build_conditions,
    collect_rows,
    export_all,
)


def _region(region_id, rgb, area, stroke, arcmin=None):
    record = {
        "region_id": region_id,
        "r": rgb[0],
        "g": rgb[1],
        "b": rgb[2],
        "area_px": area,
        "centroid_x": 10.0,
        "centroid_y": 10.0,
        "bbox_min_row": 0,
        "bbox_min_col": 0,
        "bbox_max_row": 5,
        "bbox_max_col": 5,
        "stroke_width_px": stroke,
        "color_spread": 1.0,
        "touches_border": 0,
        "lab_L": 50.0,
        "lab_a": 0.0,
        "lab_b": 0.0,
    }
    if arcmin is not None:
        record["stroke_width_arcmin"] = arcmin
    return record


def _pair(a, b, distance, stroke, background=0, arcmin=False):
    record = {
        "region_a": a,
        "region_b": b,
        "adjacent": 0,
        "shared_border_px": 0.0,
        "centroid_distance_px": distance,
        "bbox_gap_px": distance / 2.0,
        "delta_e_original": 40.0,
        "area_min_px": 100,
        "area_max_px": 200,
        "stroke_min_px": stroke,
        "stroke_max_px": stroke + 1.0,
        "color_spread_max": 2.0,
        "involves_background": background,
        "delta_e_sim_deutan_100": 2.0,
    }
    if arcmin:
        record["centroid_distance_arcmin"] = distance * 1.236
        record["bbox_gap_arcmin"] = distance / 2.0 * 1.236
        record["stroke_min_arcmin"] = stroke * 1.236
    return record


def _payload(capture_meta=None, arcmin=False):
    return {
        "image_dir": "dummy",
        "image_count": 1,
        "measurements": [
            {
                "image": "a.png",
                "width": 100,
                "height": 100,
                "capture_meta": capture_meta,
                "background_region_id": 2,
                "pairs_dropped": 0,
                "regions": [
                    _region(0, (200, 30, 30), 100, 3.0),
                    _region(1, (30, 200, 30), 120, 9.0),
                    _region(2, (255, 255, 255), 9000, 80.0),
                ],
                "pairs": [
                    _pair(0, 1, 40.0, 3.0, arcmin=arcmin),
                    _pair(0, 2, 90.0, 3.0, background=1, arcmin=arcmin),
                    _pair(1, 2, 60.0, 9.0, background=1, arcmin=arcmin),
                ],
            }
        ],
    }


class CollectRowsTests(unittest.TestCase):
    def test_flattens_and_marks_background(self):
        region_rows, pair_rows = collect_rows(_payload())
        self.assertEqual(len(region_rows), 3)
        self.assertEqual(len(pair_rows), 3)
        self.assertTrue(all(row["image"] == "a.png" for row in region_rows))
        background = [row for row in region_rows if row["is_background"]]
        self.assertEqual([row["region_id"] for row in background], [2])


class BuildConditionsTests(unittest.TestCase):
    def test_warns_when_capture_meta_missing(self):
        payload = _payload(capture_meta=None)
        text = build_conditions(payload, *collect_rows(payload))
        self.assertIn("capture_meta가 비어 있어", text)
        # 경고 문구에는 arcmin이라는 말이 나오지만 표에는 arcmin 행이 없다.
        self.assertNotIn("(arcmin) |", text)

    def test_includes_arcmin_rows_when_available(self):
        meta = {
            "screen_diagonal_inch": 15.6,
            "screen_resolution": [1920, 1080],
            "viewing_distance_cm": 50,
            "source": "테스트",
        }
        payload = _payload(capture_meta=meta, arcmin=True)
        text = build_conditions(payload, *collect_rows(payload))
        self.assertIn("(arcmin)", text)
        self.assertIn("15.6", text)

    def test_background_pairs_are_excluded_from_statistics(self):
        payload = _payload()
        text = build_conditions(payload, *collect_rows(payload))
        # 배경이 끼지 않은 쌍은 셋 중 하나뿐이다.
        self.assertIn("| 배경이 끼지 않은 쌍 | 1 |", text)

    def test_states_the_measured_threshold_caveat(self):
        payload = _payload()
        text = build_conditions(payload, *collect_rows(payload))
        self.assertIn(str(MEASURED_OPTIMAL_THRESHOLD), text)
        self.assertIn("맞붙여", text)

    def test_has_four_stimulus_conditions(self):
        payload = _payload()
        text = build_conditions(payload, *collect_rows(payload))
        for label in ("얇음(5%)", "보통(50%)", "가까움(25%)", "멂(75%)"):
            self.assertIn(label, text)


class ExportAllTests(unittest.TestCase):
    def test_writes_three_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            json_path = root / "measurements.json"
            json_path.write_text(
                json.dumps(_payload(), ensure_ascii=False), encoding="utf-8"
            )
            paths = export_all(json_path, root / "out", verbose=False)

            for key in ("regions", "pairs", "conditions"):
                self.assertTrue(Path(paths[key]).exists())

            with open(paths["pairs"], encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertIn("delta_e_sim_deutan_100", rows[0])
            self.assertNotIn("is_confused", rows[0])


if __name__ == "__main__":
    unittest.main()
