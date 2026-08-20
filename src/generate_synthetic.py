"""정답을 아는 합성 이미지 생성기.

실제 화면 캡처에는 "이 픽셀이 어떤 요소인가"라는 정답이 없다. 분할이
얼마나 실패하는지 세려면 정답이 필요하므로, 같은 그림을 두 장 그린다.

* RGB 이미지  : 4배 크기로 그린 뒤 축소해 안티앨리어싱을 만든다.
                얇은 선의 대표색이 배경색과 섞이는 실제 문제를 재현한다.
* 요소 ID 맵  : 같은 도형을 축소만 최근접으로 해 경계를 흐리지 않는다.
                요소별 정답 픽셀 위치가 된다.

색은 적록색약에서 실제로 겹치는 쌍만 쓴다. 생성할 때 Machado
시뮬레이션 색차를 함께 계산해 정답 파일에 기록한다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from distance import color_distance
from machado import simulate_color_float


# 적록색약에서 겹치기 쉬운 색쌍. 원본은 뚜렷이 다르지만 시뮬레이션
# 후에는 가까워지는 조합을 골랐고, 실제 색차는 생성 시 계산해 기록한다.
CONFUSABLE_PALETTE: tuple[tuple[int, int, int], ...] = (
    (214, 39, 40),    # 빨강
    (44, 160, 44),    # 초록
    (188, 189, 34),   # 올리브
    (140, 86, 75),    # 갈색
    (255, 127, 14),   # 주황
    (127, 127, 127),  # 회색
    (148, 103, 189),  # 보라
    (31, 119, 180),   # 파랑
)

BACKGROUND = (255, 255, 255)
SUPERSAMPLE = 4


def _new_canvases(width: int, height: int):
    from PIL import Image, ImageDraw

    scale = SUPERSAMPLE
    rgb = Image.new("RGB", (width * scale, height * scale), BACKGROUND)
    ids = Image.new("I", (width * scale, height * scale), 0)
    return rgb, ImageDraw.Draw(rgb), ids, ImageDraw.Draw(ids)


def _finish(rgb, ids, width: int, height: int):
    from PIL import Image

    color_image = rgb.resize((width, height), Image.LANCZOS)
    id_image = ids.resize((width, height), Image.NEAREST)
    return (
        np.asarray(color_image, dtype=np.uint8),
        np.asarray(id_image, dtype=np.int32),
    )


def _polyline(draw_rgb, draw_ids, points, color, width_px, element_id):
    scaled = [(x * SUPERSAMPLE, y * SUPERSAMPLE) for x, y in points]
    line_width = max(1, int(round(width_px * SUPERSAMPLE)))
    draw_rgb.line(scaled, fill=color, width=line_width, joint="curve")
    draw_ids.line(scaled, fill=element_id, width=line_width, joint="curve")


def _rectangle(draw_rgb, draw_ids, box, color, element_id):
    left, top, right, bottom = (value * SUPERSAMPLE for value in box)
    draw_rgb.rectangle((left, top, right, bottom), fill=color)
    draw_ids.rectangle((left, top, right, bottom), fill=element_id)


def _ellipse(draw_rgb, draw_ids, box, color, element_id):
    left, top, right, bottom = (value * SUPERSAMPLE for value in box)
    draw_rgb.ellipse((left, top, right, bottom), fill=color)
    draw_ids.ellipse((left, top, right, bottom), fill=element_id)


def _element(element_id, kind, color, nominal_width):
    """정답 파일에 넣을 요소 한 개의 기록을 만든다."""
    record = {
        "element_id": int(element_id),
        "kind": kind,
        "rgb": list(color),
        "nominal_stroke_width_px": float(nominal_width),
    }
    for cvd_type in ("protan", "deutan"):
        simulated = simulate_color_float(color, cvd_type, 1.0)
        record[f"sim_{cvd_type}_rgb"] = [round(v, 2) for v in simulated]
    return record


def scene_thin_lines(width: int, height: int, stroke_width: float):
    """굵기가 같은 꺾은선 8개. n_segments 검출률 측정의 기준 장면이다."""
    rgb, draw_rgb, ids, draw_ids = _new_canvases(width, height)
    elements = []
    margin = 40
    span = height - 2 * margin
    for index, color in enumerate(CONFUSABLE_PALETTE):
        element_id = index + 1
        baseline = margin + span * (index + 0.5) / len(CONFUSABLE_PALETTE)
        points = []
        for step in range(9):
            x = margin + (width - 2 * margin) * step / 8
            wobble = math.sin(step * 1.1 + index) * span * 0.045
            points.append((x, baseline + wobble))
        _polyline(draw_rgb, draw_ids, points, color, stroke_width, element_id)
        elements.append(_element(element_id, "line", color, stroke_width))
    return _finish(rgb, ids, width, height) + (elements,)


def scene_route_map(width: int, height: int):
    """노선도 모사. 굵은 노선과 역 표시가 섞여 있다."""
    rgb, draw_rgb, ids, draw_ids = _new_canvases(width, height)
    elements = []
    routes = (
        ((60, 80), (240, 80), (300, 140), (300, 300), (240, 360)),
        ((60, 160), (200, 160), (260, 220), (420, 220)),
        ((120, 40), (120, 260), (180, 320), (420, 320)),
        ((60, 240), (160, 240), (220, 300), (220, 380)),
    )
    element_id = 0
    for index, points in enumerate(routes):
        element_id += 1
        color = CONFUSABLE_PALETTE[index]
        _polyline(draw_rgb, draw_ids, points, color, 5.0, element_id)
        elements.append(_element(element_id, "route", color, 5.0))
    for index, points in enumerate(routes):
        for x, y in points[::2]:
            element_id += 1
            _ellipse(
                draw_rgb, draw_ids, (x - 4, y - 4, x + 4, y + 4),
                (30, 30, 30), element_id,
            )
            elements.append(_element(element_id, "station", (30, 30, 30), 8.0))
    return _finish(rgb, ids, width, height) + (elements,)


def scene_bar_chart(width: int, height: int):
    """큰 색면. 얇은 요소와 대비되는 조건이다."""
    rgb, draw_rgb, ids, draw_ids = _new_canvases(width, height)
    elements = []
    bar_count = 6
    margin = 50
    slot = (width - 2 * margin) / bar_count
    for index in range(bar_count):
        element_id = index + 1
        color = CONFUSABLE_PALETTE[index]
        left = margin + slot * index + slot * 0.15
        right = margin + slot * (index + 1) - slot * 0.15
        top = margin + (height - 2 * margin) * (0.15 + 0.11 * index)
        _rectangle(
            draw_rgb, draw_ids,
            (left, top, right, height - margin), color, element_id,
        )
        elements.append(_element(element_id, "bar", color, right - left))
    return _finish(rgb, ids, width, height) + (elements,)


def scene_legend_lines(width: int, height: int, stroke_width: float = 3.0):
    """범례와 선이 멀리 떨어진 장면.

    의미로는 짝이지만 공간적으로는 멀다. 거리 근사가 놓치는 쌍을
    수치로 보여주기 위한 장면이다.

    ``stroke_width``를 두껍게 준 판을 따로 만드는 이유: 3 px 선은 어떤
    설정에서도 검출되지 않아 d_max를 아무리 늘려도 범례-선 쌍이 0으로
    나온다. 거리 때문에 못 잡는 것과 선이 얇아 못 잡는 것을 갈라 보려면
    선이 검출되는 판이 하나 필요하다.
    """
    rgb, draw_rgb, ids, draw_ids = _new_canvases(width, height)
    elements = []
    element_id = 0
    legend_x = width - 90
    for index in range(5):
        color = CONFUSABLE_PALETTE[index]
        element_id += 1
        y = 60 + index * 34
        _rectangle(
            draw_rgb, draw_ids,
            (legend_x, y, legend_x + 26, y + 14), color, element_id,
        )
        elements.append(_element(element_id, "legend_swatch", color, 14.0))

        element_id += 1
        points = [
            (50 + step * (width - 200) / 6, 70 + index * 40
             + math.sin(step + index) * 12)
            for step in range(7)
        ]
        _polyline(
            draw_rgb, draw_ids, points, color, stroke_width, element_id
        )
        elements.append(_element(element_id, "line", color, stroke_width))
    return _finish(rgb, ids, width, height) + (elements,)


def scene_dot_plate(width: int, height: int, seed: int = 0):
    """색각 검사표 모사. 작은 원이 빽빽한 조건이다."""
    rgb, draw_rgb, ids, draw_ids = _new_canvases(width, height)
    elements = []
    rng = np.random.default_rng(seed)
    center = (width / 2, height / 2)
    radius = min(width, height) * 0.45
    element_id = 0
    for _ in range(320):
        angle = rng.uniform(0, 2 * math.pi)
        distance = radius * math.sqrt(rng.uniform(0, 1))
        x = center[0] + distance * math.cos(angle)
        y = center[1] + distance * math.sin(angle)
        dot_radius = rng.uniform(4, 9)
        # 가운데 세로 띠만 다른 색으로 채워 도형을 만든다.
        inside = abs(x - center[0]) < radius * 0.25
        color = CONFUSABLE_PALETTE[0] if inside else CONFUSABLE_PALETTE[1]
        element_id += 1
        _ellipse(
            draw_rgb, draw_ids,
            (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
            color, element_id,
        )
        elements.append(
            _element(element_id, "dot", color, dot_radius * 2)
        )
    return _finish(rgb, ids, width, height) + (elements,)


def _pair_reference() -> list[dict[str, object]]:
    """팔레트 색쌍의 원본·시뮬레이션 색차 표. 장면과 무관한 참고값이다."""
    rows = []
    for i, color_a in enumerate(CONFUSABLE_PALETTE):
        for color_b in CONFUSABLE_PALETTE[i + 1:]:
            row = {
                "rgb_a": list(color_a),
                "rgb_b": list(color_b),
                "delta_e_original": round(
                    color_distance(color_a, color_b), 3
                ),
            }
            for cvd_type in ("protan", "deutan"):
                simulated_a = simulate_color_float(color_a, cvd_type, 1.0)
                simulated_b = simulate_color_float(color_b, cvd_type, 1.0)
                row[f"delta_e_{cvd_type}"] = round(
                    color_distance(simulated_a, simulated_b), 3
                )
            rows.append(row)
    return rows


def generate_dataset(
    out_dir: str | Path,
    *,
    width: int = 520,
    height: int = 420,
    stroke_widths: tuple[float, ...] = (1, 2, 3, 4, 6, 8, 12, 16),
    seed: int = 0,
) -> dict[str, object]:
    """합성 이미지와 정답 파일을 한 번에 만든다.

    ``<out_dir>/images``에 PNG를, ``<out_dir>/truth``에 요소 ID 맵을,
    ``<out_dir>/ground_truth.json``에 요소 목록을 쓴다.
    """
    from PIL import Image

    root = Path(out_dir)
    image_dir = root / "images"
    truth_dir = root / "truth"
    image_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    scenes: list[tuple[str, tuple]] = []
    for stroke_width in stroke_widths:
        name = f"lines_w{int(stroke_width):02d}"
        scenes.append((name, scene_thin_lines(width, height, stroke_width)))
    scenes.append(("route_map", scene_route_map(width, height)))
    scenes.append(("bar_chart", scene_bar_chart(width, height)))
    scenes.append(("legend_lines", scene_legend_lines(width, height, 3.0)))
    scenes.append(
        ("legend_lines_w08", scene_legend_lines(width, height, 8.0))
    )
    scenes.append(("dot_plate", scene_dot_plate(width, height, seed)))

    manifest: dict[str, object] = {
        "width": width,
        "height": height,
        "supersample": SUPERSAMPLE,
        "palette_pair_reference": _pair_reference(),
        "images": {},
    }
    for name, (color_array, id_array, elements) in scenes:
        Image.fromarray(color_array).save(image_dir / f"{name}.png")
        np.save(truth_dir / f"{name}.npy", id_array.astype(np.int32))
        present, counts = np.unique(id_array, return_counts=True)
        visible = {
            int(element): int(count)
            for element, count in zip(present, counts)
            if element != 0
        }
        for element in elements:
            element["visible_px"] = visible.get(element["element_id"], 0)
        manifest["images"][name] = {
            "image": f"images/{name}.png",
            "truth": f"truth/{name}.npy",
            "elements": elements,
        }

    with open(root / "ground_truth.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


__all__ = [
    "CONFUSABLE_PALETTE",
    "generate_dataset",
    "scene_bar_chart",
    "scene_dot_plate",
    "scene_legend_lines",
    "scene_route_map",
    "scene_thin_lines",
]
