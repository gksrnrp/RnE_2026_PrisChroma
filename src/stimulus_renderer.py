"""결정론적인 2x2 선 자극을 무손실 PNG와 픽셀 명세로 만든다.

이 모듈은 응답 수집 GUI가 아니다. Pillow의 고수준 선 그리기 규칙에
의존하지 않고 반열린 정수 사각형을 직접 채워, 선 길이·굵기·끝모양과
안티앨리어싱 정책을 픽셀 단위로 재검증할 수 있게 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, __version__ as PILLOW_VERSION

from calibration_design import (
    CALIBRATION_RANDOMIZATION,
    CALIBRATION_TOTAL_TRIALS,
    block_count_from_config,
    block_trial_count,
    trial_counts_from_config,
    width_block_trial_counts,
)
from color_spaces import srgb_to_lab
from distance import delta_e_2000
from machado import simulate_color_float
from stimulus_design import generate_symmetric_pair


RENDERER_ID = "rne-pillow-raster-v1"
RENDERER_IMPLEMENTATION_VERSION = "1.0"
LOCKED_PILLOW_VERSION = "12.3.0"
CAP_POLICY = "butt"
ANTIALIASING_POLICY = "disabled"
PIXEL_SNAPPING_POLICY = "integer-centers"
RASTER_RULE = (
    "axis-aligned half-open rectangles; requested integer anchor minus "
    "floor(extent/2), with the extra pixel on the negative side for even extents"
)
VALIDATION_LEVEL_COUNT = 8
VALIDATION_BLOCK_COUNT = 3
VALIDATION_TRIALS_PER_BLOCK = 64
PRACTICE_POLICY_ID = "representative-calibration-levels-balanced-v1"
PRACTICE_LEVEL_INDICES = (1, 6, 12)
PRACTICE_FIRST_TRIAL_ID = 1_000_001


class RendererValidationError(ValueError):
    """렌더러 설정, 시행 또는 저장된 픽셀이 검증 규칙을 어겼다."""


class StimulusCompilationError(RendererValidationError):
    """시행표의 목표 색차를 실제 표시 자극으로 컴파일하지 못했다."""


def renderer_implementation_sha256() -> str:
    """현재 렌더러 구현 파일 바이트의 SHA-256을 반환한다."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _fail(message: str) -> None:
    raise RendererValidationError(message)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path}는 객체여야 합니다.")
    return value


def _required(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            _fail(f"필수값이 없습니다: {path}")
        value = value[part]
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{path}는 정수여야 합니다.")
    if minimum is not None and value < minimum:
        _fail(f"{path}는 {minimum} 이상이어야 합니다.")
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{path}는 유한한 수여야 합니다.")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{path}는 유한한 수여야 합니다.")
    return result


def _rgb(value: Any, path: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 0 <= channel <= 255
            for channel in value
        )
    ):
        _fail(f"{path}는 0~255 정수 RGB 세 값이어야 합니다.")
    return int(value[0]), int(value[1]), int(value[2])


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _orientation_class(value: Any) -> tuple[float, str]:
    orientation = _finite_number(value, "line.orientation_deg")
    normalized = orientation % 180.0
    if math.isclose(normalized, 0.0, abs_tol=1e-12) or math.isclose(
        normalized, 180.0, abs_tol=1e-12
    ):
        return 0.0, "horizontal"
    if math.isclose(normalized, 90.0, abs_tol=1e-12):
        return 90.0, "vertical"
    _fail(
        "픽셀 검증 렌더러는 안티앨리어싱 없는 0/90도 축 정렬 선만 지원합니다."
    )


def _positions(config: Mapping[str, Any]) -> list[tuple[int, int]]:
    raw = _required(config, "coordinates.positions_px")
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 4
    ):
        _fail("coordinates.positions_px에는 좌표 네 개가 필요합니다.")
    positions: list[tuple[int, int]] = []
    for index, point in enumerate(raw, start=1):
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes))
            or len(point) != 2
        ):
            _fail(f"coordinates.positions_px[{index - 1}]는 [x,y]여야 합니다.")
        positions.append(
            (
                _integer(point[0], f"coordinates.positions_px[{index - 1}][0]"),
                _integer(point[1], f"coordinates.positions_px[{index - 1}][1]"),
            )
        )

    if len(set(positions)) != 4:
        _fail("coordinates.positions_px의 네 좌표는 서로 달라야 합니다.")
    xs = sorted({point[0] for point in positions})
    ys = sorted({point[1] for point in positions})
    if len(xs) != 2 or len(ys) != 2 or set(positions) != {
        (x, y) for x in xs for y in ys
    }:
        _fail("coordinates.positions_px는 정확한 2x2 직교 격자를 이루어야 합니다.")

    spacing = _integer(
        _required(config, "coordinates.center_spacing_px"),
        "coordinates.center_spacing_px",
        minimum=1,
    )
    if xs[1] - xs[0] != spacing or ys[1] - ys[0] != spacing:
        _fail(
            "coordinates.center_spacing_px가 실제 2x2 좌표의 가로·세로 간격과 다릅니다."
        )
    return positions


def _box_for_anchor(
    anchor: tuple[int, int], length: int, width: int, orientation: str
) -> tuple[int, int, int, int]:
    x, y = anchor
    horizontal_extent = length if orientation == "horizontal" else width
    vertical_extent = width if orientation == "horizontal" else length
    left = x - horizontal_extent // 2
    top = y - vertical_extent // 2
    return left, top, left + horizontal_extent, top + vertical_extent


def _boxes_overlap(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _renderer_geometry(
    config: Mapping[str, Any], stroke_width_px: Any
) -> dict[str, Any]:
    if _required(config, "task.type") != "4afc_odd_one_out":
        _fail("task.type은 4afc_odd_one_out이어야 합니다.")
    if _required(config, "task.layout") != "2x2":
        _fail("task.layout은 2x2여야 합니다.")
    if _required(config, "task.choice_count") != 4:
        _fail("task.choice_count는 4여야 합니다.")

    canvas_width = _integer(
        _required(config, "display.resolution_width_px"),
        "display.resolution_width_px",
        minimum=1,
    )
    canvas_height = _integer(
        _required(config, "display.resolution_height_px"),
        "display.resolution_height_px",
        minimum=1,
    )
    background = _rgb(
        _required(config, "display.background_rgb"), "display.background_rgb"
    )
    line = _mapping(_required(config, "line"), "line")
    if line.get("renderer") != RENDERER_ID:
        _fail(f"line.renderer는 {RENDERER_ID!r}여야 합니다.")
    if line.get("cap") != CAP_POLICY:
        _fail("픽셀 검증 렌더러는 line.cap='butt'만 지원합니다.")
    if line.get("antialiasing") != ANTIALIASING_POLICY:
        _fail("픽셀 검증 렌더러는 line.antialiasing='disabled'만 지원합니다.")
    if line.get("pixel_snapping") != PIXEL_SNAPPING_POLICY:
        _fail(
            "픽셀 검증 렌더러는 line.pixel_snapping='integer-centers'만 지원합니다."
        )
    length = _integer(line.get("length_px"), "line.length_px", minimum=1)
    width = _integer(stroke_width_px, "trial.stroke_width_px", minimum=1)
    normalized_orientation, orientation = _orientation_class(line.get("orientation_deg"))
    positions = _positions(config)
    boxes = [_box_for_anchor(point, length, width, orientation) for point in positions]

    for index, (left, top, right, bottom) in enumerate(boxes, start=1):
        if left < 0 or top < 0 or right > canvas_width or bottom > canvas_height:
            _fail(
                f"자극 {index}의 픽셀 경계 {left, top, right, bottom}가 "
                f"{canvas_width}x{canvas_height} 캔버스를 벗어납니다."
            )
    for first in range(4):
        for second in range(first + 1, 4):
            if _boxes_overlap(boxes[first], boxes[second]):
                _fail(f"자극 위치 {first + 1}과 {second + 1}의 픽셀이 겹칩니다.")

    return {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "background": background,
        "length": length,
        "width": width,
        "orientation_deg": normalized_orientation,
        "orientation_class": orientation,
        "positions": positions,
        "boxes": boxes,
    }


def _position_labels(trial: Mapping[str, Any]) -> list[str]:
    correct_position = _integer(
        _required(trial, "correct_position"), "trial.correct_position", minimum=1
    )
    if correct_position > 4:
        _fail("trial.correct_position은 1~4여야 합니다.")
    odd_color = _required(trial, "odd_color")
    if odd_color not in {"a", "b"}:
        _fail("trial.odd_color는 'a' 또는 'b'여야 합니다.")
    common_color = "b" if odd_color == "a" else "a"
    expected = [common_color] * 4
    expected[correct_position - 1] = odd_color
    if "position_colors" in trial and list(trial["position_colors"]) != expected:
        _fail(
            "trial.position_colors가 correct_position/odd_color로 계산한 배열과 다릅니다."
        )
    return expected


def build_stimulus_spec(
    config: Mapping[str, Any], trial: Mapping[str, Any]
) -> dict[str, Any]:
    """한 시행의 픽셀 기하와 색을 JSON 직렬화 가능한 명세로 만든다."""

    _mapping(config, "config")
    _mapping(trial, "trial")
    trial_id = _integer(_required(trial, "trial_id"), "trial.trial_id", minimum=1)
    geometry = _renderer_geometry(config, _required(trial, "stroke_width_px"))
    background = geometry["background"]
    rgb_a = _rgb(_required(trial, "displayed_rgb_a"), "trial.displayed_rgb_a")
    rgb_b = _rgb(_required(trial, "displayed_rgb_b"), "trial.displayed_rgb_b")
    if rgb_a == rgb_b:
        _fail("displayed_rgb_a와 displayed_rgb_b는 서로 달라야 합니다.")
    if rgb_a == background or rgb_b == background:
        _fail("표시색이 배경색과 같으면 픽셀 수를 독립적으로 검증할 수 없습니다.")

    labels = _position_labels(trial)
    colors = {"a": rgb_a, "b": rgb_b}
    area = geometry["length"] * geometry["width"]
    objects: list[dict[str, Any]] = []
    for index, (anchor, box, label) in enumerate(
        zip(geometry["positions"], geometry["boxes"], labels), start=1
    ):
        left, top, right, bottom = box
        objects.append(
            {
                "position": index,
                "requested_anchor_px": list(anchor),
                "bbox_xyxy_exclusive": list(box),
                "pixel_index_centroid": [
                    (left + right - 1) / 2.0,
                    (top + bottom - 1) / 2.0,
                ],
                "color_label": label,
                "rgb": list(colors[label]),
                "expected_pixel_count": area,
            }
        )

    counts: Counter[tuple[int, int, int]] = Counter()
    counts[background] = (
        geometry["canvas_width"] * geometry["canvas_height"] - 4 * area
    )
    for label in labels:
        counts[colors[label]] += area
    expected_counts = [
        {"rgb": list(rgb), "count": count} for rgb, count in sorted(counts.items())
    ]

    return {
        "schema_version": "1.0",
        "artifact_type": "pixel_specified_2x2_line_stimulus",
        "renderer": {
            "id": RENDERER_ID,
            "implementation_version": RENDERER_IMPLEMENTATION_VERSION,
            "implementation_sha256": renderer_implementation_sha256(),
            "pillow_version": PILLOW_VERSION,
            "locked_pillow_version": LOCKED_PILLOW_VERSION,
            "pillow_lock_match": PILLOW_VERSION == LOCKED_PILLOW_VERSION,
            "image_format": "PNG",
            "color_mode": "RGB",
            "lossless": True,
            "cap": CAP_POLICY,
            "antialiasing": ANTIALIASING_POLICY,
            "resampling": "none",
            "pixel_snapping": PIXEL_SNAPPING_POLICY,
            "raster_rule": RASTER_RULE,
        },
        "config": {
            "config_version": config.get("config_version"),
            "canonical_sha256": _canonical_sha256(config),
            "schedule_config_sha256": trial.get("config_sha256"),
        },
        "trial": {
            "trial_id": trial_id,
            "stimulus_family_id": trial.get(
                "stimulus_family_id",
                _mapping(config.get("color_design", {}), "color_design").get(
                    "stimulus_family_id"
                ),
            ),
            "stroke_width_px": geometry["width"],
            "correct_position": int(trial["correct_position"]),
            "odd_color": trial["odd_color"],
            "position_colors": labels,
            "target_delta_e00_simulated": trial.get(
                "target_delta_e00_simulated", trial.get("delta_e00_simulated")
            ),
            "realized_delta_e00_simulated": trial.get("delta_e00_simulated"),
        },
        "canvas": {
            "width_px": geometry["canvas_width"],
            "height_px": geometry["canvas_height"],
            "background_rgb": list(background),
        },
        "geometry": {
            "line_length_px": geometry["length"],
            "stroke_width_px": geometry["width"],
            "orientation_deg": geometry["orientation_deg"],
            "orientation_class": geometry["orientation_class"],
            "cap": CAP_POLICY,
            "coordinates_are_pixel_indices": True,
            "even_extent_bias": "negative",
        },
        "display_colors": {"a": list(rgb_a), "b": list(rgb_b)},
        "objects": objects,
        "pixel_verification": {
            "status": "specified_not_yet_saved",
            "expected_counts_by_rgb": expected_counts,
            "expected_foreground_pixel_count": 4 * area,
            "decoded_rgb_sha256": None,
            "png_sha256": None,
            "unexpected_rgb_count": None,
        },
    }


def _image_from_spec(spec: Mapping[str, Any]) -> Image.Image:
    canvas = _mapping(_required(spec, "canvas"), "canvas")
    width = _integer(canvas.get("width_px"), "canvas.width_px", minimum=1)
    height = _integer(canvas.get("height_px"), "canvas.height_px", minimum=1)
    background = _rgb(canvas.get("background_rgb"), "canvas.background_rgb")
    image = Image.new("RGB", (width, height), background)
    objects = _required(spec, "objects")
    if not isinstance(objects, Sequence) or len(objects) != 4:
        _fail("objects에는 네 자극이 필요합니다.")
    boxes: list[tuple[int, int, int, int]] = []
    for index, raw in enumerate(objects, start=1):
        item = _mapping(raw, f"objects[{index - 1}]")
        box_raw = item.get("bbox_xyxy_exclusive")
        if (
            not isinstance(box_raw, Sequence)
            or isinstance(box_raw, (str, bytes))
            or len(box_raw) != 4
        ):
            _fail(f"objects[{index - 1}].bbox_xyxy_exclusive 형식이 잘못됐습니다.")
        box = tuple(
            _integer(value, f"objects[{index - 1}].bbox_xyxy_exclusive")
            for value in box_raw
        )
        left, top, right, bottom = box
        if left < 0 or top < 0 or right > width or bottom > height:
            _fail(f"objects[{index - 1}] 경계가 캔버스 밖입니다.")
        if right <= left or bottom <= top:
            _fail(f"objects[{index - 1}] 경계의 넓이 또는 높이가 0입니다.")
        expected_area = (right - left) * (bottom - top)
        if item.get("expected_pixel_count") != expected_area:
            _fail(f"objects[{index - 1}]의 예상 픽셀 수가 경계 면적과 다릅니다.")
        if any(_boxes_overlap(box, previous) for previous in boxes):
            _fail(f"objects[{index - 1}]의 픽셀이 다른 자극과 겹칩니다.")
        boxes.append(box)
        image.paste(_rgb(item.get("rgb"), f"objects[{index - 1}].rgb"), box)
    return image


def _pixel_counts(image: Image.Image) -> list[dict[str, Any]]:
    counts = Counter(tuple(pixel) for pixel in image.get_flattened_data())
    return [
        {"rgb": list(rgb), "count": count} for rgb, count in sorted(counts.items())
    ]


def _assert_image_matches_spec(image: Image.Image, spec: Mapping[str, Any]) -> str:
    if image.mode != "RGB":
        _fail(f"이미지 모드는 RGB여야 합니다: {image.mode}")
    expected = _image_from_spec(spec)
    if image.size != expected.size:
        _fail(f"이미지 크기가 픽셀 명세와 다릅니다: {image.size} != {expected.size}")
    if image.tobytes() != expected.tobytes():
        _fail("PNG를 해독한 RGB 픽셀이 픽셀 명세와 다릅니다.")
    actual_counts = _pixel_counts(image)
    expected_counts = _required(spec, "pixel_verification.expected_counts_by_rgb")
    if actual_counts != expected_counts:
        _fail("RGB별 픽셀 수가 픽셀 명세와 다릅니다.")
    return hashlib.sha256(image.tobytes()).hexdigest()


def _atomic_save_png(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        image.save(
            temporary,
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    rendered = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    try:
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def render_stimulus_png(
    config: Mapping[str, Any],
    trial: Mapping[str, Any],
    output_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """한 시행을 PNG로 저장하고 검증된 픽셀 메타데이터를 나란히 저장한다."""

    if PILLOW_VERSION != LOCKED_PILLOW_VERSION:
        _fail(
            f"Pillow 잠금 버전이 다릅니다: {PILLOW_VERSION} != {LOCKED_PILLOW_VERSION}"
        )
    output = Path(output_path)
    if output.suffix.lower() != ".png":
        _fail("output_path 확장자는 .png여야 합니다.")
    sidecar = (
        Path(metadata_path)
        if metadata_path is not None
        else output.with_name(output.stem + ".pixels.json")
    )
    if output.resolve() == sidecar.resolve():
        _fail("PNG와 픽셀 메타데이터는 서로 다른 경로여야 합니다.")
    spec = build_stimulus_spec(config, trial)
    image = _image_from_spec(spec)
    decoded_hash = _assert_image_matches_spec(image, spec)
    _atomic_save_png(image, output)

    try:
        with Image.open(output) as decoded:
            decoded.load()
            if decoded.format != "PNG":
                _fail("저장 파일이 PNG 형식이 아닙니다.")
            if decoded.info.get("icc_profile") is not None:
                _fail("렌더러 PNG에는 암묵적 색변환을 일으킬 ICC profile을 넣지 않습니다.")
            saved_hash = _assert_image_matches_spec(decoded, spec)
    except OSError as error:
        raise RendererValidationError(f"저장한 PNG를 다시 읽을 수 없습니다: {error}") from error
    if saved_hash != decoded_hash:
        _fail("PNG 저장 전후의 RGB 픽셀 해시가 다릅니다.")

    spec["pixel_verification"].update(
        {
            "status": "verified",
            "decoded_rgb_sha256": saved_hash,
            "png_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "unexpected_rgb_count": 0,
        }
    )
    _atomic_write_json(sidecar, spec)
    verify_stimulus_artifact(output, sidecar, config=config, trial=trial)
    return spec


def verify_stimulus_artifact(
    png_path: str | Path,
    metadata_path: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    trial: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """PNG와 sidecar를 다시 읽어 모든 픽셀·해시·선택적 입력 연결을 검증한다."""

    png = Path(png_path)
    sidecar = Path(metadata_path)
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RendererValidationError(f"픽셀 메타데이터를 읽을 수 없습니다: {error}") from error
    if not isinstance(metadata, dict):
        _fail("픽셀 메타데이터 최상위는 객체여야 합니다.")
    renderer = _mapping(_required(metadata, "renderer"), "renderer")
    expected_renderer = {
        "id": RENDERER_ID,
        "implementation_version": RENDERER_IMPLEMENTATION_VERSION,
        "implementation_sha256": renderer_implementation_sha256(),
        "locked_pillow_version": LOCKED_PILLOW_VERSION,
        "image_format": "PNG",
        "color_mode": "RGB",
        "lossless": True,
        "cap": CAP_POLICY,
        "antialiasing": ANTIALIASING_POLICY,
        "resampling": "none",
        "pixel_snapping": PIXEL_SNAPPING_POLICY,
        "raster_rule": RASTER_RULE,
    }
    for key, expected in expected_renderer.items():
        if renderer.get(key) != expected:
            _fail(f"renderer.{key}가 검증 정책과 다릅니다.")
    if renderer.get("pillow_version") != LOCKED_PILLOW_VERSION:
        _fail("메타데이터의 Pillow 버전이 잠금 버전과 다릅니다.")
    if renderer.get("pillow_lock_match") is not True:
        _fail("메타데이터가 Pillow 잠금 일치를 확인하지 않았습니다.")
    verification = _mapping(
        _required(metadata, "pixel_verification"), "pixel_verification"
    )
    if verification.get("status") != "verified":
        _fail("픽셀 메타데이터 상태가 verified가 아닙니다.")
    raw = png.read_bytes()
    if hashlib.sha256(raw).hexdigest() != verification.get("png_sha256"):
        _fail("PNG 파일 SHA-256이 메타데이터와 다릅니다.")
    try:
        with Image.open(png) as image:
            image.load()
            if image.format != "PNG":
                _fail("파일 형식이 PNG가 아닙니다.")
            if image.info.get("icc_profile") is not None:
                _fail("검증 대상 PNG에 ICC profile이 포함되어 있습니다.")
            decoded_hash = _assert_image_matches_spec(image, metadata)
    except OSError as error:
        raise RendererValidationError(f"PNG를 해독할 수 없습니다: {error}") from error
    if decoded_hash != verification.get("decoded_rgb_sha256"):
        _fail("해독 RGB SHA-256이 메타데이터와 다릅니다.")
    if verification.get("unexpected_rgb_count") != 0:
        _fail("unexpected_rgb_count는 0이어야 합니다.")

    if (config is None) != (trial is None):
        _fail("입력 연결을 확인하려면 config와 trial을 함께 넘겨야 합니다.")
    if config is not None and trial is not None:
        expected_spec = build_stimulus_spec(config, trial)
        for key in (
            "renderer",
            "config",
            "trial",
            "canvas",
            "geometry",
            "display_colors",
            "objects",
        ):
            if metadata.get(key) != expected_spec.get(key):
                _fail(f"메타데이터 {key}가 제공한 config/trial과 다릅니다.")
    return {
        "renderer_verified": True,
        "trial_id": _required(metadata, "trial.trial_id"),
        "png_sha256": verification["png_sha256"],
        "decoded_rgb_sha256": decoded_hash,
    }


def _canary_colors(background: tuple[int, int, int]) -> tuple[list[int], list[int]]:
    candidates = ((0, 0, 0), (255, 255, 255), (255, 0, 255))
    selected = [candidate for candidate in candidates if candidate != background][:2]
    return list(selected[0]), list(selected[1])


def verify_renderer_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    """잠금 의존성과 모든 계획 굵기의 픽셀 기하를 메모리에서 검증한다."""

    if PILLOW_VERSION != LOCKED_PILLOW_VERSION:
        _fail(
            f"Pillow 잠금 버전이 다릅니다: {PILLOW_VERSION} != {LOCKED_PILLOW_VERSION}"
        )
    background = _rgb(
        _required(config, "display.background_rgb"), "display.background_rgb"
    )
    rgb_a, rgb_b = _canary_colors(background)
    widths: list[int] = []
    calibration = _required(config, "calibration.widths")
    if not isinstance(calibration, Sequence) or isinstance(calibration, (str, bytes)):
        _fail("calibration.widths는 배열이어야 합니다.")
    for index, row in enumerate(calibration):
        widths.append(
            _integer(
                _mapping(row, f"calibration.widths[{index}]").get("width_px"),
                f"calibration.widths[{index}].width_px",
                minimum=1,
            )
        )
    validation = config.get("validation")
    if isinstance(validation, Mapping):
        candidates = validation.get("candidates", [])
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            _fail("validation.candidates는 배열이어야 합니다.")
        for index, row in enumerate(candidates):
            widths.append(
                _integer(
                    _mapping(row, f"validation.candidates[{index}]").get(
                        "rendered_width_px"
                    ),
                    f"validation.candidates[{index}].rendered_width_px",
                    minimum=1,
                )
            )
    verified_widths = sorted(set(widths))
    for trial_id, width in enumerate(verified_widths, start=1):
        trial = {
            "trial_id": trial_id,
            "stroke_width_px": width,
            "correct_position": 1,
            "odd_color": "b",
            "displayed_rgb_a": rgb_a,
            "displayed_rgb_b": rgb_b,
        }
        spec = build_stimulus_spec(config, trial)
        image = _image_from_spec(spec)
        _assert_image_matches_spec(image, spec)
    return {
        "renderer_verified": True,
        "renderer_id": RENDERER_ID,
        "implementation_version": RENDERER_IMPLEMENTATION_VERSION,
        "implementation_sha256": renderer_implementation_sha256(),
        "pillow_version": PILLOW_VERSION,
        "verified_stroke_widths_px": verified_widths,
        "human_response_collection_implemented": True,
    }


def _compile_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    color_design = _mapping(_required(config, "color_design"), "color_design")
    observer = _mapping(_required(config, "observer_profile"), "observer_profile")
    if (
        _required(config, "sampling.randomization")
        != CALIBRATION_RANDOMIZATION
    ):
        raise StimulusCompilationError(
            f"sampling.randomization은 {CALIBRATION_RANDOMIZATION}여야 합니다."
        )
    stimulus_family_id = color_design.get("stimulus_family_id")
    if not isinstance(stimulus_family_id, str) or not stimulus_family_id.strip():
        raise StimulusCompilationError(
            "color_design.stimulus_family_id가 구체 문자열이어야 합니다."
        )
    midpoint = color_design.get("midpoint_lab")
    direction = color_design.get("direction_lab")
    if midpoint is None or direction is None:
        raise StimulusCompilationError("midpoint_lab과 direction_lab이 필요합니다.")
    if color_design.get("symmetric_about_midpoint") is not True:
        raise StimulusCompilationError("symmetric_about_midpoint=true가 필요합니다.")
    if color_design.get("gamut_policy") != "reject":
        raise StimulusCompilationError("color_design.gamut_policy='reject'가 필요합니다.")
    if color_design.get("rgb_quantization") != "nearest_8bit":
        raise StimulusCompilationError(
            "color_design.rgb_quantization='nearest_8bit'가 필요합니다."
        )
    absolute_tolerance = _finite_number(
        color_design.get("target_absolute_tolerance_delta_e00"),
        "color_design.target_absolute_tolerance_delta_e00",
    )
    if absolute_tolerance <= 0:
        raise StimulusCompilationError(
            "color_design.target_absolute_tolerance_delta_e00은 0보다 커야 합니다."
        )
    relative_tolerance = _finite_number(
        color_design.get("target_relative_tolerance"),
        "color_design.target_relative_tolerance",
    )
    if not 0 <= relative_tolerance < 1:
        raise StimulusCompilationError(
            "color_design.target_relative_tolerance는 0 이상 1 미만이어야 합니다."
        )
    grid_steps = _integer(
        color_design.get("search_grid_steps"),
        "color_design.search_grid_steps",
        minimum=32,
    )
    participant_id = observer.get("participant_id")
    if not isinstance(participant_id, str) or not participant_id.strip():
        raise StimulusCompilationError("observer_profile.participant_id가 필요합니다.")
    cvd_type = observer.get("cvd_type")
    if cvd_type not in {"protan", "deutan"}:
        raise StimulusCompilationError(
            "observer_profile.cvd_type은 protan 또는 deutan이어야 합니다."
        )
    severity = _finite_number(observer.get("severity"), "observer_profile.severity")
    if not 0.0 <= severity <= 1.0:
        raise StimulusCompilationError("observer_profile.severity는 0~1이어야 합니다.")
    observer_strings: dict[str, str] = {}
    for field in (
        "severity_source",
        "severity_session_id",
        "lapse_source",
        "lapse_session_id",
    ):
        value = observer.get(field)
        if not isinstance(value, str) or not value.strip():
            raise StimulusCompilationError(f"observer_profile.{field}가 필요합니다.")
        observer_strings[field] = value
    fixed_lapse = _finite_number(
        observer.get("fixed_lapse"), "observer_profile.fixed_lapse"
    )
    if not 0 <= fixed_lapse < 0.20:
        raise StimulusCompilationError(
            "observer_profile.fixed_lapse는 0 이상 0.20 미만이어야 합니다."
        )
    return {
        "stimulus_family_id": stimulus_family_id,
        "midpoint_lab": midpoint,
        "direction_lab": direction,
        "absolute_target_tolerance": absolute_tolerance,
        "relative_target_tolerance": relative_tolerance,
        "grid_steps": grid_steps,
        "participant_id": participant_id,
        "cvd_type": cvd_type,
        "severity": severity,
        "fixed_lapse": fixed_lapse,
        **observer_strings,
    }


def compile_stimulus_manifest(
    config: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path | None = None,
    development_allow_partial_schedule: bool = False,
) -> dict[str, Any]:
    """목표 ΔE 시행표를 실제 8-bit 색쌍·픽셀 명세·선택적 PNG로 컴파일한다.

    같은 목표 색차는 한 번만 생성한다. 어느 목표라도 허용 오차, 색역 또는
    픽셀 정책을 만족하지 못하면 부분 manifest를 반환하지 않고 실패한다.
    """

    if not isinstance(development_allow_partial_schedule, bool):
        raise StimulusCompilationError(
            "development_allow_partial_schedule은 명시적인 boolean이어야 합니다."
        )
    if not isinstance(schedule, Sequence) or isinstance(schedule, (str, bytes)):
        raise StimulusCompilationError("schedule은 시행 객체 배열이어야 합니다.")
    if not schedule:
        raise StimulusCompilationError("schedule이 비었습니다.")
    try:
        settings = _compile_settings(config)
        renderer_gate = verify_renderer_environment(config)
    except StimulusCompilationError:
        raise
    except RendererValidationError as error:
        raise StimulusCompilationError(str(error)) from error
    rows = [_mapping(row, f"schedule[{index}]") for index, row in enumerate(schedule)]

    seen_trials: set[int] = set()
    targets: set[float] = set()
    schedule_hashes: set[str] = set()
    git_commits: set[str] = set()
    for index, row in enumerate(rows):
        trial_id = _integer(row.get("trial_id"), f"schedule[{index}].trial_id", minimum=1)
        if trial_id in seen_trials:
            raise StimulusCompilationError(f"schedule trial_id가 중복됩니다: {trial_id}")
        seen_trials.add(trial_id)
        target = _finite_number(
            row.get("target_delta_e00_simulated"),
            f"schedule[{index}].target_delta_e00_simulated",
        )
        if target <= 0:
            raise StimulusCompilationError("목표 시뮬레이션 ΔE00은 0보다 커야 합니다.")
        targets.add(target)
        session_block = _integer(
            row.get("session_block"), f"schedule[{index}].session_block", minimum=1
        )
        calibration_block_count = block_count_from_config(config)
        if session_block > calibration_block_count:
            raise StimulusCompilationError(
                f"schedule[{index}].session_block은 1~{calibration_block_count}여야 합니다."
            )
        try:
            _position_labels(row)
            _renderer_geometry(config, row.get("stroke_width_px"))
        except RendererValidationError as error:
            raise StimulusCompilationError(
                f"schedule[{index}]의 픽셀 시행값이 잘못됐습니다: {error}"
            ) from error
        for field, expected in (
            ("config_version", config.get("config_version")),
            ("participant_id", settings["participant_id"]),
            ("stimulus_family_id", settings["stimulus_family_id"]),
            ("cvd_type", settings["cvd_type"]),
            ("machado_severity", settings["severity"]),
            ("severity_source", settings["severity_source"]),
            ("severity_session_id", settings["severity_session_id"]),
            ("fixed_lapse", settings["fixed_lapse"]),
            ("lapse_source", settings["lapse_source"]),
            ("lapse_session_id", settings["lapse_session_id"]),
        ):
            if row.get(field) != expected:
                raise StimulusCompilationError(
                    f"schedule[{index}].{field}가 동결 config와 다릅니다."
                )
        if not isinstance(row.get("config_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", row["config_sha256"]
        ):
            raise StimulusCompilationError(
                f"schedule[{index}].config_sha256 형식이 잘못됐습니다."
            )
        if not isinstance(row.get("git_commit"), str) or not re.fullmatch(
            r"[0-9a-f]{40}", row["git_commit"]
        ):
            raise StimulusCompilationError(
                f"schedule[{index}].git_commit 형식이 잘못됐습니다."
            )
        schedule_hashes.add(row["config_sha256"])
        git_commits.add(row["git_commit"])
    if len(schedule_hashes) != 1:
        raise StimulusCompilationError("schedule 안에서 config_sha256이 바뀌었습니다.")
    if len(git_commits) != 1:
        raise StimulusCompilationError("schedule 안에서 git_commit이 바뀌었습니다.")

    configured_levels_raw = _required(config, "sampling.delta_e00_levels")
    if (
        not isinstance(configured_levels_raw, Sequence)
        or isinstance(configured_levels_raw, (str, bytes))
        or len(configured_levels_raw) != 12
    ):
        raise StimulusCompilationError(
            "sampling.delta_e00_levels에는 동결된 12수준이 필요합니다."
        )
    configured_levels = [
        _finite_number(value, f"sampling.delta_e00_levels[{index}]")
        for index, value in enumerate(configured_levels_raw)
    ]
    if (
        any(value <= 0 for value in configured_levels)
        or len(set(configured_levels)) != 12
        or configured_levels != sorted(configured_levels)
    ):
        raise StimulusCompilationError(
            "sampling.delta_e00_levels는 서로 다른 양의 오름차순 12수준이어야 합니다."
        )
    configured_set = set(configured_levels)
    if not targets <= configured_set:
        raise StimulusCompilationError(
            "schedule에 config의 sampling.delta_e00_levels 밖 목표가 있습니다."
        )
    partial_schedule = targets != configured_set
    if partial_schedule and not development_allow_partial_schedule:
        missing = sorted(configured_set - targets)
        raise StimulusCompilationError(
            "전체 자극 컴파일에는 동결된 12 target가 모두 필요합니다. "
            f"누락={missing}"
        )
    if not partial_schedule:
        calibration_rows = _required(config, "calibration.widths")
        if (
            not isinstance(calibration_rows, Sequence)
            or isinstance(calibration_rows, (str, bytes))
            or len(calibration_rows) != 3
        ):
            raise StimulusCompilationError(
                "전체 calibration 컴파일에는 굵기 세 개가 필요합니다."
            )
        configured_widths = {
            _integer(
                _mapping(item, f"calibration.widths[{index}]").get("width_px"),
                f"calibration.widths[{index}].width_px",
                minimum=1,
            )
            for index, item in enumerate(calibration_rows)
        }
        if len(configured_widths) != 3 or len(rows) != CALIBRATION_TOTAL_TRIALS:
            raise StimulusCompilationError(
                "전체 calibration schedule은 3굵기, 12수준, 총 384시행이어야 합니다."
            )
        if seen_trials != set(range(1, CALIBRATION_TOTAL_TRIALS + 1)):
            raise StimulusCompilationError(
                "전체 calibration trial_id는 정확히 1~384여야 합니다."
            )
        combinations = Counter(
            (
                int(row["stroke_width_px"]),
                float(row["target_delta_e00_simulated"]),
                int(row["correct_position"]),
                row["odd_color"],
            )
            for row in rows
        )
        width_rows = list(_required(config, "calibration.widths"))
        expected_repeats = {
            int(width_rows[index]["width_px"]): count // 96
            for index, count in enumerate(trial_counts_from_config(config))
        }
        if (
            len(combinations) != 288
            or {key[0] for key in combinations} != configured_widths
            or any(
                count != expected_repeats[width_px]
                for (width_px, _level, _position, _odd), count in combinations.items()
            )
        ):
            raise StimulusCompilationError(
                "전체 schedule의 굵기×수준×위치×odd색 조합이 완전 균형이 아닙니다."
            )
        block_counts = Counter(int(row["session_block"]) for row in rows)
        block_width_counts = Counter(
            (int(row["session_block"]), int(row["stroke_width_px"]))
            for row in rows
        )
        block_count = block_count_from_config(config)
        expected_block = block_trial_count(config)
        expected_width_block = width_block_trial_counts(config)
        expected_block_width_counts = Counter(
            {
                (block, int(width_rows[index]["width_px"])): expected_width_block[index]
                for block in range(1, block_count + 1)
                for index in range(3)
            }
        )
        if (
            block_counts != Counter(
                {block: expected_block for block in range(1, block_count + 1)}
            )
            or block_width_counts != expected_block_width_counts
        ):
            raise StimulusCompilationError(
                "전체 schedule이 동결한 384시행 세션 블록 균형을 만족하지 않습니다."
            )

    generated: dict[float, dict[str, Any]] = {}
    allowed_errors: dict[float, float] = {}
    for target in sorted(targets):
        allowed_error = max(
            settings["absolute_target_tolerance"],
            settings["relative_target_tolerance"] * target,
        )
        allowed_errors[target] = allowed_error
        try:
            pair = generate_symmetric_pair(
                settings["midpoint_lab"],
                settings["direction_lab"],
                target,
                settings["cvd_type"],
                settings["severity"],
                max_target_error=allowed_error,
                grid_steps=settings["grid_steps"],
            )
        except (TypeError, ValueError) as error:
            raise StimulusCompilationError(
                f"목표 ΔE00 {target:g} 자극을 만들지 못했습니다: {error}"
            ) from error
        try:
            actual_delta = _finite_number(
                pair["simulated_delta_e00"],
                f"목표 {target:g}의 실제 simulated_delta_e00",
            )
            reported_error = _finite_number(
                pair["target_error"], f"목표 {target:g}의 target_error"
            )
        except (KeyError, RendererValidationError) as error:
            raise StimulusCompilationError(
                f"목표 ΔE00 {target:g} 생성 결과가 불완전합니다: {error}"
            ) from error
        calculated_error = abs(actual_delta - target)
        if not math.isclose(
            reported_error, calculated_error, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise StimulusCompilationError(
                f"목표 ΔE00 {target:g}의 보고 오차가 실제 표시 색차와 다릅니다."
            )
        if reported_error > allowed_error:
            raise StimulusCompilationError(
                f"목표 ΔE00 {target:g}의 실제 오차가 허용값을 넘습니다."
            )
        generated[target] = pair

    previous_actual: float | None = None
    seen_rgb_pairs: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    for target in sorted(generated):
        pair = generated[target]
        try:
            rgb_pair = tuple(
                sorted(
                    (
                        _rgb(pair["rgb_a"], f"목표 {target:g}.rgb_a"),
                        _rgb(pair["rgb_b"], f"목표 {target:g}.rgb_b"),
                    )
                )
            )
        except (KeyError, RendererValidationError) as error:
            raise StimulusCompilationError(
                f"목표 ΔE00 {target:g}의 RGB 생성 결과가 불완전합니다: {error}"
            ) from error
        if rgb_pair in seen_rgb_pairs:
            raise StimulusCompilationError(
                "서로 다른 목표 ΔE00가 같은 8-bit RGB 자극 쌍으로 양자화됐습니다: "
                f"target={target:g}, rgb_pair={rgb_pair}"
            )
        seen_rgb_pairs.add(rgb_pair)
        actual = float(pair["simulated_delta_e00"])
        if previous_actual is not None and actual <= previous_actual:
            raise StimulusCompilationError(
                "목표 ΔE00 오름차순에서 실제 8-bit 표시 색차가 엄격히 증가하지 "
                f"않습니다: target={target:g}, actual={actual:g}, previous={previous_actual:g}"
            )
        previous_actual = actual

    raster_dir = Path(output_dir) if output_dir is not None else None
    compiled_trials: list[dict[str, Any]] = []
    for row in rows:
        target = float(row["target_delta_e00_simulated"])
        pair = generated[target]
        position_colors = ["b" if row["odd_color"] == "a" else "a"] * 4
        position_colors[int(row["correct_position"]) - 1] = row["odd_color"]
        enriched = {
            **dict(row),
            "position_colors": position_colors,
            "displayed_rgb_a": list(pair["rgb_a"]),
            "displayed_rgb_b": list(pair["rgb_b"]),
            "displayed_lab_a": list(pair["displayed_lab_a"]),
            "displayed_lab_b": list(pair["displayed_lab_b"]),
            "simulated_rgb_a": list(pair["simulated_rgb_a"]),
            "simulated_rgb_b": list(pair["simulated_rgb_b"]),
            "simulated_lab_a": list(pair["simulated_lab_a"]),
            "simulated_lab_b": list(pair["simulated_lab_b"]),
            "delta_e00_original": pair["original_delta_e00"],
            "delta_e00_simulated": pair["simulated_delta_e00"],
            "target_error_delta_e00": pair["target_error"],
            "allowed_target_error_delta_e00": allowed_errors[target],
        }
        spec = build_stimulus_spec(config, enriched)
        raster_artifact: dict[str, Any] | None = None
        if raster_dir is not None:
            filename = f"trial-{int(row['trial_id']):04d}.png"
            png_path = raster_dir / filename
            metadata_path = raster_dir / f"trial-{int(row['trial_id']):04d}.pixels.json"
            metadata = render_stimulus_png(
                config, enriched, png_path, metadata_path=metadata_path
            )
            raster_artifact = {
                "png": filename,
                "pixel_metadata": metadata_path.name,
                "png_sha256": metadata["pixel_verification"]["png_sha256"],
                "decoded_rgb_sha256": metadata["pixel_verification"][
                    "decoded_rgb_sha256"
                ],
                "pixel_metadata_sha256": hashlib.sha256(
                    metadata_path.read_bytes()
                ).hexdigest(),
            }
        compiled_trials.append(
            {
                **enriched,
                "pixel_spec": {
                    "canvas": spec["canvas"],
                    "geometry": spec["geometry"],
                    "objects": spec["objects"],
                    "expected_counts_by_rgb": spec["pixel_verification"][
                        "expected_counts_by_rgb"
                    ],
                },
                "raster_artifact": raster_artifact,
            }
        )

    color_pairs = []
    for target in sorted(generated):
        pair = generated[target]
        color_pairs.append(
            {
                "target_delta_e00_simulated": target,
                "target_error_delta_e00": pair["target_error"],
                "allowed_target_error_delta_e00": allowed_errors[target],
                "displayed_rgb_a": list(pair["rgb_a"]),
                "displayed_rgb_b": list(pair["rgb_b"]),
                "displayed_lab_a": list(pair["displayed_lab_a"]),
                "displayed_lab_b": list(pair["displayed_lab_b"]),
                "simulated_rgb_a": list(pair["simulated_rgb_a"]),
                "simulated_rgb_b": list(pair["simulated_rgb_b"]),
                "simulated_lab_a": list(pair["simulated_lab_a"]),
                "simulated_lab_b": list(pair["simulated_lab_b"]),
                "delta_e00_original": pair["original_delta_e00"],
                "delta_e00_simulated": pair["simulated_delta_e00"],
                "generation_metadata": pair["metadata"],
            }
        )
    return {
        "schema_version": "1.0",
        "artifact_type": "compiled_pixel_stimulus_manifest",
        "config_version": config.get("config_version"),
        "config_canonical_sha256": _canonical_sha256(config),
        "schedule_config_sha256": next(iter(schedule_hashes)),
        "git_commit": next(iter(git_commits)),
        "stimulus_family_id": settings["stimulus_family_id"],
        "participant_id": settings["participant_id"],
        "cvd_type": settings["cvd_type"],
        "machado_severity": settings["severity"],
        "renderer": renderer_gate,
        "human_response_collection_implemented": True,
        "human_data_collection_allowed": False,
        "blocker": "실제 수집에는 response_gui의 런타임·동의 검증이 별도로 필요합니다.",
        "color_generation": {
            "gamut_policy": "reject",
            "rgb_quantization": "nearest_8bit",
            "target_absolute_tolerance_delta_e00": settings[
                "absolute_target_tolerance"
            ],
            "target_relative_tolerance": settings["relative_target_tolerance"],
            "allowed_error_rule": "max(absolute_tolerance, relative_tolerance * target)",
            "search_grid_steps": settings["grid_steps"],
            "unique_target_count": len(color_pairs),
            "configured_target_count": len(configured_levels),
            "development_partial_schedule": partial_schedule,
        },
        "color_pairs": color_pairs,
        "trial_count": len(compiled_trials),
        "png_files_written": raster_dir is not None,
        "trials": compiled_trials,
    }


def _read_compilation_input(
    path: str | Path, label: str
) -> tuple[dict[str, Any], str]:
    file_path = Path(path)
    try:
        raw = file_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StimulusCompilationError(
            f"{label} UTF-8 JSON을 읽을 수 없습니다: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise StimulusCompilationError(f"{label} JSON 최상위는 객체여야 합니다.")
    return payload, hashlib.sha256(raw).hexdigest()


def _sha256_field(value: Any, label: str, *, length: int = 64) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        rf"[0-9a-f]{{{length}}}", value
    ):
        raise StimulusCompilationError(
            f"{label}는 소문자 16진수 {length}자리여야 합니다."
        )
    return value


def _numeric_triplet(value: Any, label: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise StimulusCompilationError(f"{label}에는 유한한 숫자 세 개가 필요합니다.")
    try:
        result = tuple(_finite_number(item, label) for item in value)
    except RendererValidationError as error:
        raise StimulusCompilationError(str(error)) from error
    return result  # type: ignore[return-value]


def _require_close(actual: Any, expected: float, label: str) -> float:
    try:
        number = _finite_number(actual, label)
    except RendererValidationError as error:
        raise StimulusCompilationError(str(error)) from error
    if not math.isclose(number, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise StimulusCompilationError(
            f"{label}가 실제 8-bit RGB에서 재계산한 값과 다릅니다: "
            f"stored={number:.12g}, recomputed={expected:.12g}"
        )
    return number


def _require_triplet_close(value: Any, expected: Sequence[float], label: str) -> None:
    observed = _numeric_triplet(value, label)
    for index, (actual, wanted) in enumerate(zip(observed, expected, strict=True)):
        if not math.isclose(actual, float(wanted), rel_tol=1e-10, abs_tol=1e-10):
            raise StimulusCompilationError(
                f"{label}[{index}]가 실제 8-bit RGB 재계산값과 다릅니다."
            )


def _validated_color_pair(
    payload: Mapping[str, Any],
    *,
    target: float,
    settings: Mapping[str, Any],
    label: str,
    rgb_fields: tuple[str, str],
    actual_field: str,
    original_field: str,
    require_regenerated_pair: bool,
) -> dict[str, Any]:
    """저장 색쌍을 8-bit RGB에서 독립 재계산하고 tolerance를 검사한다."""

    try:
        rgb_a = _rgb(payload.get(rgb_fields[0]), f"{label}.{rgb_fields[0]}")
        rgb_b = _rgb(payload.get(rgb_fields[1]), f"{label}.{rgb_fields[1]}")
    except RendererValidationError as error:
        raise StimulusCompilationError(str(error)) from error
    if rgb_a == rgb_b:
        raise StimulusCompilationError(f"{label}의 두 8-bit RGB가 서로 같습니다.")

    allowed_error = max(
        float(settings["absolute_target_tolerance"]),
        float(settings["relative_target_tolerance"]) * target,
    )
    _require_close(
        payload.get("allowed_target_error_delta_e00"),
        allowed_error,
        f"{label}.allowed_target_error_delta_e00",
    )

    displayed_lab_a = tuple(float(value) for value in srgb_to_lab(rgb_a))
    displayed_lab_b = tuple(float(value) for value in srgb_to_lab(rgb_b))
    simulated_rgb_a = tuple(
        float(value)
        for value in simulate_color_float(
            rgb_a, str(settings["cvd_type"]), float(settings["severity"])
        )
    )
    simulated_rgb_b = tuple(
        float(value)
        for value in simulate_color_float(
            rgb_b, str(settings["cvd_type"]), float(settings["severity"])
        )
    )
    simulated_lab_a = tuple(float(value) for value in srgb_to_lab(simulated_rgb_a))
    simulated_lab_b = tuple(float(value) for value in srgb_to_lab(simulated_rgb_b))
    original_delta = float(delta_e_2000(displayed_lab_a, displayed_lab_b))
    simulated_delta = float(delta_e_2000(simulated_lab_a, simulated_lab_b))

    _require_triplet_close(payload.get("displayed_lab_a"), displayed_lab_a, f"{label}.displayed_lab_a")
    _require_triplet_close(payload.get("displayed_lab_b"), displayed_lab_b, f"{label}.displayed_lab_b")
    _require_triplet_close(payload.get("simulated_rgb_a"), simulated_rgb_a, f"{label}.simulated_rgb_a")
    _require_triplet_close(payload.get("simulated_rgb_b"), simulated_rgb_b, f"{label}.simulated_rgb_b")
    _require_triplet_close(payload.get("simulated_lab_a"), simulated_lab_a, f"{label}.simulated_lab_a")
    _require_triplet_close(payload.get("simulated_lab_b"), simulated_lab_b, f"{label}.simulated_lab_b")
    _require_close(payload.get(original_field), original_delta, f"{label}.{original_field}")
    _require_close(payload.get(actual_field), simulated_delta, f"{label}.{actual_field}")
    target_error = abs(simulated_delta - target)
    _require_close(
        payload.get("target_error_delta_e00"),
        target_error,
        f"{label}.target_error_delta_e00",
    )
    if target_error > allowed_error + 1e-10:
        raise StimulusCompilationError(
            f"{label}의 실제 8-bit RGB 목표 오차가 동결 tolerance를 넘습니다: "
            f"error={target_error:.12g}, allowed={allowed_error:.12g}"
        )

    if require_regenerated_pair:
        try:
            regenerated = generate_symmetric_pair(
                settings["midpoint_lab"],
                settings["direction_lab"],
                target,
                str(settings["cvd_type"]),
                float(settings["severity"]),
                max_target_error=allowed_error,
                grid_steps=int(settings["grid_steps"]),
            )
        except (TypeError, ValueError) as error:
            raise StimulusCompilationError(
                f"{label}의 동결 target를 현재 자극 생성 계약으로 재생성할 수 없습니다: {error}"
            ) from error
        regenerated_pair = (
            tuple(int(value) for value in regenerated["rgb_a"]),
            tuple(int(value) for value in regenerated["rgb_b"]),
        )
        if (rgb_a, rgb_b) != regenerated_pair:
            raise StimulusCompilationError(
                f"{label}의 RGB 쌍이 동결 config로 결정론적으로 재생성한 쌍과 다릅니다."
            )

    return {
        "target_delta_e00_simulated": target,
        "target_error_delta_e00": target_error,
        "allowed_target_error_delta_e00": allowed_error,
        "displayed_rgb_a": list(rgb_a),
        "displayed_rgb_b": list(rgb_b),
        "displayed_lab_a": list(displayed_lab_a),
        "displayed_lab_b": list(displayed_lab_b),
        "simulated_rgb_a": list(simulated_rgb_a),
        "simulated_rgb_b": list(simulated_rgb_b),
        "simulated_lab_a": list(simulated_lab_a),
        "simulated_lab_b": list(simulated_lab_b),
        "delta_e00_original": original_delta,
        "delta_e00_simulated": simulated_delta,
    }


def _pixel_compile_trial(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    raster_dir: Path,
    filename_prefix: str,
) -> dict[str, Any]:
    enriched = dict(row)
    position_colors = _position_labels(enriched)
    enriched["position_colors"] = position_colors
    spec = build_stimulus_spec(config, enriched)
    trial_id = int(enriched["trial_id"])
    stem = f"{filename_prefix}-trial-{trial_id:07d}"
    png_path = raster_dir / f"{stem}.png"
    metadata_path = raster_dir / f"{stem}.pixels.json"
    metadata = render_stimulus_png(
        config, enriched, png_path, metadata_path=metadata_path
    )
    verification = verify_stimulus_artifact(
        png_path, metadata_path, config=config, trial=enriched
    )
    return {
        **enriched,
        "pixel_spec": {
            "canvas": spec["canvas"],
            "geometry": spec["geometry"],
            "objects": spec["objects"],
            "expected_counts_by_rgb": spec["pixel_verification"][
                "expected_counts_by_rgb"
            ],
        },
        "raster_artifact": {
            "png": png_path.name,
            "pixel_metadata": metadata_path.name,
            "png_sha256": verification["png_sha256"],
            "decoded_rgb_sha256": verification["decoded_rgb_sha256"],
            "pixel_metadata_sha256": hashlib.sha256(
                metadata_path.read_bytes()
            ).hexdigest(),
        },
    }


def _assert_calibration_source_matches_config(
    calibration: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    settings: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source = _mapping(calibration.get("source"), "calibration.source")
    if source.get("config_sha256") != config_sha256:
        raise StimulusCompilationError(
            "calibration artifact의 원 config raw SHA-256이 입력 config와 다릅니다."
        )
    for field, expected in (
        ("config_version", config.get("config_version")),
        ("stimulus_family_id", settings["stimulus_family_id"]),
        ("renderer_id", RENDERER_ID),
    ):
        if source.get(field) != expected:
            raise StimulusCompilationError(
                f"calibration.source.{field}가 입력 config/렌더러와 다릅니다."
            )
    _sha256_field(source.get("trial_file_sha256"), "calibration.source.trial_file_sha256")
    _sha256_field(source.get("git_commit"), "calibration.source.git_commit", length=40)
    participants = calibration.get("participants")
    if not isinstance(participants, Sequence) or isinstance(participants, (str, bytes)) or len(participants) != 1:
        raise StimulusCompilationError("calibration artifact에는 참가자 한 명만 있어야 합니다.")
    participant = _mapping(participants[0], "calibration.participants[0]")
    for field, expected in (
        ("participant_id", settings["participant_id"]),
        ("cvd_type", settings["cvd_type"]),
        ("machado_severity", settings["severity"]),
        ("severity_source", settings["severity_source"]),
        ("severity_session_id", settings["severity_session_id"]),
        ("fixed_lapse", settings["fixed_lapse"]),
        ("lapse_source", settings["lapse_source"]),
        ("lapse_session_id", settings["lapse_session_id"]),
    ):
        if participant.get(field) != expected:
            raise StimulusCompilationError(
                f"calibration participant의 {field}가 입력 config와 다릅니다."
            )
    if participant.get("analysis_status") != "complete":
        raise StimulusCompilationError(
            "calibration 모델 적합이 complete가 아니어서 validation을 컴파일할 수 없습니다."
        )
    return source, participant


def compile_validation_stimulus_manifest(
    config_path: str | Path,
    calibration_fits_path: str | Path,
    validation_plan_path: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """동결 validation plan 3×64회를 검증된 PNG manifest로 컴파일한다.

    세 입력은 객체가 아니라 파일 경로로만 받는다. 그래야 plan에 동결된 원
    config와 calibration artifact의 *raw file* SHA-256을 재검사할 수 있다.
    """

    config, config_sha256 = _read_compilation_input(config_path, "config")
    calibration, calibration_sha256 = _read_compilation_input(
        calibration_fits_path, "calibration model fits"
    )
    plan, plan_raw_sha256 = _read_compilation_input(
        validation_plan_path, "validation plan"
    )
    try:
        settings = _compile_settings(config)
        renderer_gate = verify_renderer_environment(config)
    except RendererValidationError as error:
        raise StimulusCompilationError(str(error)) from error

    try:
        from analyze_calibration import validate_calibration_artifact
        from validation_design import compute_plan_sha256, validate_plan
    except ImportError as error:
        raise StimulusCompilationError(
            f"validation 컴파일 의존성을 불러올 수 없습니다: {error}"
        ) from error
    try:
        validate_calibration_artifact(calibration)
        validate_plan(plan)
        canonical_plan_sha256 = compute_plan_sha256(plan)
    except (TypeError, ValueError) as error:
        raise StimulusCompilationError(
            f"validation plan/calibration artifact 검증 실패: {error}"
        ) from error

    plan_sha256 = _sha256_field(plan.get("plan_sha256"), "validation plan.plan_sha256")
    if plan_sha256 != canonical_plan_sha256:
        raise StimulusCompilationError("validation plan canonical SHA-256이 다릅니다.")
    if plan.get("config_sha256") != config_sha256:
        raise StimulusCompilationError(
            "validation plan의 원 config raw SHA-256이 입력 config와 다릅니다."
        )
    if plan.get("calibration_model_fits_sha256") != calibration_sha256:
        raise StimulusCompilationError(
            "validation plan의 calibration artifact raw SHA-256이 입력 파일과 다릅니다."
        )
    source, _ = _assert_calibration_source_matches_config(
        calibration,
        config=config,
        config_sha256=config_sha256,
        settings=settings,
    )
    if plan.get("source_calibration_trials_sha256") != source.get(
        "trial_file_sha256"
    ):
        raise StimulusCompilationError(
            "validation plan의 source calibration trials SHA-256이 calibration artifact와 다릅니다."
        )
    if plan.get("participant_id") != settings["participant_id"]:
        raise StimulusCompilationError(
            "validation plan participant_id가 config와 다릅니다."
        )
    if plan.get("data_origin") != calibration.get("data_origin"):
        raise StimulusCompilationError(
            "validation plan과 calibration artifact의 data_origin이 다릅니다."
        )

    validation_config = _mapping(config.get("validation"), "config.validation")
    for field, expected in (
        ("independent_blocks", VALIDATION_BLOCK_COUNT),
        ("delta_e00_level_count", VALIDATION_LEVEL_COUNT),
        ("trials_per_block", VALIDATION_TRIALS_PER_BLOCK),
    ):
        if validation_config.get(field) != expected:
            raise StimulusCompilationError(
                f"config.validation.{field}는 {expected}이어야 합니다."
            )
    if not isinstance(validation_config.get("randomization"), str) or not validation_config[
        "randomization"
    ].strip():
        raise StimulusCompilationError("config.validation.randomization이 필요합니다.")

    selected_width = _mapping(plan.get("selected_width"), "validation plan.selected_width")
    selected_width_px = _integer(
        selected_width.get("stroke_width_px"),
        "validation plan.selected_width.stroke_width_px",
        minimum=1,
    )
    selected_width_deg = _finite_number(
        selected_width.get("stroke_width_deg"),
        "validation plan.selected_width.stroke_width_deg",
    )
    candidates = validation_config.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise StimulusCompilationError("config.validation.candidates가 배열이 아닙니다.")
    candidate_widths = {
        (
            _integer(
                _mapping(candidate, f"config.validation.candidates[{index}]").get(
                    "rendered_width_px"
                ),
                f"config.validation.candidates[{index}].rendered_width_px",
                minimum=1,
            ),
            _finite_number(
                _mapping(candidate, f"config.validation.candidates[{index}]").get(
                    "width_deg"
                ),
                f"config.validation.candidates[{index}].width_deg",
            ),
        )
        for index, candidate in enumerate(candidates)
    }
    if not any(
        width_px == selected_width_px
        and math.isclose(width_deg, selected_width_deg, rel_tol=1e-12, abs_tol=1e-12)
        for width_px, width_deg in candidate_widths
    ):
        raise StimulusCompilationError(
            "validation selected_width가 config의 동결 후보에 없습니다."
        )

    raw_levels = plan.get("levels")
    if not isinstance(raw_levels, Sequence) or isinstance(raw_levels, (str, bytes)) or len(raw_levels) != VALIDATION_LEVEL_COUNT:
        raise StimulusCompilationError("validation plan에는 정확히 8수준이 필요합니다.")
    levels: dict[int, dict[str, Any]] = {}
    seen_pairs: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    previous_target = -math.inf
    previous_actual = -math.inf
    for offset, raw_level in enumerate(raw_levels, start=1):
        level = _mapping(raw_level, f"validation plan.levels[{offset - 1}]")
        index = _integer(
            level.get("validation_level_index"),
            f"validation plan.levels[{offset - 1}].validation_level_index",
            minimum=1,
        )
        if index != offset:
            raise StimulusCompilationError(
                "validation level index는 1~8 오름차순으로 정확히 한 번씩 있어야 합니다."
            )
        target = _finite_number(
            level.get("target_delta_e00_simulated"),
            f"validation plan.levels[{offset - 1}].target_delta_e00_simulated",
        )
        if target <= previous_target:
            raise StimulusCompilationError("validation target ΔE00가 엄격히 증가하지 않습니다.")
        normalized = _validated_color_pair(
            level,
            target=target,
            settings=settings,
            label=f"validation plan.levels[{offset - 1}]",
            rgb_fields=("rgb_a", "rgb_b"),
            actual_field="actual_delta_e00_simulated",
            original_field="original_delta_e00",
            require_regenerated_pair=True,
        )
        pair_key = tuple(
            sorted(
                (
                    tuple(normalized["displayed_rgb_a"]),
                    tuple(normalized["displayed_rgb_b"]),
                )
            )
        )
        if pair_key in seen_pairs:
            raise StimulusCompilationError(
                "validation의 서로 다른 수준이 같은 무순서 8-bit RGB 쌍입니다."
            )
        actual = float(normalized["delta_e00_simulated"])
        if actual <= previous_actual:
            raise StimulusCompilationError(
                "validation target 오름차순에서 실제 simulated ΔE00가 엄격히 증가하지 않습니다."
            )
        seen_pairs.add(pair_key)
        previous_target = target
        previous_actual = actual
        levels[index] = {
            "validation_level_index": index,
            **normalized,
            "actual_delta_e00_simulated": normalized["delta_e00_simulated"],
        }

    schedule = plan.get("schedule")
    if not isinstance(schedule, Sequence) or isinstance(schedule, (str, bytes)) or len(schedule) != VALIDATION_BLOCK_COUNT * VALIDATION_TRIALS_PER_BLOCK:
        raise StimulusCompilationError("validation schedule은 정확히 3×64=192시행이어야 합니다.")
    selection = _mapping(
        plan.get("selection_specification"), "validation plan.selection_specification"
    )
    first_trial_id = _integer(
        selection.get("first_trial_id"),
        "validation plan.selection_specification.first_trial_id",
        minimum=1,
    )
    if first_trial_id != 385:
        raise StimulusCompilationError(
            "validation 전역 trial_id는 예약 범위 385~576을 사용해야 합니다."
        )
    base_seed = _integer(
        selection.get("base_seed"),
        "validation plan.selection_specification.base_seed",
        minimum=0,
    )
    expected_ids = set(range(385, 577))
    seen_ids: set[int] = set()
    seen_global_keys: set[str] = set()
    block_combinations: Counter[tuple[str, int, int, str]] = Counter()
    within_indices: Counter[tuple[str, int]] = Counter()
    checked_rows: list[dict[str, Any]] = []
    for offset, raw_row in enumerate(schedule):
        row = _mapping(raw_row, f"validation plan.schedule[{offset}]")
        trial_id = _integer(
            row.get("trial_id"), f"validation plan.schedule[{offset}].trial_id", minimum=1
        )
        if trial_id in seen_ids:
            raise StimulusCompilationError(f"validation trial_id가 중복됩니다: {trial_id}")
        seen_ids.add(trial_id)
        expected_key = f"{settings['participant_id']}:{config_sha256}:{trial_id}"
        if row.get("global_trial_key") != expected_key:
            raise StimulusCompilationError(
                f"validation trial {trial_id}의 global_trial_key가 participant/config/trial과 다릅니다."
            )
        if expected_key in seen_global_keys:
            raise StimulusCompilationError("validation global_trial_key가 중복됩니다.")
        seen_global_keys.add(expected_key)
        for field, expected in (
            ("participant_id", settings["participant_id"]),
            ("config_sha256", config_sha256),
            ("validation_plan_sha256", plan_sha256),
            ("phase", "validation"),
            ("stroke_width_px", selected_width_px),
        ):
            if row.get(field) != expected:
                raise StimulusCompilationError(
                    f"validation trial {trial_id}의 {field}가 동결 입력과 다릅니다."
                )
        if not math.isclose(
            _finite_number(row.get("stroke_width_deg"), "validation stroke_width_deg"),
            selected_width_deg,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise StimulusCompilationError(
                f"validation trial {trial_id}의 stroke_width_deg가 selected_width와 다릅니다."
            )
        level_index = _integer(
            row.get("validation_level_index"),
            f"validation trial {trial_id}.validation_level_index",
            minimum=1,
        )
        if level_index not in levels:
            raise StimulusCompilationError(
                f"validation trial {trial_id}가 정의되지 않은 수준을 참조합니다."
            )
        level = levels[level_index]
        _require_close(
            row.get("target_delta_e00_simulated"),
            float(level["target_delta_e00_simulated"]),
            f"validation trial {trial_id}.target_delta_e00_simulated",
        )
        _require_close(
            row.get("actual_delta_e00_simulated"),
            float(level["delta_e00_simulated"]),
            f"validation trial {trial_id}.actual_delta_e00_simulated",
        )
        block = row.get("validation_block_id")
        if block not in {"V01", "V02", "V03"}:
            raise StimulusCompilationError(
                f"validation trial {trial_id}의 validation_block_id가 잘못됐습니다."
            )
        within = _integer(
            row.get("within_block_index"),
            f"validation trial {trial_id}.within_block_index",
            minimum=1,
        )
        if within > VALIDATION_TRIALS_PER_BLOCK:
            raise StimulusCompilationError("validation within_block_index는 1~64여야 합니다.")
        expected_seed = base_seed + int(str(block)[1:]) - 1
        if row.get("random_seed") != expected_seed:
            raise StimulusCompilationError(
                f"validation trial {trial_id}의 block seed가 동결 계획과 다릅니다."
            )
        try:
            position_colors = _position_labels(row)
            _renderer_geometry(config, selected_width_px)
        except RendererValidationError as error:
            raise StimulusCompilationError(
                f"validation trial {trial_id} 픽셀 계약 오류: {error}"
            ) from error
        position = int(row["correct_position"])
        odd = str(row["odd_color"])
        block_combinations[(str(block), level_index, position, odd)] += 1
        within_indices[(str(block), within)] += 1
        checked_rows.append(
            {
                **dict(row),
                "config_version": config.get("config_version"),
                "git_commit": source["git_commit"],
                "stimulus_family_id": settings["stimulus_family_id"],
                "cvd_type": settings["cvd_type"],
                "machado_severity": settings["severity"],
                "severity_source": settings["severity_source"],
                "severity_session_id": settings["severity_session_id"],
                "fixed_lapse": settings["fixed_lapse"],
                "lapse_source": settings["lapse_source"],
                "lapse_session_id": settings["lapse_session_id"],
                "position_colors": position_colors,
                **{
                    key: value
                    for key, value in level.items()
                    if key != "validation_level_index"
                },
                "actual_delta_e00_simulated": level["delta_e00_simulated"],
            }
        )
    if seen_ids != expected_ids:
        raise StimulusCompilationError(
            "validation trial_id는 동결 first_trial_id부터 192개 연속 전역 ID여야 합니다."
        )
    expected_combinations = {
        (f"V{block:02d}", level, position, odd)
        for block in range(1, VALIDATION_BLOCK_COUNT + 1)
        for level in range(1, VALIDATION_LEVEL_COUNT + 1)
        for position in range(1, 5)
        for odd in ("a", "b")
    }
    if set(block_combinations) != expected_combinations or set(block_combinations.values()) != {1}:
        raise StimulusCompilationError(
            "validation 각 block은 8수준×4위치×2 odd 조합을 정확히 한 번씩 가져야 합니다."
        )
    expected_within = {
        (f"V{block:02d}", within)
        for block in range(1, VALIDATION_BLOCK_COUNT + 1)
        for within in range(1, VALIDATION_TRIALS_PER_BLOCK + 1)
    }
    if set(within_indices) != expected_within or set(within_indices.values()) != {1}:
        raise StimulusCompilationError(
            "validation 각 block의 within_block_index는 1~64를 정확히 한 번씩 가져야 합니다."
        )

    raster_dir = Path(output_dir)
    compiled_trials = [
        _pixel_compile_trial(
            config,
            row,
            raster_dir=raster_dir,
            filename_prefix="validation",
        )
        for row in checked_rows
    ]
    return {
        "schema_version": "1.0",
        "artifact_type": "compiled_validation_pixel_stimulus_manifest",
        "phase": "validation",
        "data_origin": plan.get("data_origin"),
        "config_version": config.get("config_version"),
        "config_canonical_sha256": _canonical_sha256(config),
        "schedule_config_sha256": config_sha256,
        "source_config_raw_sha256": config_sha256,
        "validation_plan_sha256": plan_sha256,
        "validation_plan_raw_file_sha256": plan_raw_sha256,
        "calibration_model_fits_sha256": calibration_sha256,
        "source_calibration_trials_sha256": source["trial_file_sha256"],
        "git_commit": source["git_commit"],
        "stimulus_family_id": settings["stimulus_family_id"],
        "participant_id": settings["participant_id"],
        "cvd_type": settings["cvd_type"],
        "machado_severity": settings["severity"],
        "renderer": renderer_gate,
        "selected_width": {
            "stroke_width_px": selected_width_px,
            "stroke_width_deg": selected_width_deg,
        },
        "validation_design": {
            "level_count": VALIDATION_LEVEL_COUNT,
            "independent_blocks": VALIDATION_BLOCK_COUNT,
            "trials_per_block": VALIDATION_TRIALS_PER_BLOCK,
            "balance": "8-levels-x-4-positions-x-2-odd-identities",
            "randomization": validation_config["randomization"],
            "first_trial_id": first_trial_id,
            "last_trial_id": first_trial_id + len(compiled_trials) - 1,
        },
        "color_generation": {
            "gamut_policy": "reject",
            "rgb_quantization": "nearest_8bit",
            "target_absolute_tolerance_delta_e00": settings[
                "absolute_target_tolerance"
            ],
            "target_relative_tolerance": settings["relative_target_tolerance"],
            "allowed_error_rule": "max(absolute_tolerance, relative_tolerance * target)",
            "search_grid_steps": settings["grid_steps"],
            "unique_target_count": VALIDATION_LEVEL_COUNT,
        },
        "color_pairs": [levels[index] for index in range(1, VALIDATION_LEVEL_COUNT + 1)],
        "trial_count": len(compiled_trials),
        "png_files_written": True,
        "human_response_collection_implemented": True,
        "human_data_collection_allowed": False,
        "blocker": "실제 수집 전 GUI·동의·화면·Git 런타임 gate를 별도로 통과해야 합니다.",
        "trials": compiled_trials,
    }


def _validate_calibration_manifest_for_practice(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    settings: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[int], str]:
    if manifest.get("schema_version") != "1.0" or manifest.get(
        "artifact_type"
    ) != "compiled_pixel_stimulus_manifest":
        raise StimulusCompilationError(
            "practice 입력은 전체 compiled calibration pixel manifest여야 합니다."
        )
    if manifest.get("config_canonical_sha256") != _canonical_sha256(config):
        raise StimulusCompilationError(
            "calibration manifest canonical config SHA-256이 입력 config와 다릅니다."
        )
    if manifest.get("schedule_config_sha256") != config_sha256:
        raise StimulusCompilationError(
            "calibration manifest raw config SHA-256이 입력 config와 다릅니다."
        )
    for field, expected in (
        ("config_version", config.get("config_version")),
        ("stimulus_family_id", settings["stimulus_family_id"]),
        ("participant_id", settings["participant_id"]),
        ("cvd_type", settings["cvd_type"]),
        ("machado_severity", settings["severity"]),
    ):
        if manifest.get(field) != expected:
            raise StimulusCompilationError(
                f"calibration manifest의 {field}가 입력 config와 다릅니다."
            )
    git_commit = _sha256_field(
        manifest.get("git_commit"), "calibration manifest.git_commit", length=40
    )
    renderer = _mapping(manifest.get("renderer"), "calibration manifest.renderer")
    current_renderer = verify_renderer_environment(config)
    for field in (
        "renderer_id",
        "implementation_version",
        "implementation_sha256",
        "pillow_version",
    ):
        if renderer.get(field) != current_renderer.get(field):
            raise StimulusCompilationError(
                f"calibration manifest.renderer.{field}가 현재 renderer와 다릅니다."
            )

    raw_trials = manifest.get("trials")
    if not isinstance(raw_trials, Sequence) or isinstance(raw_trials, (str, bytes)) or len(raw_trials) != CALIBRATION_TOTAL_TRIALS or manifest.get("trial_count") != CALIBRATION_TOTAL_TRIALS:
        raise StimulusCompilationError("practice 입력 calibration manifest는 전체 384시행이어야 합니다.")
    trial_ids: set[int] = set()
    combinations: Counter[tuple[int, float, int, str]] = Counter()
    block_counts: Counter[int] = Counter()
    block_cells: Counter[tuple[int, int, float]] = Counter()
    for offset, raw_trial in enumerate(raw_trials):
        trial = _mapping(raw_trial, f"calibration manifest.trials[{offset}]")
        trial_id = _integer(trial.get("trial_id"), "calibration trial_id", minimum=1)
        if trial_id in trial_ids:
            raise StimulusCompilationError("calibration manifest trial_id가 중복됩니다.")
        trial_ids.add(trial_id)
        if trial.get("phase", "calibration") != "calibration":
            raise StimulusCompilationError("calibration manifest에 다른 phase가 섞였습니다.")
        block = _integer(trial.get("session_block"), "calibration session_block", minimum=1)
        level_target = _finite_number(
            trial.get("target_delta_e00_simulated"), "calibration target ΔE00"
        )
        position = _integer(trial.get("correct_position"), "calibration correct_position", minimum=1)
        odd = trial.get("odd_color")
        if position > 4 or odd not in {"a", "b"}:
            raise StimulusCompilationError("calibration 위치/odd identity가 잘못됐습니다.")
        width = _integer(trial.get("stroke_width_px"), "calibration width", minimum=1)
        combinations[(width, level_target, position, str(odd))] += 1
        block_counts[block] += 1
        block_cells[(block, width, level_target)] += 1
    calibration_block_count = block_count_from_config(config)
    expected_block_trials = block_trial_count(config)
    if len(trial_ids) != CALIBRATION_TOTAL_TRIALS or block_counts != Counter(
        {block: expected_block_trials for block in range(1, calibration_block_count + 1)}
    ):
        raise StimulusCompilationError("calibration manifest의 384시행 block 구성이 잘못됐습니다.")
    if trial_ids != set(range(1, CALIBRATION_TOTAL_TRIALS + 1)):
        raise StimulusCompilationError("calibration manifest trial_id는 정확히 1~384여야 합니다.")
    configured_width_rows = list(_required(config, "calibration.widths"))
    expected_repeats = {
        int(configured_width_rows[index]["width_px"]): count // 96
        for index, count in enumerate(trial_counts_from_config(config))
    }
    if len(combinations) != 288 or any(
        count != expected_repeats[width]
        for (width, _target, _position, _odd), count in combinations.items()
    ):
        raise StimulusCompilationError(
            "calibration manifest의 굵기×수준×위치×odd 완전균형이 깨졌습니다."
        )

    configured_targets = [
        _finite_number(value, f"sampling.delta_e00_levels[{index}]")
        for index, value in enumerate(_required(config, "sampling.delta_e00_levels"))
    ]
    configured_widths = {
        _integer(
            _mapping(row, f"calibration.widths[{index}]").get("width_px"),
            f"calibration.widths[{index}].width_px",
            minimum=1,
        )
        for index, row in enumerate(_required(config, "calibration.widths"))
    }
    expected_combinations = {
        (width, target, position, odd)
        for width in configured_widths
        for target in configured_targets
        for position in range(1, 5)
        for odd in ("a", "b")
    }
    if set(combinations) != expected_combinations:
        raise StimulusCompilationError(
            "calibration manifest가 config의 3굵기×12수준 조합과 다릅니다."
        )
    expected_width_block = width_block_trial_counts(config)
    observed_width_block = Counter(
        (block, width)
        for (block, width, _target), count in block_cells.items()
        for _ in range(count)
    )
    expected_width_block_counter = Counter(
        {
            (block, int(configured_width_rows[index]["width_px"])): expected_width_block[index]
            for block in range(1, calibration_block_count + 1)
            for index in range(3)
        }
    )
    if observed_width_block != expected_width_block_counter:
        raise StimulusCompilationError(
            "calibration manifest의 block별 굵기 시행 수가 동결 설계와 다릅니다."
        )
    raw_pairs = manifest.get("color_pairs")
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)) or len(raw_pairs) != 12:
        raise StimulusCompilationError("calibration manifest에는 12개 color pair가 필요합니다.")
    normalized_pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    previous_actual = -math.inf
    for offset, raw_pair in enumerate(raw_pairs, start=1):
        pair = _mapping(raw_pair, f"calibration manifest.color_pairs[{offset - 1}]")
        target = _finite_number(
            pair.get("target_delta_e00_simulated"),
            f"calibration manifest.color_pairs[{offset - 1}].target",
        )
        if offset > len(configured_targets) or not math.isclose(
            target, configured_targets[offset - 1], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise StimulusCompilationError(
                "calibration manifest color pair 순서/target가 config의 12수준과 다릅니다."
            )
        normalized = _validated_color_pair(
            pair,
            target=target,
            settings=settings,
            label=f"calibration manifest.color_pairs[{offset - 1}]",
            rgb_fields=("displayed_rgb_a", "displayed_rgb_b"),
            actual_field="delta_e00_simulated",
            original_field="delta_e00_original",
            require_regenerated_pair=True,
        )
        key = tuple(
            sorted(
                (
                    tuple(normalized["displayed_rgb_a"]),
                    tuple(normalized["displayed_rgb_b"]),
                )
            )
        )
        if key in seen_pairs:
            raise StimulusCompilationError(
                "calibration manifest의 서로 다른 수준이 같은 무순서 RGB 쌍입니다."
            )
        actual = float(normalized["delta_e00_simulated"])
        if actual <= previous_actual:
            raise StimulusCompilationError(
                "calibration manifest의 실제 simulated ΔE00가 엄격히 증가하지 않습니다."
            )
        seen_pairs.add(key)
        previous_actual = actual
        normalized_pairs.append(
            {"calibration_level_index": offset, **normalized}
        )
    return normalized_pairs, trial_ids, git_commit


def validate_calibration_manifest_structure(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    """GUI 실행 전에 전체 calibration 구조와 config 결합을 검증한다."""

    try:
        settings = _compile_settings(config)
        pairs, trial_ids, git_commit = _validate_calibration_manifest_for_practice(
            manifest,
            config=config,
            config_sha256=config_sha256,
            settings=settings,
        )
    except RendererValidationError as error:
        raise StimulusCompilationError(str(error)) from error

    expected_generation = {
        "gamut_policy": "reject",
        "rgb_quantization": "nearest_8bit",
        "target_absolute_tolerance_delta_e00": settings[
            "absolute_target_tolerance"
        ],
        "target_relative_tolerance": settings["relative_target_tolerance"],
        "allowed_error_rule": "max(absolute_tolerance, relative_tolerance * target)",
        "search_grid_steps": settings["grid_steps"],
        "unique_target_count": 12,
        "configured_target_count": 12,
        "development_partial_schedule": False,
    }
    generation = _mapping(
        manifest.get("color_generation"), "calibration manifest.color_generation"
    )
    for field, expected in expected_generation.items():
        if generation.get(field) != expected:
            raise StimulusCompilationError(
                f"calibration manifest.color_generation.{field}가 config와 다릅니다."
            )

    width_rows = list(_required(config, "calibration.widths"))
    width_by_px = {
        _integer(
            _mapping(row, "calibration width").get("width_px"),
            "calibration width_px",
            minimum=1,
        ): (
            index,
            _finite_number(
                _mapping(row, "calibration width").get("width_deg"),
                "calibration width_deg",
            ),
        )
        for index, row in enumerate(width_rows, start=1)
    }
    targets = [
        _finite_number(value, f"sampling.delta_e00_levels[{index}]")
        for index, value in enumerate(_required(config, "sampling.delta_e00_levels"))
    ]
    expected_trial_fields = {
        "config_version": config.get("config_version"),
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "stimulus_family_id": settings["stimulus_family_id"],
        "participant_id": settings["participant_id"],
        "cvd_type": settings["cvd_type"],
        "machado_severity": settings["severity"],
        "severity_source": settings["severity_source"],
        "severity_session_id": settings["severity_session_id"],
        "fixed_lapse": settings["fixed_lapse"],
        "lapse_source": settings["lapse_source"],
        "lapse_session_id": settings["lapse_session_id"],
    }
    for offset, raw_trial in enumerate(manifest["trials"]):
        trial = _mapping(raw_trial, f"calibration manifest.trials[{offset}]")
        trial_id = _integer(
            trial.get("trial_id"), "calibration trial_id", minimum=1
        )
        for field, expected in expected_trial_fields.items():
            if trial.get(field) != expected:
                raise StimulusCompilationError(
                    f"calibration trial {trial_id}.{field}가 config/manifest와 다릅니다."
                )
        width_px = _integer(
            trial.get("stroke_width_px"), "calibration width", minimum=1
        )
        if width_px not in width_by_px:
            raise StimulusCompilationError(
                f"calibration trial {trial_id}에 config 밖 굵기가 있습니다."
            )
        width_index, width_deg = width_by_px[width_px]
        if trial.get("width_index") != width_index or not math.isclose(
            _finite_number(trial.get("stroke_width_deg"), "calibration width_deg"),
            width_deg,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise StimulusCompilationError(
                f"calibration trial {trial_id}의 width index/시각도가 config와 다릅니다."
            )
        level_index = _integer(
            trial.get("delta_e00_level_index"),
            "calibration level index",
            minimum=1,
        )
        if level_index > len(targets) or not math.isclose(
            _finite_number(
                trial.get("target_delta_e00_simulated"), "calibration target"
            ),
            targets[level_index - 1],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise StimulusCompilationError(
                f"calibration trial {trial_id}의 level index/target이 config와 다릅니다."
            )
        try:
            _position_labels(trial)
        except RendererValidationError as error:
            raise StimulusCompilationError(
                f"calibration trial {trial_id}의 위치/odd 구성이 잘못됐습니다: {error}"
            ) from error
    return {
        "normalized_color_pairs": pairs,
        "trial_ids": trial_ids,
        "git_commit": git_commit,
    }


def validate_practice_manifest_structure(
    manifest: Mapping[str, Any],
    *,
    calibration_manifest: Mapping[str, Any],
    calibration_manifest_sha256: str,
    config: Mapping[str, Any],
    config_sha256: str,
) -> None:
    """GUI 실행 전에 practice 출처·결정 규칙·균형을 검증한다."""

    calibration = validate_calibration_manifest_structure(
        calibration_manifest, config=config, config_sha256=config_sha256
    )
    settings = _compile_settings(config)
    practice_trials = _integer(
        _required(config, "sampling.practice_trials"),
        "sampling.practice_trials",
        minimum=0,
    )
    if not 6 <= practice_trials <= 10:
        raise StimulusCompilationError("practice_trials는 6~10이어야 합니다.")
    expected_headers = {
        "schema_version": "1.0",
        "artifact_type": "compiled_practice_pixel_stimulus_manifest",
        "phase": "practice",
        "config_version": config.get("config_version"),
        "config_canonical_sha256": _canonical_sha256(config),
        "schedule_config_sha256": config_sha256,
        "source_config_raw_sha256": config_sha256,
        "source_calibration_manifest_sha256": calibration_manifest_sha256,
        "git_commit": calibration["git_commit"],
        "stimulus_family_id": settings["stimulus_family_id"],
        "participant_id": settings["participant_id"],
        "cvd_type": settings["cvd_type"],
        "machado_severity": settings["severity"],
        "trial_count": practice_trials,
        "png_files_written": True,
    }
    for field, expected in expected_headers.items():
        if manifest.get(field) != expected:
            raise StimulusCompilationError(
                f"practice manifest.{field}가 동결 config/calibration과 다릅니다."
            )
    renderer = _mapping(manifest.get("renderer"), "practice manifest.renderer")
    current_renderer = verify_renderer_environment(config)
    for field in (
        "renderer_id",
        "implementation_version",
        "implementation_sha256",
        "pillow_version",
    ):
        if renderer.get(field) != current_renderer.get(field):
            raise StimulusCompilationError(
                f"practice manifest.renderer.{field}가 현재 renderer와 다릅니다."
            )

    design = _mapping(
        manifest.get("practice_design"), "practice manifest.practice_design"
    )
    expected_width = sorted(
        (
            _integer(
                _mapping(row, "calibration width").get("width_px"),
                "width_px",
                minimum=1,
            ),
            _finite_number(
                _mapping(row, "calibration width").get("width_deg"), "width_deg"
            ),
        )
        for row in _required(config, "calibration.widths")
    )[1]
    expected_seed = _practice_seed(config_sha256, calibration["git_commit"])
    expected_design = {
        "policy_id": PRACTICE_POLICY_ID,
        "policy_version": "1",
        "representative_calibration_level_indices": list(PRACTICE_LEVEL_INDICES),
        "derived_seed": expected_seed,
        "first_trial_id": PRACTICE_FIRST_TRIAL_ID,
        "last_trial_id": PRACTICE_FIRST_TRIAL_ID + practice_trials - 1,
        "width_policy": "middle-of-three-sorted-calibration-widths",
        "stroke_width_px": expected_width[0],
        "stroke_width_deg": expected_width[1],
        "practice_feedback": config["task"]["practice_feedback"],
        "main_feedback_ignored": True,
        "balance_rule": "minimum-count-range-for-position-odd-and-representative-level",
    }
    for field, expected in expected_design.items():
        if design.get(field) != expected:
            raise StimulusCompilationError(
                f"practice manifest.practice_design.{field}가 동결 정책과 다릅니다."
            )

    raw_trials = manifest.get("trials")
    if (
        not isinstance(raw_trials, Sequence)
        or isinstance(raw_trials, (str, bytes))
        or len(raw_trials) != practice_trials
    ):
        raise StimulusCompilationError("practice manifest 시행 수가 config와 다릅니다.")
    expected_ids = set(
        range(PRACTICE_FIRST_TRIAL_ID, PRACTICE_FIRST_TRIAL_ID + practice_trials)
    )
    position_counts: Counter[int] = Counter()
    odd_counts: Counter[str] = Counter()
    level_counts: Counter[int] = Counter()
    observed_ids: set[int] = set()
    pair_by_level = {
        int(pair["calibration_level_index"]): pair
        for pair in calibration["normalized_color_pairs"]
    }
    for offset, raw_trial in enumerate(raw_trials, start=1):
        trial = _mapping(raw_trial, f"practice manifest.trials[{offset - 1}]")
        trial_id = _integer(trial.get("trial_id"), "practice trial_id", minimum=1)
        observed_ids.add(trial_id)
        level_index = _integer(
            trial.get("calibration_level_index"),
            "practice level index",
            minimum=1,
        )
        if level_index not in PRACTICE_LEVEL_INDICES:
            raise StimulusCompilationError(
                "practice 시행이 대표 세 수준 밖을 사용합니다."
            )
        expected_fields = {
            "global_trial_key": f"{settings['participant_id']}:{config_sha256}:{trial_id}",
            "practice_order_index": offset,
            "phase": "practice",
            "feedback_enabled": config["task"]["practice_feedback"],
            "practice_policy_id": PRACTICE_POLICY_ID,
            "stroke_width_px": expected_width[0],
            "stroke_width_deg": expected_width[1],
            "random_seed": expected_seed,
            "config_version": config.get("config_version"),
            "config_sha256": config_sha256,
            "source_calibration_manifest_sha256": calibration_manifest_sha256,
            "git_commit": calibration["git_commit"],
            "stimulus_family_id": settings["stimulus_family_id"],
            "participant_id": settings["participant_id"],
            "cvd_type": settings["cvd_type"],
            "machado_severity": settings["severity"],
            "severity_source": settings["severity_source"],
            "severity_session_id": settings["severity_session_id"],
            "fixed_lapse": settings["fixed_lapse"],
            "lapse_source": settings["lapse_source"],
            "lapse_session_id": settings["lapse_session_id"],
        }
        for field, expected in expected_fields.items():
            actual = trial.get(field)
            if isinstance(expected, float):
                equal = isinstance(actual, (int, float)) and math.isclose(
                    float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
                )
            else:
                equal = actual == expected
            if not equal:
                raise StimulusCompilationError(
                    f"practice trial {trial_id}.{field}가 동결 정책과 다릅니다."
                )
        pair = pair_by_level[level_index]
        for field in (
            "target_delta_e00_simulated",
            "delta_e00_simulated",
            "displayed_rgb_a",
            "displayed_rgb_b",
        ):
            if trial.get(field) != pair.get(field):
                raise StimulusCompilationError(
                    f"practice trial {trial_id}.{field}가 calibration color pair와 다릅니다."
                )
        try:
            _position_labels(trial)
        except RendererValidationError as error:
            raise StimulusCompilationError(
                f"practice trial {trial_id}의 위치/odd 구성이 잘못됐습니다: {error}"
            ) from error
        position_counts[int(trial["correct_position"])] += 1
        odd_counts[str(trial["odd_color"])] += 1
        level_counts[level_index] += 1
    if observed_ids != expected_ids:
        raise StimulusCompilationError("practice trial_id 예약 범위가 잘못됐습니다.")
    for counts, expected_keys, label in (
        (position_counts, {1, 2, 3, 4}, "위치"),
        (odd_counts, {"a", "b"}, "odd identity"),
        (level_counts, set(PRACTICE_LEVEL_INDICES), "대표 수준"),
    ):
        if set(counts) != expected_keys or max(counts.values()) - min(counts.values()) > 1:
            raise StimulusCompilationError(f"practice {label} 빈도가 균형이 아닙니다.")


def _practice_seed(config_sha256: str, git_commit: str) -> int:
    material = f"{PRACTICE_POLICY_ID}|{config_sha256}|{git_commit}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def compile_practice_stimulus_manifest(
    config_path: str | Path,
    calibration_manifest_path: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """동결 calibration manifest의 대표 세 수준으로 practice PNG를 만든다."""

    config, config_sha256 = _read_compilation_input(config_path, "config")
    calibration_manifest, calibration_manifest_sha256 = _read_compilation_input(
        calibration_manifest_path, "calibration manifest"
    )
    try:
        settings = _compile_settings(config)
        renderer_gate = verify_renderer_environment(config)
    except RendererValidationError as error:
        raise StimulusCompilationError(str(error)) from error
    practice_trials = _integer(
        _required(config, "sampling.practice_trials"),
        "sampling.practice_trials",
        minimum=0,
    )
    if not 6 <= practice_trials <= 10:
        raise StimulusCompilationError(
            "practice manifest는 config.sampling.practice_trials가 6~10일 때만 만듭니다."
        )
    practice_feedback = _required(config, "task.practice_feedback")
    if not isinstance(practice_feedback, bool):
        raise StimulusCompilationError("task.practice_feedback는 boolean이어야 합니다.")

    calibration_validation = validate_calibration_manifest_structure(
        calibration_manifest,
        config=config,
        config_sha256=config_sha256,
    )
    pairs = calibration_validation["normalized_color_pairs"]
    calibration_trial_ids = calibration_validation["trial_ids"]
    git_commit = calibration_validation["git_commit"]
    selected_pairs = {
        index: pairs[index - 1] for index in PRACTICE_LEVEL_INDICES
    }
    seed = _practice_seed(config_sha256, git_commit)
    generator = random.Random(seed)

    base_combinations = [
        (1, "a"),
        (2, "b"),
        (3, "a"),
        (4, "b"),
        (1, "b"),
        (2, "a"),
        (3, "b"),
        (4, "a"),
    ]
    position_permutation = [1, 2, 3, 4]
    generator.shuffle(position_permutation)
    invert_odd = bool(generator.getrandbits(1))
    combinations = []
    for position, odd in (base_combinations * 2)[:practice_trials]:
        mapped_odd = (
            ("b" if odd == "a" else "a") if invert_odd else odd
        )
        combinations.append((position_permutation[position - 1], mapped_odd))
    representative_sequence = list(PRACTICE_LEVEL_INDICES) * 4
    representative_offset = generator.randrange(len(PRACTICE_LEVEL_INDICES))
    representative_sequence = (
        representative_sequence[representative_offset:]
        + representative_sequence[:representative_offset]
    )[:practice_trials]
    planned = list(zip(combinations, representative_sequence, strict=True))
    generator.shuffle(planned)

    width_rows = _required(config, "calibration.widths")
    if not isinstance(width_rows, Sequence) or isinstance(width_rows, (str, bytes)) or len(width_rows) != 3:
        raise StimulusCompilationError("practice width 선택에는 calibration 굵기 세 개가 필요합니다.")
    sorted_widths = sorted(
        (
            _integer(
                _mapping(row, f"calibration.widths[{index}]").get("width_px"),
                f"calibration.widths[{index}].width_px",
                minimum=1,
            ),
            _finite_number(
                _mapping(row, f"calibration.widths[{index}]").get("width_deg"),
                f"calibration.widths[{index}].width_deg",
            ),
        )
        for index, row in enumerate(width_rows)
    )
    practice_width_px, practice_width_deg = sorted_widths[1]
    first_trial_id = PRACTICE_FIRST_TRIAL_ID
    practice_ids = set(range(first_trial_id, first_trial_id + practice_trials))
    overlap = practice_ids & calibration_trial_ids
    if overlap:
        raise StimulusCompilationError(
            f"practice와 calibration 전역 trial_id가 겹칩니다: {sorted(overlap)}"
        )

    checked_rows: list[dict[str, Any]] = []
    for order, ((position, odd), level_index) in enumerate(planned, start=1):
        trial_id = first_trial_id + order - 1
        pair = selected_pairs[level_index]
        base = "b" if odd == "a" else "a"
        position_colors = [base] * 4
        position_colors[position - 1] = odd
        checked_rows.append(
            {
                "trial_id": trial_id,
                "global_trial_key": f"{settings['participant_id']}:{config_sha256}:{trial_id}",
                "practice_order_index": order,
                "phase": "practice",
                "feedback_enabled": practice_feedback,
                "practice_policy_id": PRACTICE_POLICY_ID,
                "calibration_level_index": level_index,
                "stroke_width_px": practice_width_px,
                "stroke_width_deg": practice_width_deg,
                "target_delta_e00_simulated": pair[
                    "target_delta_e00_simulated"
                ],
                "actual_delta_e00_simulated": pair["delta_e00_simulated"],
                "correct_position": position,
                "odd_color": odd,
                "position_colors": position_colors,
                "random_seed": seed,
                "config_version": config.get("config_version"),
                "config_sha256": config_sha256,
                "source_calibration_manifest_sha256": calibration_manifest_sha256,
                "git_commit": git_commit,
                "stimulus_family_id": settings["stimulus_family_id"],
                "participant_id": settings["participant_id"],
                "cvd_type": settings["cvd_type"],
                "machado_severity": settings["severity"],
                "severity_source": settings["severity_source"],
                "severity_session_id": settings["severity_session_id"],
                "fixed_lapse": settings["fixed_lapse"],
                "lapse_source": settings["lapse_source"],
                "lapse_session_id": settings["lapse_session_id"],
                **{
                    key: value
                    for key, value in pair.items()
                    if key
                    not in {
                        "calibration_level_index",
                        "target_delta_e00_simulated",
                    }
                },
            }
        )
    position_counts = Counter(int(row["correct_position"]) for row in checked_rows)
    odd_counts = Counter(str(row["odd_color"]) for row in checked_rows)
    level_counts = Counter(int(row["calibration_level_index"]) for row in checked_rows)
    if max(position_counts.values()) - min(position_counts.values()) > 1:
        raise StimulusCompilationError("practice 위치 빈도가 가능한 최대로 균형화되지 않았습니다.")
    if max(odd_counts.values()) - min(odd_counts.values()) > 1:
        raise StimulusCompilationError("practice odd identity가 가능한 최대로 균형화되지 않았습니다.")
    if max(level_counts.values()) - min(level_counts.values()) > 1:
        raise StimulusCompilationError("practice 대표 수준 빈도가 가능한 최대로 균형화되지 않았습니다.")

    raster_dir = Path(output_dir)
    compiled_trials = [
        _pixel_compile_trial(
            config,
            row,
            raster_dir=raster_dir,
            filename_prefix="practice",
        )
        for row in checked_rows
    ]
    return {
        "schema_version": "1.0",
        "artifact_type": "compiled_practice_pixel_stimulus_manifest",
        "phase": "practice",
        "config_version": config.get("config_version"),
        "config_canonical_sha256": _canonical_sha256(config),
        "schedule_config_sha256": config_sha256,
        "source_config_raw_sha256": config_sha256,
        "source_calibration_manifest_sha256": calibration_manifest_sha256,
        "git_commit": git_commit,
        "stimulus_family_id": settings["stimulus_family_id"],
        "participant_id": settings["participant_id"],
        "cvd_type": settings["cvd_type"],
        "machado_severity": settings["severity"],
        "renderer": renderer_gate,
        "practice_design": {
            "policy_id": PRACTICE_POLICY_ID,
            "policy_version": "1",
            "representative_calibration_level_indices": list(
                PRACTICE_LEVEL_INDICES
            ),
            "seed_derivation": "first-64-bits-of-sha256(policy-id|config-raw-sha256|git-commit)",
            "derived_seed": seed,
            "first_trial_id": first_trial_id,
            "last_trial_id": first_trial_id + practice_trials - 1,
            "width_policy": "middle-of-three-sorted-calibration-widths",
            "stroke_width_px": practice_width_px,
            "stroke_width_deg": practice_width_deg,
            "feedback_policy_source": "config.task.practice_feedback",
            "practice_feedback": practice_feedback,
            "main_feedback_ignored": True,
            "balance_rule": "minimum-count-range-for-position-odd-and-representative-level",
            "counts": {
                "correct_position": {
                    str(key): position_counts.get(key, 0) for key in range(1, 5)
                },
                "odd_color": {key: odd_counts.get(key, 0) for key in ("a", "b")},
                "calibration_level_index": {
                    str(key): level_counts.get(key, 0)
                    for key in PRACTICE_LEVEL_INDICES
                },
            },
        },
        "color_generation": {
            "source": "verified-compiled-calibration-manifest",
            "gamut_policy": "reject",
            "rgb_quantization": "nearest_8bit",
            "target_absolute_tolerance_delta_e00": settings[
                "absolute_target_tolerance"
            ],
            "target_relative_tolerance": settings["relative_target_tolerance"],
            "allowed_error_rule": "max(absolute_tolerance, relative_tolerance * target)",
            "search_grid_steps": settings["grid_steps"],
            "unique_target_count": len(PRACTICE_LEVEL_INDICES),
        },
        "color_pairs": [selected_pairs[index] for index in PRACTICE_LEVEL_INDICES],
        "trial_count": len(compiled_trials),
        "png_files_written": True,
        "human_response_collection_implemented": True,
        "human_data_collection_allowed": False,
        "blocker": "실제 수집 전 GUI·동의·화면·Git 런타임 gate를 별도로 통과해야 합니다.",
        "trials": compiled_trials,
    }


def _read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RendererValidationError(f"{label} JSON을 읽을 수 없습니다: {error}") from error
    if not isinstance(payload, dict):
        _fail(f"{label} JSON 최상위는 객체여야 합니다.")
    return payload


def _make_raster_paths_manifest_relative(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    raster_dir: str | Path,
) -> None:
    """CLI 산출물의 상대 경로를 manifest 위치에 안전하게 묶는다."""

    manifest_parent = Path(manifest_path).resolve().parent
    resolved_raster = Path(raster_dir).resolve()
    try:
        prefix = resolved_raster.relative_to(manifest_parent)
    except ValueError as error:
        raise StimulusCompilationError(
            "PNG 폴더는 response GUI가 안전하게 찾도록 manifest 폴더 안에 있어야 합니다."
        ) from error
    rows = manifest.get("trials", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        artifact = row.get("raster_artifact")
        if not isinstance(artifact, dict):
            continue
        for field in ("png", "pixel_metadata"):
            value = artifact.get(field)
            if isinstance(value, str):
                artifact[field] = (prefix / value).as_posix()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="2x2 선 자극 PNG·픽셀 메타데이터 생성/검증(응답 GUI 미포함)"
    )
    actions = parser.add_subparsers(dest="action", required=True)
    render = actions.add_parser("render", help="시행 JSON 하나를 PNG로 렌더링")
    render.add_argument("--config", required=True)
    render.add_argument("--trial", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--metadata")
    verify = actions.add_parser("verify", help="기존 PNG와 픽셀 메타데이터 재검증")
    verify.add_argument("--png", required=True)
    verify.add_argument("--metadata", required=True)
    compile_parser = actions.add_parser(
        "compile", help="시행표 목표 ΔE를 실제 색쌍과 선택적 PNG로 컴파일"
    )
    compile_parser.add_argument("--config", required=True)
    compile_parser.add_argument("--schedule", required=True)
    compile_parser.add_argument("--manifest", required=True)
    compile_parser.add_argument("--png-dir")
    validation_compile = actions.add_parser(
        "compile-validation",
        help="동결 validation plan 3x64를 검증된 PNG manifest로 컴파일",
    )
    validation_compile.add_argument("--config", required=True)
    validation_compile.add_argument("--calibration-fits", required=True)
    validation_compile.add_argument("--validation-plan", required=True)
    validation_compile.add_argument("--manifest", required=True)
    validation_compile.add_argument("--png-dir", required=True)
    practice_compile = actions.add_parser(
        "compile-practice",
        help="전체 calibration manifest의 고정 대표 수준으로 practice PNG 컴파일",
    )
    practice_compile.add_argument("--config", required=True)
    practice_compile.add_argument("--calibration-manifest", required=True)
    practice_compile.add_argument("--manifest", required=True)
    practice_compile.add_argument("--png-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "verify":
            result = verify_stimulus_artifact(args.png, args.metadata)
        elif args.action == "compile-validation":
            manifest = compile_validation_stimulus_manifest(
                args.config,
                args.calibration_fits,
                args.validation_plan,
                output_dir=args.png_dir,
            )
            _make_raster_paths_manifest_relative(
                manifest,
                manifest_path=args.manifest,
                raster_dir=args.png_dir,
            )
            _atomic_write_json(Path(args.manifest), manifest)
            result = {
                "renderer_verified": True,
                "artifact_type": manifest["artifact_type"],
                "phase": manifest["phase"],
                "trial_count": manifest["trial_count"],
                "manifest": str(Path(args.manifest)),
                "png_files_written": True,
                "human_data_collection_allowed": False,
            }
        elif args.action == "compile-practice":
            manifest = compile_practice_stimulus_manifest(
                args.config,
                args.calibration_manifest,
                output_dir=args.png_dir,
            )
            _make_raster_paths_manifest_relative(
                manifest,
                manifest_path=args.manifest,
                raster_dir=args.png_dir,
            )
            _atomic_write_json(Path(args.manifest), manifest)
            result = {
                "renderer_verified": True,
                "artifact_type": manifest["artifact_type"],
                "phase": manifest["phase"],
                "trial_count": manifest["trial_count"],
                "manifest": str(Path(args.manifest)),
                "png_files_written": True,
                "human_data_collection_allowed": False,
            }
        else:
            config = _read_json_object(args.config, "config")
            if args.action == "render":
                trial = _read_json_object(args.trial, "trial")
                metadata = render_stimulus_png(
                    config, trial, args.output, metadata_path=args.metadata
                )
                result = {
                    "renderer_verified": True,
                    "trial_id": metadata["trial"]["trial_id"],
                    "png_sha256": metadata["pixel_verification"]["png_sha256"],
                    "human_response_collection_implemented": True,
                }
            else:
                schedule_payload = _read_json_object(args.schedule, "schedule")
                schedule = schedule_payload.get("trials")
                if not isinstance(schedule, list):
                    _fail("schedule JSON에는 trials 배열이 필요합니다.")
                manifest = compile_stimulus_manifest(
                    config, schedule, output_dir=args.png_dir
                )
                if args.png_dir is not None:
                    _make_raster_paths_manifest_relative(
                        manifest,
                        manifest_path=args.manifest,
                        raster_dir=args.png_dir,
                    )
                _atomic_write_json(Path(args.manifest), manifest)
                result = {
                    "renderer_verified": True,
                    "trial_count": manifest["trial_count"],
                    "manifest": str(Path(args.manifest)),
                    "png_files_written": manifest["png_files_written"],
                    "human_response_collection_implemented": True,
                }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (RendererValidationError, OSError, ValueError) as error:
        print(f"렌더링 차단됨: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANTIALIASING_POLICY",
    "CAP_POLICY",
    "LOCKED_PILLOW_VERSION",
    "PIXEL_SNAPPING_POLICY",
    "RENDERER_ID",
    "RendererValidationError",
    "StimulusCompilationError",
    "build_stimulus_spec",
    "compile_practice_stimulus_manifest",
    "compile_stimulus_manifest",
    "compile_validation_stimulus_manifest",
    "main",
    "render_stimulus_png",
    "renderer_implementation_sha256",
    "verify_renderer_environment",
    "verify_stimulus_artifact",
    "validate_calibration_manifest_structure",
    "validate_practice_manifest_structure",
]
