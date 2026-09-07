"""선 굵기별 색 구별 실험을 위한 단위 변환과 사전 계산.

이 모듈은 사람의 C80을 임의로 정하지 않는다. 화면의 물리 조건을 시각도로
바꾸고, Szafir(2018)의 정상색각 관찰자 선 모델에서 *상대적인 굵기 효과*만
계산한다. 실제 적록색각이상 참가자의 C80은 별도 실험으로 측정해야 한다.

Szafir 모델은 CIELAB 축별 유클리드 차이를 사용한다. 따라서 여기서 반환하는
값을 CIEDE2000 값이나 본 연구의 4AFC 80% 역치로 직접 해석하면 안 된다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real


SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG = 0.05
SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG = 0.35

# Eq. 14--16, Szafir (2018), ``Modeling Color Difference for
# Visualization Design``. ND_axis(p, s) = p / (a - b / s), s는 시각도.
_SZAFIR_LINE_COEFFICIENTS = {
    "L": (0.0742, 0.0023),
    "a": (0.0623, 0.0015),
    "b": (0.0425, 0.0009),
}


def _positive_finite(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name}은 0보다 큰 유한한 숫자여야 합니다.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name}은 0보다 큰 유한한 숫자여야 합니다.")
    return number


def pixel_pitch_mm(display_width_mm: Real, horizontal_resolution_px: Real) -> float:
    """화면 가로 길이와 가로 해상도로 한 픽셀의 물리 폭(mm)을 구한다."""
    width_mm = _positive_finite(display_width_mm, "display_width_mm")
    resolution_px = _positive_finite(
        horizontal_resolution_px, "horizontal_resolution_px"
    )
    return width_mm / resolution_px


def pixels_to_visual_angle_deg(
    width_px: Real,
    *,
    display_width_mm: Real,
    horizontal_resolution_px: Real,
    viewing_distance_mm: Real,
) -> float:
    """픽셀 폭을 관찰자의 시각도(degree)로 바꾼다.

    화면은 관찰자 정면에 있고 픽셀이 정사각형이라는 조건의 기하학적 변환이다.
    실제 실험에서는 화면 가로 길이와 관찰 거리를 직접 측정해 전달해야 한다.
    """
    pixels = _positive_finite(width_px, "width_px")
    distance_mm = _positive_finite(viewing_distance_mm, "viewing_distance_mm")
    physical_width_mm = pixels * pixel_pitch_mm(
        display_width_mm, horizontal_resolution_px
    )
    angle_rad = 2.0 * math.atan(physical_width_mm / (2.0 * distance_mm))
    return math.degrees(angle_rad)


def visual_angle_deg_to_pixels(
    visual_angle_deg: Real,
    *,
    display_width_mm: Real,
    horizontal_resolution_px: Real,
    viewing_distance_mm: Real,
) -> float:
    """시각도(degree)를 해당 화면의 픽셀 폭으로 바꾼다."""
    angle_deg = _positive_finite(visual_angle_deg, "visual_angle_deg")
    if angle_deg >= 180.0:
        raise ValueError("visual_angle_deg는 180보다 작아야 합니다.")
    distance_mm = _positive_finite(viewing_distance_mm, "viewing_distance_mm")
    physical_width_mm = 2.0 * distance_mm * math.tan(
        math.radians(angle_deg) / 2.0
    )
    return physical_width_mm / pixel_pitch_mm(
        display_width_mm, horizontal_resolution_px
    )


def _check_szafir_angle(
    thickness_deg: Real, *, allow_extrapolation: bool
) -> float:
    thickness = _positive_finite(thickness_deg, "thickness_deg")
    if not isinstance(allow_extrapolation, bool):
        raise TypeError("allow_extrapolation은 bool이어야 합니다.")
    if not allow_extrapolation and not (
        SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG
        <= thickness
        <= SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG
    ):
        raise ValueError(
            "Szafir 선 모델의 측정 범위는 0.05~0.35도입니다. "
            "범위 밖 값을 연구 결과로 쓰지 마세요. 탐색 목적으로만 "
            "외삽하려면 allow_extrapolation=True를 명시해야 합니다."
        )
    return thickness


def szafir_line_axis_nd(
    axis: str,
    probability: Real,
    thickness_deg: Real,
    *,
    allow_extrapolation: bool = False,
) -> float:
    """Szafir(2018) 선 모델의 축별 normalized difference를 계산한다.

    ``probability``는 원 논문의 이진 동일/상이 과제 모델에 들어가는 p다.
    본 연구의 4AFC 실제 정답률이나 C80과 같은 값이 아니다.
    """
    if axis not in _SZAFIR_LINE_COEFFICIENTS:
        raise ValueError("axis는 'L', 'a', 'b' 중 하나여야 합니다.")
    if isinstance(probability, bool) or not isinstance(probability, Real):
        raise TypeError("probability는 0보다 크고 1보다 작은 숫자여야 합니다.")
    p = float(probability)
    if not math.isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError("probability는 0보다 크고 1보다 작아야 합니다.")

    thickness = _check_szafir_angle(
        thickness_deg, allow_extrapolation=allow_extrapolation
    )
    intercept, inverse_term = _SZAFIR_LINE_COEFFICIENTS[axis]
    denominator = intercept - inverse_term / thickness
    if denominator <= 0.0:
        raise ValueError("이 굵기에서는 Szafir 식의 분모가 양수가 아닙니다.")
    return p / denominator


def szafir_line_direction_threshold(
    direction_lab: Sequence[Real],
    probability: Real,
    thickness_deg: Real,
    *,
    allow_extrapolation: bool = False,
) -> float:
    """고정 CIELAB 방향에서 Szafir normalized distance가 1인 크기를 구한다.

    방향 벡터는 함수 안에서 길이 1로 정규화한다. 반환값은 CIELAB 직선상의
    이동 크기이며, CIEDE2000이나 적록색각이상자의 개인 C80이 아니다.
    """
    if len(direction_lab) != 3:
        raise ValueError("direction_lab은 (dL, da, db) 세 값이어야 합니다.")
    components = []
    for index, value in enumerate(direction_lab):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"direction_lab[{index}]는 유한한 숫자여야 합니다.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"direction_lab[{index}]는 유한한 숫자여야 합니다.")
        components.append(number)

    norm = math.sqrt(sum(value * value for value in components))
    if norm == 0.0:
        raise ValueError("direction_lab은 영벡터일 수 없습니다.")
    unit = [value / norm for value in components]
    axes = ("L", "a", "b")
    axis_nd = [
        szafir_line_axis_nd(
            axis,
            probability,
            thickness_deg,
            allow_extrapolation=allow_extrapolation,
        )
        for axis in axes
    ]
    normalized_squared = sum(
        (component / nd) ** 2 for component, nd in zip(unit, axis_nd)
    )
    return 1.0 / math.sqrt(normalized_squared)


def szafir_relative_scale(
    direction_lab: Sequence[Real],
    probability: Real,
    thickness_deg: Real,
    reference_thickness_deg: Real,
    *,
    allow_extrapolation: bool = False,
) -> float:
    """참조 굵기 대비 Szafir 선 모델의 상대 역치 배율을 반환한다."""
    target = szafir_line_direction_threshold(
        direction_lab,
        probability,
        thickness_deg,
        allow_extrapolation=allow_extrapolation,
    )
    reference = szafir_line_direction_threshold(
        direction_lab,
        probability,
        reference_thickness_deg,
        allow_extrapolation=allow_extrapolation,
    )
    return target / reference


def geometric_validation_widths(
    calibration_widths: Sequence[Real],
) -> tuple[float, float]:
    """세 보정 굵기 사이의 사전등록용 기하평균 후보 두 개를 만든다."""
    if len(calibration_widths) != 3:
        raise ValueError("calibration_widths에는 정확히 세 굵기가 필요합니다.")
    widths = [
        _positive_finite(value, f"calibration_widths[{index}]")
        for index, value in enumerate(calibration_widths)
    ]
    if not widths[0] < widths[1] < widths[2]:
        raise ValueError("세 굵기는 작은 값부터 엄격히 증가해야 합니다.")
    return math.sqrt(widths[0] * widths[1]), math.sqrt(widths[1] * widths[2])


__all__ = [
    "SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG",
    "SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG",
    "geometric_validation_widths",
    "pixel_pitch_mm",
    "pixels_to_visual_angle_deg",
    "szafir_line_axis_nd",
    "szafir_line_direction_threshold",
    "szafir_relative_scale",
    "visual_angle_deg_to_pixels",
]
