"""Machado 방식의 적록 색각이상 시뮬레이터.

입출력은 일반적인 8-bit sRGB ``(R, G, B)`` 튜플이다. 행렬은 빛의
세기에 비례하는 선형 RGB에 적용해야 하므로, 이 모듈은 다음 순서를
사용한다.

    8-bit sRGB -> 선형 RGB -> Machado 행렬 -> gamut clip -> 8-bit sRGB

행렬 출처:
G. M. Machado, M. M. Oliveira, L. A. F. Fernandes (2009),
"A Physiologically-based Model for Simulation of Color Vision Deficiency".
저자 보충 자료의 0.0~1.0 행렬 표를 사용한다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from color_spaces import (
    linear_rgb_to_srgb,
    srgb_to_linear_rgb,
    validate_rgb,
)


Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


# 표의 행렬 순서는 severity 0.0, 0.1, ..., 1.0이다.
_PROTAN_MATRICES: Final[tuple[Matrix3, ...]] = (
    ((1.000000, 0.000000, -0.000000), (0.000000, 1.000000, 0.000000), (-0.000000, -0.000000, 1.000000)),
    ((0.856167, 0.182038, -0.038205), (0.029342, 0.955115, 0.015544), (-0.002880, -0.001563, 1.004443)),
    ((0.734766, 0.334872, -0.069637), (0.051840, 0.919198, 0.028963), (-0.004928, -0.004209, 1.009137)),
    ((0.630323, 0.465641, -0.095964), (0.069181, 0.890046, 0.040773), (-0.006308, -0.007724, 1.014032)),
    ((0.539009, 0.579343, -0.118352), (0.082546, 0.866121, 0.051332), (-0.007136, -0.011959, 1.019095)),
    ((0.458064, 0.679578, -0.137642), (0.092785, 0.846313, 0.060902), (-0.007494, -0.016807, 1.024301)),
    ((0.385450, 0.769005, -0.154455), (0.100526, 0.829802, 0.069673), (-0.007442, -0.022190, 1.029632)),
    ((0.319627, 0.849633, -0.169261), (0.106241, 0.815969, 0.077790), (-0.007025, -0.028051, 1.035076)),
    ((0.259411, 0.923008, -0.182420), (0.110296, 0.804340, 0.085364), (-0.006276, -0.034346, 1.040622)),
    ((0.203876, 0.990338, -0.194214), (0.112975, 0.794542, 0.092483), (-0.005222, -0.041043, 1.046265)),
    ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
)

_DEUTAN_MATRICES: Final[tuple[Matrix3, ...]] = (
    ((1.000000, 0.000000, -0.000000), (0.000000, 1.000000, 0.000000), (-0.000000, -0.000000, 1.000000)),
    ((0.866435, 0.177704, -0.044139), (0.049567, 0.939063, 0.011370), (-0.003453, 0.007233, 0.996220)),
    ((0.760729, 0.319078, -0.079807), (0.090568, 0.889315, 0.020117), (-0.006027, 0.013325, 0.992702)),
    ((0.675425, 0.433850, -0.109275), (0.125303, 0.847755, 0.026942), (-0.007950, 0.018572, 0.989378)),
    ((0.605511, 0.528560, -0.134071), (0.155318, 0.812366, 0.032316), (-0.009376, 0.023176, 0.986200)),
    ((0.547494, 0.607765, -0.155259), (0.181692, 0.781742, 0.036566), (-0.010410, 0.027275, 0.983136)),
    ((0.498864, 0.674741, -0.173604), (0.205199, 0.754872, 0.039929), (-0.011131, 0.030969, 0.980162)),
    ((0.457771, 0.731899, -0.189670), (0.226409, 0.731012, 0.042579), (-0.011595, 0.034333, 0.977261)),
    ((0.422823, 0.781057, -0.203881), (0.245752, 0.709602, 0.044646), (-0.011843, 0.037423, 0.974421)),
    ((0.392952, 0.823610, -0.216562), (0.263559, 0.690210, 0.046232), (-0.011910, 0.040281, 0.971630)),
    ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881)),
)

_MATRICES: Final = {
    "protan": _PROTAN_MATRICES,
    "deutan": _DEUTAN_MATRICES,
}

_TYPE_ALIASES: Final = {
    "protan": "protan",
    "protanomaly": "protan",
    "protanopia": "protan",
    "deutan": "deutan",
    "deuteranomaly": "deutan",
    "deuteranopia": "deutan",
}


def _canonical_cvd_type(cvd_type: str) -> str:
    """사용자 입력을 ``protan`` 또는 ``deutan``으로 정규화한다."""
    if not isinstance(cvd_type, str):
        raise TypeError("cvd_type은 문자열이어야 합니다.")
    try:
        return _TYPE_ALIASES[cvd_type.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "cvd_type은 'protan' 또는 'deutan'이어야 합니다."
        ) from exc


def _validate_severity(severity: float) -> float:
    if isinstance(severity, bool) or not isinstance(severity, (int, float)):
        raise TypeError("severity는 0.0~1.0 사이의 숫자여야 합니다.")
    value = float(severity)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("severity는 0.0~1.0 범위여야 합니다.")
    return value


def get_machado_matrix(cvd_type: str, severity: float) -> Matrix3:
    """CVD 종류와 severity에 해당하는 Machado 3x3 행렬을 반환한다.

    표에 없는 severity는 인접한 0.1 간격 행렬 사이에서 선형 보간한다.
    예를 들어 0.15는 0.1 행렬과 0.2 행렬을 절반씩 섞는다.
    """
    kind = _canonical_cvd_type(cvd_type)
    value = _validate_severity(severity)

    position = value * 10.0
    lower_index = min(math.floor(position + 1e-12), 10)
    upper_index = min(lower_index + 1, 10)
    fraction = position - lower_index

    lower = _MATRICES[kind][lower_index]
    if lower_index == upper_index or abs(fraction) < 1e-12:
        return lower

    upper = _MATRICES[kind][upper_index]
    return tuple(
        tuple(
            lower[row][column]
            + fraction * (upper[row][column] - lower[row][column])
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def simulate_color_float(
    rgb: Sequence[int | float],
    cvd_type: str,
    severity: float,
) -> tuple[float, float, float]:
    """반올림 전 0~255 sRGB 시뮬레이션 결과를 반환한다.

    색차 계산과 최적화에서는 8-bit 반올림 오차를 피하기 위해 이 함수를
    사용하고, 화면에 표시할 최종 결과에는 :func:`simulate_color`를 쓴다.
    """
    red, green, blue = validate_rgb(rgb)
    value = _validate_severity(severity)
    kind = _canonical_cvd_type(cvd_type)
    if value == 0.0:
        return red, green, blue

    linear_rgb = srgb_to_linear_rgb((red, green, blue))
    matrix = get_machado_matrix(kind, value)
    simulated_linear = tuple(
        sum(matrix[row][column] * linear_rgb[column] for column in range(3))
        for row in range(3)
    )
    return linear_rgb_to_srgb(simulated_linear, clip=True)


def simulate_color(
    rgb: Sequence[int | float],
    cvd_type: str,
    severity: float,
) -> tuple[int, int, int]:
    """8-bit sRGB 색 하나를 Machado 방식으로 시뮬레이션한다.

    Args:
        rgb: 각 채널이 0~255인 ``(R, G, B)`` 값.
        cvd_type: ``"protan"`` 또는 ``"deutan"``.
        severity: 0.0(정상 색각)~1.0(이색형 색각)의 연속 값.

    Returns:
        gamut에 맞게 자른 8-bit sRGB ``(R, G, B)`` 튜플.
    """
    result = simulate_color_float(rgb, cvd_type, severity)
    return tuple(min(255, max(0, round(channel))) for channel in result)


__all__ = ["get_machado_matrix", "simulate_color", "simulate_color_float"]
