"""색차 계산 함수.

혼동 탐지와 최종 평가는 CIEDE2000을, 보정 이동량의 목적함수는
OKLab 유클리드 거리를 사용한다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from color_spaces import srgb_to_lab, srgb_to_oklab


def delta_e_2000(
    lab_a: Sequence[float] | np.ndarray,
    lab_b: Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """두 CIELAB 색의 CIEDE2000 색차를 계산한다.

    배열의 마지막 축은 ``(L*, a*, b*)``이어야 하며 NumPy broadcasting을
    지원한다. 구현 검증에는 Sharma, Wu, Dalal(2005)의 보충 데이터를 쓴다.
    """
    color_1 = np.asarray(lab_a, dtype=float)
    color_2 = np.asarray(lab_b, dtype=float)
    if color_1.shape[-1:] != (3,) or color_2.shape[-1:] != (3,):
        raise ValueError("CIELAB 배열의 마지막 축 크기는 3이어야 합니다.")

    l1, a1, b1 = np.moveaxis(color_1, -1, 0)
    l2, a2, b2 = np.moveaxis(color_2, -1, 0)

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_mean = (c1 + c2) / 2.0
    c_mean_7 = c_mean**7
    g = 0.5 * (1.0 - np.sqrt(c_mean_7 / (c_mean_7 + 25.0**7)))

    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = np.hypot(a1_prime, b1)
    c2_prime = np.hypot(a2_prime, b2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0
    h1_prime = np.where(c1_prime == 0.0, 0.0, h1_prime)
    h2_prime = np.where(c2_prime == 0.0, 0.0, h2_prime)

    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime
    hue_difference = h2_prime - h1_prime
    zero_chroma = c1_prime * c2_prime == 0.0
    delta_h_prime = np.where(
        zero_chroma,
        0.0,
        np.where(
            hue_difference > 180.0,
            hue_difference - 360.0,
            np.where(hue_difference < -180.0, hue_difference + 360.0, hue_difference),
        ),
    )
    delta_big_h_prime = (
        2.0
        * np.sqrt(c1_prime * c2_prime)
        * np.sin(np.radians(delta_h_prime) / 2.0)
    )

    l_mean_prime = (l1 + l2) / 2.0
    c_mean_prime = (c1_prime + c2_prime) / 2.0
    hue_sum = h1_prime + h2_prime
    hue_abs_difference = np.abs(h1_prime - h2_prime)
    h_mean_prime = np.where(
        zero_chroma,
        hue_sum,
        np.where(
            hue_abs_difference <= 180.0,
            hue_sum / 2.0,
            np.where(hue_sum < 360.0, (hue_sum + 360.0) / 2.0, (hue_sum - 360.0) / 2.0),
        ),
    )

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_mean_prime - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_mean_prime))
        + 0.32 * np.cos(np.radians(3.0 * h_mean_prime + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_mean_prime - 63.0))
    )
    delta_theta = 30.0 * np.exp(-((h_mean_prime - 275.0) / 25.0) ** 2)
    c_mean_prime_7 = c_mean_prime**7
    r_c = 2.0 * np.sqrt(c_mean_prime_7 / (c_mean_prime_7 + 25.0**7))
    s_l = 1.0 + 0.015 * (l_mean_prime - 50.0) ** 2 / np.sqrt(
        20.0 + (l_mean_prime - 50.0) ** 2
    )
    s_c = 1.0 + 0.045 * c_mean_prime
    s_h = 1.0 + 0.015 * c_mean_prime * t
    r_t = -np.sin(np.radians(2.0 * delta_theta)) * r_c

    lightness_term = delta_l_prime / s_l
    chroma_term = delta_c_prime / s_c
    hue_term = delta_big_h_prime / s_h
    result = np.sqrt(
        lightness_term**2
        + chroma_term**2
        + hue_term**2
        + r_t * chroma_term * hue_term
    )
    if result.ndim == 0:
        return float(result)
    return result


def oklab_distance(
    oklab_a: Sequence[float] | np.ndarray,
    oklab_b: Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """OKLab 좌표 사이의 유클리드 거리를 계산한다."""
    color_1 = np.asarray(oklab_a, dtype=float)
    color_2 = np.asarray(oklab_b, dtype=float)
    if color_1.shape[-1:] != (3,) or color_2.shape[-1:] != (3,):
        raise ValueError("OKLab 배열의 마지막 축 크기는 3이어야 합니다.")
    result = np.linalg.norm(color_1 - color_2, axis=-1)
    if result.ndim == 0:
        return float(result)
    return result


def color_distance(
    color_a: Sequence[int | float],
    color_b: Sequence[int | float],
    *,
    method: str = "ciede2000",
) -> float:
    """두 0~255 sRGB 색의 색차를 계산한다."""
    normalized_method = method.strip().lower()
    if normalized_method in {"ciede2000", "deltae00", "de2000"}:
        return float(delta_e_2000(srgb_to_lab(color_a), srgb_to_lab(color_b)))
    if normalized_method in {"oklab", "ok"}:
        return float(oklab_distance(srgb_to_oklab(color_a), srgb_to_oklab(color_b)))
    raise ValueError("method는 'ciede2000' 또는 'oklab'이어야 합니다.")


__all__ = ["color_distance", "delta_e_2000", "oklab_distance"]
