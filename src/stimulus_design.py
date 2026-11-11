"""통제된 두 색 자극을 D65 CIELAB 좌표에서 생성한다.

연속 Lab 좌표에서는 지정한 중점을 기준으로 두 색을 정확히 대칭 배치한다.
실험에서 보이는 색은 8-bit sRGB이므로, 최종 후보의 모든 색차는 반올림된
표시 RGB를 다시 변환하고 Machado 시뮬레이션한 값으로 계산한다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from color_spaces import is_lab_in_srgb_gamut, lab_to_srgb, srgb_to_lab
from distance import delta_e_2000
from machado import simulate_color_float


Color3 = tuple[float, float, float]
RGB8 = tuple[int, int, int]


def _finite_color3(value: Sequence[int | float], *, name: str) -> Color3:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name}에는 숫자 3개가 있어야 합니다.")
    result = tuple(float(channel) for channel in value)
    if not all(math.isfinite(channel) for channel in result):
        raise ValueError(f"{name} 값은 유한한 숫자여야 합니다.")
    return result  # type: ignore[return-value]


def _symmetric_labs(midpoint: Color3, direction: Color3, scale: float) -> tuple[Color3, Color3]:
    return (
        tuple(midpoint[i] - scale * direction[i] for i in range(3)),
        tuple(midpoint[i] + scale * direction[i] for i in range(3)),
    )  # type: ignore[return-value]


def _display_rgb(lab: Color3) -> tuple[Color3, RGB8]:
    rgb_float = lab_to_srgb(lab)
    rgb = tuple(min(255, max(0, round(channel))) for channel in rgb_float)
    return rgb_float, rgb  # type: ignore[return-value]


def _candidate(
    midpoint: Color3,
    direction: Color3,
    scale: float,
    cvd_type: str,
    severity: float,
) -> dict[str, Any]:
    continuous_a, continuous_b = _symmetric_labs(midpoint, direction, scale)
    float_a, rgb_a = _display_rgb(continuous_a)
    float_b, rgb_b = _display_rgb(continuous_b)
    displayed_a = srgb_to_lab(rgb_a)
    displayed_b = srgb_to_lab(rgb_b)
    simulated_rgb_a = simulate_color_float(rgb_a, cvd_type, severity)
    simulated_rgb_b = simulate_color_float(rgb_b, cvd_type, severity)
    simulated_a = srgb_to_lab(simulated_rgb_a)
    simulated_b = srgb_to_lab(simulated_rgb_b)
    return {
        "scale": scale,
        "continuous_lab_a": continuous_a,
        "continuous_lab_b": continuous_b,
        "rgb_float_a": float_a,
        "rgb_float_b": float_b,
        "rgb_a": rgb_a,
        "rgb_b": rgb_b,
        "displayed_lab_a": displayed_a,
        "displayed_lab_b": displayed_b,
        "simulated_rgb_a": simulated_rgb_a,
        "simulated_rgb_b": simulated_rgb_b,
        "simulated_lab_a": simulated_a,
        "simulated_lab_b": simulated_b,
        "original_delta_e00": float(delta_e_2000(displayed_a, displayed_b)),
        "simulated_delta_e00": float(delta_e_2000(simulated_a, simulated_b)),
    }


def _maximum_symmetric_gamut_scale(
    midpoint: Color3,
    direction: Color3,
    *,
    maximum_scale: float,
) -> tuple[float, bool]:
    """주어진 Lab 직선에서 양 끝이 색역 안인 최대 scale을 찾는다."""
    if not is_lab_in_srgb_gamut(midpoint):
        raise ValueError("midpoint_lab가 sRGB gamut 밖에 있습니다.")

    def valid(scale: float) -> bool:
        first, second = _symmetric_labs(midpoint, direction, scale)
        return is_lab_in_srgb_gamut(first) and is_lab_in_srgb_gamut(second)

    low = 0.0
    high = min(1.0, maximum_scale)
    while high < maximum_scale and valid(high):
        low = high
        high = min(maximum_scale, high * 2.0)
    if valid(high):
        return high, False

    for _ in range(60):
        middle = (low + high) / 2.0
        if valid(middle):
            low = middle
        else:
            high = middle
    return low, True


def generate_symmetric_pair(
    midpoint_lab: Sequence[int | float],
    direction_lab: Sequence[int | float],
    target_simulated_delta_e00: float,
    cvd_type: str,
    severity: float,
    *,
    max_target_error: float = 0.5,
    maximum_scale: float = 256.0,
    grid_steps: int = 4096,
) -> dict[str, Any]:
    """목표 Machado 후 ΔE00에 가장 가까운 대칭 표시색 쌍을 생성한다.

    연속 Lab 직선에서 색역 안 후보를 탐색한 뒤 각 후보를 8-bit RGB로
    반올림한다. 반환 색차는 이 실제 표시 RGB를 기준으로 다시 계산된다.
    8-bit 양자화 때문에 임의의 목표를 정확히 만들 수 있다는 보장은 없으며,
    가장 가까운 후보도 ``max_target_error``를 넘으면 ``ValueError``가 난다.
    """
    midpoint = _finite_color3(midpoint_lab, name="midpoint_lab")
    direction = _finite_color3(direction_lab, name="direction_lab")
    target = float(target_simulated_delta_e00)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_simulated_delta_e00은 0보다 큰 유한한 값이어야 합니다.")
    if not math.isfinite(max_target_error) or max_target_error < 0.0:
        raise ValueError("max_target_error는 0 이상의 유한한 값이어야 합니다.")
    if not math.isfinite(maximum_scale) or maximum_scale <= 0.0:
        raise ValueError("maximum_scale은 0보다 큰 유한한 값이어야 합니다.")
    if isinstance(grid_steps, bool) or not isinstance(grid_steps, int) or grid_steps < 32:
        raise ValueError("grid_steps는 32 이상의 정수여야 합니다.")

    norm = math.sqrt(sum(channel * channel for channel in direction))
    if norm <= 1e-12:
        raise ValueError("direction_lab은 영벡터일 수 없습니다.")
    unit = tuple(channel / norm for channel in direction)
    gamut_scale, boundary_found = _maximum_symmetric_gamut_scale(
        midpoint, unit, maximum_scale=maximum_scale
    )
    if gamut_scale <= 0.0:
        raise ValueError("주어진 중점과 방향에서 서로 다른 sRGB 자극을 만들 수 없습니다.")

    # 같은 8-bit 쌍은 한 번만 평가한다. 동점은 더 작은 scale, 이어서 RGB
    # 사전식 순서로 고정해 플랫폼과 실행 순서에 관계없이 결정론적으로 고른다.
    unique: dict[tuple[RGB8, RGB8], dict[str, Any]] = {}
    for scale in np.linspace(0.0, gamut_scale, grid_steps + 1):
        continuous_a, continuous_b = _symmetric_labs(midpoint, unit, float(scale))
        if not (
            is_lab_in_srgb_gamut(continuous_a)
            and is_lab_in_srgb_gamut(continuous_b)
        ):
            continue
        candidate = _candidate(midpoint, unit, float(scale), cvd_type, severity)
        key = candidate["rgb_a"], candidate["rgb_b"]
        previous = unique.get(key)
        if previous is None or candidate["scale"] < previous["scale"]:
            unique[key] = candidate

    candidates = [candidate for candidate in unique.values() if candidate["rgb_a"] != candidate["rgb_b"]]
    if not candidates:
        raise ValueError("주어진 중점과 방향에서 서로 다른 8-bit sRGB 자극을 만들 수 없습니다.")
    best = min(
        candidates,
        key=lambda item: (
            abs(item["simulated_delta_e00"] - target),
            item["scale"],
            item["rgb_a"],
            item["rgb_b"],
        ),
    )
    error = abs(best["simulated_delta_e00"] - target)
    if error > max_target_error:
        reachable = max(item["simulated_delta_e00"] for item in candidates)
        raise ValueError(
            "목표 Machado 후 ΔE00를 허용 오차 안에서 만들 수 없습니다: "
            f"target={target:.6g}, best_error={error:.6g}, "
            f"searched_max={reachable:.6g}, tolerance={max_target_error:.6g}"
        )

    midpoint_reconstructed = tuple(
        (best["continuous_lab_a"][i] + best["continuous_lab_b"][i]) / 2.0
        for i in range(3)
    )
    return {
        "rgb_a": best["rgb_a"],
        "rgb_b": best["rgb_b"],
        "continuous_lab_a": best["continuous_lab_a"],
        "continuous_lab_b": best["continuous_lab_b"],
        "displayed_lab_a": best["displayed_lab_a"],
        "displayed_lab_b": best["displayed_lab_b"],
        "simulated_rgb_a": best["simulated_rgb_a"],
        "simulated_rgb_b": best["simulated_rgb_b"],
        "simulated_lab_a": best["simulated_lab_a"],
        "simulated_lab_b": best["simulated_lab_b"],
        "original_delta_e00": best["original_delta_e00"],
        "simulated_delta_e00": best["simulated_delta_e00"],
        "target_simulated_delta_e00": target,
        "target_error": error,
        "metadata": {
            "scale": best["scale"],
            "direction_unit": unit,
            "continuous_midpoint": midpoint_reconstructed,
            "continuous_symmetry_max_abs_error": max(
                abs(midpoint_reconstructed[i] - midpoint[i]) for i in range(3)
            ),
            "gamut": {
                "all_continuous_candidates_in_gamut": True,
                "maximum_symmetric_scale": gamut_scale,
                "boundary_found_before_maximum_scale": boundary_found,
            },
            "quantization": {
                "rgb_float_a": best["rgb_float_a"],
                "rgb_float_b": best["rgb_float_b"],
                "rounding_error_a": tuple(
                    best["rgb_a"][i] - best["rgb_float_a"][i] for i in range(3)
                ),
                "rounding_error_b": tuple(
                    best["rgb_b"][i] - best["rgb_float_b"][i] for i in range(3)
                ),
            },
            "search": {
                "method": "uniform_scale_grid_with_8bit_deduplication",
                "grid_steps": grid_steps,
                "unique_rgb_pairs_evaluated": len(unique),
                "max_target_error": max_target_error,
            },
        },
    }


__all__ = ["generate_symmetric_pair"]
