"""측정한 Lab 중점·방향 안에서만 C80 제약을 만족시키는 보정.

한 색 방향에서 잰 scalar C80을 임의 RGB 방향으로 일반화하면 행동 근거가
사라진다. 이 모듈은 원 자극과 후보를 모두 같은 연속 Lab 중점·방향의 대칭
쌍으로 제한한다. 반환값의 'minimum'은 연속 전체 색공간이 아니라 명시한
scale grid를 8-bit sRGB로 양자화해 얻은 유일 후보 집합 안의 정확한 최소다.
실현가능 후보는 표시 RGB에서 다시 계산한 Machado 후 ΔE00가 목표 이상인
후보 전부이며, 목표 초과량에 별도 상한을 두지 않는다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np

from color_spaces import srgb_to_oklab, validate_rgb
from stimulus_design import (
    _candidate,
    _finite_color3,
    _maximum_symmetric_gamut_scale,
)


def _positive(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name}은 0보다 큰 유한한 숫자여야 합니다.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name}은 0보다 큰 유한한 숫자여야 합니다.")
    return result


def _movement(
    source_a: tuple[int, int, int],
    source_b: tuple[int, int, int],
    candidate_a: tuple[int, int, int],
    candidate_b: tuple[int, int, int],
) -> tuple[float, float, float]:
    source_ok_a = np.asarray(srgb_to_oklab(source_a), dtype=float)
    source_ok_b = np.asarray(srgb_to_oklab(source_b), dtype=float)
    candidate_ok_a = np.asarray(srgb_to_oklab(candidate_a), dtype=float)
    candidate_ok_b = np.asarray(srgb_to_oklab(candidate_b), dtype=float)
    first = float(np.linalg.norm(candidate_ok_a - source_ok_a))
    second = float(np.linalg.norm(candidate_ok_b - source_ok_b))
    return first, second, first + second


def find_path_constrained_correction(
    midpoint_lab: Sequence[int | float],
    direction_lab: Sequence[int | float],
    source_rgb_a: Sequence[int | float],
    source_rgb_b: Sequence[int | float],
    target_simulated_delta_e00: Real,
    cvd_type: str,
    severity: Real,
    *,
    maximum_scale: Real = 256.0,
    grid_steps: int = 4096,
) -> dict[str, Any]:
    """동일 자극 경로의 유한 8-bit 후보 중 OKLab 이동 합 최소를 찾는다.

    source RGB도 같은 ``midpoint_lab``·``direction_lab``·grid에서 생성된 쌍이어야
    한다. 이 조건을 확인하지 못하면 임의 RGB에 한 방향의 C80을 쓰는 오류를
    막기 위해 보정을 거부한다. 실현가능성은 ``simulated_delta_e00 >= target``을
    그대로 사용하며, 자극 생성의 목표 근사 허용오차로 후보를 다시 자르지 않는다.
    """

    midpoint = _finite_color3(midpoint_lab, name="midpoint_lab")
    direction = _finite_color3(direction_lab, name="direction_lab")
    validated_a = validate_rgb(source_rgb_a)
    validated_b = validate_rgb(source_rgb_b)
    if any(not float(value).is_integer() for value in (*validated_a, *validated_b)):
        raise ValueError("source RGB는 실제 표시된 8-bit 정수여야 합니다.")
    source_a = tuple(int(value) for value in validated_a)
    source_b = tuple(int(value) for value in validated_b)
    if source_a == source_b:
        raise ValueError("source RGB 두 색은 서로 달라야 합니다.")
    target = _positive(target_simulated_delta_e00, "target_simulated_delta_e00")
    max_scale = _positive(maximum_scale, "maximum_scale")
    if isinstance(grid_steps, bool) or not isinstance(grid_steps, int) or grid_steps < 32:
        raise ValueError("grid_steps는 32 이상의 정수여야 합니다.")
    if cvd_type not in {"protan", "deutan"}:
        raise ValueError("cvd_type은 protan 또는 deutan이어야 합니다.")
    if isinstance(severity, bool) or not isinstance(severity, Real):
        raise ValueError("severity는 0~1 유한한 숫자여야 합니다.")
    severity_value = float(severity)
    if not math.isfinite(severity_value) or not 0 <= severity_value <= 1:
        raise ValueError("severity는 0~1 유한한 숫자여야 합니다.")

    norm = math.sqrt(sum(channel * channel for channel in direction))
    if norm <= 1e-12:
        raise ValueError("direction_lab은 영벡터일 수 없습니다.")
    unit = tuple(channel / norm for channel in direction)
    gamut_scale, boundary_found = _maximum_symmetric_gamut_scale(
        midpoint, unit, maximum_scale=max_scale
    )

    unique: dict[tuple[tuple[int, int, int], tuple[int, int, int]], dict[str, Any]] = {}
    for scale in np.linspace(0.0, gamut_scale, grid_steps + 1):
        candidate = _candidate(
            midpoint, unit, float(scale), cvd_type, severity_value
        )
        key = candidate["rgb_a"], candidate["rgb_b"]
        if key[0] == key[1]:
            continue
        previous = unique.get(key)
        if previous is None or float(candidate["scale"]) < float(previous["scale"]):
            unique[key] = candidate
    source = unique.get((source_a, source_b))
    if source is None:
        raise ValueError(
            "source RGB가 동결 중점·방향·grid에서 생성된 대칭 자극 쌍이 아닙니다."
        )

    source_distance = float(source["simulated_delta_e00"])
    feasible = [
        candidate
        for candidate in unique.values()
        if float(candidate["simulated_delta_e00"]) >= target
    ]
    if not feasible:
        reachable = max(float(item["simulated_delta_e00"]) for item in unique.values())
        raise ValueError(
            "같은 자극 경로와 sRGB 색역에서 목표 C80에 도달할 수 없습니다: "
            f"target={target:.6g}, reachable_max={reachable:.6g}"
        )

    ranked: list[
        tuple[
            float,
            float,
            tuple[int, int, int],
            tuple[int, int, int],
            dict[str, Any],
            float,
            float,
        ]
    ] = []
    for candidate in feasible:
        move_a, move_b, total = _movement(
            source_a,
            source_b,
            candidate["rgb_a"],
            candidate["rgb_b"],
        )
        ranked.append(
            (
                total,
                float(candidate["scale"]),
                candidate["rgb_a"],
                candidate["rgb_b"],
                candidate,
                move_a,
                move_b,
            )
        )
    total, _scale, _rgb_a, _rgb_b, best, move_a, move_b = min(ranked)
    achieved = float(best["simulated_delta_e00"])
    overshoot = achieved - target
    status = (
        "unchanged"
        if best["rgb_a"] == source_a and best["rgb_b"] == source_b
        else "corrected"
    )
    if achieved < target:  # pragma: no cover
        raise RuntimeError("내부 오류: 목표 C80 미만 후보가 feasible 집합에 포함됐습니다.")

    return {
        "status": status,
        "optimization_scope": "exact_over_finite_symmetric_path_grid_not_continuous_global",
        "behavioral_scope": "same_lab_midpoint_direction_as_c80_measurement",
        "source_rgb_a": source_a,
        "source_rgb_b": source_b,
        "corrected_rgb_a": best["rgb_a"],
        "corrected_rgb_b": best["rgb_b"],
        "source_simulated_delta_e00": source_distance,
        "target_simulated_delta_e00": target,
        "achieved_simulated_delta_e00": achieved,
        "target_overshoot_delta_e00": overshoot,
        "color_change_a_oklab": move_a,
        "color_change_b_oklab": move_b,
        "total_color_change_oklab": total,
        "continuous_lab_a": best["continuous_lab_a"],
        "continuous_lab_b": best["continuous_lab_b"],
        "displayed_lab_a": best["displayed_lab_a"],
        "displayed_lab_b": best["displayed_lab_b"],
        "simulated_rgb_a": best["simulated_rgb_a"],
        "simulated_rgb_b": best["simulated_rgb_b"],
        "simulated_lab_a": best["simulated_lab_a"],
        "simulated_lab_b": best["simulated_lab_b"],
        "search": {
            "method": "exhaustive_uniform_scale_grid_with_8bit_deduplication",
            "grid_steps": grid_steps,
            "unique_rgb_pairs_evaluated": len(unique),
            "feasible_pair_count": len(feasible),
            "maximum_scale": max_scale,
            "maximum_symmetric_gamut_scale": gamut_scale,
            "gamut_boundary_found": boundary_found,
            "constraint": "simulated_delta_e00 >= target_simulated_delta_e00",
            "constraint_boundary_tolerance_delta_e00": 0.0,
        },
    }


__all__ = ["find_path_constrained_correction"]
