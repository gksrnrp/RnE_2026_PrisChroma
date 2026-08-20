"""두 색이 색각이상 조건에서 혼동되는지 판정한다."""

from __future__ import annotations

import math
from collections.abc import Sequence

from distance import color_distance
from machado import simulate_color_float


def _validate_threshold(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name}은 0 이상의 숫자여야 합니다.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name}은 0 이상의 유한한 값이어야 합니다.")
    return result


def is_confusing(
    color_a: Sequence[int | float],
    color_b: Sequence[int | float],
    cvd_type: str,
    severity: float,
    original_threshold: float,
    confusion_threshold: float,
) -> dict[str, object]:
    """두 색이 색각이상 때문에 혼동되는 색쌍인지 판정한다.

    판정 조건은 다음 두 조건을 동시에 만족하는 것이다.

    1. 원본 ΔE00가 ``original_threshold``보다 크다.
    2. Machado 시뮬레이션 후 ΔE00가 ``confusion_threshold``보다 작다.

    임계값은 사용자 실험으로 조정해야 하며 이 함수는 임의의 기본값을
    사용하지 않는다.
    """
    original_limit = _validate_threshold("original_threshold", original_threshold)
    confusion_limit = _validate_threshold("confusion_threshold", confusion_threshold)

    simulated_a = simulate_color_float(color_a, cvd_type, severity)
    simulated_b = simulate_color_float(color_b, cvd_type, severity)
    original_distance = color_distance(color_a, color_b, method="ciede2000")
    simulated_distance = color_distance(simulated_a, simulated_b, method="ciede2000")

    return {
        "original_distance": original_distance,
        "simulated_distance": simulated_distance,
        "original_threshold": original_limit,
        "confusion_threshold": confusion_limit,
        "originally_distinct": original_distance > original_limit,
        "simulated_close": simulated_distance < confusion_limit,
        "confusing": (
            original_distance > original_limit
            and simulated_distance < confusion_limit
        ),
        "simulated_A": simulated_a,
        "simulated_B": simulated_b,
    }


__all__ = ["is_confusing"]
