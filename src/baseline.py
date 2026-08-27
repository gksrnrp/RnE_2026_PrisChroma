"""두 색용 규칙 기반 개인화 색보정 baseline 통합 함수."""

from __future__ import annotations

from collections.abc import Sequence

from config import ConfusionThresholds, SearchConfig
from confusion import is_confusing
from correction import find_minimum_correction
from distance import color_distance
from machado import simulate_color_float


def run_baseline(
    color_a: Sequence[int | float],
    color_b: Sequence[int | float],
    cvd_type: str,
    severity: float,
    thresholds: ConfusionThresholds,
    *,
    search_config: SearchConfig | None = None,
    palette: Sequence[Sequence[int | float]] | None = None,
) -> dict[str, object]:
    """혼동되지 않으면 유지하고, 혼동되면 최소 색이동을 탐색한다.

    ``palette``에 장면의 다른 색들을 넘기면 보정 결과가 그 색들과 새로
    혼동되지 않도록 재혼동 제약이 함께 적용된다.
    """
    before = is_confusing(
        color_a,
        color_b,
        cvd_type,
        severity,
        thresholds.original_min,
        thresholds.simulated_max,
    )
    original_a = tuple(int(round(float(value))) for value in color_a)
    original_b = tuple(int(round(float(value))) for value in color_b)

    if not bool(before["confusing"]):
        return {
            "status": "unchanged",
            "reason": "이 임계값 설정에서는 색각이상 때문에 생긴 혼동 쌍이 아닙니다.",
            "input": {
                "color_A": original_a,
                "color_B": original_b,
                "cvd_type": cvd_type,
                "severity": severity,
            },
            "before": before,
            "corrected_A": original_a,
            "corrected_B": original_b,
            "after": before,
            "correction": None,
        }

    correction = find_minimum_correction(
        color_a,
        color_b,
        cvd_type,
        severity,
        thresholds.target,
        search_config=search_config,
        palette=palette,
    )
    corrected_a = correction["corrected_A"]
    corrected_b = correction["corrected_B"]
    simulated_a = simulate_color_float(corrected_a, cvd_type, severity)
    simulated_b = simulate_color_float(corrected_b, cvd_type, severity)
    simulated_distance = color_distance(simulated_a, simulated_b, method="ciede2000")
    after = {
        "original_distance": color_distance(corrected_a, corrected_b, method="ciede2000"),
        "simulated_distance": simulated_distance,
        "target_threshold": thresholds.target,
        "target_met": simulated_distance >= thresholds.target,
        "simulated_A": simulated_a,
        "simulated_B": simulated_b,
    }
    return {
        "status": "corrected" if correction["success"] else "search_failed",
        "reason": correction["reason"],
        "input": {
            "color_A": original_a,
            "color_B": original_b,
            "cvd_type": cvd_type,
            "severity": severity,
        },
        "before": before,
        "corrected_A": corrected_a,
        "corrected_B": corrected_b,
        "after": after,
        "correction": correction,
    }


__all__ = ["run_baseline"]
