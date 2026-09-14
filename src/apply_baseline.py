"""②단계 측정 결과를 이창민의 ``run_baseline()``에 연결하는 어댑터.

``baseline.py``는 색 두 개를 받는 함수다. 이 모듈은 SLIC이 뽑은 영역
목록을 그 입력 형태로 바꾸고, 결과를 영역 단위로 되돌린다.
``baseline.py``는 읽기만 하고 고치지 않는다.

임계값에는 안전한 기본값이 없다. 호출자는 합성 점검값인지 실제 실험으로
확정한 값인지 출처와 함께 명시적으로 넘겨야 한다.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from baseline import run_baseline
from config import ConfusionThresholds, SearchConfig


def select_candidate_pairs(
    pairs: list[dict],
    thresholds: ConfusionThresholds,
    *,
    condition_key: str,
    exclude_background: bool = True,
    min_area_px: int = 0,
    limit: int | None = None,
) -> list[dict]:
    """주어진 임계값 설정으로 보정 후보 쌍을 고른다.

    측정 단계에서 이미 계산해 둔 색차로 거르므로 run_baseline이 안에서
    다시 판정하는 것과 같은 결과가 나오되, 수천 쌍에 최소 색이동 탐색을
    돌리는 낭비를 막는다. 이 선별은 넘겨받은 설정값을 적용한 것이지
    연구가 정한 혼동 기준이 아니다.
    """
    selected = []
    for pair in pairs:
        if exclude_background and pair.get("involves_background"):
            continue
        if pair["area_min_px"] < min_area_px:
            continue
        simulated = pair.get(f"delta_e_sim_{condition_key}")
        if simulated is None:
            raise KeyError(
                f"측정 결과에 delta_e_sim_{condition_key} 열이 없습니다. "
                "measure_image의 cvd_conditions와 맞는지 확인하세요."
            )
        if (
            pair["delta_e_original"] > thresholds.original_min
            and simulated < thresholds.simulated_max
        ):
            selected.append(pair)

    selected.sort(key=lambda pair: pair[f"delta_e_sim_{condition_key}"])
    if limit is not None and len(selected) > limit:
        return selected[:limit]
    return selected


def correct_measurement(
    measurement: dict,
    *,
    cvd_type: str,
    severity: float,
    thresholds: ConfusionThresholds,
    search_config: SearchConfig | None = None,
    use_palette: bool = True,
    exclude_background: bool = True,
    limit: int | None = 40,
    verbose: bool = False,
) -> dict[str, object]:
    """측정 결과 한 장을 run_baseline에 통과시킨다.

    Returns:
        후보 쌍별 baseline 결과와, 영역별로 제안된 보정색 요약.
        한 영역이 여러 쌍에 걸려 서로 다른 색을 제안받는 경우가 있는데,
        그 충돌 횟수를 그대로 센다. 선택적 보정에서 영역 단위 결론을
        어떻게 낼지는 아직 정해진 문제가 아니기 때문이다.
    """
    condition_key = f"{cvd_type}_{int(round(float(severity) * 100)):03d}"
    regions = {
        record["region_id"]: record
        for record in measurement["regions"]
    }
    candidates = select_candidate_pairs(
        measurement["pairs"],
        thresholds,
        condition_key=condition_key,
        exclude_background=exclude_background,
        limit=limit,
    )

    palette = None
    if use_palette:
        palette = [
            (record["r"], record["g"], record["b"])
            for record in measurement["regions"]
        ]

    proposals: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    results = []
    for pair in candidates:
        color_a = _region_rgb(regions, pair["region_a"])
        color_b = _region_rgb(regions, pair["region_b"])
        outcome = run_baseline(
            color_a,
            color_b,
            cvd_type,
            severity,
            thresholds,
            search_config=search_config,
            palette=palette,
        )
        proposals[pair["region_a"]].append(tuple(outcome["corrected_A"]))
        proposals[pair["region_b"]].append(tuple(outcome["corrected_B"]))
        results.append(
            {
                "region_a": pair["region_a"],
                "region_b": pair["region_b"],
                "rgb_a": list(color_a),
                "rgb_b": list(color_b),
                "corrected_a": list(outcome["corrected_A"]),
                "corrected_b": list(outcome["corrected_B"]),
                "status": outcome["status"],
                "delta_e_original": pair["delta_e_original"],
                "delta_e_simulated_before": round(
                    float(outcome["before"]["simulated_distance"]), 4
                ),
                "delta_e_simulated_after": round(
                    float(outcome["after"]["simulated_distance"]), 4
                ),
                "target_met": int(bool(outcome["after"]["target_met"])),
                "stroke_min_px": pair["stroke_min_px"],
                "centroid_distance_px": pair["centroid_distance_px"],
            }
        )
        if verbose:
            print(
                f"  {pair['region_a']}-{pair['region_b']}: "
                f"{outcome['status']} "
                f"ΔE_sim {outcome['before']['simulated_distance']:.2f}"
                f" → {outcome['after']['simulated_distance']:.2f}"
            )

    conflicts = {
        region_id: [list(color) for color in sorted(set(colors))]
        for region_id, colors in proposals.items()
        if len(set(colors)) > 1
    }
    # 서로 다른 쌍에서 한 영역에 상충하는 색을 제안했다면 첫 제안을 임의로
    # 채택하지 않는다. 전역 팔레트 최적화가 없는 현재 단계에서는 해당 영역을
    # 미해결로 남기는 것이 유일하게 정직한 동작이다.
    region_colors = {
        region_id: list(colors[0])
        for region_id, colors in proposals.items()
        if len(set(colors)) == 1
    }
    return {
        "image": measurement["image"],
        "cvd_type": cvd_type,
        "severity": float(severity),
        "thresholds": {
            "original_min": thresholds.original_min,
            "simulated_max": thresholds.simulated_max,
            "target": thresholds.target,
            "note": (
                "호출자가 명시적으로 제공한 값이다. 실제 실험값인지 "
                "합성 점검값인지는 원자료 메타데이터로 별도 확인해야 한다."
            ),
        },
        "candidate_pair_count": len(candidates),
        "corrected_pair_count": sum(
            1 for record in results if record["status"] == "corrected"
        ),
        "target_met_count": sum(record["target_met"] for record in results),
        "region_conflict_count": len(conflicts),
        "region_conflicts": conflicts,
        "unresolved_region_ids": sorted(conflicts),
        "region_colors": region_colors,
        "pairs": results,
    }


def _region_rgb(regions: dict, region_id: int) -> tuple[int, int, int]:
    record = regions[region_id]
    return (record["r"], record["g"], record["b"])


def render_corrected_image(
    image: np.ndarray,
    labels: np.ndarray,
    region_colors: dict[int, list[int]],
) -> np.ndarray:
    """제안된 보정색으로 영역을 칠한 이미지를 만든다.

    영역 대표색을 통째로 갈아끼우므로 안티앨리어싱된 경계는 원본 그대로
    남는다. ``correct_measurement``에서 충돌한 영역은 ``region_colors``에
    포함되지 않아 원본으로 남는다. 보고서 진단 그림용이지 최종 보정
    결과물이 아니다.
    """
    corrected = image.copy()
    for region_id, color in region_colors.items():
        corrected[labels == region_id] = np.asarray(color, dtype=np.uint8)
    return corrected


__all__ = [
    "correct_measurement",
    "render_corrected_image",
    "select_candidate_pairs",
]
