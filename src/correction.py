"""혼동 색쌍의 규칙 기반 최소 색이동 탐색.

초기 baseline은 OKLab radial grid로 넓게 탐색한 뒤, 이동량이 작은 여러
후보 주변의 8-bit RGB 정수색을 합쳐 다시 확인하는 결정론적 multi-start
탐색을 사용한다. 양쪽 이동은 개선이 멈출 때까지 새 해 주변을 반복 탐색한다.
반환되는 결과는 연속 공간의 수학적 전역 최적해가 아니라, 지정된 탐색과
정밀화 범위에서 발견한 최소 변화 해이다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from itertools import product

import numpy as np

from color_spaces import srgb_to_oklab, validate_rgb
from config import SearchConfig
from distance import color_distance, delta_e_2000
from machado import get_machado_matrix, simulate_color_float


_EPSILON = 1e-12
_SINGLE_RGB_REFINEMENT_RADIUS = 16
_PAIR_RGB_REFINEMENT_RADIUS = 4
_SINGLE_REFINEMENT_STARTS = 16
_PAIR_REFINEMENT_STARTS = 3
_PAIR_REFINEMENT_MAX_PASSES = 20
_RECONFUSION_CANDIDATE_CHUNK = 1024
_RECONFUSION_PALETTE_CHUNK = 128
_RECONFUSION_TOLERANCE = 1e-9


def _srgb_array_to_linear(rgb_255: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb_255, dtype=float) / 255.0
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def _linear_array_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(linear_rgb, dtype=float), 0.0, 1.0)
    return 255.0 * np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * values ** (1.0 / 2.4) - 0.055,
    )


def _srgb_array_to_oklab(rgb_255: np.ndarray) -> np.ndarray:
    linear = _srgb_array_to_linear(rgb_255)
    l = 0.4122214708 * linear[..., 0] + 0.5363325363 * linear[..., 1] + 0.0514459929 * linear[..., 2]
    m = 0.2119034982 * linear[..., 0] + 0.6806995451 * linear[..., 1] + 0.1073969566 * linear[..., 2]
    s = 0.0883024619 * linear[..., 0] + 0.2817188376 * linear[..., 1] + 0.6299787005 * linear[..., 2]
    roots = np.cbrt(np.stack((l, m, s), axis=-1))
    return np.stack(
        (
            0.2104542553 * roots[..., 0] + 0.7936177850 * roots[..., 1] - 0.0040720468 * roots[..., 2],
            1.9779984951 * roots[..., 0] - 2.4285922050 * roots[..., 1] + 0.4505937099 * roots[..., 2],
            0.0259040371 * roots[..., 0] + 0.7827717662 * roots[..., 1] - 0.8086757660 * roots[..., 2],
        ),
        axis=-1,
    )


def _oklab_array_to_linear(oklab: np.ndarray) -> np.ndarray:
    values = np.asarray(oklab, dtype=float)
    lightness, a, b = values[..., 0], values[..., 1], values[..., 2]
    l_root = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_root = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_root = lightness - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_root**3, m_root**3, s_root**3
    return np.stack(
        (
            4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ),
        axis=-1,
    )


def _srgb_array_to_lab(rgb_255: np.ndarray) -> np.ndarray:
    linear = _srgb_array_to_linear(rgb_255)
    x = 0.4124564 * linear[..., 0] + 0.3575761 * linear[..., 1] + 0.1804375 * linear[..., 2]
    y = 0.2126729 * linear[..., 0] + 0.7151522 * linear[..., 1] + 0.0721750 * linear[..., 2]
    z = 0.0193339 * linear[..., 0] + 0.1191920 * linear[..., 1] + 0.9503041 * linear[..., 2]
    xyz = np.stack((x / 0.95047, y, z / 1.08883), axis=-1)
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        (
            116.0 * transformed[..., 1] - 16.0,
            500.0 * (transformed[..., 0] - transformed[..., 1]),
            200.0 * (transformed[..., 1] - transformed[..., 2]),
        ),
        axis=-1,
    )


def _simulate_array_to_lab(
    rgb_255: np.ndarray, cvd_type: str, severity: float
) -> np.ndarray:
    matrix = np.asarray(get_machado_matrix(cvd_type, severity), dtype=float)
    linear = _srgb_array_to_linear(rgb_255)
    simulated_linear = np.clip(linear @ matrix.T, 0.0, 1.0)
    return _srgb_array_to_lab(_linear_array_to_srgb(simulated_linear))


@lru_cache(maxsize=None)
def _search_directions(grid_radius: int) -> np.ndarray:
    """정수 격자에서 중복된 방향을 제거한 단위벡터 집합을 만든다."""
    unique: dict[tuple[float, float, float], np.ndarray] = {}
    values = range(-grid_radius, grid_radius + 1)
    for vector_tuple in product(values, repeat=3):
        vector = np.asarray(vector_tuple, dtype=float)
        length = float(np.linalg.norm(vector))
        if length == 0.0:
            continue
        direction = vector / length
        key = tuple(float(value) for value in np.round(direction, 12))
        unique[key] = direction
    return np.stack(list(unique.values()), axis=0)


def _make_candidates(
    rgb: Sequence[int | float], search: SearchConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """후보 8-bit RGB, OKLab, 이동량, 원본 OKLab을 반환한다."""
    original_rgb = np.asarray(validate_rgb(rgb), dtype=float)
    original_oklab = np.asarray(srgb_to_oklab(original_rgb), dtype=float)
    directions = _search_directions(search.direction_grid_radius)
    radii = np.arange(search.step, search.max_shift + search.step * 0.5, search.step)
    offsets = (directions[:, None, :] * radii[None, :, None]).reshape(-1, 3)
    proposed_oklab = np.concatenate((original_oklab[None, :], original_oklab + offsets), axis=0)

    proposed_linear = _oklab_array_to_linear(proposed_oklab)
    valid = np.all((proposed_linear >= -1e-10) & (proposed_linear <= 1.0 + 1e-10), axis=1)
    valid_linear = np.clip(proposed_linear[valid], 0.0, 1.0)
    rgb_8bit = np.rint(_linear_array_to_srgb(valid_linear)).astype(np.uint8)

    # 여러 연속 후보가 같은 8-bit 색이 되면 한 번만 계산한다.
    rgb_8bit = np.unique(rgb_8bit, axis=0)
    actual_oklab = _srgb_array_to_oklab(rgb_8bit)
    movements = np.linalg.norm(actual_oklab - original_oklab, axis=1)
    order = np.argsort(movements, kind="stable")
    return rgb_8bit[order], actual_oklab[order], movements[order], original_oklab


def _python_rgb(rgb: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(value) for value in rgb)  # type: ignore[return-value]


_ACHROMATIC_CHROMA = 1e-4


def _rgb_in_srgb_gamut(rgb: Sequence[int | float]) -> bool:
    """반환 색이 실제로 표현 가능한 8-bit sRGB 값인지 확인한다.

    후보 생성 단계에서 gamut 밖 좌표를 걸러내므로 정상 동작에서는 항상
    참이지만, 이 값을 하드코딩하면 후보 생성이 망가져도 결과에는
    ``in_srgb_gamut=True``가 그대로 남는다. 그래서 반환 직전에 다시
    확인한다.
    """
    return all(
        float(channel).is_integer() and 0.0 <= float(channel) <= 255.0
        for channel in rgb
    )


def _hue_shift_degrees(
    original_oklab: np.ndarray, corrected_oklab: np.ndarray
) -> float:
    """OKLab 색상각이 몇 도 회전했는지 -180~180 범위로 반환한다.

    무채색에 가까우면 색상각이 정의되지 않으므로 0.0을 반환한다.
    """
    chroma_before = float(np.hypot(original_oklab[1], original_oklab[2]))
    chroma_after = float(np.hypot(corrected_oklab[1], corrected_oklab[2]))
    if min(chroma_before, chroma_after) < _ACHROMATIC_CHROMA:
        return 0.0
    before = math.degrees(math.atan2(original_oklab[2], original_oklab[1]))
    after = math.degrees(math.atan2(corrected_oklab[2], corrected_oklab[1]))
    return (after - before + 180.0) % 360.0 - 180.0


def _make_result(
    strategy: str,
    rgb_a: np.ndarray,
    rgb_b: np.ndarray,
    oklab_a: np.ndarray,
    oklab_b: np.ndarray,
    original_oklab_a: np.ndarray,
    original_oklab_b: np.ndarray,
    simulated_distance: float,
) -> dict[str, object]:
    delta_a = oklab_a - original_oklab_a
    delta_b = oklab_b - original_oklab_b
    change_a = float(np.linalg.norm(delta_a))
    change_b = float(np.linalg.norm(delta_b))
    corrected_a = _python_rgb(rgb_a)
    corrected_b = _python_rgb(rgb_b)
    hue_shift_a = _hue_shift_degrees(original_oklab_a, oklab_a)
    hue_shift_b = _hue_shift_degrees(original_oklab_b, oklab_b)
    return {
        "strategy": strategy,
        "corrected_A": corrected_a,
        "corrected_B": corrected_b,
        "delta_A": tuple(float(value) for value in delta_a),
        "delta_B": tuple(float(value) for value in delta_b),
        "color_change_A": change_a,
        "color_change_B": change_b,
        "total_color_change": change_a + change_b,
        "hue_shift_A_degrees": hue_shift_a,
        "hue_shift_B_degrees": hue_shift_b,
        "max_hue_shift_degrees": max(abs(hue_shift_a), abs(hue_shift_b)),
        "simulated_distance": float(simulated_distance),
        "in_srgb_gamut": (
            _rgb_in_srgb_gamut(corrected_a) and _rgb_in_srgb_gamut(corrected_b)
        ),
    }


def _prepare_palette(
    palette: Sequence[Sequence[int | float]] | None,
    original_a: np.ndarray,
    original_b: np.ndarray,
) -> np.ndarray:
    """재혼동 검사에 쓸 주변 색들을 8-bit 배열로 정규화한다.

    보정 대상인 A, B 자신은 제외한다. 포함하면 "원본에서 멀어져라"라는
    잘못된 제약이 되어 불필요하게 큰 이동을 강제한다.
    """
    if palette is None:
        return np.empty((0, 3), dtype=np.uint8)
    rows = [
        np.rint(np.asarray(validate_rgb(color), dtype=float)).astype(np.uint8)
        for color in palette
    ]
    if not rows:
        return np.empty((0, 3), dtype=np.uint8)
    stacked = np.unique(np.stack(rows, axis=0), axis=0)
    excluded = np.any(
        np.all(stacked[:, None, :] == np.stack((original_a, original_b))[None, :, :], axis=2),
        axis=1,
    )
    return stacked[~excluded]


def _reconfusion_mask(
    simulated_lab: np.ndarray,
    palette_lab: np.ndarray,
    target: float,
    original_simulated_lab: np.ndarray,
) -> np.ndarray:
    """새 팔레트 충돌을 막고 기존 충돌은 더 악화시키지 않는다.

    원래부터 ``target`` 미만이던 제3색까지 억지로 해결하면 현재 색쌍의
    보정 비용에 다른 문제의 해결 비용이 섞인다. 따라서 제3색별 하한은
    ``min(보정 전 거리, target)`` 이다. 원래 안전했던 관계에는 target을,
    이미 가까웠던 관계에는 보정 전 거리를 적용한다.
    """
    if palette_lab.shape[0] == 0:
        return np.ones(simulated_lab.shape[0], dtype=bool)

    # 정밀 탐색은 후보가 수만 개까지 늘 수 있다. 후보×팔레트 전체 거리
    # 행렬을 한 번에 만들면 실제 이미지 팔레트에서 메모리가 급증하므로
    # 양쪽 축을 나누어 후보별 최근접 거리만 유지한다.
    original_distances = np.asarray(
        delta_e_2000(
            np.asarray(original_simulated_lab, dtype=float)[None, :],
            palette_lab,
        )
    ).reshape(-1)
    required_distances = np.minimum(original_distances, target)
    safe = np.ones(simulated_lab.shape[0], dtype=bool)
    for candidate_start in range(
        0,
        simulated_lab.shape[0],
        _RECONFUSION_CANDIDATE_CHUNK,
    ):
        candidate_chunk = simulated_lab[
            candidate_start : candidate_start + _RECONFUSION_CANDIDATE_CHUNK
        ]
        chunk_safe = np.ones(candidate_chunk.shape[0], dtype=bool)
        for palette_start in range(
            0,
            palette_lab.shape[0],
            _RECONFUSION_PALETTE_CHUNK,
        ):
            palette_chunk = palette_lab[
                palette_start : palette_start + _RECONFUSION_PALETTE_CHUNK
            ]
            distances = np.asarray(
                delta_e_2000(
                    candidate_chunk[:, None, :],
                    palette_chunk[None, :, :],
                )
            )
            required = required_distances[
                palette_start : palette_start + palette_chunk.shape[0]
            ]
            chunk_safe &= np.all(
                distances + _RECONFUSION_TOLERANCE >= required[None, :],
                axis=1,
            )
            if not np.any(chunk_safe):
                break
        safe[
            candidate_start : candidate_start + candidate_chunk.shape[0]
        ] = chunk_safe
    return safe


def _rgb_cube(
    center: Sequence[int | float], radius: int
) -> np.ndarray:
    """중심색 주변의 모든 8-bit RGB 정수 후보를 만든다."""
    center_rgb = np.rint(np.asarray(validate_rgb(center), dtype=float)).astype(int)
    axes = [
        np.arange(
            max(0, int(channel) - radius),
            min(255, int(channel) + radius) + 1,
            dtype=np.int16,
        )
        for channel in center_rgb
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, 3).astype(np.uint8)


def _refine_single_candidates(
    initials: Sequence[dict[str, object]],
    *,
    moving_a: bool,
    original_a: np.ndarray,
    original_b: np.ndarray,
    origin_a: np.ndarray,
    origin_b: np.ndarray,
    fixed_simulated_lab: np.ndarray,
    original_moving_simulated_lab: np.ndarray,
    cvd_type: str,
    severity: float,
    target: float,
    palette_lab: np.ndarray,
) -> tuple[dict[str, object], int]:
    """여러 radial grid 해 주변을 합쳐 단일 이동 해를 정밀화한다."""
    if not initials:
        raise ValueError("단일 이동 정밀 탐색에는 초기 후보가 필요합니다.")
    initial = min(
        initials,
        key=lambda result: float(result["total_color_change"]),
    )
    key = "corrected_A" if moving_a else "corrected_B"
    rgb_candidates = np.unique(
        np.concatenate(
            [
                _rgb_cube(result[key], _SINGLE_RGB_REFINEMENT_RADIUS)
                for result in initials
            ],
            axis=0,
        ),
        axis=0,
    )
    oklab_candidates = _srgb_array_to_oklab(rgb_candidates)
    origin = origin_a if moving_a else origin_b
    movements = np.linalg.norm(oklab_candidates - origin, axis=1)
    simulated_lab = _simulate_array_to_lab(rgb_candidates, cvd_type, severity)
    safe = _reconfusion_mask(
        simulated_lab,
        palette_lab,
        target,
        original_moving_simulated_lab,
    )
    distances = np.asarray(delta_e_2000(simulated_lab, fixed_simulated_lab))
    feasible = np.flatnonzero(
        (movements > _EPSILON) & safe & (distances >= target)
    )
    if not feasible.size:
        return initial, int(rgb_candidates.shape[0])

    best_index = int(feasible[np.argmin(movements[feasible])])
    if moving_a:
        refined = _make_result(
            "A_only",
            rgb_candidates[best_index],
            original_b,
            oklab_candidates[best_index],
            origin_b,
            origin_a,
            origin_b,
            float(distances[best_index]),
        )
    else:
        refined = _make_result(
            "B_only",
            original_a,
            rgb_candidates[best_index],
            origin_a,
            oklab_candidates[best_index],
            origin_a,
            origin_b,
            float(distances[best_index]),
        )

    if float(refined["total_color_change"]) < float(initial["total_color_change"]):
        return refined, int(rgb_candidates.shape[0])
    return initial, int(rgb_candidates.shape[0])


def _search_pair_pool(
    rgb_a: np.ndarray,
    rgb_b: np.ndarray,
    initial: dict[str, object],
    *,
    origin_a: np.ndarray,
    origin_b: np.ndarray,
    original_simulated_lab_a: np.ndarray,
    original_simulated_lab_b: np.ndarray,
    cvd_type: str,
    severity: float,
    target: float,
    palette_lab: np.ndarray,
    pair_chunk_size: int,
) -> dict[str, object]:
    """두 RGB 후보 집합에서 현재 해보다 작은 양쪽 이동 해를 찾는다."""
    oklab_a = _srgb_array_to_oklab(rgb_a)
    oklab_b = _srgb_array_to_oklab(rgb_b)
    movement_a = np.linalg.norm(oklab_a - origin_a, axis=1)
    movement_b = np.linalg.norm(oklab_b - origin_b, axis=1)
    simulated_lab_a = _simulate_array_to_lab(rgb_a, cvd_type, severity)
    simulated_lab_b = _simulate_array_to_lab(rgb_b, cvd_type, severity)
    safe_a = _reconfusion_mask(
        simulated_lab_a, palette_lab, target, original_simulated_lab_a
    )
    safe_b = _reconfusion_mask(
        simulated_lab_b, palette_lab, target, original_simulated_lab_b
    )
    moving_a = np.flatnonzero((movement_a > _EPSILON) & safe_a)
    moving_b = np.flatnonzero((movement_b > _EPSILON) & safe_b)
    if not moving_a.size or not moving_b.size:
        return initial

    best_cost = float(initial["total_color_change"])
    best_pair: tuple[int, int, float] | None = None
    for start in range(0, moving_a.size, pair_chunk_size):
        a_indices = moving_a[start : start + pair_chunk_size]
        distances = np.asarray(
            delta_e_2000(
                simulated_lab_a[a_indices, None, :],
                simulated_lab_b[None, moving_b, :],
            )
        )
        costs = movement_a[a_indices, None] + movement_b[None, moving_b]
        feasible = (distances >= target) & (costs < best_cost)
        if not np.any(feasible):
            continue
        masked_costs = np.where(feasible, costs, np.inf)
        row, column = np.unravel_index(
            int(np.argmin(masked_costs)),
            masked_costs.shape,
        )
        best_cost = float(masked_costs[row, column])
        best_pair = (
            int(a_indices[row]),
            int(moving_b[column]),
            float(distances[row, column]),
        )

    if best_pair is None:
        return initial
    index_a, index_b, separation = best_pair
    return _make_result(
        "both",
        rgb_a[index_a],
        rgb_b[index_b],
        oklab_a[index_a],
        oklab_b[index_b],
        origin_a,
        origin_b,
        separation,
    )


def _refine_pair_candidates(
    initials: Sequence[dict[str, object]],
    *,
    origin_a: np.ndarray,
    origin_b: np.ndarray,
    original_simulated_lab_a: np.ndarray,
    original_simulated_lab_b: np.ndarray,
    cvd_type: str,
    severity: float,
    target: float,
    palette_lab: np.ndarray,
    pair_chunk_size: int,
) -> tuple[dict[str, object], int, int, int]:
    """multi-start 해를 합친 뒤 작은 RGB cube로 수렴할 때까지 정밀화한다."""
    if not initials:
        raise ValueError("양쪽 이동 정밀 탐색에는 초기 후보가 필요합니다.")
    both_initials = [result for result in initials if result["strategy"] == "both"]
    if not both_initials:
        raise ValueError("양쪽 이동 정밀 탐색에는 both 초기 후보가 필요합니다.")
    current = min(
        both_initials,
        key=lambda result: float(result["total_color_change"]),
    )
    rgb_a = np.unique(
        np.concatenate(
            [
                _rgb_cube(result["corrected_A"], _PAIR_RGB_REFINEMENT_RADIUS)
                for result in initials
            ],
            axis=0,
        ),
        axis=0,
    )
    rgb_b = np.unique(
        np.concatenate(
            [
                _rgb_cube(result["corrected_B"], _PAIR_RGB_REFINEMENT_RADIUS)
                for result in initials
            ],
            axis=0,
        ),
        axis=0,
    )
    count_a, count_b = int(rgb_a.shape[0]), int(rgb_b.shape[0])
    passes = 1
    current = _search_pair_pool(
        rgb_a,
        rgb_b,
        current,
        origin_a=origin_a,
        origin_b=origin_b,
        original_simulated_lab_a=original_simulated_lab_a,
        original_simulated_lab_b=original_simulated_lab_b,
        cvd_type=cvd_type,
        severity=severity,
        target=target,
        palette_lab=palette_lab,
        pair_chunk_size=pair_chunk_size,
    )

    # 첫 cube의 경계에서 해가 나오면 한 번의 정밀화만으로는 같은 방향으로
    # 더 이동할 수 없다. 새 해를 중심으로 다시 훑어 개선이 멈출 때 종료한다.
    for _ in range(1, _PAIR_REFINEMENT_MAX_PASSES):
        local_a = _rgb_cube(current["corrected_A"], _PAIR_RGB_REFINEMENT_RADIUS)
        local_b = _rgb_cube(current["corrected_B"], _PAIR_RGB_REFINEMENT_RADIUS)
        count_a += int(local_a.shape[0])
        count_b += int(local_b.shape[0])
        refined = _search_pair_pool(
            local_a,
            local_b,
            current,
            origin_a=origin_a,
            origin_b=origin_b,
            original_simulated_lab_a=original_simulated_lab_a,
            original_simulated_lab_b=original_simulated_lab_b,
            cvd_type=cvd_type,
            severity=severity,
            target=target,
            palette_lab=palette_lab,
            pair_chunk_size=pair_chunk_size,
        )
        passes += 1
        if float(refined["total_color_change"]) >= float(
            current["total_color_change"]
        ) - _EPSILON:
            break
        current = refined
    return current, count_a, count_b, passes


def find_minimum_correction(
    color_a: Sequence[int | float],
    color_b: Sequence[int | float],
    cvd_type: str,
    severity: float,
    target_threshold: float,
    *,
    search_config: SearchConfig | None = None,
    palette: Sequence[Sequence[int | float]] | None = None,
) -> dict[str, object]:
    """세 이동 전략을 비교해 탐색 범위 안의 최소 변화 보정색을 찾는다.

    구별 조건에는 Machado 시뮬레이션 후 CIEDE2000을, 최소화 목표에는
    두 색의 OKLab 이동 거리 합을 사용한다.

    ``palette``에 주변 색들을 넘기면 재혼동(re-confusion) 제약이 함께
    적용된다. 원래 target 이상이던 제3색 관계는 target 미만으로 만들지
    않고, 원래부터 target 미만이던 관계는 보정 전보다 더 가깝게 만들지
    않는다. 이 제약은 :func:`~confusion.is_confusing`으로는 대체할 수
    없다. 그 함수는 "원본에서는 달랐는가"를 함께 묻기 때문에, 보정이
    만들어낸 새 충돌은 색각이상 탓이 아니라는 이유로 통과시킨다.
    """
    if isinstance(target_threshold, bool) or not isinstance(target_threshold, (int, float)):
        raise TypeError("target_threshold는 0 이상의 숫자여야 합니다.")
    target = float(target_threshold)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("target_threshold는 0 이상의 유한한 값이어야 합니다.")

    search = search_config or SearchConfig()
    original_a = np.rint(np.asarray(validate_rgb(color_a), dtype=float)).astype(np.uint8)
    original_b = np.rint(np.asarray(validate_rgb(color_b), dtype=float)).astype(np.uint8)
    simulated_before = color_distance(
        simulate_color_float(original_a, cvd_type, severity),
        simulate_color_float(original_b, cvd_type, severity),
        method="ciede2000",
    )

    original_oklab_a = np.asarray(srgb_to_oklab(original_a), dtype=float)
    original_oklab_b = np.asarray(srgb_to_oklab(original_b), dtype=float)
    if simulated_before >= target:
        unchanged = _make_result(
            "unchanged",
            original_a,
            original_b,
            original_oklab_a,
            original_oklab_b,
            original_oklab_a,
            original_oklab_b,
            simulated_before,
        )
        return {
            "success": True,
            "reason": "두 색이 이미 target_threshold 이상으로 구별됩니다.",
            **unchanged,
            "target_threshold": target,
            "alternatives": {"A_only": None, "B_only": None, "both": None},
            "search": {
                "method": "radial_grid_with_multistart_rgb_refinement",
                "step": search.step,
                "max_shift": search.max_shift,
                "direction_grid_radius": search.direction_grid_radius,
                "candidate_count_A": 1,
                "candidate_count_B": 1,
                "palette_size": 0,
                "reconfusion_checked": False,
                "safe_candidate_count_A": 1,
                "safe_candidate_count_B": 1,
                "single_refinement_radius": _SINGLE_RGB_REFINEMENT_RADIUS,
                "pair_refinement_radius": _PAIR_RGB_REFINEMENT_RADIUS,
                "single_refinement_max_starts": _SINGLE_REFINEMENT_STARTS,
                "pair_refinement_max_starts": _PAIR_REFINEMENT_STARTS + 2,
                "pair_refinement_max_passes": _PAIR_REFINEMENT_MAX_PASSES,
                "pair_refinement_passes": 0,
                "refinement_start_counts": {
                    "A_only": 0,
                    "B_only": 0,
                    "both": 0,
                },
                "refinement_candidate_counts": {
                    "A_only": 0,
                    "B_only": 0,
                    "both_A": 0,
                    "both_B": 0,
                },
                "refinement_performed": False,
                "refinement_improved": False,
                "refinement_gain": {
                    "A_only": 0.0,
                    "B_only": 0.0,
                    "both": 0.0,
                },
                "search_skipped": True,
                "approximate": True,
            },
        }

    rgb_a, oklab_a, movement_a, origin_a = _make_candidates(original_a, search)
    rgb_b, oklab_b, movement_b, origin_b = _make_candidates(original_b, search)
    simulated_lab_a = _simulate_array_to_lab(rgb_a, cvd_type, severity)
    simulated_lab_b = _simulate_array_to_lab(rgb_b, cvd_type, severity)
    fixed_lab_a = _simulate_array_to_lab(original_a[None, :], cvd_type, severity)[0]
    fixed_lab_b = _simulate_array_to_lab(original_b[None, :], cvd_type, severity)[0]

    palette_rgb = _prepare_palette(palette, original_a, original_b)
    palette_lab = (
        _simulate_array_to_lab(palette_rgb.astype(float), cvd_type, severity)
        if palette_rgb.shape[0]
        else np.empty((0, 3), dtype=float)
    )
    safe_a = _reconfusion_mask(
        simulated_lab_a, palette_lab, target, fixed_lab_a
    )
    safe_b = _reconfusion_mask(
        simulated_lab_b, palette_lab, target, fixed_lab_b
    )

    alternatives: dict[str, dict[str, object] | None] = {
        "A_only": None,
        "B_only": None,
        "both": None,
    }
    refinement_seeds: dict[str, list[dict[str, object]]] = {
        "A_only": [],
        "B_only": [],
        "both": [],
    }

    distances_a_only = np.asarray(delta_e_2000(simulated_lab_a, fixed_lab_b))
    feasible_a = np.flatnonzero(
        (movement_a > _EPSILON) & safe_a & (distances_a_only >= target)
    )
    if feasible_a.size:
        ordered_a = feasible_a[np.argsort(movement_a[feasible_a], kind="stable")]
        for index in ordered_a[:_SINGLE_REFINEMENT_STARTS]:
            refinement_seeds["A_only"].append(
                _make_result(
                    "A_only",
                    rgb_a[index],
                    original_b,
                    oklab_a[index],
                    origin_b,
                    origin_a,
                    origin_b,
                    float(distances_a_only[index]),
                )
            )
        alternatives["A_only"] = refinement_seeds["A_only"][0]

    distances_b_only = np.asarray(delta_e_2000(fixed_lab_a, simulated_lab_b))
    feasible_b = np.flatnonzero(
        (movement_b > _EPSILON) & safe_b & (distances_b_only >= target)
    )
    if feasible_b.size:
        ordered_b = feasible_b[np.argsort(movement_b[feasible_b], kind="stable")]
        for index in ordered_b[:_SINGLE_REFINEMENT_STARTS]:
            refinement_seeds["B_only"].append(
                _make_result(
                    "B_only",
                    original_a,
                    rgb_b[index],
                    origin_a,
                    oklab_b[index],
                    origin_a,
                    origin_b,
                    float(distances_b_only[index]),
                )
            )
        alternatives["B_only"] = refinement_seeds["B_only"][0]

    moving_a = np.flatnonzero((movement_a > _EPSILON) & safe_a)
    moving_b = np.flatnonzero((movement_b > _EPSILON) & safe_b)
    pair_seed_candidates: list[tuple[float, int, int, float]] = []
    for start in range(0, moving_a.size, search.pair_chunk_size):
        a_indices = moving_a[start : start + search.pair_chunk_size]
        distances = np.asarray(
            delta_e_2000(
                simulated_lab_a[a_indices, None, :],
                simulated_lab_b[None, moving_b, :],
            )
        )
        costs = movement_a[a_indices, None] + movement_b[None, moving_b]
        masked_costs = np.where(distances >= target, costs, np.inf)
        finite_flat = np.flatnonzero(np.isfinite(masked_costs))
        if not finite_flat.size:
            continue
        if finite_flat.size > _PAIR_REFINEMENT_STARTS:
            finite_costs = masked_costs.ravel()[finite_flat]
            selected = finite_flat[
                np.argpartition(
                    finite_costs,
                    _PAIR_REFINEMENT_STARTS - 1,
                )[:_PAIR_REFINEMENT_STARTS]
            ]
        else:
            selected = finite_flat
        for flat_index in selected:
            row, column = np.unravel_index(int(flat_index), masked_costs.shape)
            pair_seed_candidates.append(
                (
                    float(masked_costs[row, column]),
                    int(a_indices[row]),
                    int(moving_b[column]),
                    float(distances[row, column]),
                )
            )

    pair_seed_candidates.sort(key=lambda item: item[0])
    for _, index_a, index_b, separation in pair_seed_candidates[
        :_PAIR_REFINEMENT_STARTS
    ]:
        refinement_seeds["both"].append(
            _make_result(
                "both",
                rgb_a[index_a],
                rgb_b[index_b],
                oklab_a[index_a],
                oklab_b[index_b],
                origin_a,
                origin_b,
                separation,
            )
        )
    if refinement_seeds["both"]:
        alternatives["both"] = refinement_seeds["both"][0]

    refinement_counts = {
        "A_only": 0,
        "B_only": 0,
        "both_A": 0,
        "both_B": 0,
    }
    pair_refinement_passes = 0
    # 정밀 탐색이 "실행됐는가"와 "결과를 실제로 줄였는가"는 다르다.
    # 후보 개수는 실행만 하면 항상 0보다 크므로, 개선 여부는 이동량을
    # 직접 비교해서 따로 기록한다.
    totals_before_refinement = {
        name: (None if result is None else float(result["total_color_change"]))
        for name, result in alternatives.items()
    }
    if alternatives["A_only"] is not None:
        alternatives["A_only"], refinement_counts["A_only"] = (
            _refine_single_candidates(
                refinement_seeds["A_only"],
                moving_a=True,
                original_a=original_a,
                original_b=original_b,
                origin_a=origin_a,
                origin_b=origin_b,
                fixed_simulated_lab=fixed_lab_b,
                original_moving_simulated_lab=fixed_lab_a,
                cvd_type=cvd_type,
                severity=severity,
                target=target,
                palette_lab=palette_lab,
            )
        )
    if alternatives["B_only"] is not None:
        alternatives["B_only"], refinement_counts["B_only"] = (
            _refine_single_candidates(
                refinement_seeds["B_only"],
                moving_a=False,
                original_a=original_a,
                original_b=original_b,
                origin_a=origin_a,
                origin_b=origin_b,
                fixed_simulated_lab=fixed_lab_a,
                original_moving_simulated_lab=fixed_lab_b,
                cvd_type=cvd_type,
                severity=severity,
                target=target,
                palette_lab=palette_lab,
            )
        )
    if alternatives["both"] is not None:
        # 양쪽 이동의 최적해가 단일 이동 해 바로 옆에 있는 경우가 많다.
        # 정밀화된 A_only/B_only도 cube 중심으로 사용하되, 둘 다 실제로
        # 움직인 후보만 both 결과로 채택한다.
        for name in ("A_only", "B_only"):
            if alternatives[name] is not None:
                refinement_seeds["both"].append(alternatives[name])
        (
            alternatives["both"],
            refinement_counts["both_A"],
            refinement_counts["both_B"],
            pair_refinement_passes,
        ) = _refine_pair_candidates(
            refinement_seeds["both"],
            origin_a=origin_a,
            origin_b=origin_b,
            original_simulated_lab_a=fixed_lab_a,
            original_simulated_lab_b=fixed_lab_b,
            cvd_type=cvd_type,
            severity=severity,
            target=target,
            palette_lab=palette_lab,
            pair_chunk_size=search.pair_chunk_size,
        )

    refinement_gain = {
        name: (
            0.0
            if before is None or alternatives[name] is None
            else before - float(alternatives[name]["total_color_change"])
        )
        for name, before in totals_before_refinement.items()
    }

    successful = [result for result in alternatives.values() if result is not None]
    search_metadata = {
        "method": "radial_grid_with_multistart_rgb_refinement",
        "step": search.step,
        "max_shift": search.max_shift,
        "direction_grid_radius": search.direction_grid_radius,
        "candidate_count_A": int(rgb_a.shape[0]),
        "candidate_count_B": int(rgb_b.shape[0]),
        "palette_size": int(palette_rgb.shape[0]),
        "reconfusion_checked": bool(palette_rgb.shape[0]),
        "reconfusion_policy": "prevent_new_or_worsened",
        "safe_candidate_count_A": int(np.count_nonzero(safe_a)),
        "safe_candidate_count_B": int(np.count_nonzero(safe_b)),
        "single_refinement_radius": _SINGLE_RGB_REFINEMENT_RADIUS,
        "pair_refinement_radius": _PAIR_RGB_REFINEMENT_RADIUS,
        "single_refinement_max_starts": _SINGLE_REFINEMENT_STARTS,
        "pair_refinement_max_starts": _PAIR_REFINEMENT_STARTS + 2,
        "pair_refinement_max_passes": _PAIR_REFINEMENT_MAX_PASSES,
        "pair_refinement_passes": pair_refinement_passes,
        "refinement_start_counts": {
            name: len(seeds) for name, seeds in refinement_seeds.items()
        },
        "refinement_candidate_counts": refinement_counts,
        # 정밀 탐색을 돌렸는지 여부.
        "refinement_performed": any(refinement_counts.values()),
        # 정밀 탐색이 총 이동량을 실제로 줄였는지 여부와 그 감소량.
        "refinement_improved": any(gain > 0.0 for gain in refinement_gain.values()),
        "refinement_gain": refinement_gain,
        "search_skipped": False,
        "approximate": True,
    }
    if not successful:
        return {
            "success": False,
            "reason": (
                "현재 탐색 범위에서 target_threshold를 만족하는 sRGB 색을 "
                "찾지 못했습니다. max_shift를 키워도 sRGB gamut이 먼저 후보를 "
                "제한하므로 거의 효과가 없습니다. step을 줄이거나 "
                "direction_grid_radius를 늘리고, 그래도 실패하면 "
                "target_threshold가 이 색쌍에 대해 과도한지 재검토하세요."
                + (
                    " 팔레트 재혼동 제약이 후보를 모두 배제했을 수도 있습니다."
                    if palette_rgb.shape[0]
                    else ""
                )
            ),
            "strategy": None,
            "corrected_A": _python_rgb(original_a),
            "corrected_B": _python_rgb(original_b),
            "delta_A": (0.0, 0.0, 0.0),
            "delta_B": (0.0, 0.0, 0.0),
            "color_change_A": 0.0,
            "color_change_B": 0.0,
            "total_color_change": 0.0,
            "hue_shift_A_degrees": 0.0,
            "hue_shift_B_degrees": 0.0,
            "max_hue_shift_degrees": 0.0,
            "simulated_distance": simulated_before,
            "target_threshold": target,
            "in_srgb_gamut": (
                _rgb_in_srgb_gamut(_python_rgb(original_a))
                and _rgb_in_srgb_gamut(_python_rgb(original_b))
            ),
            "alternatives": alternatives,
            "search": search_metadata,
        }

    best = min(successful, key=lambda result: float(result["total_color_change"]))
    return {
        "success": True,
        "reason": (
            "radial grid와 multi-start RGB 정밀 탐색에서 총 OKLab "
            "이동량이 가장 작은 결과입니다."
        ),
        **best,
        "target_threshold": target,
        "alternatives": alternatives,
        "search": search_metadata,
    }


__all__ = ["find_minimum_correction"]
