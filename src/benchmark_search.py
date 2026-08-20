"""최소 색이동 탐색의 step 의존성을 재현 가능하게 측정한다.

연구 보고서에 탐색 안정성 수치를 사용할 때는 이 파일이 저장한 JSON과
함께 random seed, 색쌍, 실행 환경, 편차 계산식을 보관한다.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from config import SearchConfig
from correction import find_minimum_correction
from distance import color_distance
from machado import simulate_color_float


DEFAULT_SEED = 20260820
DEFAULT_STEPS = (0.02, 0.01, 0.005)
ORIGINAL_MIN = 20.0
SIMULATED_MAX = 15.0
CVD_TYPE = "deutan"
SEVERITY = 0.8
TARGET = 15.0


def _pair_distances(
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
) -> tuple[float, float]:
    original = color_distance(color_a, color_b, method="ciede2000")
    simulated = color_distance(
        simulate_color_float(color_a, CVD_TYPE, SEVERITY),
        simulate_color_float(color_b, CVD_TYPE, SEVERITY),
        method="ciede2000",
    )
    return original, simulated


def generate_confusing_pairs(
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    """고정 seed로 원본은 다르고 시뮬레이션 후 가까운 색쌍을 만든다."""
    rng = np.random.default_rng(seed)
    pairs: list[dict[str, object]] = []
    attempts = 0
    maximum_attempts = max(10_000, count * 10_000)
    while len(pairs) < count and attempts < maximum_attempts:
        attempts += 1
        colors = rng.integers(0, 256, size=(2, 3))
        color_a = tuple(int(value) for value in colors[0])
        color_b = tuple(int(value) for value in colors[1])
        original, simulated = _pair_distances(color_a, color_b)
        if original <= ORIGINAL_MIN or simulated >= SIMULATED_MAX:
            continue
        pairs.append(
            {
                "color_A": color_a,
                "color_B": color_b,
                "original_distance": original,
                "simulated_distance": simulated,
            }
        )
    if len(pairs) != count:
        raise RuntimeError(
            f"{maximum_attempts}번 안에 혼동 색쌍 {count}개를 만들지 못했습니다."
        )
    return pairs


def _relative_spread_percent(values: list[float]) -> float | None:
    """최솟값 대비 step 결과 범위의 비율을 백분율로 계산한다."""
    best = min(values)
    if best <= 1e-15:
        return 0.0 if max(values) <= 1e-15 else None
    return 100.0 * (max(values) - best) / best


def run_benchmark(
    pair_count: int,
    seed: int,
    steps: tuple[float, ...],
    max_shift: float,
) -> dict[str, object]:
    pairs = generate_confusing_pairs(pair_count, seed)
    records: list[dict[str, object]] = []
    spreads: list[float] = []
    undefined_spread_count = 0
    nonmonotonic_count = 0

    # 큰 step에서 작은 step 순으로 비교해야 "더 촘촘해졌는데 악화"를 찾을 수 있다.
    ordered_steps = tuple(sorted(steps, reverse=True))
    for pair in pairs:
        results: dict[str, object] = {}
        totals: list[float] = []
        for step in ordered_steps:
            result = find_minimum_correction(
                pair["color_A"],
                pair["color_B"],
                CVD_TYPE,
                SEVERITY,
                TARGET,
                search_config=SearchConfig(step=step, max_shift=max_shift),
            )
            if not bool(result["success"]):
                raise RuntimeError(
                    f"색쌍 {pair['color_A']}/{pair['color_B']}에서 "
                    f"step={step:g} 탐색에 실패했습니다."
                )
            total = float(result["total_color_change"])
            totals.append(total)
            results[f"{step:g}"] = {
                "success": bool(result["success"]),
                "strategy": result["strategy"],
                "corrected_A": result["corrected_A"],
                "corrected_B": result["corrected_B"],
                "total_color_change": total,
                "simulated_distance": float(result["simulated_distance"]),
            }

        spread = _relative_spread_percent(totals)
        worsened_when_finer = any(
            totals[index + 1] > totals[index] + 1e-12
            for index in range(len(totals) - 1)
        )
        if spread is None:
            undefined_spread_count += 1
        else:
            spreads.append(spread)
        nonmonotonic_count += int(worsened_when_finer)
        records.append(
            {
                **pair,
                "relative_spread_percent": spread,
                "worsened_when_finer": worsened_when_finer,
                "results": results,
            }
        )

    spread_array = np.asarray(spreads, dtype=float)
    if spread_array.size:
        median_spread: float | None = float(np.median(spread_array))
        mean_spread: float | None = float(np.mean(spread_array))
        max_spread: float | None = float(np.max(spread_array))
        pairs_over_five = int(np.count_nonzero(spread_array > 5.0))
    else:
        median_spread = None
        mean_spread = None
        max_spread = None
        pairs_over_five = 0
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "configuration": {
            "random_seed": seed,
            "pair_count": pair_count,
            "cvd_type": CVD_TYPE,
            "severity": SEVERITY,
            "original_min": ORIGINAL_MIN,
            "simulated_max": SIMULATED_MAX,
            "target": TARGET,
            "steps_coarse_to_fine": ordered_steps,
            "max_shift": max_shift,
            "spread_definition": "100 * (max(total)-min(total)) / min(total)",
        },
        "summary": {
            "median_relative_spread_percent": median_spread,
            "mean_relative_spread_percent": mean_spread,
            "max_relative_spread_percent": max_spread,
            "pairs_over_5_percent": pairs_over_five,
            "undefined_spread_pairs": undefined_spread_count,
            "nonmonotonic_pairs": nonmonotonic_count,
        },
        "pairs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="최소 색이동 step 안정성 benchmark")
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", nargs="+", type=float, default=DEFAULT_STEPS)
    parser.add_argument("--max-shift", type=float, default=0.30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs는 1 이상이어야 합니다.")
    if not args.steps:
        parser.error("--steps에는 값이 하나 이상 있어야 합니다.")

    report = run_benchmark(
        args.pairs,
        args.seed,
        tuple(args.steps),
        args.max_shift,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is None:
        print(text)
        return
    args.output.write_text(text + "\n", encoding="utf-8")
    print(f"benchmark 결과 저장: {args.output.resolve()}")


if __name__ == "__main__":
    main()
