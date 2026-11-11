"""규칙 기반 baseline의 실험 변수.

색차 임계값은 아직 실험으로 확정되지 않았으므로 기본 정답값을 두지 않는다.
사용자 실험 또는 명시적인 실험 설정으로 반드시 전달해야 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real


@dataclass(frozen=True)
class ConfusionThresholds:
    """혼동 탐지와 보정 목표에 사용하는 ΔE00 임계값 묶음."""

    original_min: float
    simulated_max: float
    target: float

    def __post_init__(self) -> None:
        for name, value in (
            ("original_min", self.original_min),
            ("simulated_max", self.simulated_max),
            ("target", self.target),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name}은 0 이상의 숫자여야 합니다.")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name}은 0 이상의 유한한 값이어야 합니다.")
        if self.simulated_max > self.original_min:
            raise ValueError(
                "original_min은 simulated_max 이상이어야 합니다. "
                "그렇지 않으면 원래부터 비슷한 색쌍이 색각이상 때문에 혼동되는 "
                "것으로 잘못 분류됩니다."
            )
        if self.target < self.simulated_max:
            raise ValueError("target은 simulated_max 이상이어야 합니다.")


@dataclass(frozen=True)
class SearchConfig:
    """OKLab radial grid와 후속 multi-start RGB 정밀 탐색 설정.

    ``step``은 첫 radial grid의 거리 간격이다. grid 해 주변을 정수 RGB
    cube로 다시 훑되, 각 전략에서 이동량이 작은 여러 grid 후보를 시작점으로
    합쳐 탐색한다. 양쪽 이동은 새 해를 중심으로 개선이 멈출 때까지 반복한다.
    단일 시작점이나 한 번의 cube 경계에 갇히는 문제는 줄어들지만 전체
    색공간의 수학적 전역 최적해를 보장하지는 않는다.

    기본값은 보수적인 0.005로 유지한다. step별 안정성은 회귀 테스트로,
    더 넓은 표본의 편차는 고정 seed와 계산식을 기록하는
    ``benchmark_search.py``로 검증한다.
    """

    step: float = 0.005
    max_shift: float = 0.30
    direction_grid_radius: int = 2
    pair_chunk_size: int = 64

    def __post_init__(self) -> None:
        for name, value in (
            ("step", self.step),
            ("max_shift", self.max_shift),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name}은 유한한 숫자여야 합니다.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name}은 유한한 숫자여야 합니다.")

        if float(self.step) <= 0.0:
            raise ValueError("step은 0보다 커야 합니다.")
        if float(self.max_shift) < float(self.step):
            raise ValueError("max_shift는 step 이상이어야 합니다.")

        for name, value in (
            ("direction_grid_radius", self.direction_grid_radius),
            ("pair_chunk_size", self.pair_chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name}는 1 이상의 정수여야 합니다.")
            if int(value) < 1:
                raise ValueError(f"{name}는 1 이상의 정수여야 합니다.")

        # NumPy 숫자형을 받아도 이후 range()/JSON 출력에서 문제가 없도록
        # 내부 값은 파이썬 기본 숫자형으로 정규화한다.
        object.__setattr__(self, "step", float(self.step))
        object.__setattr__(self, "max_shift", float(self.max_shift))
        object.__setattr__(self, "direction_grid_radius", int(self.direction_grid_radius))
        object.__setattr__(self, "pair_chunk_size", int(self.pair_chunk_size))


__all__ = ["ConfusionThresholds", "SearchConfig"]
