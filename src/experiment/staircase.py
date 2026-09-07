"""검증 굵기에서 쓰는 적응형 계단법.

교정 측정은 고정자극(method of constant stimuli)을 쓴다. 심리측정 함수 전체를
적합해야 C80 이 나오기 때문이다. 반면 **검증**은 이미 예측한 C80 이 맞는지만
보면 되므로, 역치 근처에 시행을 모으는 계단법이 효율적이다.

1-up / 2-down 은 약 70.7% 정답 지점으로 수렴한다. C80 은 그보다 위이므로
계단으로 얻은 자료도 **심리측정 함수를 적합해서** C80 을 뽑는다. 계단은
자극을 어디에 놓을지 정하는 장치일 뿐이다.

세 세션을 독립적으로 돌리고 각각 따로 초기화한다. 시작점은 특정 모델에
유리하지 않도록 사전에 정한 공통 규칙을 쓴다 (연구계획서 §마).
"""

from __future__ import annotations

import numpy as np


class Staircase:
    """log 축 위의 1-up / 2-down."""

    def __init__(self, start, x_min, x_max, max_trials=120,
                 step_big=0.20, step_small=0.10, n_big=2):
        self.x = float(np.clip(start, x_min, x_max))
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.max_trials = int(max_trials)
        self.step_big, self.step_small, self.n_big = step_big, step_small, n_big
        self.n_correct_run = 0
        self.last_dir = 0
        self.reversals = []
        self.n_trials = 0
        self.history = []

    @property
    def step(self):
        return self.step_big if len(self.reversals) < self.n_big else self.step_small

    @property
    def done(self):
        return self.n_trials >= self.max_trials

    def update(self, correct):
        self.n_trials += 1
        self.history.append((self.x, bool(correct)))
        if correct:
            self.n_correct_run += 1
            if self.n_correct_run < 2:
                return
            self.n_correct_run = 0
            d = -1                      # 두 번 맞히면 더 어렵게
        else:
            self.n_correct_run = 0
            d = +1                      # 틀리면 쉽게
        if self.last_dir and d != self.last_dir:
            self.reversals.append(self.x)
        self.last_dir = d
        self.x = float(np.clip(10 ** (np.log10(max(self.x, 1e-6)) + d * self.step),
                               self.x_min, self.x_max))

    def levels(self):
        return np.array([h[0] for h in self.history])

    def responses(self):
        return np.array([h[1] for h in self.history], dtype=bool)


def default_start(expected_c80, x_max):
    """공통 시작점 규칙.

    예상 C80 의 1.6 배에서 시작해 내려온다. 특정 모델의 예측값에서 시작하면
    그 모델에 유리해지므로, **교정 자료의 개인 평균**에서만 계산한다.
    """
    return float(min(expected_c80 * 1.6, x_max * 0.9))


def default_range(expected_c80, x_max):
    """계단이 훑을 범위."""
    return float(max(expected_c80 * 0.15, 0.05)), float(x_max)


__all__ = ["Staircase", "default_start", "default_range"]
