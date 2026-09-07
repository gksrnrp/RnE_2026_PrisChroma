"""심리측정 함수와 C80 정의.

4지선다 강제선택 과제이므로 아무렇게나 찍어도 25% 는 맞는다. 실제로 색을
구별해서 맞히는 부분만 떼어내려면 그 몫을 빼고 봐야 한다.

    p(정답 | x) = γ + (1 − γ − λ) · F[(x − θ) / w],   γ = 1/4

    x  자극의 색차. **Machado 시뮬레이션 후 ΔE00** 로 표현한다.
    θ  위치. 곡선이 가운데를 지나는 지점.
    w  기울기 폭. 작을수록 급격히 갈린다.
    λ  lapse. 딴 짓·오조작으로 아무리 쉬워도 틀리는 비율.
    F  로지스틱

C80 은 **정답률이 0.80 이 되는 x** 다. 역산하면

    q = (0.80 − γ) / (1 − γ − λ)
    C80 = θ + w · ln(q / (1 − q))

주의: λ 가 0.20 을 넘으면 천장이 0.80 아래로 내려가 C80 이 존재하지 않는다.
그 경우 값을 지어내지 말고 None 을 돌려준다.
"""

from __future__ import annotations

import numpy as np

GUESS_RATE = 0.25          # 4지선다


def p_correct(x, theta, w, lapse, guess=GUESS_RATE):
    """정답 확률. x, theta, w, lapse 모두 broadcasting 된다."""
    z = np.clip((np.asarray(x, dtype=float) - theta) / w, -60.0, 60.0)
    return guess + (1.0 - guess - lapse) / (1.0 + np.exp(-z))


def p_discriminate(x, theta, w):
    """찍기와 실수를 뺀 순수 구별 확률. 임계값을 정의할 때 쓴다."""
    z = np.clip((np.asarray(x, dtype=float) - theta) / w, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def threshold_at_accuracy(theta, w, lapse, p=0.80, guess=GUESS_RATE):
    """정답률이 p 가 되는 색차. 천장을 넘으면 None."""
    ceiling = 1.0 - lapse
    if not (guess < p < ceiling):
        return None
    q = (p - guess) / (1.0 - guess - lapse)
    return float(theta + w * np.log(q / (1.0 - q)))


def c80(theta, w, lapse, guess=GUESS_RATE):
    """C80. 이 연구의 주 지표."""
    return threshold_at_accuracy(theta, w, lapse, 0.80, guess)


def log_likelihood(x, correct, theta, w, lapse, guess=GUESS_RATE):
    p = np.clip(p_correct(x, theta, w, lapse, guess), 1e-12, 1 - 1e-12)
    correct = np.asarray(correct, dtype=bool)
    return float(np.sum(np.where(correct, np.log(p), np.log1p(-p))))


def simulate_responses(x, theta, w, lapse, rng, guess=GUESS_RATE):
    """가상 참가자의 응답. parameter recovery 와 모의검증에 쓴다."""
    p = p_correct(x, theta, w, lapse, guess)
    return rng.random(np.shape(p)) < p


def stimulus_levels(c80_guess, n_levels=8, lo=0.35, hi=1.9):
    """고정자극(method of constant stimuli) 에 쓸 색차 수준.

    C80 추정이 목적이므로 곡선의 **윗부분**에 수준을 더 둔다. 역치 근처에만
    몰아 놓으면 정답률 0.80 지점이 외삽이 되어 불안정해진다.

    c80_guess 는 사전 정보(다른 굵기나 예비 측정)에서 온 대략값이면 된다.
    """
    return np.geomspace(c80_guess * lo, c80_guess * hi, int(n_levels))


__all__ = ["GUESS_RATE", "p_correct", "p_discriminate", "threshold_at_accuracy",
           "c80", "log_likelihood", "simulate_responses", "stimulus_levels"]
