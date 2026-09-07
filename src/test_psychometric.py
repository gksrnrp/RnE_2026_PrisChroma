"""심리측정 함수와 C80 계산 검사."""

import numpy as np
import pytest

from experiment.psychometric import (GUESS_RATE, c80, log_likelihood,
                                     p_correct, p_discriminate,
                                     simulate_responses, stimulus_levels,
                                     threshold_at_accuracy)


def test_guess_rate_is_quarter():
    # 4지선다이므로 찍어서 맞을 확률은 1/4 이다. 여기가 틀리면 모든 역치가 틀린다.
    assert GUESS_RATE == 0.25


def test_floor_and_ceiling():
    # 색차가 아주 작으면 찍기 확률, 아주 크면 1 - lapse 에 수렴한다.
    assert p_correct(-1e4, 3.0, 1.0, 0.05) == pytest.approx(0.25, abs=1e-6)
    assert p_correct(1e4, 3.0, 1.0, 0.05) == pytest.approx(0.95, abs=1e-6)


def test_monotone_increasing():
    x = np.linspace(0, 20, 200)
    p = p_correct(x, 4.0, 1.0, 0.03)
    assert np.all(np.diff(p) > -1e-12)


def test_c80_round_trips():
    # C80 에서의 정답률은 정의상 0.80 이어야 한다.
    for th, w, lp in [(1.5, 0.32, 0.02), (3.0, 1.0, 0.05), (6.0, 2.2, 0.05)]:
        x = c80(th, w, lp)
        assert p_correct(x, th, w, lp) == pytest.approx(0.80, abs=1e-9)


def test_theta_is_the_midpoint_of_discrimination():
    # theta 에서 순수 구별 확률은 0.5 다. 정답률 0.5 가 아니라는 점이 중요하다.
    assert p_discriminate(3.0, 3.0, 1.0) == pytest.approx(0.5, abs=1e-12)


def test_c80_unreachable_returns_none_not_garbage():
    # lapse 가 크면 정확도 기준 역치는 도달 불가능할 수 있다.
    # 그때 조용히 이상한 값을 내놓으면 안 된다.
    assert threshold_at_accuracy(3.0, 1.0, 0.12, p=0.99) is None
    # lapse 가 0.20 을 넘으면 천장이 0.80 아래이므로 C80 자체가 없다.
    assert c80(3.0, 1.0, 0.25) is None


def test_c80_grows_with_theta_and_w():
    base = c80(3.0, 1.0, 0.05)
    assert c80(6.0, 1.0, 0.05) > base      # 역치가 오른쪽으로
    assert c80(3.0, 2.0, 0.05) > base      # 곡선이 완만해도 C80 은 커진다


def test_stimulus_levels_bracket_the_guess():
    lv = stimulus_levels(4.0, n_levels=8)
    assert len(lv) == 8
    assert lv[0] < 4.0 < lv[-1]            # 예상 C80 을 사이에 둔다
    assert np.all(np.diff(lv) > 0)
    assert lv[0] > 0


def test_log_likelihood_prefers_the_truth():
    rng = np.random.default_rng(0)
    x = np.repeat(stimulus_levels(4.0, n_levels=8), 60)
    y = simulate_responses(x, theta=3.0, w=1.0, lapse=0.05, rng=rng)
    truth = log_likelihood(x, y, 3.0, 1.0, 0.05)
    for bad in [(1.0, 1.0, 0.05), (9.0, 1.0, 0.05), (3.0, 4.7, 0.05)]:
        assert truth > log_likelihood(x, y, *bad)


def test_simulate_responses_hits_expected_rate():
    rng = np.random.default_rng(1)
    x = np.full(6000, 3.0)
    y = simulate_responses(x, 3.0, 1.0, 0.05, rng)
    assert y.mean() == pytest.approx(p_correct(3.0, 3.0, 1.0, 0.05), abs=0.02)
