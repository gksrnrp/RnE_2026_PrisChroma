"""적응형 계단법 검사."""

import numpy as np
import pytest

from experiment.psychometric import c80, p_correct, simulate_responses
from experiment.staircase import Staircase, default_range, default_start


def test_two_correct_go_down_one_wrong_goes_up():
    sc = Staircase(start=4.0, x_min=0.1, x_max=20.0)
    sc.update(True)
    assert sc.x == pytest.approx(4.0)      # 한 번 맞혀서는 안 내려간다
    sc.update(True)
    assert sc.x < 4.0                      # 두 번 맞히면 어려워진다
    here = sc.x
    sc.update(False)
    assert sc.x > here                     # 한 번 틀리면 쉬워진다


def test_correct_run_resets_after_a_miss():
    sc = Staircase(start=4.0, x_min=0.1, x_max=20.0)
    sc.update(True)
    sc.update(False)
    sc.update(True)
    here = sc.x
    sc.update(True)
    assert sc.x < here                     # 틀린 뒤 다시 두 번 맞혀야 내려간다


def test_stays_inside_range():
    sc = Staircase(start=4.0, x_min=1.0, x_max=6.0)
    for _ in range(80):
        sc.update(True)
    assert sc.x >= 1.0
    for _ in range(80):
        sc.update(False)
    assert sc.x <= 6.0


def test_step_shrinks_after_reversals():
    sc = Staircase(start=4.0, x_min=0.1, x_max=20.0, step_big=0.2,
                   step_small=0.1, n_big=2)
    assert sc.step == 0.2
    for r in (True, True, False, True, True, False, True, True):
        sc.update(r)
    assert len(sc.reversals) >= 2
    assert sc.step == 0.1


def test_stops_at_max_trials():
    sc = Staircase(4.0, 0.1, 20.0, max_trials=30)
    n = 0
    while not sc.done:
        sc.update(n % 3 != 0)
        n += 1
    assert n == 30
    assert len(sc.history) == 30


def test_converges_near_seventy_percent_for_a_virtual_observer():
    # 1-up/2-down 은 약 70.7% 정답 지점으로 모인다. 그 근처인지만 확인한다.
    theta, w, lapse = 3.0, 1.0, 0.03
    rng = np.random.default_rng(0)
    finals = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        sc = Staircase(start=9.0, x_min=0.3, x_max=20.0, max_trials=120)
        while not sc.done:
            p = float(p_correct(sc.x, theta, w, lapse))
            sc.update(rng.random() < p)
        finals.append(np.mean(sc.levels()[-40:]))
    got = float(np.mean(finals))
    target = float(theta + w * np.log(0.6093 / (1 - 0.6093)))   # 70.7% 정답 지점
    assert abs(got - target) < 1.0


def test_staircase_data_can_recover_c80():
    # 계단은 자극 배치 장치일 뿐이고, C80 은 적합으로 뽑는다.
    from experiment.fit_c80 import fit_single
    theta, w, lapse = 3.0, 1.0, 0.03
    true = c80(theta, w, lapse)
    got = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        sc = Staircase(start=true * 1.6, x_min=0.3, x_max=20.0, max_trials=120)
        while not sc.done:
            sc.update(rng.random() < float(p_correct(sc.x, theta, w, lapse)))
        r = fit_single(sc.levels(), sc.responses())
        if r["c80"] is not None:
            got.append(r["c80"])
    assert len(got) >= 10
    assert abs(np.median(got) / true - 1) < 0.25


def test_start_rule_does_not_exceed_screen_limit():
    # 시작점이 화면이 낼 수 있는 최대 색차를 넘으면 안 된다.
    assert default_start(expected_c80=9.0, x_max=10.0) <= 10.0
    lo, hi = default_range(4.0, 15.0)
    assert 0 < lo < 4.0 < hi
