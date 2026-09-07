"""모의검증과 모델 비교 검사. 사람 실험 전에 통과해야 한다."""

import numpy as np
import pytest

from experiment.compare_models import (MODELS, compare, divergence, fit_all,
                                       predict_at, required_trials)
from experiment.fit_c80 import (bootstrap_c80, combine_staircases, fit_shared_w,
                                fit_single)
from experiment.parameter_recovery import TRUTHS, recover_one
from experiment.psychometric import c80, simulate_responses, stimulus_levels


def _data(theta, w, lapse, n, seed, c80_guess=None):
    rng = np.random.default_rng(seed)
    g = c80_guess if c80_guess else c80(theta, w, lapse)
    x = np.repeat(stimulus_levels(g, n_levels=8), n // 8)
    return x, simulate_responses(x, theta, w, lapse, rng)


# ------------------------------------------------------------- 회복

@pytest.mark.parametrize("truth", TRUTHS, ids=[t["name"] for t in TRUTHS])
def test_c80_is_recovered_without_bias(truth):
    r = recover_one(truth, n_trials=96, repeats=60, seed=7)
    assert abs(r["c80_bias"]) < 0.15
    assert r["c80_rel_sd"] < 0.35


def test_precision_improves_with_more_trials():
    # ④ 표의 근거. 이 관계가 깨지면 시행 수 권고가 무의미해진다.
    sds = [recover_one(TRUTHS[1], n, 60, seed=3)["c80_rel_sd"]
           for n in (48, 96, 192)]
    assert sds[0] > sds[1] > sds[2]
    # 대략 1/sqrt(n) 이어야 한다.
    assert 1.2 < sds[0] / sds[2] < 2.6


def test_fixing_lapse_hurts_a_careless_observer():
    # 실수가 잦은 사람에게 lapse 를 낮게 고정하면 편향이 커진다.
    # 모의검증에서 확인한 사실이므로 여기에 고정해 둔다.
    careless = TRUTHS[3]
    free = recover_one(careless, 96, 60, seed=5)
    fixed = recover_one(careless, 96, 60, seed=5, fixed_lapse=0.05)
    assert abs(fixed["c80_bias"]) > abs(free["c80_bias"])


def test_misplaced_levels_are_worse_than_correct_ones():
    ok = recover_one(TRUTHS[1], 96, 60, seed=9, guess_err=1.0)
    off = recover_one(TRUTHS[1], 96, 60, seed=9, guess_err=2.0)
    assert off["c80_rel_sd"] > ok["c80_rel_sd"]


# ------------------------------------------------------------- 적합

def test_bootstrap_ci_covers_the_truth():
    theta, w, lapse = 3.0, 1.0, 0.05
    true = c80(theta, w, lapse)
    hits = 0
    for s in range(20):
        x, y = _data(theta, w, lapse, 96, s)
        b = bootstrap_c80(x, y, n=150, seed=s)
        hits += int(b["ci_low"] <= true <= b["ci_high"])
    assert hits >= 15          # 95% 구간이 지나치게 좁지 않은지


def test_shared_slope_fit_finds_the_common_w():
    conds = {}
    for i, th in enumerate((2.0, 3.0, 4.5)):
        conds["w%d" % i] = _data(th, 1.0, 0.05, 96, 40 + i, c80_guess=6.0)
    r = fit_shared_w(conds)
    assert 0.47 <= r["w"] <= 2.2                    # 격자에서 참값 1.0 근처
    ths = [r["conditions"][k]["theta"] for k in ("w0", "w1", "w2")]
    assert ths[0] < ths[1] < ths[2]                 # 순서는 지켜져야 한다


def test_combine_refuses_when_sessions_disagree():
    ok = combine_staircases([4.0, 4.2, 3.9])
    assert ok["usable"] and ok["combined"] == pytest.approx(4.0333, abs=1e-3)
    bad = combine_staircases([2.0, 4.0, 7.0])
    assert not bad["usable"] and bad["combined"] is None
    assert "확정하지 않는다" in bad["reason"]


# ------------------------------------------------------------- 모델 비교

def test_each_model_recovers_its_own_shape():
    w = np.array([0.06, 0.14, 0.33])
    for nm, (_, pred, _, _) in MODELS.items():
        truth = {"constant": [5.0], "inverse": [3.0, 0.15],
                 "power": [2.0, -0.4], "saturating": [3.0, 4.0, 0.1]}[nm]
        y = pred(w, truth)
        f = fit_all(w, y)[nm]
        got = f["predict"](w)
        assert np.allclose(got, y, rtol=0.05), nm


def test_constant_model_loses_when_width_matters():
    w = np.array([0.06, 0.14, 0.33])
    y = 2.0 + 0.20 / w                    # 굵기 의존이 뚜렷한 참값
    fits = fit_all(w, y)
    assert fits["inverse"]["sse"] < fits["constant"]["sse"]
    assert fits["inverse"]["aic"] < fits["constant"]["aic"]


def test_models_diverge_more_outside_the_measured_range():
    # 잰 굵기 사이에서는 후보들이 비슷한 값을 내고, 바깥으로 나갈수록 갈린다.
    # 그래서 검증 굵기를 잰 범위 안쪽에 두면 모델을 못 가른다.
    w = np.array([0.06, 0.14, 0.33])
    fits = fit_all(w, 2.0 + 0.20 / w)
    inside = divergence(fits, 0.14)["spread_rel"]
    outside = divergence(fits, 0.60)["spread_rel"]
    assert outside > inside * 2

    # 굵기 의존이 없는 참값이면 어디서든 후보들이 거의 일치한다.
    flat = fit_all(w, np.full_like(w, 5.0))
    assert divergence(flat, 0.60)["spread_rel"] < 0.05


def test_required_trials_matches_the_120_recommendation():
    # 예측 차이가 6.8% 뿐이면 96 시행 x 3 세션으로는 못 가른다.
    r = required_trials(0.068, rel_sd_at_96=0.171, n_sessions=3, k=2.0)
    assert r["n_trials"] > 96
    # 차이가 크면 시행 수가 줄어든다.
    assert required_trials(0.30)["n_trials"] < r["n_trials"]


def test_compare_flags_an_unresolvable_condition():
    # 모델 예측이 거의 같은 굵기에서는 우열을 확정하면 안 된다.
    cal = {0.06: _data(2.0, 0.7, 0.04, 96, 11, c80_guess=4.0),
           0.14: _data(3.0, 1.0, 0.04, 96, 12, c80_guess=5.0),
           0.33: _data(3.6, 1.2, 0.04, 96, 13, c80_guess=6.0)}
    val = {0.10: _data(2.6, 0.9, 0.04, 96, 14, c80_guess=4.5)}
    res = compare(cal, val, n_boot=120)
    v = res["validation"][0]
    assert v["measured_c80"] is not None
    assert set(v["predictions"]) == set(MODELS)
    # 잰 굵기 사이의 내삽점이므로 모델들이 거의 같은 값을 낸다.
    assert v["model_spread_rel"] < 0.5
    assert v["resolvable"] is False
