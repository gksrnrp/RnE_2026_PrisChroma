"""모의검증과 예측 모델 비교 검사. 사람 실험 전에 통과해야 한다."""

import numpy as np
import pytest

from experiment.compare_models import (beta_meaning, beta_power,
                                       build_participant_thresholds,
                                       combined_rel_sd,
                                       choose_validation_width, compare,
                                       fit_models, plan_check, required_trials,
                                       snap_to_pixel, szafir_T,
                                       szafir_in_range, validation_candidates)
from experiment.fit_c80 import (bootstrap_c80, combine_staircases, fit_shared_w,
                                fit_single)
from experiment.parameter_recovery import TRUTHS, recover_one
from experiment.psychometric import c80, simulate_responses, stimulus_levels


def _data(theta, w, lapse, n, seed, c80_guess=None):
    rng = np.random.default_rng(seed)
    g = c80_guess if c80_guess else c80(theta, w, lapse)
    x = np.repeat(stimulus_levels(g, n_levels=8), max(1, n // 8))
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


# ------------------------------------------------------- Szafir 굵기 함수

def test_szafir_reconstruction_matches_the_reference_table():
    # 계수는 논문에서 직접 옮긴 것이 아니라 related-work §7-2 표를 되맞춘
    # 값이다. 표를 재현하지 못하면 그 근거가 무너진다.
    for s_deg, t in [(0.050, 15.48), (0.080, 11.48), (0.120, 10.04),
                     (0.250, 8.88), (0.350, 8.62)]:
        assert float(szafir_T(s_deg)) == pytest.approx(t, abs=0.01)


def test_szafir_diverges_below_its_root():
    with pytest.raises(ValueError):
        szafir_T(0.02)
    assert not szafir_in_range(0.041)      # 2 px 는 적용 범위 밖
    assert szafir_in_range(0.062)          # 3 px 는 안


def test_szafir_effect_is_small_as_documented():
    # 전 구간 배율이 1.8 배뿐이라는 것이 이 연구의 핵심 난점이다.
    assert float(szafir_T(0.05) / szafir_T(0.35)) == pytest.approx(1.80, abs=0.02)


# ------------------------------------------------------------- 네 모델

WIDTHS = [0.062, 0.144, 0.330]
SCREEN = dict(screen_width_px=1920, screen_width_cm=34.54, distance_cm=50.0)


def _cal(c80s, widths=None, n=96, seed=0):
    """굵기별 참 C80 을 주면 그에 맞는 가상 응답을 만든다."""
    widths = widths or WIDTHS
    out = {}
    for i, (w, target) in enumerate(zip(widths, c80s)):
        wid, lapse = 1.0, 0.05
        theta = target - wid * np.log(0.55 / (0.70 - 0.55))
        out[w] = _data(theta, wid, lapse, n, seed + i, c80_guess=target)
    return out


def test_four_models_are_all_fitted():
    m = fit_models(_cal([5.0, 4.6, 4.4]))
    assert set(m) >= {"B96", "B288", "C96", "D288"}
    assert m["B96"].beta == 0.0 and m["B288"].beta == 0.0
    assert m["C96"].beta == 1.0
    assert m["B96"].n_trials == 96 and m["B288"].n_trials == 288


def test_b_models_ignore_width_and_c96_follows_szafir():
    m = fit_models(_cal([5.0, 4.6, 4.4]))
    assert m["B96"].predict_one(0.05) == m["B96"].predict_one(0.35)
    # C96 은 가운데 굵기에서 B96 과 같은 값을 내고, 다른 굵기에서만 갈린다.
    s2 = m["middle_width"]
    assert m["C96"].predict_one(s2) == pytest.approx(m["B96"].predict_one(s2),
                                                     rel=1e-9)
    assert m["C96"].predict_one(0.05) > m["C96"].predict_one(0.35)


def test_beta_recovers_no_width_effect():
    # 굵기와 무관한 참값이면 beta 가 0 근처여야 한다.
    m = fit_models(_cal([4.5, 4.5, 4.5]),
                   per_width_c80=dict(zip(WIDTHS, [4.5, 4.5, 4.5])))
    assert abs(m["D288"].beta) < 0.05
    assert "거의 없다" in beta_meaning(m["D288"].beta)


def test_beta_recovers_szafir_shape():
    # 참값이 Szafir 형태 그대로면 beta 가 1 근처여야 한다.
    true = [4.0 * float(szafir_T(w)) / float(szafir_T(WIDTHS[1]))
            for w in WIDTHS]
    m = fit_models(_cal(true), per_width_c80=dict(zip(WIDTHS, true)))
    assert m["D288"].beta == pytest.approx(1.0, abs=0.05)
    assert "전이 성공" in beta_meaning(m["D288"].beta)


def test_beta_flags_a_larger_effect_in_cvd():
    # 굵기 효과가 Szafir 예측보다 훨씬 크면 그것이 결과가 되어야 한다.
    true = [4.0 * (float(szafir_T(w)) / float(szafir_T(WIDTHS[1]))) ** 3.0
            for w in WIDTHS]
    m = fit_models(_cal(true), per_width_c80=dict(zip(WIDTHS, true)))
    assert m["D288"].beta > 1.5
    assert "그대로 쓸 수 없다" in beta_meaning(m["D288"].beta)


# --------------------------------------------------- 5단계: 검증 굵기

def test_validation_candidates_are_the_geometric_means():
    c = validation_candidates(WIDTHS)
    assert c[0] == pytest.approx(np.sqrt(0.062 * 0.144))
    assert c[1] == pytest.approx(np.sqrt(0.144 * 0.330))
    # 두 후보 모두 교정 굵기 사이에 있다. 그래서 모델이 잘 안 갈린다.
    assert WIDTHS[0] < c[0] < WIDTHS[1] < c[1] < WIDTHS[2]


def test_validation_candidates_snap_to_the_documented_pixels():
    # related-work §7-2 가 계산해 둔 5 px / 11 px 와 일치해야 한다.
    got = [snap_to_pixel(c, **SCREEN)["px"] for c in validation_candidates(WIDTHS)]
    assert got == [5, 11]


def test_validation_width_is_chosen_from_calibration_only():
    m = fit_models(_cal([5.0, 4.6, 4.4]))
    pick = choose_validation_width(m, WIDTHS, screen=SCREEN)
    assert pick["chosen"]["px"] == 5              # 예측이 더 갈리는 쪽
    assert pick["chosen"]["c96_vs_b96_rel"] == pytest.approx(0.087, abs=0.005)
    assert len(pick["candidates"]) == 2
    assert "교정 자료만" in pick["rule"]


def test_combined_sd_uses_the_measured_anchor_not_the_optimistic_formula():
    # 60x3 의 실측 상대SD 는 9.6% 다. 해석식 0.134/sqrt(3)=7.7% 를 쓰면
    # 필요한 시행 수를 절반 가까이 과소평가한다.
    assert combined_rel_sd(60, 3) == pytest.approx(0.096, abs=1e-6)
    assert combined_rel_sd(240, 3) == pytest.approx(0.048, abs=1e-6)
    assert combined_rel_sd(120, 3) < combined_rel_sd(60, 3)


def test_plan_check_rejects_sixty_by_three_and_also_one_twenty():
    # 계획서의 60 x 3 으로는 8.7% 차이를 가를 수 없다. 자료를 보기 전에
    # 이 사실이 드러나야 한다. 120 x 3 으로 늘려도 마찬가지다.
    m = fit_models(_cal([5.0, 4.6, 4.4]))
    for n in (60, 120, 192):
        chk = plan_check(m, WIDTHS, screen=SCREEN, planned_trials=n)
        assert chk["sufficient"] is False, n
        assert "구분할 수 없다" in chk["message"]
    chk = plan_check(m, WIDTHS, screen=SCREEN, planned_trials=60)
    assert chk["required_trials_per_session"] > 200
    assert chk["feasible"] is False        # 시행을 늘려서 될 문제가 아니다


def test_required_trials_shrinks_as_the_gap_grows():
    a = required_trials(0.068)
    b = required_trials(0.30)
    assert b["n_trials"] < a["n_trials"]
    assert required_trials(0.0)["feasible"] is False
    # 기하평균 검증점의 예측 차이로는 감당 못 할 시행 수가 나온다.
    assert required_trials(0.087)["feasible"] is False


# --------------------------------- β 검정력: 이 연구의 실제 답이 나오는 곳

def test_beta_is_recovered_without_bias_from_calibration_alone():
    # 검증 굵기의 예측 대조는 갈리지 않지만, 교정 288 시행의 β 는 잡힌다.
    for true_b in (0.0, 1.0, 2.0):
        r = beta_power(WIDTHS, true_b, repeats=30, seed=1)
        assert abs(r["mean"] - true_b) < 0.25, true_b


def test_beta_separates_no_effect_and_a_much_larger_effect_from_szafir():
    assert beta_power(WIDTHS, 0.0, repeats=30, seed=2)["separable_from_szafir"]
    assert beta_power(WIDTHS, 3.0, repeats=30, seed=2)["separable_from_szafir"]


def test_beta_does_not_falsely_reject_szafir():
    # 참값이 Szafir 그대로일 때 "다르다" 고 하면 안 된다.
    assert not beta_power(WIDTHS, 1.0, repeats=30,
                          seed=3)["separable_from_szafir"]


def test_beta_resolution_is_about_one_unit():
    # β 를 ±0.5 수준으로 재므로, 1 에서 1 정도는 벗어나야 구분된다.
    # 이보다 작은 차이를 주장하면 안 된다.
    r = beta_power(WIDTHS, 1.0, repeats=30, seed=4)
    assert 0.3 < r["sd"] < 0.8


# ------------------------------------------------------------- 전체 비교

def _val(seed, n=60):
    return {sid: _data(3.0, 1.0, 0.05, n, seed + i, c80_guess=4.6)
            for i, sid in enumerate(("S1", "S2", "S3"))}


def test_compare_reports_unresolvable_instead_of_a_winner():
    # 검증점이 교정 굵기 사이라 모델 예측이 거의 같다. 60 시행이면 측정
    # 오차가 더 크므로 "구분 불가" 가 나와야 한다.
    res = compare(_cal([5.0, 4.6, 4.4]), 0.1031, _val(70), n_boot=120)
    assert set(res["predictions"]) == {"B96", "B288", "C96", "D288"}
    assert len(res["comparisons"]) == 3
    assert res["measured_ci"][0] < res["measured_c80"] < res["measured_ci"][1]
    assert not any(c["resolvable"] for c in res["comparisons"])
    for c in res["comparisons"]:
        assert c["winner"] is None
        assert "확정하지 않는다" in c["verdict"]


def test_compare_covers_the_three_main_questions():
    res = compare(_cal([5.0, 4.6, 4.4]), 0.1031, _val(75), n_boot=100)
    pairs = {(c["a"], c["b"]) for c in res["comparisons"]}
    assert pairs == {("C96", "B96"), ("D288", "B288"), ("C96", "D288")}


def test_compare_reports_session_reproducibility():
    res = compare(_cal([5.0, 4.6, 4.4]), 0.1031, _val(80), n_boot=100)
    assert set(res["sessions"]) == {"S1", "S2", "S3"}
    assert res["reproducibility"]["cv"] is not None
    assert res["reproducibility"]["cv_limit"] == 0.20


# ------------------------------------------------------------- 산출물

def test_output_separates_measured_from_predicted():
    res = compare(_cal([5.0, 4.6, 4.4]), 0.1031, _val(90), n_boot=100)
    out = build_participant_thresholds([{
        "participant_id": "P01", "cvd_type": "deutan", "severity": 0.7,
        "severity_source": "independent_patch_session", "result": res,
        "screen": SCREEN}])

    assert out["schema_version"] == "1.0"
    assert out["coordinate"] == "post_machado_delta_e00"
    p = out["participants"][0]
    assert p["participant_id"] == "P01"
    assert p["severity_source"] == "independent_patch_session"

    assert {x["value_type"] for x in p["predictions"]} == {"measured",
                                                           "predicted"}
    meas = [x for x in p["predictions"] if x["value_type"] == "measured"]
    pred = [x for x in p["predictions"] if x["value_type"] == "predicted"]
    assert len(meas) == 4                      # 교정 3 + 검증 1
    assert len(pred) == 4                      # 모델 4 개
    assert {x["model"] for x in pred} == {"B96", "B288", "C96", "D288"}
    for x in p["predictions"]:
        assert x["stroke_width_px"] is not None
        assert x["stroke_width_deg"] > 0
        assert x["c80_sim"] > 0
    # 검증 굵기의 실측에는 신뢰구간이 붙어 있어야 한다.
    v = [x for x in meas if x["stroke_width_px"] == 5][0]
    assert v["ci_low"] < v["c80_sim"] < v["ci_high"]


def test_output_records_beta_and_refuses_to_confirm_when_unresolvable():
    res = compare(_cal([5.0, 4.6, 4.4]), 0.1031, _val(95), n_boot=100)
    out = build_participant_thresholds([{
        "participant_id": "P01", "cvd_type": "protan", "severity": 0.94,
        "severity_source": "patch_session_PO3", "result": res,
        "screen": None}])
    p = out["participants"][0]
    assert p["beta"] is not None and p["beta_meaning"]
    assert "일반화하지 않는다" in out["caveat"]
    # 구분 불가이면 confirmed 가 서면 안 된다.
    assert p["confirmed"] is False


def test_output_is_json_serializable(tmp_path):
    import json

    from experiment.compare_models import save_participant_thresholds
    res = compare(_cal([5.0, 4.6, 4.4]), 0.1031, _val(99), n_boot=80)
    path = str(tmp_path / "participant_thresholds.json")
    save_participant_thresholds([{
        "participant_id": "P01", "cvd_type": "protan", "severity": 0.94,
        "severity_source": "patch_session_PO3", "result": res,
        "screen": SCREEN}], path)
    again = json.load(open(path, encoding="utf-8"))
    assert again["participants"][0]["participant_id"] == "P01"
    # 원자료에 개인정보가 없듯 산출물에도 없어야 한다.
    raw = open(path, encoding="utf-8").read()
    for banned in ("이름", "학번", "student_id"):
        assert banned not in raw
