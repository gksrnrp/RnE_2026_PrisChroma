"""예측 모델 비교 (계획서 4단계) 와 미측정 굵기 검증 (5단계).

네 모델
-------
    B96    가운데 굵기 96 시행으로 적합한 개인별 **단일** 역치
    B288   전체 288 시행으로 적합한 개인별 **단일** 역치
    C96    가운데 굵기 96 시행 + Szafir 상대 굵기 함수
    D288   전체 288 시행으로 적합한 개인별 **굵기** 모델

네 모델은 사실 하나의 식에 들어간다.

    C80(s) = A · T(s)^β        T(s) 는 Szafir 상대역치

    B96  / B288   β = 0        굵기와 무관. A 를 각각 96 / 288 시행으로 정한다
    C96           β = 1        Szafir 형태를 그대로 가져다 쓴다. A 는 96 시행
    D288          β 자유       형태까지 개인 자료로 정한다. 288 시행

그래서 **β 하나가 주 연구질문의 답**이 된다.

    β ≈ 0   굵기는 상관없다
    β ≈ 1   정상 관찰자에서 얻은 굵기 함수가 색각이상자에게도 그대로 맞는다
    β > 1   색각이상자에서 굵기 효과가 더 크다
            -> 정상 관찰자 모델을 그대로 쓸 수 없다는 결과

세 가지 주 비교
---------------
    C96 vs B96     같은 자료량에서 선행모델 전이가 도움이 되는가
    D288 vs B288   굵기 정보를 쓰는 것이 도움이 되는가
    C96 vs D288    적은 개인 측정으로 어디까지 예측할 수 있는가

우열은 **교정 자료의 적합도가 아니라 재지 않은 굵기의 예측 오차**로 가린다.
점 세 개에 파라미터 두 개를 맞추면 어느 모델이든 잘 맞기 때문이다.

보편 고정 임계값은 주 모델이 아니다. 기존 방식과의 보조 비교에만 쓴다.
"""

from __future__ import annotations

import json
import math

import numpy as np

from .fit_c80 import bootstrap_c80, combine_staircases, fit_single
from .psychometric import c80

# ------------------------------------------------------- Szafir 상대 굵기 함수
#
# 형태:  T(s) = A + B / (s − s0)
#
# ⚠️ 출처에 대하여
#     아래 계수는 Szafir(2018) 논문에서 직접 옮긴 것이 아니라,
#     `docs/related-work.md` §7-2 에 계산해 둔 상대역치 표를 되맞춘 값이다.
#     그 표의 다섯 점을 소수 셋째 자리까지 재현한다(`_self_check`).
#     s0 = 0.0241 은 §7-6 이 적어둔 a* 축 발산점 0.024 도와 일치한다.
#     우리 자극이 a* 축을 따라 움직이므로 a* 채널이 지배한다.
#
#     **최종 보고 전에 논문 원문의 계수와 대조할 것.**

SZAFIR_A = 8.0277
SZAFIR_B = 0.19303
SZAFIR_S0 = 0.02411

# Szafir 선 모델의 적용 범위 (도). 밖으로 나가면 외삽이다.
SZAFIR_RANGE_DEG = (0.05, 0.35)

# related-work §7-2 의 표. 되맞춤이 맞는지 확인하는 데 쓴다.
_REFERENCE_TABLE = [(0.050, 15.48), (0.080, 11.48), (0.120, 10.04),
                    (0.250, 8.88), (0.350, 8.62)]


def szafir_T(s_deg):
    """굵기 s(도) 에서의 상대역치. 값 자체가 아니라 **비율**만 의미가 있다."""
    s = np.asarray(s_deg, dtype=float)
    if np.any(s <= SZAFIR_S0):
        raise ValueError("굵기 %.4f 도는 Szafir 함수의 발산점 %.4f 도 이하다"
                         % (float(np.min(s)), SZAFIR_S0))
    return SZAFIR_A + SZAFIR_B / (s - SZAFIR_S0)


def szafir_in_range(s_deg):
    lo, hi = SZAFIR_RANGE_DEG
    return bool(lo <= float(s_deg) <= hi)


def _self_check():
    got = szafir_T([s for s, _ in _REFERENCE_TABLE])
    ref = np.array([t for _, t in _REFERENCE_TABLE])
    err = float(np.max(np.abs(got - ref)))
    if err > 0.01:
        raise AssertionError("Szafir 되맞춤이 표와 어긋난다 (최대 오차 %.4f)" % err)
    return err


_self_check()


# ------------------------------------------------------------------ 모델

class _Model:
    """C80(s) = A · T(s)^beta."""

    def __init__(self, name, A, beta, n_trials, source, note=""):
        self.name = name
        self.A = float(A)
        self.beta = float(beta)
        self.n_trials = int(n_trials)
        self.source = source
        self.note = note

    def predict(self, s_deg):
        s = np.asarray(s_deg, dtype=float)
        if self.beta == 0.0:
            return np.full(s.shape, self.A) if s.shape else float(self.A)
        return self.A * szafir_T(s) ** self.beta

    def predict_one(self, s_deg):
        return float(np.atleast_1d(self.predict(np.atleast_1d(s_deg)))[0])

    def as_dict(self):
        return {"model": self.name, "A": self.A, "beta": self.beta,
                "n_trials": self.n_trials, "source": self.source,
                "note": self.note}


def _pool(calibration):
    x = np.concatenate([np.asarray(calibration[w][0], float)
                        for w in sorted(calibration)])
    y = np.concatenate([np.asarray(calibration[w][1], bool)
                        for w in sorted(calibration)])
    return x, y


def middle_width(calibration):
    """가운데 굵기. 세 굵기의 가운데 값이다."""
    w = sorted(calibration)
    if len(w) != 3:
        raise ValueError("교정 굵기는 세 개여야 한다 (지금 %d 개)" % len(w))
    return w[1]


def fit_models(calibration, per_width_c80=None):
    """네 모델을 모두 적합한다.

    calibration: {굵기(도): (x, correct)} 세 개
    """
    widths = sorted(calibration)
    s2 = middle_width(calibration)

    if per_width_c80 is None:
        per_width_c80 = {}
        for w in widths:
            r = fit_single(*calibration[w])
            if r["c80"] is None:
                raise ValueError("굵기 %.4f 도에서 C80 을 추정하지 못했다" % w)
            per_width_c80[w] = r["c80"]

    n_mid = len(np.asarray(calibration[s2][0]))
    n_all = sum(len(np.asarray(calibration[w][0])) for w in widths)

    # B96 — 가운데 굵기만, 굵기 무관
    b96 = _Model("B96", per_width_c80[s2], 0.0, n_mid,
                 "가운데 굵기 %.4f 도" % s2,
                 "굵기를 무시한 개인별 단일 역치")

    # B288 — 전체를 한 덩어리로 적합, 굵기 무관
    rp = fit_single(*_pool(calibration))
    if rp["c80"] is None:
        raise ValueError("전체 자료에서 C80 을 추정하지 못했다")
    b288 = _Model("B288", rp["c80"], 0.0, n_all, "전체 %d 시행" % n_all,
                  "굵기를 구분하지 않고 모두 합쳐 하나의 역치로 본다")

    # C96 — 가운데 굵기에 Szafir 형태를 얹는다
    a_c = per_width_c80[s2] / float(szafir_T(s2))
    c96 = _Model("C96", a_c, 1.0, n_mid, "가운데 굵기 %.4f 도 + Szafir" % s2,
                 "형태는 선행연구에서 빌리고 위치만 개인 자료로 맞춘다")

    # D288 — 형태까지 개인 자료로. log C80 = log A + beta · log T(s)
    lt = np.log(szafir_T(widths))
    lc = np.log([per_width_c80[w] for w in widths])
    beta, log_a = np.polyfit(lt, lc, 1)
    resid = lc - (log_a + beta * lt)
    d288 = _Model("D288", math.exp(log_a), beta, n_all,
                  "전체 %d 시행" % n_all,
                  "굵기 효과의 크기까지 개인 자료로 정한다")
    d288.rmse_log = float(np.sqrt(np.mean(resid ** 2)))

    return {"B96": b96, "B288": b288, "C96": c96, "D288": d288,
            "per_width_c80": per_width_c80, "middle_width": s2}


def beta_meaning(beta):
    """D288 의 beta 가 뜻하는 것. 주 연구질문의 답이다."""
    if beta < 0.35:
        return "굵기 효과가 거의 없다"
    if beta < 1.5:
        return "Szafir 형태가 색각이상자에게도 대체로 맞는다 (전이 성공)"
    return "색각이상자에서 굵기 효과가 더 크다. 정상 관찰자 모델을 그대로 쓸 수 없다"


# ------------------------------------------------- 5단계: 검증 굵기 선택

def validation_candidates(widths):
    """검증 후보는 인접한 교정 굵기의 기하평균이다."""
    s = sorted(widths)
    if len(s) != 3:
        raise ValueError("교정 굵기는 세 개여야 한다")
    return [math.sqrt(s[0] * s[1]), math.sqrt(s[1] * s[2])]


def snap_to_pixel(s_deg, screen_width_px, screen_width_cm, distance_cm):
    """도 단위 후보를 실제로 그릴 수 있는 정수 px 로 맞춘다."""
    from .stimulus import visual_angle_deg
    px_deg = [(p, visual_angle_deg(p, screen_width_px, screen_width_cm,
                                   distance_cm))
              for p in range(1, 60)]
    p, d = min(px_deg, key=lambda t: abs(t[1] - s_deg))
    return {"px": p, "deg": d, "requested_deg": float(s_deg),
            "in_szafir_range": szafir_in_range(d)}


def choose_validation_width(models, widths, screen=None):
    """검증 굵기를 고른다. **교정 자료만** 쓰고 검증 응답을 보기 전에 고정한다.

    주 비교가 C96 vs B96 이므로, 두 모델의 예측이 가장 크게 갈리는 후보를
    고른다. 두 모델의 예측 비는 T(sv)/T(s2) 이다.
    """
    s2 = models["middle_width"]
    out = []
    for cand in validation_candidates(widths):
        pick = (snap_to_pixel(cand, **screen) if screen else
                {"deg": cand, "px": None, "requested_deg": cand,
                 "in_szafir_range": szafir_in_range(cand)})
        d = pick["deg"]
        gap = abs(float(szafir_T(d)) / float(szafir_T(s2)) - 1.0)
        preds = {nm: m.predict_one(d) for nm, m in models.items()
                 if isinstance(m, _Model)}
        v = np.array(list(preds.values()))
        out.append({**pick, "c96_vs_b96_rel": gap, "predictions": preds,
                    "model_spread_rel": float((v.max() - v.min())
                                              / float(np.median(v)))})
    best = max(out, key=lambda c: c["c96_vs_b96_rel"])
    return {"chosen": best, "candidates": out,
            "rule": "C96 와 B96 의 예측 차이가 큰 쪽. 교정 자료만 사용."}


# --------------------------------------------- 구분 가능한 조건인가

def required_trials(divergence_rel, rel_sd_60x3=0.096, n_sessions=3, k=2.0):
    """모델을 가르는 데 필요한 **세션당** 시행 수.

    기준값 0.096 은 계단 60 회를 3 세션 합쳤을 때의 C80 상대SD 다
    (related-work §7-3, 모의검증 300 회 반복으로 **측정한** 값).

    ⚠️ 해석적으로 계산하면 0.134/sqrt(3) = 0.077 이 나오지만 실제는 0.096 이다.
        세션별 C80 을 각각 적합한 뒤 평균하기 때문에 시행을 전부 합쳐 한 번에
        적합하는 것만큼 줄지 않는다. 낙관적인 0.077 을 쓰면 필요한 시행 수를
        절반 가까이 과소평가하게 되므로 **측정값을 쓴다.**

    총 시행 수에 반비례해 분산이 줄어든다고 보고 환산한다.

        SD(n, s) = 0.096 · sqrt(180 / (n · s))

    k 는 예측 차이가 측정 표준편차의 몇 배여야 구분했다고 볼지다. 기본 2.0 은
    계획서 §6.3 의 결정 규칙(`lowerCI[δ] > h`, 95% 구간)과 같은 수준이다.
    """
    if divergence_rel <= 0:
        return {"feasible": False, "n_trials": None,
                "reason": "예측 차이가 0 이다"}
    need_sd = divergence_rel / k
    n = (180.0 / n_sessions) * (rel_sd_60x3 / need_sd) ** 2
    n_round = int(math.ceil(n / 12.0) * 12)
    return {
        "divergence_rel": float(divergence_rel), "k": float(k),
        "n_sessions": int(n_sessions),
        "required_combined_rel_sd": float(need_sd),
        "n_trials": n_round, "total_trials": n_round * int(n_sessions),
        "feasible": bool(n <= 200),
        "reason": ("" if n <= 200 else
                   "세션당 %d 시행(합계 %d)이 필요하다. 한 사람이 감당하기 "
                   "어렵다. 검증 굵기를 옮기거나 질문의 구조를 바꿔야 한다."
                   % (n_round, n_round * int(n_sessions))),
    }


def combined_rel_sd(n_trials, n_sessions=3, rel_sd_60x3=0.096):
    """계단 n 회 x s 세션을 합쳤을 때 기대되는 C80 상대SD."""
    return float(rel_sd_60x3 * math.sqrt(180.0 / (n_trials * n_sessions)))


def plan_check(models, widths, screen=None, planned_trials=60, n_sessions=3):
    """**자료를 보기 전에** 돌린다. 계획대로 해서 답이 나오는가?"""
    pick = choose_validation_width(models, widths, screen)
    ch = pick["chosen"]
    need = required_trials(ch["c96_vs_b96_rel"], n_sessions=n_sessions)
    enough = planned_trials >= (need["n_trials"] or 10 ** 9)
    return {
        "validation": ch, "candidates": pick["candidates"],
        "planned_trials_per_session": int(planned_trials),
        "required_trials_per_session": need["n_trials"],
        "sufficient": bool(enough),
        "expected_rel_sd": combined_rel_sd(planned_trials, n_sessions),
        "feasible": need["feasible"],
        "message": ("계획한 %d x %d 로 구분 가능하다."
                    % (planned_trials, n_sessions) if enough else
                    "계획한 %d x %d 로는 구분할 수 없다. 예측 차이 %.1f%% 에 "
                    "견주어 측정 상대SD 가 %.1f%% 다. 세션당 %d 시행(합계 %d)이 "
                    "필요하다. 이대로 진행하면 '구분 불가'로 끝난다."
                    % (planned_trials, n_sessions, ch["c96_vs_b96_rel"] * 100,
                       combined_rel_sd(planned_trials, n_sessions) * 100,
                       need["n_trials"], need["total_trials"])),
    }


def beta_power(widths, true_beta, anchor_c80=4.0, n_per_width=96, repeats=60,
               seed=0, theta_w=1.0, lapse=0.05):
    """교정 자료 288 시행으로 β 를 얼마나 정확히 잴 수 있는지 모의로 확인한다.

    **이것이 이 연구의 주 검정력이다.** 검증 굵기에서의 예측 대조는 예측 차이가
    작아 시행을 늘려도 잘 갈리지 않지만(§plan_check), β 는 교정 굵기 세 개가
    T 를 1.66 배 훑기 때문에 훨씬 잘 잡힌다.
    """
    from .psychometric import simulate_responses, stimulus_levels

    widths = sorted(widths)
    ref = float(szafir_T(widths[1]))
    truth = [anchor_c80 * (float(szafir_T(w)) / ref) ** true_beta
             for w in widths]
    got = []
    for r in range(int(repeats)):
        cal = {}
        for i, (w, t) in enumerate(zip(widths, truth)):
            theta = t - theta_w * math.log(0.55 / (0.70 - 0.55))
            rng = np.random.default_rng(seed + r * 17 + i)
            x = np.repeat(stimulus_levels(t, n_levels=8), n_per_width // 8)
            cal[w] = (x, simulate_responses(x, theta, theta_w, lapse, rng))
        try:
            got.append(fit_models(cal)["D288"].beta)
        except ValueError:
            continue
    g = np.array(got)
    lo, hi = np.percentile(g, [2.5, 97.5])
    return {"true_beta": float(true_beta), "mean": float(g.mean()),
            "sd": float(g.std()), "ci_low": float(lo), "ci_high": float(hi),
            "ci_width": float(hi - lo), "n_ok": int(len(g)),
            "separable_from_szafir": bool(lo > 1.0 or hi < 1.0),
            "separable_from_zero": bool(lo > 0.0 or hi < 0.0)}


# ------------------------------------------------------------ 전체 비교

def compare(calibration, validation_width, validation_sessions, n_boot=200,
            seed=0):
    """교정으로 적합하고, 검증 굵기의 실측과 대조한다.

    validation_sessions: {계단ID: (x, correct)} — 독립 세션 셋
    """
    widths = sorted(calibration)
    models = fit_models(calibration)
    sv = float(validation_width)

    # --- 검증: 세션별 C80 과 재현성
    per_session, boots = {}, {}
    for sid in sorted(validation_sessions):
        x, y = validation_sessions[sid]
        r = fit_single(x, y)
        per_session[sid] = r["c80"]
        boots[sid] = bootstrap_c80(x, y, n=n_boot, seed=seed + hash(sid) % 999)
    repro = combine_staircases(list(per_session.values()))

    # 세 세션을 합친 자료로 실측 C80 과 신뢰구간
    vx = np.concatenate([np.asarray(validation_sessions[s][0], float)
                         for s in sorted(validation_sessions)])
    vy = np.concatenate([np.asarray(validation_sessions[s][1], bool)
                         for s in sorted(validation_sessions)])
    measured = fit_single(vx, vy)
    vboot = bootstrap_c80(vx, vy, n=n_boot, seed=seed + 1)

    # --- 각 모델의 예측과 오차
    rows = {}
    for nm in ("B96", "B288", "C96", "D288"):
        p = models[nm].predict_one(sv)
        rows[nm] = {
            "prediction": p,
            "rel_error": (p / measured["c80"] - 1) if measured["c80"] else None,
            "abs_error": (abs(p - measured["c80"]) if measured["c80"] else None),
            "inside_ci": (bool(vboot["ci_low"] <= p <= vboot["ci_high"])
                          if vboot else None),
        }

    # --- 주 비교 세 가지
    def _pair(a, b, question):
        ea, eb = rows[a]["abs_error"], rows[b]["abs_error"]
        gap = abs(rows[a]["prediction"] - rows[b]["prediction"])
        rel = gap / measured["c80"] if measured["c80"] else 0.0
        sd = vboot["rel_sd"] if vboot else None
        resolvable = bool(sd is not None and rel > 2 * sd)
        return {"question": question, "a": a, "b": b,
                "prediction_gap_rel": float(rel),
                "measurement_rel_sd": sd,
                "resolvable": resolvable,
                "winner": (None if not resolvable else
                           (a if ea < eb else b)),
                "verdict": ("두 모델의 예측 차이(%.1f%%)가 측정 오차(%.1f%%)에 "
                            "묻힌다. 우열을 확정하지 않는다."
                            % (rel * 100, (sd or 0) * 100) if not resolvable
                            else "%s 가 더 정확하다 (오차 %.1f%% vs %.1f%%)"
                            % (a if ea < eb else b,
                               min(ea, eb) / measured["c80"] * 100,
                               max(ea, eb) / measured["c80"] * 100))}

    comparisons = [
        _pair("C96", "B96", "같은 자료량에서 선행모델 전이가 도움이 되는가"),
        _pair("D288", "B288", "굵기 정보를 쓰는 것이 도움이 되는가"),
        _pair("C96", "D288", "적은 개인 측정으로 어디까지 예측할 수 있는가"),
    ]

    return {
        "calibration_widths_deg": widths,
        "middle_width_deg": models["middle_width"],
        "per_width_c80": models["per_width_c80"],
        "models": {nm: models[nm].as_dict() for nm in
                   ("B96", "B288", "C96", "D288")},
        "beta": models["D288"].beta,
        "beta_meaning": beta_meaning(models["D288"].beta),
        "validation_width_deg": sv,
        "validation_in_szafir_range": szafir_in_range(sv),
        "measured_c80": measured["c80"],
        "measured_ci": [vboot["ci_low"], vboot["ci_high"]] if vboot else None,
        "measured_rel_sd": vboot["rel_sd"] if vboot else None,
        "sessions": per_session,
        "reproducibility": repro,
        "predictions": rows,
        "comparisons": comparisons,
    }


# ------------------------------------------------------------ 산출물

def build_participant_thresholds(entries, schema_version="1.0"):
    """`results/participant_thresholds.json` 을 만든다.

    entries: [{participant_id, cvd_type, severity, severity_source,
               result(=compare 반환값), screen}]

    측정값과 예측값을 `value_type` 으로 반드시 구분한다.
    """
    out = {"schema_version": schema_version,
           "coordinate": "post_machado_delta_e00",
           "participants": []}

    for e in entries:
        res = e["result"]
        screen = e.get("screen")
        preds = []

        def _px(deg):
            if not screen:
                return None
            from .stimulus import visual_angle_deg
            best = min(range(1, 60),
                       key=lambda p: abs(visual_angle_deg(p, **screen) - deg))
            return best

        # 교정 굵기에서 실제로 잰 값
        for w, c in sorted(res["per_width_c80"].items()):
            preds.append({"model": "measured", "stroke_width_px": _px(w),
                          "stroke_width_deg": round(float(w), 5),
                          "c80_sim": round(float(c), 3),
                          "ci_low": None, "ci_high": None,
                          "value_type": "measured"})

        # 검증 굵기에서 실제로 잰 값
        sv = res["validation_width_deg"]
        ci = res["measured_ci"] or [None, None]
        preds.append({"model": "measured", "stroke_width_px": _px(sv),
                      "stroke_width_deg": round(float(sv), 5),
                      "c80_sim": round(float(res["measured_c80"]), 3),
                      "ci_low": round(ci[0], 3) if ci[0] else None,
                      "ci_high": round(ci[1], 3) if ci[1] else None,
                      "value_type": "measured"})

        # 검증 굵기에서의 예측값
        for nm, row in res["predictions"].items():
            preds.append({"model": nm, "stroke_width_px": _px(sv),
                          "stroke_width_deg": round(float(sv), 5),
                          "c80_sim": round(float(row["prediction"]), 3),
                          "ci_low": None, "ci_high": None,
                          "value_type": "predicted"})

        resolvable = any(c["resolvable"] for c in res["comparisons"])
        out["participants"].append({
            "participant_id": e["participant_id"],
            "cvd_type": e["cvd_type"],
            "severity": e["severity"],
            "severity_source": e["severity_source"],
            "confirmed": bool(res["reproducibility"]["usable"] and resolvable),
            "beta": round(float(res["beta"]), 4),
            "beta_meaning": res["beta_meaning"],
            "staircase_cv": (round(float(res["reproducibility"]["cv"]), 4)
                             if res["reproducibility"]["cv"] else None),
            "predictions": preds,
        })

    out["caveat"] = ("참가자 수가 적다. 여기 담긴 개인 결과를 protan 또는 "
                     "deutan 전체의 성질로 일반화하지 않는다.")
    return out


def save_participant_thresholds(entries, path, **kw):
    data = build_participant_thresholds(entries, **kw)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


# ------------------------------------------------------------ 출력

def report(res):
    print("=" * 78)
    print("예측 모델 비교 — B96 / B288 / C96 / D288")
    print("=" * 78)

    print("\n교정 굵기에서 잰 C80")
    print("%-12s %10s" % ("굵기(도)", "C80"))
    for w, c in sorted(res["per_width_c80"].items()):
        print("%-12.4f %10.2f" % (w, c))

    print("\n적합된 모델   C80(s) = A · T(s)^β")
    print("%-8s %10s %8s %8s   %s" % ("모델", "A", "β", "시행", "근거"))
    for nm in ("B96", "B288", "C96", "D288"):
        m = res["models"][nm]
        print("%-8s %10.4f %8.3f %8d   %s"
              % (nm, m["A"], m["beta"], m["n_trials"], m["source"]))

    print("\nβ = %.3f  ->  %s" % (res["beta"], res["beta_meaning"]))

    sv = res["validation_width_deg"]
    print("\n검증 굵기 %.4f 도 (재지 않은 굵기)" % sv)
    if not res["validation_in_szafir_range"]:
        print("  ⚠ Szafir 모델 적용 범위 %.2f~%.2f 도 밖이다. 외삽이다."
              % SZAFIR_RANGE_DEG)
    print("  실측 C80 %.2f" % res["measured_c80"], end="")
    if res["measured_ci"]:
        print("  (95%% 구간 %.2f ~ %.2f, 상대SD %.1f%%)"
              % (res["measured_ci"][0], res["measured_ci"][1],
                 res["measured_rel_sd"] * 100))
    else:
        print()

    print("\n  %-8s %10s %10s %8s" % ("모델", "예측", "상대오차", "구간 안"))
    for nm in sorted(res["predictions"],
                     key=lambda n: abs(res["predictions"][n]["rel_error"] or 9)):
        r = res["predictions"][nm]
        print("  %-8s %10.2f %9.1f%% %8s"
              % (nm, r["prediction"], (r["rel_error"] or 0) * 100,
                 "예" if r["inside_ci"] else "아니오"))

    print("\n계단 세 세션의 재현성")
    for sid, c in sorted(res["sessions"].items()):
        print("  %-6s C80 %.2f" % (sid, c))
    rp = res["reproducibility"]
    print("  변동계수 %.3f (기준 %.2f) -> %s"
          % (rp["cv"], rp["cv_limit"], "통합" if rp["usable"] else "통합 안 함"))
    if not rp["usable"]:
        print("  " + rp["reason"])

    print("\n주 비교")
    for c in res["comparisons"]:
        print("  %s vs %s — %s" % (c["a"], c["b"], c["question"]))
        print("     %s" % c["verdict"])

    if not any(c["resolvable"] for c in res["comparisons"]):
        print("\n  ** 어느 비교도 구분 가능하지 않다. 모델 우열을 주장하지 않고")
        print("     '구분 불가' 로 보고한다. **")
    print("=" * 78)
    return res


__all__ = ["szafir_T", "szafir_in_range", "SZAFIR_RANGE_DEG", "fit_models",
           "beta_meaning", "validation_candidates", "snap_to_pixel",
           "choose_validation_width", "required_trials", "combined_rel_sd",
           "plan_check", "beta_power",
           "compare", "build_participant_thresholds",
           "save_participant_thresholds", "report", "middle_width"]
