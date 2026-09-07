"""굵기에 따라 C80 이 어떻게 변하는지, 후보 모델을 견주는 도구.

무엇을 견주는가
---------------
교정 굵기 세 개에서 C80 을 재고, 각 모델을 그 세 점에 맞춘다. 그다음 **재지
않은 검증 굵기**에서 C80 을 예측하고, 실제로 잰 값과 비교한다.

    교정 굵기(3개) --적합--> 모델 --예측--> 검증 굵기 --대조--> 실측 C80

교정 자료에 대한 AIC 만 보고 우열을 말하면 안 된다. 점 세 개에 파라미터
두세 개를 맞추면 어느 모델이든 잘 맞는다. 우열은 **검증 굵기의 예측 오차**로
가린다.

⚠️ 라벨에 대하여
    굵기 계획서에 적힌 후보 이름(B96, C96, D288 등)의 정의는 이 저장소에
    들어와 있지 않다. 여기서는 모델을 이름으로 박아두지 않고 인자로 받는다.
    계획서의 후보가 확정되면 `MODELS` 를 그 정의로 바꾸면 된다. 아래 기본
    후보 네 개는 계획서와 무관하게, 물리적으로 말이 되는 형태를 넣어둔 것이다.

⚠️ 모델을 가릴 수 있는가는 미리 확인해야 한다
    후보들의 예측이 검증점에서 8.7% / 6.8% 밖에 안 벌어지는데 측정의 표준편차가
    9.6% 면, 어떤 결과가 나와도 우열을 말할 수 없다. `required_trials` 로
    **자료를 보기 전에** 필요한 시행 수를 정하라.
"""

from __future__ import annotations

import numpy as np

from .fit_c80 import bootstrap_c80, fit_single

# ---------------------------------------------------------------- 후보 모델
#
# 각 항목: (파라미터 개수, 예측 함수, 초기값, 설명)
# 폭 w 는 **시야각(도)** 이다. px 는 화면마다 뜻이 달라 모델에 쓰면 안 된다.

MODELS = {
    "constant": (
        1, lambda w, p: np.full_like(np.asarray(w, float), p[0]),
        lambda w, c: [np.mean(c)],
        "굵기와 무관하다. 굵기 효과가 없다는 귀무 모델.",
    ),
    "inverse": (
        2, lambda w, p: p[0] + p[1] / np.asarray(w, float),
        lambda w, c: [min(c), (max(c) - min(c)) * min(w)],
        "가는 선일수록 급격히 어려워지고 굵어지면 바닥에 수렴한다.",
    ),
    "power": (
        2, lambda w, p: p[0] * np.asarray(w, float) ** p[1],
        lambda w, c: [np.mean(c), -0.3],
        "거듭제곱. 지수를 자유롭게 둔다.",
    ),
    "saturating": (
        3, lambda w, p: p[0] + p[1] * np.exp(-np.asarray(w, float) / max(p[2], 1e-6)),
        lambda w, c: [min(c), max(c) - min(c), np.mean(w)],
        "어느 굵기부터는 더 굵어져도 나아지지 않는다.",
    ),
}


def _fit_model(name, w, c80s, weights=None):
    """가중 최소제곱으로 모델 하나를 적합한다."""
    from scipy.optimize import minimize

    n_par, pred, init, _ = MODELS[name]
    w = np.asarray(w, dtype=float)
    y = np.asarray(c80s, dtype=float)
    sw = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)

    def sse(p):
        r = (pred(w, p) - y) * sw
        return float(np.sum(r * r))

    best = minimize(sse, init(w, y), method="Nelder-Mead",
                    options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
    resid = pred(w, best.x) - y
    n = len(y)
    # 잔차 정규분포 가정의 AIC. 점이 셋뿐이므로 참고값으로만 본다.
    aic = n * np.log(max(float(np.sum(resid ** 2)) / n, 1e-12)) + 2 * (n_par + 1)
    return {"name": name, "params": [float(v) for v in best.x],
            "n_params": n_par, "sse": float(np.sum(resid ** 2)),
            "aic": float(aic), "predict": lambda ww, p=best.x: pred(ww, p)}


def fit_all(widths_deg, c80s, weights=None):
    """모든 후보를 교정 굵기에 적합한다."""
    return {nm: _fit_model(nm, widths_deg, c80s, weights) for nm in MODELS}


def predict_at(fits, width_deg):
    return {nm: float(np.atleast_1d(f["predict"](np.array([width_deg])))[0])
            for nm, f in fits.items()}


def divergence(fits, width_deg):
    """검증점에서 후보들의 예측이 서로 얼마나 벌어지는가 (상대 비율).

    이 값이 측정 표준편차보다 작으면 실험을 해도 우열이 안 갈린다.
    """
    p = predict_at(fits, width_deg)
    v = np.array(list(p.values()))
    ref = float(np.median(v))
    return {"predictions": p, "spread_rel": float((v.max() - v.min()) / ref),
            "pairwise_min_rel": float(min(
                abs(a - b) / ref
                for i, a in enumerate(v) for b in v[i + 1:]) if len(v) > 1 else 0.0)}


def required_trials(divergence_rel, rel_sd_at_96=0.171, n_sessions=3, k=2.0):
    """모델을 가르는 데 필요한 세션당 시행 수.

    C80 의 상대SD 는 시행 수의 제곱근에 반비례한다 (모의검증 ④ 표).
    세션을 n 번 합치면 다시 sqrt(n) 만큼 줄어든다. 두 모델의 예측 차이가
    측정 표준편차의 k 배는 되어야 구분했다고 말할 수 있다.

    반환값의 `n_trials` 는 **세션당** 시행 수다.
    """
    if divergence_rel <= 0:
        return {"feasible": False, "reason": "예측 차이가 0 이다"}
    need_sd = divergence_rel / k                       # 요구되는 통합 상대SD
    need_session_sd = need_sd * np.sqrt(n_sessions)    # 세션 하나의 상대SD
    n = 96.0 * (rel_sd_at_96 / need_session_sd) ** 2
    return {
        "divergence_rel": float(divergence_rel),
        "k": float(k), "n_sessions": int(n_sessions),
        "required_session_rel_sd": float(need_session_sd),
        "n_trials": int(np.ceil(n / 12.0) * 12),       # 수준 8개에 맞춰 올림
        "feasible": bool(n <= 400),
        "reason": ("" if n <= 400 else
                   "세션당 %d 시행이 필요하다. 한 사람이 감당하기 어렵다. "
                   "검증 굵기를 예측이 더 벌어지는 곳으로 옮겨야 한다." % int(n)),
    }


def compare(calibration, validation, guess_c80=None, n_boot=400, seed=0):
    """전체 비교.

    calibration: {굵기(도): (x, correct)}   교정 굵기 3개
    validation:  {굵기(도): (x, correct)}   검증 굵기 1개 (세 세션을 합친 것)
    """
    w_cal, c_cal, sd_cal = [], [], []
    for w in sorted(calibration):
        x, y = calibration[w]
        r = fit_single(x, y)
        b = bootstrap_c80(x, y, n=n_boot, seed=seed)
        if r["c80"] is None:
            raise ValueError("굵기 %.4f 도에서 C80 을 추정하지 못했다" % w)
        w_cal.append(w)
        c_cal.append(r["c80"])
        sd_cal.append(b["rel_sd"] if b else None)

    # 정밀도가 나쁜 점의 영향을 줄인다.
    wt = np.array([1.0 / s if s else 1.0 for s in sd_cal])
    fits = fit_all(w_cal, c_cal, weights=wt / wt.mean())

    out = {"calibration": [{"width_deg": w, "c80": c, "rel_sd": s}
                           for w, c, s in zip(w_cal, c_cal, sd_cal)],
           "fits": {nm: {k: f[k] for k in ("name", "params", "n_params",
                                           "sse", "aic")}
                    for nm, f in fits.items()},
           "validation": []}

    for w in sorted(validation):
        x, y = validation[w]
        r = fit_single(x, y)
        b = bootstrap_c80(x, y, n=n_boot, seed=seed + 1)
        pred = predict_at(fits, w)
        div = divergence(fits, w)
        errs = {nm: (p / r["c80"] - 1) for nm, p in pred.items()} if r["c80"] else {}
        inside = ({nm: bool(b["ci_low"] <= p <= b["ci_high"])
                   for nm, p in pred.items()} if b else {})
        out["validation"].append({
            "width_deg": w, "measured_c80": r["c80"],
            "ci": [b["ci_low"], b["ci_high"]] if b else None,
            "rel_sd": b["rel_sd"] if b else None,
            "predictions": pred, "rel_error": errs,
            "prediction_inside_ci": inside,
            "model_spread_rel": div["spread_rel"],
            "resolvable": bool(b and div["spread_rel"] > 2 * b["rel_sd"]),
        })
    return out


def report(res):
    print("=" * 76)
    print("굵기-C80 모델 비교")
    print("=" * 76)
    print("\n교정 굵기에서 잰 C80")
    print("%-12s %10s %10s" % ("굵기(도)", "C80", "상대SD"))
    for r in res["calibration"]:
        print("%-12.4f %10.2f %10s" %
              (r["width_deg"], r["c80"],
               "%.3f" % r["rel_sd"] if r["rel_sd"] else "-"))

    print("\n교정 자료 적합 (점이 셋뿐이므로 AIC 는 참고용)")
    print("%-12s %8s %10s   %s" % ("모델", "파라미터", "AIC", "설명"))
    for nm, f in sorted(res["fits"].items(), key=lambda kv: kv[1]["aic"]):
        print("%-12s %8d %10.2f   %s" % (nm, f["n_params"], f["aic"],
                                         MODELS[nm][3]))

    for v in res["validation"]:
        print("\n검증 굵기 %.4f 도 — 재지 않은 곳에서의 예측" % v["width_deg"])
        print("  실측 C80 %.2f" % v["measured_c80"], end="")
        if v["ci"]:
            print("  (95%% 구간 %.2f ~ %.2f, 상대SD %.3f)"
                  % (v["ci"][0], v["ci"][1], v["rel_sd"]))
        else:
            print()
        print("  %-12s %10s %10s %8s" % ("모델", "예측", "상대오차", "구간 안"))
        for nm in sorted(v["predictions"], key=lambda n: abs(v["rel_error"].get(n, 9))):
            print("  %-12s %10.2f %9.0f%% %8s"
                  % (nm, v["predictions"][nm], v["rel_error"].get(nm, 0) * 100,
                     "예" if v["prediction_inside_ci"].get(nm) else "아니오"))
        print("  모델 간 예측 폭 %.1f%%, 측정 상대SD %.1f%%"
              % (v["model_spread_rel"] * 100, (v["rel_sd"] or 0) * 100))
        if v["resolvable"]:
            print("  -> 구분 가능한 조건이다.")
        else:
            print("  -> 예측 차이가 측정 오차에 묻힌다. **이 굵기에서는 모델")
            print("     우열을 확정하지 않는다.** 시행 수를 늘리거나 검증")
            print("     굵기를 옮겨야 한다.")
    print("=" * 76)
    return res


__all__ = ["MODELS", "fit_all", "predict_at", "divergence", "required_trials",
           "compare", "report"]
