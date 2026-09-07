"""심리측정 함수 적합과 C80 추정.

세 가지를 제공한다.

    fit_single      한 조건(굵기 하나)을 적합
    fit_shared_w    여러 굵기를 함께 적합하되 기울기 w 를 공유
    bootstrap_c80   C80 의 신뢰구간

왜 격자 탐색인가
----------------
파라미터가 셋(θ, w, λ)뿐이고, 96 시행처럼 자료가 적으면 가능도면이 평평하거나
경계에 붙는 경우가 잦다. 경사법은 그런 곳에서 시작점에 따라 다른 답을 낸다.
격자는 느리지만 **같은 자료에 항상 같은 답**을 주므로 사전등록한 분석에 맞다.

기울기 공유 (fit_shared_w)
--------------------------
96 시행으로 θ, w, λ 를 모두 안정적으로 추정하지 못할 수 있다. 그때 쓰는
사전 대안이 **굵기가 달라도 곡선의 기울기는 같다**고 두는 것이다. 위치 θ 만
굵기마다 따로 두므로 파라미터가 크게 줄어든다.

이 가정이 타당한지는 공통 w 모델과 굵기별 w 모델의 AIC 로 사후 점검한다
(연구계획서 §6.2).
"""

from __future__ import annotations

import numpy as np

from .psychometric import GUESS_RATE, c80, log_likelihood, p_correct

# 격자. 색차 단위는 Machado 시뮬레이션 후 ΔE00.
THETA_GRID = np.round(np.concatenate([np.arange(0.05, 8.0, 0.05),
                                      np.arange(8.0, 40.0, 0.25)]), 4)
W_GRID = np.array([0.10, 0.15, 0.22, 0.32, 0.47, 0.68, 1.0, 1.5, 2.2, 3.2, 4.7])
LAPSE_GRID = np.array([0.0, 0.01, 0.02, 0.04, 0.07, 0.12])


def _grid_ll(x, correct, thetas, ws, lapses, guess=GUESS_RATE):
    """(θ, w, λ) 격자 전체의 로그가능도. 브로드캐스팅으로 한 번에 계산한다."""
    x = np.asarray(x, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    z = np.clip((x[None, None, None, :] - thetas[:, None, None, None])
                / ws[None, :, None, None], -60.0, 60.0)
    p = guess + (1.0 - guess - lapses[None, None, :, None]) / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.sum(np.where(correct, np.log(p), np.log1p(-p)), axis=-1)


def fit_single(x, correct, thetas=None, ws=None, lapses=None,
               fixed_lapse=None, guess=GUESS_RATE):
    """한 조건의 (θ, w, λ) 최대가능도 추정과 C80.

    fixed_lapse 를 주면 λ 를 그 값으로 고정한다. parameter recovery 에서
    λ 가 식별되지 않으면 **실제 자료를 보기 전에** 이 대안으로 전환한다.
    """
    thetas = THETA_GRID if thetas is None else np.asarray(thetas)
    ws = W_GRID if ws is None else np.asarray(ws)
    lapses = (np.array([float(fixed_lapse)]) if fixed_lapse is not None
              else (LAPSE_GRID if lapses is None else np.asarray(lapses)))

    ll = _grid_ll(x, correct, thetas, ws, lapses, guess)
    i, j, k = np.unravel_index(np.argmax(ll), ll.shape)
    th, w, lp = float(thetas[i]), float(ws[j]), float(lapses[k])
    n = len(np.asarray(x))
    n_par = 2 if fixed_lapse is not None else 3
    return {
        "theta": th, "w": w, "lapse": lp,
        "c80": c80(th, w, lp, guess),
        "loglik": float(ll[i, j, k]),
        "n_trials": int(n),
        "n_params": n_par,
        "aic": float(2 * n_par - 2 * ll[i, j, k]),
        "at_theta_edge": bool(i == 0 or i == len(thetas) - 1),
        "at_w_edge": bool(j == 0 or j == len(ws) - 1),
        "at_lapse_edge": bool(len(lapses) > 1 and (k == 0 or k == len(lapses) - 1)),
    }


def fit_shared_w(conditions, thetas=None, ws=None, lapses=None,
                 fixed_lapse=None, guess=GUESS_RATE):
    """여러 굵기를 함께 적합. w 와 λ 는 공유, θ 만 굵기마다 따로.

    conditions: {이름: (x, correct)} — 굵기 조건별 자료
    """
    thetas = THETA_GRID if thetas is None else np.asarray(thetas)
    ws = W_GRID if ws is None else np.asarray(ws)
    lapses = (np.array([float(fixed_lapse)]) if fixed_lapse is not None
              else (LAPSE_GRID if lapses is None else np.asarray(lapses)))

    names = list(conditions)
    # 조건마다 (θ, w, λ) 격자 가능도를 구한 뒤, w·λ 를 고정하고 θ 만 최적화한다.
    per = [_grid_ll(conditions[nm][0], conditions[nm][1], thetas, ws, lapses, guess)
           for nm in names]
    best_theta_ll = [g.max(axis=0) for g in per]          # (w, λ) 별 최적 θ 의 가능도
    total = np.sum(best_theta_ll, axis=0)                 # w·λ 를 공유하므로 합산
    j, k = np.unravel_index(np.argmax(total), total.shape)
    w, lp = float(ws[j]), float(lapses[k])

    out = {"w": w, "lapse": lp, "loglik": float(total[j, k]), "conditions": {}}
    n_total = 0
    for nm, g in zip(names, per):
        i = int(np.argmax(g[:, j, k]))
        th = float(thetas[i])
        n = len(np.asarray(conditions[nm][0]))
        n_total += n
        out["conditions"][nm] = {
            "theta": th, "w": w, "lapse": lp, "c80": c80(th, w, lp, guess),
            "n_trials": n, "at_theta_edge": bool(i == 0 or i == len(thetas) - 1),
        }
    n_par = len(names) + (1 if fixed_lapse is not None else 2)
    out["n_params"] = n_par
    out["n_trials"] = n_total
    out["aic"] = float(2 * n_par - 2 * out["loglik"])
    return out


def bootstrap_c80(x, correct, n=400, seed=0, fixed_lapse=None,
                  guess=GUESS_RATE, **grid):
    """시행을 복원추출해 C80 의 신뢰구간을 구한다.

    C80 은 곡선의 가파른 윗부분이라 θ 보다 불확실성이 크다. 점추정만 보고
    모델 우열을 말하면 안 된다.
    """
    x = np.asarray(x, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n)):
        idx = rng.integers(0, len(x), len(x))
        r = fit_single(x[idx], correct[idx], fixed_lapse=fixed_lapse,
                       guess=guess, **grid)
        if r["c80"] is not None:
            vals.append(r["c80"])
    if len(vals) < 20:
        return None
    v = np.array(vals)
    return {"median": float(np.median(v)),
            "ci_low": float(np.percentile(v, 2.5)),
            "ci_high": float(np.percentile(v, 97.5)),
            "rel_sd": float(v.std() / v.mean()),
            "n_ok": len(v)}


def combine_staircases(c80_values, cv_limit=0.20):
    """독립 계단 세 번의 C80 을 합칠지 판단한다 (연구계획서 §마).

    변동계수가 기준 이하일 때만 통합한다. 넘으면 통합하지 않고 세션별로
    보고하며, 그 굵기에서 모델 우열을 확정하지 않는다.
    """
    v = np.asarray([c for c in c80_values if c is not None], dtype=float)
    if len(v) == 0:
        return {"combined": None, "cv": None, "usable": False,
                "reason": "유효한 C80 이 없다"}
    cv = float(v.std(ddof=0) / v.mean())
    ok = cv <= cv_limit
    return {
        "sessions": [float(c) for c in v],
        "cv": cv, "cv_limit": float(cv_limit), "usable": ok,
        "combined": float(v.mean()) if ok else None,
        "reason": ("" if ok else
                   "변동계수 %.3f 가 기준 %.2f 를 넘는다. 통합하지 않고 "
                   "세션별로 보고하며 모델 우열을 확정하지 않는다."
                   % (cv, cv_limit)),
    }


__all__ = ["fit_single", "fit_shared_w", "bootstrap_c80", "combine_staircases",
           "THETA_GRID", "W_GRID", "LAPSE_GRID"]
