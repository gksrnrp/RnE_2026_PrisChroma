"""개인화 모델의 예측 성능 검증.

추정에 **쓰지 않은 색쌍**을 모델이 맞히는지 본다.

왜 필요한가
-----------
같은 자료로 맞추고 같은 자료로 평가하면 잘 맞는 게 당연하다. 파라미터가
자료를 외운 것인지, 그 사람의 색각을 실제로 담아낸 것인지 구별되지 않는다.
그래서 **기준색 단위로** 잘라서, 한 번도 본 적 없는 색으로 검증한다.

    조건(기준색)을 k 묶음으로 나눈다
      -> k-1 묶음으로 (type, severity, θ, w, λ) 추정
      -> 남은 1 묶음의 응답을 예측
      -> 모두 돌아가며 반복

시행 단위로 자르면 안 된다. 같은 기준색의 다른 시행이 훈련 쪽에 남아 있으면
"새로운 색"이 아니기 때문이다.

무엇과 비교하나
---------------
모델이 잘 맞는지 보려면 기준선이 있어야 한다.

    찍기        정답률 25% 로 고정 (4지선다)
    평균만      훈련 자료의 전체 정답률을 그대로 예측값으로
    정상 색각   색각이상이 아니라고 보고 예측
    개인화 모델 추정한 (type, severity, θ, w, λ) 로 예측

'평균만'을 못 이기면 개인화가 아무 값도 못 한 것이다. 색차에 따라 정답률이
달라지는 부분을 하나도 설명하지 못했다는 뜻이기 때문이다.

실행
    python -m colorvision.validate data/colorvision/PO3_..._trials.csv
    python -m colorvision.validate a_trials.csv b_trials.csv --pool
"""

import argparse
import json
import os
import sys

import numpy as np

from .estimate import (GUESS_RATE, fit, load_trials, p_correct,
                       perceived_delta_e)


def _take(data, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) else v)
            for k, v in data.items()}


def pool(datas):
    """여러 세션을 하나로 합친다."""
    out = {}
    for k in datas[0]:
        vals = [d[k] for d in datas]
        out[k] = np.concatenate(vals) if isinstance(vals[0], np.ndarray) else vals[0]
    return out


def group_key(data):
    """검증 단위. 같은 기준색은 반드시 같은 묶음에 들어가야 한다."""
    return np.array(["%s|%s" % (a, b) for a, b in
                     zip(data["condition_id"], data["dir_name"])])


def log_likelihood(pred, correct):
    p = np.clip(pred, 1e-9, 1 - 1e-9)
    return float(np.sum(np.log(np.where(correct, p, 1 - p))))


def cross_validate(data, k=4, sev_step=0.02, seed=0, verbose=True):
    """기준색 단위 k-겹 교차검증."""
    keys = group_key(data)
    uniq = sorted(set(keys.tolist()))
    if len(uniq) < k:
        k = max(2, len(uniq))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    folds = [[uniq[i] for i in order[j::k]] for j in range(k)]

    rows = []
    for fi, held in enumerate(folds):
        te_mask = np.isin(keys, held)
        tr_mask = ~te_mask
        if te_mask.sum() < 5 or tr_mask.sum() < 20:
            continue
        tr, te = _take(data, tr_mask), _take(data, te_mask)

        best = fit(tr, sev_step=sev_step)[0]
        d = perceived_delta_e(te["c1"], te["c2"], best["type"], best["severity"])
        pred = p_correct(d, best["threshold"], best["width"], best["lapse"])
        obs = te["correct"]

        # 기준선들
        rate = float(tr["correct"].mean())
        nrm = fit(tr, sev_step=sev_step)
        normal = next(r for r in nrm if r["type"] == "normal")
        d_n = perceived_delta_e(te["c1"], te["c2"], "normal", 0.0)
        pred_n = p_correct(d_n, normal["threshold"], normal["width"], normal["lapse"])

        rows.append({
            "fold": fi,
            "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
            "held_out": held,
            "type": best["type"], "severity": best["severity"],
            "threshold": best["threshold"],
            "observed": float(obs.mean()), "predicted": float(pred.mean()),
            "ll_model": log_likelihood(pred, obs),
            "ll_rate": log_likelihood(np.full(len(obs), rate), obs),
            "ll_chance": log_likelihood(np.full(len(obs), GUESS_RATE), obs),
            "ll_normal": log_likelihood(pred_n, obs),
        })
        if verbose:
            print("  묶음 %d: 훈련 %3d / 검증 %3d  -> %s s=%.2f"
                  % (fi, tr_mask.sum(), te_mask.sum(), best["type"], best["severity"]))
    return rows


def report(rows, data):
    L = []
    A = L.append
    n_test = sum(r["n_test"] for r in rows)
    tot = {k: sum(r[k] for r in rows)
           for k in ("ll_model", "ll_rate", "ll_chance", "ll_normal")}

    A("=" * 70)
    A("개인화 모델 예측 성능 (추정에 쓰지 않은 색쌍)")
    A("=" * 70)
    A("  %d 겹 교차검증, 검증 시행 합계 %d" % (len(rows), n_test))
    A("")
    A("  %-14s %12s %14s" % ("예측 방법", "로그가능도", "시행당"))
    A("  " + "-" * 44)
    for label, key in (("개인화 모델", "ll_model"), ("정상 색각", "ll_normal"),
                       ("평균만 쓰기", "ll_rate"), ("찍기 (25%)", "ll_chance")):
        A("  %-14s %12.1f %14.3f" % (label, tot[key], tot[key] / n_test))
    A("  " + "-" * 44)

    beat_rate = tot["ll_model"] - tot["ll_rate"]
    beat_norm = tot["ll_model"] - tot["ll_normal"]
    A("")
    A("  개인화 모델 - 평균만 쓰기 = %+.1f" % beat_rate)
    if beat_rate > 0:
        A("    -> 색차에 따른 정답률 변화를 실제로 설명한다.")
    else:
        A("    -> 개인화가 아무 값도 못 했다. 파라미터가 자료를 외운 것일 수 있다.")
    A("  개인화 모델 - 정상 색각  = %+.1f" % beat_norm)
    if beat_norm > 0:
        A("    -> 색각이상을 반영한 것이 예측에 도움이 된다.")
    else:
        A("    -> 색각이상 모델이 정상 색각보다 낫지 않다. 결과를 의심할 것.")

    A("")
    A("  묶음별 상세")
    A("  %-5s %7s %7s %-8s %9s %10s %10s" %
      ("묶음", "훈련", "검증", "유형", "severity", "관측정답률", "예측정답률"))
    for r in rows:
        A("  %-5d %7d %7d %-8s %9.2f %10.2f %10.2f" %
          (r["fold"], r["n_train"], r["n_test"], r["type"], r["severity"],
           r["observed"], r["predicted"]))

    types = [r["type"] for r in rows]
    sevs = [r["severity"] for r in rows]
    A("")
    A("  파라미터 안정성 (묶음마다 다른 색으로 추정한 값)")
    A("    유형     %s" % ("모두 %s" % types[0] if len(set(types)) == 1
                          else "갈림 -> %s" % types))
    A("    severity %.2f ~ %.2f (평균 %.2f)" % (min(sevs), max(sevs), np.mean(sevs)))
    if len(set(types)) > 1:
        A("    ! 색을 바꿔 추정했더니 유형이 갈렸다. 자극 설계나 자료량을 의심할 것.")
    A("=" * 70)
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description="개인화 모델 예측 성능 검증")
    p.add_argument("csv", nargs="+", help="세션 CSV (여러 개면 --pool 로 합침)")
    p.add_argument("--pool", action="store_true", help="여러 세션을 합쳐서 검증")
    p.add_argument("-k", type=int, default=4, help="교차검증 묶음 수")
    p.add_argument("--sev-step", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", nargs="?", const="auto", default=None)
    args = p.parse_args(argv)

    datas = [load_trials(f) for f in args.csv]
    if len(datas) > 1 and not args.pool:
        raise SystemExit("세션이 여러 개면 --pool 을 붙여라.")
    data = pool(datas) if len(datas) > 1 else datas[0]

    print("교차검증 중...")
    rows = cross_validate(data, k=args.k, sev_step=args.sev_step, seed=args.seed)
    if not rows:
        raise SystemExit("검증할 묶음을 만들지 못했다. 시행 수가 너무 적다.")
    print()
    print(report(rows, data))

    if args.json is not None:
        stem = os.path.splitext(args.csv[0])[0].replace("_trials", "")
        path = stem + "_validation.json" if args.json == "auto" else args.json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"sources": args.csv, "folds": rows}, f,
                      ensure_ascii=False, indent=2)
        print("저장: %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
