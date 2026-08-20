"""색쌍 응답으로부터 개인 파라미터 (type, severity, threshold) 를 추정한다.

실행계획 v4 §3-3 의 3단계에 해당한다. 이시하라로 유형을 '확정'하지 않고,
**실제 응답을 가장 잘 설명하는 시뮬레이터를 고르는** 구조다.

모델
----
시행 하나에서 보여준 두 색을 c1, c2 라 하자. 후보 파라미터 (type, s) 에서
그 사람이 실제로 느끼는 색차는

    d = ΔE00( Sim(c1; type, s),  Sim(c2; type, s) )

이고, 4지선다이므로 정답 확률은

    P(정답) = γ + (1 - γ - λ) · σ( (d - θ) / w ),   γ = 1/4

    θ : 구별 역치 (ΔE00 단위)  — confusion_threshold
    w : 역치 부근의 완만함
    λ : 실수율 (딴 짓, 오조작). 아주 쉬운 확인용 시행이 이 값을 잡아준다.

(type, s, θ, w, λ) 를 격자 탐색으로 최대가능도 추정한다. type 은 protan,
deutan, 그리고 색각이상이 아닌 경우를 위한 normal 세 가지를 모두 시도한다.

실행
    python -m colorvision.estimate data/P01_..._trials.csv
    python -m colorvision.estimate data/P01_..._trials.csv --plot out.png
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from . import machado as mc
from .colorspace import delta_e00_linear

GUESS_RATE = 0.25          # 4지선다


# ---------------------------------------------------------------- 데이터

def load_trials(path, include_practice=False):
    """CSV -> dict of arrays. 실제로 보여준 선형 RGB 두 색과 정오만 있으면 된다."""
    c1, c2, correct, cond, dirname, tval, catch = [], [], [], [], [], [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["phase"] == "practice" and not include_practice:
                continue
            c1.append([float(row["a_lin_r"]), float(row["a_lin_g"]), float(row["a_lin_b"])])
            c2.append([float(row["b_lin_r"]), float(row["b_lin_g"]), float(row["b_lin_b"])])
            correct.append(int(row["correct"]))
            cond.append(row["condition_id"])
            dirname.append(row["dir_name"])
            tval.append(float(row["t"]))
            catch.append(int(row["is_catch"]))
    if not c1:
        raise SystemExit("시행이 없다: %s" % path)
    return {
        "c1": np.array(c1), "c2": np.array(c2),
        "correct": np.array(correct, dtype=bool),
        "condition_id": np.array(cond), "dir_name": np.array(dirname),
        "t": np.array(tval), "is_catch": np.array(catch, dtype=bool),
    }


# ---------------------------------------------------------------- 모델

def perceived_delta_e(c1, c2, cvd_type, severity):
    """후보 파라미터에서 이 사람이 느낄 색차 ΔE00."""
    return delta_e00_linear(mc.simulate_linear(c1, cvd_type, severity),
                            mc.simulate_linear(c2, cvd_type, severity))


def p_correct(d, theta, w, lapse):
    z = np.clip((d - theta) / w, -60, 60)
    return GUESS_RATE + (1.0 - GUESS_RATE - lapse) / (1.0 + np.exp(-z))


def log_likelihood(d, correct, theta, w, lapse):
    p = np.clip(p_correct(d, theta, w, lapse), 1e-9, 1 - 1e-9)
    return float(np.sum(np.log(np.where(correct, p, 1 - p))))


def fit(data, sev_step=0.02, verbose=False):
    """격자 탐색으로 최대가능도 파라미터를 찾는다.

    (type, s) 마다 d 는 한 번만 계산하면 되므로, 안쪽 (θ, w, λ) 탐색은 값싸다.
    반환: 결과 dict 들의 리스트. 로그가능도 내림차순.
    """
    thetas = np.concatenate([np.arange(0.2, 6.0, 0.15), np.arange(6.0, 20.0, 0.5)])
    ws = np.array([0.3, 0.5, 0.8, 1.2, 1.8, 2.6, 4.0, 6.0])
    lapses = np.array([0.0, 0.01, 0.02, 0.04, 0.07, 0.12])

    correct = data["correct"]
    candidates = [("normal", 0.0)]
    sevs = np.round(np.arange(sev_step, 1.0 + 1e-9, sev_step), 4)
    for t in mc.TYPES:
        candidates += [(t, float(s)) for s in sevs]

    # (θ, w, λ) 격자를 브로드캐스팅으로 한 번에 계산한다. 부트스트랩과
    # 복원 검증이 fit 을 수백 번 부르므로 이 부분의 속도가 곧 전체 속도다.
    TH = thetas[:, None, None, None]
    W = ws[None, :, None, None]
    LP = lapses[None, None, :, None]

    results = []
    for cvd_type, s in candidates:
        d = perceived_delta_e(data["c1"], data["c2"], cvd_type, s)
        z = np.clip((d[None, None, None, :] - TH) / W, -60, 60)
        p = GUESS_RATE + (1.0 - GUESS_RATE - LP) / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        ll_grid = np.sum(np.where(correct, np.log(p), np.log1p(-p)), axis=-1)
        i, j, k = np.unravel_index(np.argmax(ll_grid), ll_grid.shape)
        ll = float(ll_grid[i, j, k])
        th, w, lp = float(thetas[i]), float(ws[j]), float(lapses[k])
        results.append({"type": cvd_type, "severity": s, "threshold": th,
                        "width": w, "lapse": lp, "loglik": ll,
                        "n_params": 4})
        if verbose:
            print("  %-7s s=%.2f  LL=%9.3f  θ=%.2f w=%.2f λ=%.2f"
                  % (cvd_type, s, ll, th, w, lp))
    results.sort(key=lambda r: -r["loglik"])
    return results


def fit_threshold(data, cvd_type, severity):
    """유형과 severity 를 고정하고 역치만 다시 맞춘다.

    기하 조건(크기, 거리, 배경)을 바꿔가며 역치가 어떻게 변하는지 볼 때 쓴다.
    유형과 severity 는 그 사람의 생리적 특성이라 화면 조건에 따라 변할 리 없다.
    그걸 자유롭게 두면 기하 효과를 severity 가 흡수해버려서, 정작 알고 싶은
    역치 변화가 가려진다.

    반환: {"threshold", "width", "lapse", "loglik", "n"}
    """
    thetas = np.concatenate([np.arange(0.1, 8.0, 0.05), np.arange(8.0, 40.0, 0.25)])
    ws = np.array([0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.6, 4.0, 6.0, 9.0])
    lapses = np.array([0.0, 0.01, 0.02, 0.04, 0.07, 0.12])

    d = perceived_delta_e(data["c1"], data["c2"], cvd_type, severity)
    correct = data["correct"]
    z = np.clip((d[None, None, None, :] - thetas[:, None, None, None])
                / ws[None, :, None, None], -60, 60)
    p = GUESS_RATE + (1.0 - GUESS_RATE - lapses[None, None, :, None]) / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ll = np.sum(np.where(correct, np.log(p), np.log1p(-p)), axis=-1)
    i, j, k = np.unravel_index(np.argmax(ll), ll.shape)
    return {"threshold": float(thetas[i]), "width": float(ws[j]),
            "lapse": float(lapses[k]), "loglik": float(ll[i, j, k]),
            "n": int(len(correct))}


def threshold_interval(data, cvd_type, severity, best_ll, drop=1.92):
    """역치의 프로파일 가능도 구간. width 와 lapse 는 각 역치에서 최적화한다."""
    thetas = np.concatenate([np.arange(0.1, 8.0, 0.05), np.arange(8.0, 40.0, 0.25)])
    ws = np.array([0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.6, 4.0, 6.0, 9.0])
    lapses = np.array([0.0, 0.01, 0.02, 0.04, 0.07, 0.12])
    d = perceived_delta_e(data["c1"], data["c2"], cvd_type, severity)
    correct = data["correct"]
    z = np.clip((d[None, None, None, :] - thetas[:, None, None, None])
                / ws[None, :, None, None], -60, 60)
    p = GUESS_RATE + (1.0 - GUESS_RATE - lapses[None, None, :, None]) / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ll = np.sum(np.where(correct, np.log(p), np.log1p(-p)), axis=-1)
    prof = ll.reshape(len(thetas), -1).max(axis=1)
    ok = thetas[prof >= best_ll - drop]
    return (float(ok.min()), float(ok.max())) if len(ok) else None


def best_per_type(results):
    out = {}
    for r in results:
        if r["type"] not in out:
            out[r["type"]] = r
    return out


def severity_interval(results, cvd_type, drop=1.92):
    """프로파일 가능도 구간. 최대에서 LL 이 drop 이상 떨어지지 않는 severity 범위.

    drop=1.92 는 자유도 1 카이제곱 95% 에 해당한다 (0.5·χ²(0.95,1)).
    """
    rows = [r for r in results if r["type"] == cvd_type]
    if not rows:
        return None
    top = max(r["loglik"] for r in rows)
    ok = [r["severity"] for r in rows if r["loglik"] >= top - drop]
    return (min(ok), max(ok)) if ok else None


def bootstrap(data, n=200, sev_step=0.05, seed=0):
    """시행을 복원추출해 (유형, severity) 추정의 안정성을 본다."""
    rng = np.random.default_rng(seed)
    N = len(data["correct"])
    types, sevs = [], []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        sub = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
        r = fit(sub, sev_step=sev_step)[0]
        types.append(r["type"])
        sevs.append(r["severity"])
    types = np.array(types)
    sevs = np.array(sevs)
    share = {t: float(np.mean(types == t)) for t in set(types.tolist())}
    same = sevs[types == max(share, key=share.get)]
    return {
        "type_share": share,
        "severity_ci": [float(np.percentile(same, 2.5)), float(np.percentile(same, 97.5))]
        if len(same) > 2 else None,
    }


# ---------------------------------------------------------------- 보고

def condition_summary(data, best):
    """조건별 정답률과, 최적 모델이 예측한 정답률을 나란히 놓는다."""
    d = perceived_delta_e(data["c1"], data["c2"], best["type"], best["severity"])
    pred = p_correct(d, best["threshold"], best["width"], best["lapse"])
    rows = []
    for cid in sorted(set(data["condition_id"].tolist())):
        m = (data["condition_id"] == cid) & ~data["is_catch"]
        if not m.any():
            continue
        rows.append({
            "condition_id": cid,
            "dir_name": data["dir_name"][m][0],
            "n": int(m.sum()),
            "observed": float(data["correct"][m].mean()),
            "predicted": float(pred[m].mean()),
            "median_t": float(np.median(data["t"][m])),
            "median_perceived_de": float(np.median(d[m])),
        })
    return rows


def direction_summary(data):
    rows = []
    for name in sorted(set(data["dir_name"].tolist())):
        m = (data["dir_name"] == name) & ~data["is_catch"]
        if not m.any():
            continue
        rows.append({"dir_name": name, "n": int(m.sum()),
                     "accuracy": float(data["correct"][m].mean()),
                     "median_t": float(np.median(data["t"][m]))})
    return rows


def report(data, results, boot=None):
    best = results[0]
    per_type = best_per_type(results)
    lines = []
    A = lines.append

    A("=" * 66)
    A("추정 결과")
    A("=" * 66)
    A("  유형          %s" % best["type"])
    A("  severity      %.3f" % best["severity"])
    A("  구별 역치 θ   %.2f  (ΔE00, 시뮬레이션된 지각 공간에서)" % best["threshold"])
    A("  완만함 w      %.2f" % best["width"])
    A("  실수율 λ      %.3f" % best["lapse"])
    A("  로그가능도    %.3f  (시행 %d)" % (best["loglik"], len(data["correct"])))
    if best["type"] != "normal":
        ci = severity_interval(results, best["type"])
        if ci:
            A("  severity 구간 %.2f ~ %.2f  (프로파일 가능도 95%%)" % ci)
        if best["severity"] >= 0.995:
            A("  ! severity 가 격자 상한(1.0)에 붙었다. '정확히 1.0'이 아니라")
            A("    '1.0 이상, 모델이 표현할 수 있는 최대'로 읽어야 한다.")

    A("")
    A("유형별 최적 적합 (로그가능도가 클수록 응답을 잘 설명)")
    for t in ("protan", "deutan", "normal"):
        if t not in per_type:
            continue
        r = per_type[t]
        mark = "  <-- 선택" if r is best else ""
        A("  %-7s s=%.2f  LL=%9.3f  ΔLL=%7.3f%s"
          % (t, r["severity"], r["loglik"], r["loglik"] - best["loglik"], mark))

    if best["type"] in mc.TYPES:
        other = "deutan" if best["type"] == "protan" else "protan"
        if other in per_type:
            dll = best["loglik"] - per_type[other]["loglik"]
            verdict = ("확실" if dll > 4.6 else "보통" if dll > 2.3 else "불충분")
            A("")
            A("  유형 판정 근거: %s 가 %s 보다 ΔLL=%.2f 우세 -> 증거 %s"
              % (best["type"], other, dll, verdict))
            A("  (ΔLL 2.3 은 우도비 10배, 4.6 은 100배에 해당)")
        if "normal" in per_type:
            dll = best["loglik"] - per_type["normal"]["loglik"]
            A("  색각이상 여부: normal 모델보다 ΔLL=%.2f %s"
              % (dll, "우세" if dll > 2.3 else "차이 미미 -> 이상 근거 약함"))

    if boot:
        A("")
        A("부트스트랩 (%d 회)" % boot.get("n", 0))
        for t, sh in sorted(boot["type_share"].items(), key=lambda kv: -kv[1]):
            A("  유형이 %-7s 로 나온 비율  %.2f" % (t, sh))
        if boot["severity_ci"]:
            A("  severity 95%% 구간  %.2f ~ %.2f" % tuple(boot["severity_ci"]))

    A("")
    A("방향별 정답률 (색각이상이면 해당 유형의 혼동 방향에서 낮게 나온다)")
    A("  %-14s %5s %10s %10s" % ("방향", "시행", "정답률", "중앙 t"))
    for r in direction_summary(data):
        A("  %-14s %5d %10.2f %10.4f" % (r["dir_name"], r["n"], r["accuracy"], r["median_t"]))

    A("")
    A("조건별 관측 vs 모델 예측 정답률")
    A("  %-16s %5s %10s %10s %12s" % ("조건", "시행", "관측", "예측", "지각ΔE00"))
    for r in condition_summary(data, best):
        A("  %-16s %5d %10.2f %10.2f %12.2f"
          % (r["condition_id"], r["n"], r["observed"], r["predicted"],
             r["median_perceived_de"]))

    if data["is_catch"].any():
        m = data["is_catch"]
        acc = data["correct"][m].mean()
        A("")
        A("확인용 쉬운 시행 정답률 %.2f (%d 회)" % (acc, int(m.sum())))
        if acc < 0.9:
            # 확인용 시행이 색각과 무관한 조건에서 나왔는지 확인한다.
            # 옛 데이터는 각 조건의 최대 색차를 썼기 때문에, 혼동축 조건의
            # 확인용 시행은 해당 유형에게 여전히 불가능하다.
            catch_conds = set(data["condition_id"][m].tolist())
            if any(not c.startswith("catch") for c in catch_conds):
                A("  주의: 이 데이터는 확인용 시행을 각 조건의 최대 색차로 만든")
                A("  옛 방식이다. 혼동축 조건의 확인용 시행은 해당 유형에게")
                A("  여전히 불가능하므로, 낮은 정답률이 집중도 저하를 뜻하지 않는다.")
                A("  실수율 λ 가 과대추정되었을 수 있다.")
            else:
                A("  0.9 미만이다 — 집중도 저하나 조작 실수를 의심할 것.")

    A("=" * 66)
    return "\n".join(lines)


# ---------------------------------------------------------------- 그림

def make_plot(data, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = results[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # 1) severity 프로파일 가능도
    ax = axes[0]
    for t, style in (("protan", "-o"), ("deutan", "-s")):
        rows = sorted([r for r in results if r["type"] == t], key=lambda r: r["severity"])
        if rows:
            ax.plot([r["severity"] for r in rows], [r["loglik"] for r in rows],
                    style, ms=3, lw=1.2, label=t)
    nrm = [r for r in results if r["type"] == "normal"]
    if nrm:
        ax.axhline(nrm[0]["loglik"], ls=":", c="gray", label="normal")
    ax.axhline(best["loglik"] - 1.92, ls="--", c="k", lw=0.8, label="95% cutoff")
    ax.set_xlabel("simulation severity")
    ax.set_ylabel("log likelihood")
    ax.set_title("severity profile")
    ax.legend(fontsize=8)
    lo = best["loglik"] - 25
    ax.set_ylim(lo, best["loglik"] + 2)

    # 2) 심리측정 곡선
    ax = axes[1]
    d = perceived_delta_e(data["c1"], data["c2"], best["type"], best["severity"])
    edges = np.quantile(d, np.linspace(0, 1, 9))
    edges = np.unique(edges)
    for i in range(len(edges) - 1):
        m = (d >= edges[i]) & (d <= edges[i + 1])
        if m.sum() >= 3:
            ax.plot(d[m].mean(), data["correct"][m].mean(), "ko", ms=6)
    xs = np.linspace(0, max(d.max(), best["threshold"] * 2), 200)
    ax.plot(xs, p_correct(xs, best["threshold"], best["width"], best["lapse"]),
            "-", lw=1.8, label="fitted")
    ax.axhline(GUESS_RATE, ls=":", c="gray")
    ax.axvline(best["threshold"], ls="--", c="r", lw=1,
               label="θ = %.2f" % best["threshold"])
    ax.set_xlabel("perceived ΔE00 under fitted model")
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0, 1.02)
    ax.set_title("psychometric function")
    ax.legend(fontsize=8)

    # 3) 방향별 정답률
    ax = axes[2]
    rows = direction_summary(data)
    names = [r["dir_name"] for r in rows]
    ax.bar(range(len(rows)), [r["accuracy"] for r in rows], color="0.4")
    ax.axhline(GUESS_RATE, ls=":", c="r", label="chance")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("accuracy by probe direction")
    ax.legend(fontsize=8)

    fig.suptitle("%s  severity=%.2f  θ=%.2f" % (best["type"], best["severity"],
                                                best["threshold"]))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description="색쌍 응답 -> 개인 파라미터 추정")
    p.add_argument("csv", help="experiment.py 가 저장한 *_trials.csv")
    p.add_argument("--sev-step", type=float, default=0.02, help="severity 격자 간격")
    p.add_argument("--bootstrap", type=int, default=0, help="부트스트랩 반복 수 (0이면 생략)")
    p.add_argument("--plot", nargs="?", const="auto", default=None,
                   help="그림 저장 경로 (생략하면 csv 옆에 만든다)")
    p.add_argument("--json", nargs="?", const="auto", default=None,
                   help="추정 결과를 JSON 으로 저장할 경로")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    data = load_trials(args.csv)
    results = fit(data, sev_step=args.sev_step, verbose=args.verbose)

    boot = None
    if args.bootstrap:
        boot = bootstrap(data, n=args.bootstrap)
        boot["n"] = args.bootstrap

    print(report(data, results, boot))

    stem = os.path.splitext(args.csv)[0].replace("_trials", "")
    if args.plot is not None:
        path = stem + "_fit.png" if args.plot == "auto" else args.plot
        make_plot(data, results, path)
        print("그림 저장: %s" % path)
    if args.json is not None:
        path = stem + "_params.json" if args.json == "auto" else args.json
        best = results[0]
        payload = {
            "source_csv": os.path.abspath(args.csv),
            "n_trials": int(len(data["correct"])),
            "type": best["type"],
            "severity": best["severity"],
            "confusion_threshold": best["threshold"],
            "width": best["width"],
            "lapse": best["lapse"],
            "loglik": best["loglik"],
            "severity_ci_profile": severity_interval(results, best["type"]),
            "per_type": {k: {kk: vv for kk, vv in v.items()}
                         for k, v in best_per_type(results).items()},
            "bootstrap": boot,
            "conditions": condition_summary(data, best),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("파라미터 저장: %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
