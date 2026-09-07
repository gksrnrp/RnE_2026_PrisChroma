"""사람 실험 전 모의검증 (연구계획서 2단계).

가상 참가자를 만들어 다음 네 가지를 확인한다.

    1. 알려진 C80 을 다시 추정할 수 있는가
    2. 심리측정 기울기 w 를 회복할 수 있는가
    3. lapse λ 를 동시에 추정할 수 있는가
    4. 96 시행으로 추정이 불안정하지 않은가

**식별이 불안정하면 실제 자료를 보기 전에** 아래 중 하나로 전환한다.

    - lapse 를 사전값으로 고정
    - 굵기 간 기울기 공유
    - 시행 수 조정

이 판단은 여기서 내리고 문서에 남긴다. 사람 자료를 본 뒤에 유리한 쪽을
고르면 안 된다.

실행
    python -m experiment.parameter_recovery              전체 점검
    python -m experiment.parameter_recovery --quick      빠른 확인
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .fit_c80 import fit_single, fit_shared_w
from .psychometric import c80, simulate_responses, stimulus_levels

# 가상 참가자 참값. 기존 패치 실험에서 얻은 범위를 참고해 넓게 잡았다.
TRUTHS = [
    {"name": "예민·급격", "theta": 1.5, "w": 0.32, "lapse": 0.02},
    {"name": "보통",      "theta": 3.0, "w": 1.00, "lapse": 0.05},
    {"name": "둔감·완만", "theta": 6.0, "w": 2.20, "lapse": 0.05},
    {"name": "실수 많음", "theta": 3.0, "w": 1.00, "lapse": 0.12},
]


def _levels(truth, n_trials, n_levels=8, guess_err=1.0):
    """자극 수준. guess_err 는 사전 C80 추정이 빗나간 정도(배수)."""
    true_c80 = c80(truth["theta"], truth["w"], truth["lapse"])
    lv = stimulus_levels(true_c80 * guess_err, n_levels=n_levels)
    return np.repeat(lv, max(1, n_trials // n_levels))[:n_trials]


def recover_one(truth, n_trials=96, repeats=300, seed=0, fixed_lapse=None,
                n_levels=8, guess_err=1.0):
    """한 참값에 대해 반복 모의 -> 회복 정도."""
    rng = np.random.default_rng(seed)
    true_c80 = c80(truth["theta"], truth["w"], truth["lapse"])
    x = _levels(truth, n_trials, n_levels, guess_err)

    got = {"c80": [], "theta": [], "w": [], "lapse": [], "edge": 0}
    for _ in range(repeats):
        y = simulate_responses(x, truth["theta"], truth["w"], truth["lapse"], rng)
        r = fit_single(x, y, fixed_lapse=fixed_lapse)
        if r["c80"] is None:
            continue
        got["c80"].append(r["c80"])
        got["theta"].append(r["theta"])
        got["w"].append(r["w"])
        got["lapse"].append(r["lapse"])
        got["edge"] += int(r["at_theta_edge"] or r["at_w_edge"])

    c = np.array(got["c80"])
    out = {
        "truth": truth, "n_trials": n_trials, "true_c80": true_c80,
        "c80_mean": float(c.mean()), "c80_bias": float(c.mean() / true_c80 - 1),
        "c80_rel_sd": float(c.std() / c.mean()),
        "w_mean": float(np.mean(got["w"])),
        "w_bias": float(np.mean(got["w"]) / truth["w"] - 1),
        "lapse_mean": float(np.mean(got["lapse"])),
        "edge_rate": float(got["edge"] / max(1, len(c))),
        "n_ok": len(c),
    }
    return out


def _print_table(title, rows):
    print("\n" + title)
    print("-" * 86)
    print("%-12s %8s %9s %9s %9s %9s %8s" %
          ("참값", "참 C80", "추정 C80", "편향", "상대SD", "w 편향", "경계"))
    print("-" * 86)
    for r in rows:
        print("%-12s %8.2f %9.2f %8.0f%% %9.3f %8.0f%% %7.0f%%" %
              (r["truth"]["name"], r["true_c80"], r["c80_mean"],
               r["c80_bias"] * 100, r["c80_rel_sd"], r["w_bias"] * 100,
               r["edge_rate"] * 100))
    print("-" * 86)


def run(repeats=300, n_trials=96, seed=0, quick=False):
    if quick:
        repeats = 80
    results = {"n_trials": n_trials, "repeats": repeats}

    print("=" * 86)
    print("사람 실험 전 모의검증 — 96 시행으로 C80 을 회복할 수 있는가")
    print("=" * 86)
    print("자극 수준 8개, 고정자극(method of constant stimuli), 4지선다")

    free = [recover_one(t, n_trials, repeats, seed + i)
            for i, t in enumerate(TRUTHS)]
    _print_table("① θ, w, λ 를 모두 자유롭게 추정", free)
    results["free"] = free

    fixed = [recover_one(t, n_trials, repeats, seed + 100 + i, fixed_lapse=0.05)
             for i, t in enumerate(TRUTHS)]
    _print_table("② λ 를 0.05 로 고정 (대안 1)", fixed)
    results["fixed_lapse"] = fixed

    # 사전 C80 추정이 빗나갔을 때
    off = [recover_one(TRUTHS[1], n_trials, repeats, seed + 200 + i,
                       guess_err=g) for i, g in enumerate((0.5, 0.7, 1.4, 2.0))]
    print("\n③ 자극 수준을 잘못 배치했을 때 (참값 '보통' 기준)")
    print("-" * 60)
    print("%-16s %10s %10s %10s" % ("사전 추정 배수", "추정 C80", "편향", "상대SD"))
    for g, r in zip((0.5, 0.7, 1.4, 2.0), off):
        print("%-16.1f %10.2f %9.0f%% %10.3f" %
              (g, r["c80_mean"], r["c80_bias"] * 100, r["c80_rel_sd"]))
    results["level_misplacement"] = [
        dict(r, guess_err=g) for g, r in zip((0.5, 0.7, 1.4, 2.0), off)]

    # 시행 수를 바꿔가며
    print("\n④ 시행 수에 따른 C80 정밀도 (참값 '보통')")
    print("-" * 60)
    print("%-12s %10s %10s %10s" % ("시행 수", "추정 C80", "편향", "상대SD"))
    by_n = []
    for n in (48, 96, 144, 192):
        r = recover_one(TRUTHS[1], n, repeats, seed + 300 + n)
        by_n.append(dict(r, n=n))
        print("%-12d %10.2f %9.0f%% %10.3f" %
              (n, r["c80_mean"], r["c80_bias"] * 100, r["c80_rel_sd"]))
    results["by_n_trials"] = by_n

    # 판정
    print("\n" + "=" * 86)
    print("판정")
    print("=" * 86)
    verdict = {}

    sd_free = float(np.mean([r["c80_rel_sd"] for r in free]))
    sd_fixed = float(np.mean([r["c80_rel_sd"] for r in fixed]))
    bias_free = float(np.mean([abs(r["c80_bias"]) for r in free]))
    w_bias_free = float(np.mean([abs(r["w_bias"]) for r in free]))

    print("  C80 상대SD   자유 추정 %.3f  /  λ 고정 %.3f" % (sd_free, sd_fixed))
    print("  C80 편향     자유 추정 %.0f%%" % (bias_free * 100))
    print("  w 편향       자유 추정 %.0f%%" % (w_bias_free * 100))

    verdict["c80_recoverable"] = bias_free < 0.15 and sd_free < 0.25
    verdict["w_recoverable"] = w_bias_free < 0.30
    verdict["lapse_helps_to_fix"] = sd_fixed < sd_free * 0.9
    verdict["c80_rel_sd_free"] = sd_free
    verdict["c80_rel_sd_fixed"] = sd_fixed

    print()
    print("  [%s] C80 회복 (편향 15%% 이내, 상대SD 0.25 이내)"
          % ("통과" if verdict["c80_recoverable"] else "실패"))
    print("  [%s] w 회복 (편향 30%% 이내)"
          % ("통과" if verdict["w_recoverable"] else "실패"))
    print("  [%s] λ 고정이 도움이 되는가"
          % ("예" if verdict["lapse_helps_to_fix"] else "아니오"))

    print("\n  권고:")
    if not verdict["w_recoverable"]:
        print("    - w 가 회복되지 않는다. **굵기 간 기울기 공유**(fit_shared_w)를")
        print("      주 분석으로 사전등록할 것.")
    if verdict["lapse_helps_to_fix"]:
        print("    - λ 를 고정하면 정밀도가 개선된다. 고정값을 사전등록할 것.")
    if not verdict["c80_recoverable"]:
        print("    - 96 시행으로는 C80 이 불안정하다. 시행 수를 늘리거나")
        print("      위 ④ 표를 근거로 조정할 것.")
    if verdict["c80_recoverable"] and verdict["w_recoverable"]:
        print("    - 96 시행 자유 추정으로 진행 가능. 다만 자극 수준 배치가")
        print("      어긋나면 ③ 처럼 나빠지므로 사전 C80 추정을 확보할 것.")
    print("=" * 86)

    results["verdict"] = verdict
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description="사람 실험 전 모의검증")
    p.add_argument("--repeats", type=int, default=300)
    p.add_argument("--n-trials", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--json", default=None, help="결과를 JSON 으로 저장")
    args = p.parse_args(argv)

    res = run(repeats=args.repeats, n_trials=args.n_trials, seed=args.seed,
              quick=args.quick)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("저장: %s" % args.json)
    return 0 if res["verdict"]["c80_recoverable"] else 1


if __name__ == "__main__":
    sys.exit(main())
