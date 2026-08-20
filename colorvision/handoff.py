"""추정한 개인 파라미터를 팀 baseline 이 쓰는 형태로 넘긴다.

이창민 파트의 `config.ConfusionThresholds` 는 세 값을 요구하는데, 그 모듈은
일부러 기본값을 두지 않는다 ("임계값은 사용자 실험으로 조정해야 한다").
그 세 값을 채우는 것이 이 파트의 최종 산출물이다.

    original_min   원본에서 구분되던 색쌍인가 (정상 색각 기준)
    simulated_max  시뮬레이션 후 이 값보다 가까우면 혼동으로 본다
    target         보정 후 이만큼은 벌려야 한다

어떻게 정하는가
---------------
estimate.py 가 맞춘 심리측정 함수는

    P(정답) = γ + (1 - γ - λ) · σ((d - θ) / w),   γ = 1/4

인데, 여기서 γ 는 4지선다라 찍어서 맞는 몫이고 λ 는 딴 짓하다 틀리는 몫이다.
둘 다 색 지각과는 무관하다. 그래서 가운데의 **순수 구별 확률**만 쓴다.

    P(구별) = σ((d - θ) / w)

이걸 뒤집으면 "구별 확률이 p 가 되는 색차" 가 나온다.

    d(p) = θ + w · ln( p / (1 - p) )

이 값으로 임계값을 정한다. 손으로 고르지 않고 측정에서 끌어내는 것이다.

    simulated_max = d(p_low)     기본 0.50  — 반반도 못 가리면 혼동으로 본다
    target        = d(p_high)    기본 0.95  — 이만큼 벌리면 거의 항상 가려진다

p_low 을 0.50 으로 두는 이유
----------------------------
처음에는 0.25 로 잡았다. "확실히 못 볼 때만 혼동으로 치자"는 생각이었는데,
색보정에서는 그 방향이 틀렸다. 놓치면 사용자가 여전히 그림을 못 읽지만,
조금 과하게 고치면 색이 약간 달라질 뿐이다. **놓치는 쪽이 더 나쁘다.**

실제로 protan 혼동선 위 색쌍 1412개로 재보니 차이가 컸다.

    기준 0.62 (25% 지점)  ->  37% 만 혼동으로 잡힘
    기준 0.95 (50% 지점)  ->  70% 가 잡힘
    기준 1.28 (75% 지점)  ->  86%
    기준 3.00 (선행 논문)  ->  100%

0.50 은 자의성도 덜하다. 그 지점이 곧 실험이 직접 추정한 역치 θ 이고,
심리물리학에서 역치를 말할 때 관례적으로 쓰는 지점이다.

⚠️ 이걸 바꿨다고 문제가 다 풀린 것은 아니다. 0.95 도 결국 **큰 패치를
붙여놓고 잰 값**이라 얇은 선에는 여전히 작을 수 있다. 그건 기하 조건을
바꿔 다시 재야 알 수 있다 (experiment_geometry.py).

λ 를 빼고 계산하는 이유가 하나 더 있다. 실수율이 조금만 높아도 정답률
천장이 0.93 같은 값이 되어 "정답률 95%" 라는 목표 자체가 도달 불가능해진다.
구별 확률로 쓰면 그런 문제가 없다.

⚠️ 이 역치는 **가장 좋은 조건에서 잰 값**이다. 큰 균일 패치를 나란히 놓고
시간 제한 없이 고르게 한 결과이므로, 실제 이미지의 얇은 선이나 작은 글씨,
텍스처가 섞인 영역에서는 이보다 나쁘다. 그래서 `margin` 으로 여유를 줄 수
있게 해 두었다. 여유를 얼마나 줘야 하는지는 실제 이미지 실험으로 답할 문제다.

실행
    python -m colorvision.handoff data/P01_..._params.json
"""

import argparse
import json
import math
import os
import sys

from .estimate import GUESS_RATE

# 원본 색쌍이 "원래 구분되던 색" 인지 보는 하한의 기본값.
# 2017 이주선·한동일 논문이 쓴 3 을 잠정값으로 둔다 (실행계획 §12 미해결 항목).
# 정상 색각자 대상 측정이 생기면 그 값으로 교체할 것.
DEFAULT_ORIGINAL_MIN = 3.0


def delta_e_at_discrimination(threshold, width, p):
    """구별 확률이 p 가 되는 지각 색차 ΔE00 (찍기와 실수를 제외한 값)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p 는 0 과 1 사이여야 한다. 받은 값 %.3f" % p)
    return threshold + width * math.log(p / (1.0 - p))


def accuracy_at(threshold, width, lapse, d):
    """그 색차에서 실제로 관찰될 4지선다 정답률 (찍기와 실수 포함)."""
    z = (d - threshold) / width
    return GUESS_RATE + (1.0 - GUESS_RATE - lapse) / (1.0 + math.exp(-z))


def baseline_thresholds(params, p_low=0.50, p_high=0.95, margin=1.0,
                        original_min=DEFAULT_ORIGINAL_MIN):
    """추정 결과 dict -> 창민 파트의 ConfusionThresholds 세 값.

    margin 은 실험실 역치가 실제 이미지에서 낙관적이라는 점을 보정하는
    배수다. 1.0 이면 측정값 그대로 쓴다.
    """
    th = float(params["confusion_threshold"])
    w = float(params["width"])
    lp = float(params["lapse"])

    sim_max = delta_e_at_discrimination(th, w, p_low) * margin
    target = delta_e_at_discrimination(th, w, p_high) * margin
    sim_max = max(sim_max, 0.0)
    target = max(target, sim_max)
    # 창민 모듈의 제약: original_min >= simulated_max
    orig_min = max(float(original_min), sim_max)

    return {
        "cvd_type": params["type"],
        "severity": float(params["severity"]),
        "original_min": orig_min,
        "simulated_max": sim_max,
        "target": target,
        "derived_from": {
            "confusion_threshold": th, "width": w, "lapse": lp,
            "p_low": p_low, "p_high": p_high, "margin": margin,
            "original_min_source": ("측정값" if orig_min > original_min
                                    else "잠정값 %.1f (2017 논문)" % original_min),
        },
    }


def performance_table(params, points=(0.05, 0.25, 0.50, 0.75, 0.95, 0.99)):
    """구별 확률별 필요한 색차. 임계값을 왜 그렇게 잡았는지 보여주는 표."""
    th, w, lp = (float(params["confusion_threshold"]), float(params["width"]),
                 float(params["lapse"]))
    return [(p, delta_e_at_discrimination(th, w, p),
             accuracy_at(th, w, lp, delta_e_at_discrimination(th, w, p)))
            for p in points]


def render(out, params):
    L = []
    A = L.append
    A("=" * 62)
    A("팀 baseline 인계값")
    A("=" * 62)
    A("  유형          %s" % out["cvd_type"])
    A("  severity      %.3f" % out["severity"])
    A("")
    A("  original_min   %6.2f   (%s)" % (out["original_min"],
                                         out["derived_from"]["original_min_source"]))
    A("  simulated_max  %6.2f   (구별 확률 %.0f%% 지점)"
      % (out["simulated_max"], out["derived_from"]["p_low"] * 100))
    A("  target         %6.2f   (구별 확률 %.0f%% 지점)"
      % (out["target"], out["derived_from"]["p_high"] * 100))
    if out["derived_from"]["margin"] != 1.0:
        A("  (여유 배수 %.2f 적용됨)" % out["derived_from"]["margin"])
    A("")
    A("  구별 확률별 필요한 지각 색차 ΔE00")
    A("    %8s  %8s  %s" % ("구별확률", "ΔE00", "실제 정답률(찍기·실수 포함)"))
    for p, d, acc in performance_table(params):
        A("    %7.0f%%  %8.2f  %14.2f" % (p * 100, d, acc))
    A("")
    A("창민 파트에 그대로 넣을 코드")
    A("-" * 62)
    A("from config import ConfusionThresholds")
    A("from baseline import run_baseline")
    A("")
    A("thresholds = ConfusionThresholds(")
    A("    original_min=%.2f," % out["original_min"])
    A("    simulated_max=%.2f," % out["simulated_max"])
    A("    target=%.2f," % out["target"])
    A(")")
    A("result = run_baseline(color_a, color_b,")
    A("                      cvd_type=%r, severity=%.3f," % (out["cvd_type"], out["severity"]))
    A("                      thresholds=thresholds)")
    A("-" * 62)
    A("")
    A("주의: 이 역치는 큰 균일 패치를 나란히 놓고 시간 제한 없이 잰 값이라")
    A("      가장 좋은 조건의 수치다. 얇은 선이나 작은 글씨에서는 더 나쁘다.")
    A("      실제 이미지에서 부족하면 --margin 으로 여유를 줄 것.")
    A("=" * 62)
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description="개인 파라미터 -> 팀 baseline 임계값")
    p.add_argument("params", help="estimate.py 가 저장한 *_params.json")
    p.add_argument("--p-low", type=float, default=0.50,
                   help="이 구별 확률 아래는 '혼동'으로 본다 (0.50 = 측정된 역치 자체)")
    p.add_argument("--p-high", type=float, default=0.95,
                   help="보정 목표로 삼을 구별 확률")
    p.add_argument("--margin", type=float, default=1.0,
                   help="실험실 역치에 곱할 여유 배수")
    p.add_argument("--original-min", type=float, default=DEFAULT_ORIGINAL_MIN)
    p.add_argument("--json", nargs="?", const="auto", default=None,
                   help="결과를 JSON 으로 저장")
    args = p.parse_args(argv)

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)

    out = baseline_thresholds(params, p_low=args.p_low, p_high=args.p_high,
                              margin=args.margin, original_min=args.original_min)
    print(render(out, params))

    if args.json is not None:
        path = (os.path.splitext(args.params)[0].replace("_params", "") +
                "_baseline.json") if args.json == "auto" else args.json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("저장: %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
