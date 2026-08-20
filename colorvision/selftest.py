"""전체 점검. 실험 전에 한 번 돌려서 다 초록인지 확인한다.

    python -m colorvision.selftest

화면 미리보기 PNG 도 함께 만든다 (전체화면으로 띄우지 않고 실제 자극이
어떻게 보이는지 확인하는 용도).
"""

import os
import sys

import numpy as np

# 화면 없는 환경에서도 pygame 이 뜨도록
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append(bool(cond))
    print("  [%s] %s %s" % ("OK  " if cond else "FAIL", name, extra))
    return bool(cond)


def section(title):
    print("\n" + title)
    print("-" * len(title))


# ---------------------------------------------------------------- 1. 색차

def test_colorspace():
    from .colorspace import delta_e00, srgb_to_linear, linear_to_srgb

    section("1. 색공간 / ΔE00")
    # Sharma, Wu, Dalal (2005) 검증 데이터 일부
    cases = [
        ((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
        ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
        ((50, -1.3802, -84.2814), (50, 0, -82.7485), 1.0000),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
        ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
        ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
    ]
    worst = 0.0
    for a, b, want in cases:
        got = float(delta_e00(np.array(a), np.array(b)))
        worst = max(worst, abs(got - want))
    check("ΔE00 이 Sharma 검증 데이터와 일치", worst < 1e-3, "최대오차 %.2e" % worst)

    x = np.random.default_rng(0).random((200, 3))
    err = float(np.abs(linear_to_srgb(srgb_to_linear(x)) - x).max())
    check("sRGB <-> 선형 왕복 일치", err < 1e-12, "최대오차 %.2e" % err)


# ---------------------------------------------------------------- 2. 시뮬레이터

def test_machado():
    from .machado import _self_check
    section("2. Machado 시뮬레이터")
    RESULTS.append(bool(_self_check()))


# ---------------------------------------------------------------- 3. 계단법

def test_staircase():
    from .stimuli import Staircase
    section("3. 계단법 수렴")
    rng = np.random.default_rng(1)
    true_t = 0.03
    ests = []
    for _ in range(400):
        sc = Staircase(t_start=0.12, t_max=0.20, max_trials=24, max_reversals=99)
        while not sc.done:
            p = 0.25 + 0.75 / (1 + np.exp(-(np.log10(sc.t) - np.log10(true_t)) / 0.10))
            sc.update(rng.random() < p * 0.98)
        th = sc.threshold()
        if th:
            ests.append(th)
    e = np.array(ests)
    check("역치 추정이 참값 부근으로 수렴", abs(np.median(e) / true_t - 1) < 0.35,
          "중앙값 %.4f / 참값 %.4f" % (np.median(e), true_t))
    check("반전이 충분히 쌓여 역치가 나옴", len(e) / 400 > 0.95,
          "수렴률 %.2f" % (len(e) / 400))


# ---------------------------------------------------------------- 4. 조건 설계

def test_design():
    from .design import load_design
    from .stimuli import build_conditions
    section("4. 조건 설계")
    d = load_design()
    if not check("conditions.json 존재", d is not None,
                 "" if d else "-> python -m colorvision.design 실행 필요"):
        return
    check("유형 식별력이 측정오차보다 큼", d["identifiability"] > 0.3,
          "%.3f" % d["identifiability"])
    conds = build_conditions()
    check("조건이 실제로 만들어짐", len(conds) >= 8, "%d 개" % len(conds))
    gaps = [c["log_thr_protan"][3] - c["log_thr_deutan"][3] for c in d["conditions"]]
    check("두 유형의 역치 차이 부호가 섞여 있음",
          min(gaps) < -0.1 and max(gaps) > 0.1,
          "%.2f ~ %.2f" % (min(gaps), max(gaps)))
    from .stimuli import build_catch_condition
    from .machado import simulate_linear
    from .colorspace import delta_e00_linear
    cc = build_catch_condition()
    if check("확인용 시행 조건 존재", cc is not None):
        a, b = cc.pair(cc.catch_t)
        des = {"%s %.1f" % (t, sv): float(delta_e00_linear(
            simulate_linear(a, t, sv), simulate_linear(b, t, sv)))
            for t, sv in [("normal", 0.0), ("protan", 1.0), ("deutan", 1.0)]}
        check("확인용 시행이 모든 유형에게 잘 보임", min(des.values()) > 15,
              "  ".join("%s=%.1f" % kv for kv in des.items()))

    lo = d.get("identifiability_by_severity", {})
    if lo:
        k = sorted(lo, key=float)[0]
        print("     참고: severity %.1f 에서 식별력 %.2f — 경증은 유형 판정이 어렵다"
              % (float(k), lo[k]["protan_true"]))


# ---------------------------------------------------------------- 5. 복원

def test_recovery():
    from .estimate import fit
    from .virtual import VirtualObserver, run_virtual_session, rows_to_data
    from .machado import normalize_type
    section("5. 파라미터 복원 (가상 관찰자)")
    truths = [("protan", 0.4), ("deutan", 0.4), ("protan", 0.8), ("deutan", 0.8)]
    hits, serr = [], []
    for ti, (tt, ts) in enumerate(truths):
        for rep in range(3):
            obs = VirtualObserver(tt, ts, threshold=2.5, seed=7000 + 100 * ti + rep)
            rows = run_virtual_session(obs, seed=7000 + 100 * ti + rep)
            b = fit(rows_to_data(rows), sev_step=0.05)[0]
            hits.append(b["type"] == normalize_type(tt))
            serr.append(abs(b["severity"] - ts))
    check("유형 복원율", np.mean(hits) >= 0.9, "%.0f%% (%d 세션)" % (np.mean(hits) * 100, len(hits)))
    check("severity 평균 오차", np.mean(serr) < 0.10, "%.3f" % np.mean(serr))


# ---------------------------------------------------------------- 6. 화면

def test_display(preview_path="data/colorvision/preview_trial.png"):
    section("6. 화면 렌더링 (헤드리스)")
    import pygame
    from .experiment import Experiment, build_parser, make_patch
    from .colorspace import srgb_to_linear, linear_to_srgb, delta_e00_linear

    # 디더링이 실제로 8비트 아래 색차를 살려내는지
    rng = np.random.default_rng(0)
    base = np.array([0.45, 0.45, 0.45])
    tiny = base + np.array([0.0015, 0.0, 0.0])
    a = np.asarray(pygame.surfarray.array3d(make_patch((160, 160), base, rng)), float).mean(axis=(0, 1))
    b = np.asarray(pygame.surfarray.array3d(make_patch((160, 160), tiny, rng)), float).mean(axis=(0, 1))
    check("디더링이 8비트 미만 색차를 보존", abs((b - a)[0] - 0.0015 * 255) < 0.05,
          "요청 %.3f계조 / 실제 %.3f계조" % (0.0015 * 255, (b - a)[0]))

    args = build_parser().parse_args(["--windowed", "--width", "1000",
                                      "--height", "720", "--practice", "0"])
    exp = Experiment(args)
    os.makedirs(os.path.dirname(preview_path) or ".", exist_ok=True)

    # 배치가 매 시행 실제로 달라지는지 확인하면서 미리보기도 만든다
    seen = set()
    for k in range(6):
        cond = exp.conditions[k % len(exp.conditions)]
        j = cond.sample_jitter(exp.nprng, cond.t_max * 0.6, j_max=args.jitter)
        srgb_a, srgb_b = cond.pair_srgb(cond.t_max * 0.6, j)
        colors = [srgb_a] * 4
        colors[k % 4] = srgb_b
        layout = exp.sample_layout()
        seen.add(tuple((r.x, r.y, r.w, sh) for r, sh, _ in layout))
        exp.draw_trial(colors, "미리보기 %d" % (k + 1), layout)
        pygame.image.save(exp.screen,
                          preview_path.replace(".png", "_%d.png" % (k + 1)))
        if k == 0:
            pygame.image.save(exp.screen, preview_path)
    check("시행 화면 렌더링 및 저장", os.path.exists(preview_path), preview_path)
    # 기본은 고정 2x2 다. 배치가 바뀌면 사람의 역치도 달라질 수 있는데 그걸
    # 검증할 방법이 없어서, 색만 흔들고 배치는 고정해 둔다.
    check("기본 배치는 매 시행 동일", len(seen) == 1,
          "6회 중 서로 다른 배치 %d개" % len(seen))

    args.varied_layout = True
    varied = {tuple((r.x, r.y, r.w, sh) for r, sh, _ in exp.sample_layout())
              for _ in range(6)}
    check("--varied-layout 이면 배치가 달라짐", len(varied) == 6,
          "6회 중 서로 다른 배치 %d개" % len(varied))
    args.varied_layout = False
    exp.message_screen(["안내 화면", "이렇게 보입니다."], wait_key=False)
    pygame.image.save(exp.screen, preview_path.replace(".png", "_message.png"))
    pygame.quit()


def main():
    print("=" * 62)
    print("colorvision 전체 점검")
    print("=" * 62)
    test_colorspace()
    test_machado()
    test_staircase()
    test_design()
    test_recovery()
    test_display()
    print("\n" + "=" * 62)
    n_fail = RESULTS.count(False)
    print("%d개 항목 중 %d개 통과%s"
          % (len(RESULTS), len(RESULTS) - n_fail,
             "" if n_fail == 0 else "  <-- %d개 실패" % n_fail))
    print("=" * 62)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
