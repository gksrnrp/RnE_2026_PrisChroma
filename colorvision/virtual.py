"""가상 관찰자로 실험 전체를 사람 없이 돌려본다.

두 가지 용도가 있다.

1. 개발 검증 — 참값 (type, severity, θ) 을 아는 관찰자를 만들고,
   실험 절차와 추정기를 통과시켰을 때 그 값이 되돌아오는지 확인한다
   (parameter recovery). 되돌아오지 않으면 실험 설계가 잘못된 것이다.

2. 실행계획 §3-5 의 대비책 — 실제 피험자를 못 구했을 때
   가상 사용자로 개인화 구조가 작동함을 보이는 데 쓴다.

실행
    python -m colorvision.virtual --type deutan --severity 0.6
    python -m colorvision.virtual --recovery        # 여러 참값 일괄 검증
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np

from . import machado as mc
from .colorspace import delta_e00_linear, hex_of
from .estimate import GUESS_RATE, fit, best_per_type
from .stimuli import build_conditions, make_staircases, run_session

CSV_FIELDS = [
    "trial_index", "phase", "condition_id", "base_hex", "dir_name", "angle_deg",
    "t", "delta_e_original", "color_a_hex", "color_b_hex",
    "a_lin_r", "a_lin_g", "a_lin_b", "b_lin_r", "b_lin_g", "b_lin_b",
    "odd_color", "correct_pos", "response_pos", "correct", "rt_ms",
    "is_catch", "jitter_r", "jitter_g", "jitter_b", "n_reversals", "timestamp",
]


class VirtualObserver:
    """참값 파라미터를 가진 가상 피험자.

    estimate.py 의 모델과 같은 식으로 정답 확률을 만든다. 즉 추정기가
    자기 자신의 생성 모델을 복원하는지 보는 것이므로, 이 검증이 통과한다고
    해서 실제 사람에게 모델이 맞다는 뜻은 아니다. 절차와 정보량의 검증이다.
    """

    def __init__(self, cvd_type, severity, threshold=2.5, width=1.0, lapse=0.02,
                 seed=0):
        self.type = mc.normalize_type(cvd_type)
        self.severity = float(severity)
        self.threshold = float(threshold)
        self.width = float(width)
        self.lapse = float(lapse)
        self.rng = np.random.default_rng(seed)

    def perceived(self, c1, c2):
        return float(delta_e00_linear(mc.simulate_linear(c1, self.type, self.severity),
                                      mc.simulate_linear(c2, self.type, self.severity)))

    def p_correct(self, c1, c2):
        d = self.perceived(c1, c2)
        z = np.clip((d - self.threshold) / self.width, -60, 60)
        return GUESS_RATE + (1 - GUESS_RATE - self.lapse) / (1 + np.exp(-z))

    def respond(self, c1, c2):
        return bool(self.rng.random() < self.p_correct(c1, c2))


def run_virtual_session(observer, n_bases=None, max_trials=16, catch_rate=0.08,
                        seed=0, jitter_max=None):
    """실제 실험과 같은 진행 논리로 시행을 돌리고 CSV 행 목록을 만든다."""
    conditions = build_conditions(n_bases=n_bases, seed=seed)
    stairs = make_staircases(conditions, max_trials)
    rows = []
    rng = np.random.default_rng(seed + 12345)

    def respond(cond, t, is_catch, jitter):
        lin_a, lin_b = cond.pair(t, jitter)
        srgb_a, srgb_b = cond.pair_srgb(t, jitter)
        correct = observer.respond(lin_a, lin_b)
        correct_pos = int(rng.integers(0, 4))
        rows.append({
            "trial_index": len(rows), "phase": "main", "condition_id": cond.cid,
            "base_hex": cond.base_hex, "dir_name": cond.dir_name,
            "angle_deg": "" if cond.angle_deg is None else round(cond.angle_deg, 3),
            "t": round(float(t), 6),
            "delta_e_original": round(cond.delta_e_original(t, jitter), 4),
            "color_a_hex": hex_of(srgb_a), "color_b_hex": hex_of(srgb_b),
            "a_lin_r": lin_a[0], "a_lin_g": lin_a[1], "a_lin_b": lin_a[2],
            "b_lin_r": lin_b[0], "b_lin_g": lin_b[1], "b_lin_b": lin_b[2],
            "odd_color": "b", "correct_pos": correct_pos,
            "response_pos": correct_pos if correct else (correct_pos + 1) % 4,
            "correct": int(correct), "rt_ms": "",
            "is_catch": int(is_catch),
            "jitter_r": jitter[0], "jitter_g": jitter[1], "jitter_b": jitter[2],
            "n_reversals": (len(stairs[cond.cid].reversals)
                            if cond.cid in stairs else ""),
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        })
        return correct

    kw = {} if jitter_max is None else {"jitter_max": jitter_max}
    run_session(conditions, respond, stairs=stairs, max_trials=max_trials,
                catch_rate=catch_rate, seed=seed, **kw)
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def rows_to_data(rows):
    """estimate.load_trials 와 같은 형태를 메모리에서 바로 만든다."""
    return {
        "c1": np.array([[r["a_lin_r"], r["a_lin_g"], r["a_lin_b"]] for r in rows]),
        "c2": np.array([[r["b_lin_r"], r["b_lin_g"], r["b_lin_b"]] for r in rows]),
        "correct": np.array([bool(r["correct"]) for r in rows]),
        "condition_id": np.array([r["condition_id"] for r in rows]),
        "dir_name": np.array([r["dir_name"] for r in rows]),
        "t": np.array([r["t"] for r in rows]),
        "is_catch": np.array([bool(r["is_catch"]) for r in rows]),
    }


# ---------------------------------------------------------------- 복원 검증

def recovery_test(truths, n_bases=None, max_trials=16, repeats=5, sev_step=0.05,
                  seed=0):
    """여러 참값에 대해 추정이 얼마나 되돌아오는지 표로 출력한다."""
    n_cond = len(build_conditions(n_bases=n_bases))
    print("복원 검증  (%s 설계, 조건 %d개, 조건당 최대 %d시행, 참값당 %d회 반복)"
          % ("휴리스틱" if n_bases else "최적화", n_cond, max_trials, repeats))
    print("-" * 78)
    print("%-9s %8s %8s | %-9s %14s %8s %10s"
          % ("참 유형", "참 sev", "참 θ", "유형적중", "추정 sev", "추정 θ", "평균 시행"))
    print("-" * 78)

    all_ok = True
    for ti, (ttype, tsev, tth) in enumerate(truths):
        types_hit, sevs, ths, ns = [], [], [], []
        for rep in range(repeats):
            obs = VirtualObserver(ttype, tsev, threshold=tth,
                                  seed=seed + 1000 * ti + rep)
            rows = run_virtual_session(obs, n_bases=n_bases, max_trials=max_trials,
                                       seed=seed + 1000 * ti + rep)
            res = fit(rows_to_data(rows), sev_step=sev_step)
            b = res[0]
            types_hit.append(b["type"] == mc.normalize_type(ttype))
            sevs.append(b["severity"])
            ths.append(b["threshold"])
            ns.append(len(rows))
        hit = float(np.mean(types_hit))
        all_ok = all_ok and hit >= 0.8
        print("%-9s %8.2f %8.2f | %8.0f%% %6.2f (±%.2f) %8.2f %10.0f"
              % (ttype, tsev, tth, hit * 100, np.mean(sevs), np.std(sevs),
                 np.mean(ths), np.mean(ns)))
    print("-" * 78)
    return all_ok


DEFAULT_TRUTHS = [
    ("protan", 0.30, 2.5),
    ("protan", 0.60, 2.5),
    ("protan", 0.90, 2.5),
    ("deutan", 0.30, 2.5),
    ("deutan", 0.60, 2.5),
    ("deutan", 0.90, 2.5),
    ("deutan", 0.60, 5.0),
]


def main(argv=None):
    p = argparse.ArgumentParser(description="가상 관찰자로 실험/추정 검증")
    p.add_argument("--recovery", action="store_true", help="여러 참값 일괄 복원 검증")
    p.add_argument("--geometry-recovery", action="store_true",
                   help="기하 조건별 역치 배수를 되찾는지 검증")
    p.add_argument("--type", default="deutan")
    p.add_argument("--severity", type=float, default=0.6)
    p.add_argument("--threshold", type=float, default=2.5)
    p.add_argument("--lapse", type=float, default=0.02)
    p.add_argument("--bases", type=int, default=None,
                   help="주면 옛 휴리스틱 설계를 쓴다 (설계 비교용)")
    p.add_argument("--max-trials", type=int, default=16)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="CSV 저장 경로 (단일 세션 모드)")
    args = p.parse_args(argv)

    if args.geometry_recovery:
        ok = geometry_recovery(cvd_type=args.type, severity=args.severity,
                               base_threshold=args.threshold,
                               repeats=args.repeats, seed=args.seed)
        print("모든 조건에서 오차 35%% 이내인가: %s" % ("예" if ok else "아니오"))
        return 0 if ok else 1

    if args.recovery:
        ok = recovery_test(DEFAULT_TRUTHS, n_bases=args.bases,
                           max_trials=args.max_trials, repeats=args.repeats,
                           seed=args.seed)
        print("유형 적중률이 모든 참값에서 80%% 이상인가: %s" % ("예" if ok else "아니오"))
        return 0 if ok else 1

    obs = VirtualObserver(args.type, args.severity, threshold=args.threshold,
                          lapse=args.lapse, seed=args.seed)
    rows = run_virtual_session(obs, n_bases=args.bases, max_trials=args.max_trials,
                               seed=args.seed)
    out = args.out or os.path.join(
        "data", "virtual_%s%.2f_%s_trials.csv"
        % (mc.normalize_type(args.type), args.severity,
           datetime.now().strftime("%Y%m%d_%H%M%S")))
    write_csv(rows, out)
    print("가상 세션 %d 시행 -> %s" % (len(rows), out))

    from .estimate import report
    data = rows_to_data(rows)
    results = fit(data, sev_step=0.02)
    print(report(data, results))
    print("참값: type=%s severity=%.2f θ=%.2f"
          % (mc.normalize_type(args.type), args.severity, args.threshold))
    return 0

# ---------------------------------------------------------------- 기하 조건 복원

class GeometryObserver(VirtualObserver):
    """기하 조건에 따라 역치가 달라지는 가상 피험자.

    조건마다 참 배수를 심어두고, analyze_geometry 가 그 배수를 되찾는지 본다.
    되찾지 못하면 실험 설계나 분석 쪽이 잘못된 것이다.
    """

    def __init__(self, cvd_type, severity, base_threshold, ratios, **kw):
        super().__init__(cvd_type, severity, threshold=base_threshold, **kw)
        self.base_threshold = float(base_threshold)
        self.ratios = dict(ratios)

    def respond_for(self, c1, c2, geometry):
        self.threshold = self.base_threshold * self.ratios.get(geometry, 1.0)
        return self.respond(c1, c2)


def run_geometry_session(observer, max_trials=14, catch_rate=0.06, seed=0,
                         n_colors=3, geometries=None):
    """기하 실험과 같은 진행 논리로 돌리고, 분석기가 읽는 행 목록을 만든다."""
    from .experiment_geometry import pick_color_conditions, _clone_condition
    from .geometry import default_geometries, visual_angle_deg
    from .stimuli import build_catch_condition, make_staircases

    geoms = geometries or default_geometries()
    colors = pick_color_conditions(n_colors, observer.type, observer.severity)
    W, H = 1920, 1080

    pairs, conds = {}, []
    for c in colors:
        for g in geoms:
            cid = "%s__%s" % (c.cid, g.name)
            pairs[cid] = (c, g)
            conds.append(_clone_condition(c, cid))

    stairs = make_staircases(conds, max_trials)
    rows = []

    def respond(cond, t, is_catch, jitter):
        color_cond, geom = pairs.get(cond.cid, (cond, geoms[0]))
        lin_a, lin_b = cond.pair(t, jitter)
        correct = observer.respond_for(lin_a, lin_b, geom.name)
        m = geom.measured(W, H)
        rows.append({
            "phase": "main", "geometry": geom.name, "color_cid": color_cond.cid,
            "t": t, "correct": "1" if correct else "0",
            "is_catch": "1" if is_catch else "0",
            "a_lin_r": lin_a[0], "a_lin_g": lin_a[1], "a_lin_b": lin_a[2],
            "b_lin_r": lin_b[0], "b_lin_g": lin_b[1], "b_lin_b": lin_b[2],
            "background_hex": hex_of(np.array(geom.background)),
            "min_width_px": m["min_width_px"],
            "element_area_px": m["element_area_px"],
            "neighbor_center_distance_px": m["neighbor_center_distance_px"],
            "element_angle_deg": "", "separation_angle_deg": "",
        })
        return correct

    run_session(conds, respond, stairs=stairs, max_trials=max_trials,
                catch_rate=catch_rate, seed=seed,
                catch_condition=build_catch_condition())
    return rows


DEFAULT_GEOMETRY_RATIOS = {
    "reference": 1.0,
    "small": 2.0,
    "tiny": 3.5,
    "thin_line": 3.0,
    "far": 2.2,
    "light_bg": 1.3,
}


def geometry_recovery(cvd_type="protan", severity=0.95, base_threshold=1.0,
                      ratios=None, repeats=3, max_trials=14, seed=0):
    """심어둔 배수를 분석기가 되찾는지 확인한다."""
    from .analyze_geometry import analyze

    ratios = ratios or DEFAULT_GEOMETRY_RATIOS
    got = {}
    for rep in range(repeats):
        obs = GeometryObserver(cvd_type, severity, base_threshold, ratios,
                               seed=seed + 17 * rep)
        rows = run_geometry_session(obs, max_trials=max_trials, seed=seed + 17 * rep)
        res, _ = analyze(rows, cvd_type, severity, n_bootstrap=0)
        for r in res:
            if r["ratio"]:
                got.setdefault(r["geometry"], []).append(r["ratio"])

    print("기하 조건 복원 검증  (참 역치 %.2f, %d회 반복, 조건당 최대 %d시행)"
          % (base_threshold, repeats, max_trials))
    print("-" * 62)
    print("%-12s %10s %10s %10s %8s" % ("조건", "참 배수", "추정 배수", "표준편차", "오차"))
    print("-" * 62)
    ok = True
    for name, truth in ratios.items():
        vals = np.array(got.get(name, []))
        if len(vals) == 0:
            print("%-12s %10.2f %10s" % (name, truth, "측정 실패"))
            ok = False
            continue
        est = float(np.mean(vals))
        err = abs(est - truth) / truth
        ok = ok and err < 0.35
        print("%-12s %10.2f %10.2f %10.2f %7.0f%%"
              % (name, truth, est, float(np.std(vals)), err * 100))
    print("-" * 62)
    return ok


if __name__ == "__main__":
    sys.exit(main())
