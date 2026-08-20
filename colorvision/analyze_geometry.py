"""기하 조건별 역치와 그 비율을 계산한다.

유형과 severity 는 본 실험에서 확정한 값으로 **고정**하고, 조건마다
역치 θ 만 다시 맞춘다. 유형·severity 는 그 사람의 생리적 특성이라 화면
조건에 따라 변할 리 없다. 자유롭게 두면 기하 효과를 severity 가 흡수해
정작 보려는 변화가 가려진다.

⚠️ 조건마다 따로 낸다. **모든 상황에 같은 배수를 쓸 수 있다고 가정하지 않는다.**
얇은 선의 배수와 멀리 떨어진 영역의 배수가 다를 수 있고, 그 자체가 결과다.

측정된 정밀도 (가상 관찰자, 참 배수를 심어두고 8회 반복)
-------------------------------------------------------
    설정                총시행   조건당   평균 편향   상대 SD
    기하6 x 색2 x 14     179      30     불안정      1.34
    기하6 x 색3 x 14     270      45       -8%      0.71
    기하4 x 색3 x 14     178      45      -13%      0.86

**색 조건 2개로는 안 된다.** 계단법 하나가 수렴하지 못하면 그 조건의 배수가
통째로 튀는데, 색이 2개뿐이면 그걸 메울 자료가 없다. 3개면 편향이 10% 안으로
들어온다.

다만 상대 표준편차가 0.7 수준이라 **한 세션의 배수는 넓은 구간으로 봐야 한다.**
「2배와 5배」는 갈라도 「2배와 2.5배」는 못 가린다. 그래서 점추정값만 쓰지 말고
부트스트랩 구간을 함께 보고, 결론이 필요하면 세션을 여러 번 돌려 모을 것.

실행
    python -m colorvision.analyze_geometry data/P01_..._geom_trials.csv \
        --type protan --severity 0.95 --plot
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from .estimate import fit_threshold, threshold_interval

REFERENCE = "reference"


def load(path, include_practice=False):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["phase"] == "practice" and not include_practice:
                continue
            rows.append(r)
    if not rows:
        raise SystemExit("시행이 없다: %s" % path)
    return rows


def subset(rows, geometry=None, drop_catch=True):
    """estimate.fit_threshold 가 받는 형태로 추린다."""
    sel = [r for r in rows
           if (geometry is None or r["geometry"] == geometry)
           and not (drop_catch and r["is_catch"] == "1")]
    if not sel:
        return None
    return {
        "c1": np.array([[float(r["a_lin_r"]), float(r["a_lin_g"]), float(r["a_lin_b"])]
                        for r in sel]),
        "c2": np.array([[float(r["b_lin_r"]), float(r["b_lin_g"]), float(r["b_lin_b"])]
                        for r in sel]),
        "correct": np.array([r["correct"] == "1" for r in sel]),
    }, sel


def geometry_stats(sel):
    """그 조건에서 실제로 화면에 그려진 크기·거리."""
    def num(key):
        vals = [float(r[key]) for r in sel if r[key] not in ("", None)]
        return float(np.median(vals)) if vals else None
    return {
        "min_width_px": num("min_width_px"),
        "element_area_px": num("element_area_px"),
        "center_distance_px": num("neighbor_center_distance_px"),
        "element_angle_deg": num("element_angle_deg"),
        "separation_angle_deg": num("separation_angle_deg"),
        "background_hex": sel[0]["background_hex"],
        "n": len(sel),
        "accuracy": float(np.mean([r["correct"] == "1" for r in sel])),
        "median_t": float(np.median([float(r["t"]) for r in sel])),
    }


def bootstrap_ratios(rows, cvd_type, severity, n=200, seed=0,
                     reference=REFERENCE):
    """조건별 역치 비율의 불확실성을 부트스트랩으로 잰다.

    한 세션에서 나온 배수는 생각보다 많이 흔들린다. 가상 관찰자로 재보니
    상대 표준편차가 0.3~0.4 였다. 「2배」와 「5배」는 구별하지만 「2배」와
    「2.5배」는 구별하지 못하는 정밀도다. 그러니 점추정값만 보고 결론을
    내리면 안 되고, 이 구간을 함께 봐야 한다.

    조건 안에서만 복원추출해 설계 균형(조건별 시행 수)을 유지한다.
    """
    rng = np.random.default_rng(seed)
    by_geom = {}
    for r in rows:
        if r["is_catch"] == "1":
            continue
        by_geom.setdefault(r["geometry"], []).append(r)
    if reference not in by_geom:
        return {}

    acc = {}
    for _ in range(n):
        thetas = {}
        for name, sel in by_geom.items():
            idx = rng.integers(0, len(sel), len(sel))
            boot = [sel[k] for k in idx]
            data, _ = subset(boot, geometry=None, drop_catch=False)
            thetas[name] = fit_threshold(data, cvd_type, severity)["threshold"]
        ref = thetas.get(reference)
        if not ref:
            continue
        for name, th in thetas.items():
            acc.setdefault(name, []).append(th / ref)

    return {name: {"lo": float(np.percentile(v, 2.5)),
                   "hi": float(np.percentile(v, 97.5)),
                   "median": float(np.median(v))}
            for name, v in acc.items()}


def analyze(rows, cvd_type, severity, n_bootstrap=200):
    names = []
    for r in rows:
        if r["geometry"] not in names and r["is_catch"] != "1":
            names.append(r["geometry"])
    if REFERENCE not in names:
        raise SystemExit(
            "'%s' 조건이 데이터에 없다. 비율을 계산할 기준이 없으므로 중단한다."
            % REFERENCE)
    # 기준을 먼저 계산
    names = [REFERENCE] + [n for n in names if n != REFERENCE]

    out = []
    ref_theta = None
    for name in names:
        got = subset(rows, geometry=name)
        if got is None:
            continue
        data, sel = got
        fit = fit_threshold(data, cvd_type, severity)
        ci = threshold_interval(data, cvd_type, severity, fit["loglik"])
        rec = {"geometry": name, **fit, "ci": ci, **geometry_stats(sel)}
        if name == REFERENCE:
            ref_theta = fit["threshold"]
        rec["ratio"] = (fit["threshold"] / ref_theta) if ref_theta else None
        out.append(rec)

    boot = bootstrap_ratios(rows, cvd_type, severity, n=n_bootstrap) if n_bootstrap else {}
    for rec in out:
        rec["ratio_ci"] = boot.get(rec["geometry"])
    return out, {"n_bootstrap": n_bootstrap, "reference": REFERENCE}


def report(results, joint, cvd_type, severity):
    L = []
    A = L.append
    A("=" * 78)
    A("기하 조건별 색 구별 역치")
    A("=" * 78)
    A("  유형 %s / severity %.3f 고정, 역치 θ 만 조건별로 재추정" % (cvd_type, severity))
    A("")
    A("  %-12s %5s %8s %8s %16s %9s %9s" %
      ("조건", "시행", "역치 θ", "비율", "비율 95% 구간", "최소폭px", "중심거리px"))
    A("  " + "-" * 74)
    for r in results:
        ratio = "   -  " if r["ratio"] is None else "%5.2fx" % r["ratio"]
        ci = r.get("ratio_ci")
        cis = "      -       " if not ci else "%5.2f ~ %5.2f  " % (ci["lo"], ci["hi"])
        A("  %-12s %5d %8.2f %8s %16s %9.0f %9.0f" %
          (r["geometry"], r["n"], r["threshold"], ratio, cis,
           r["min_width_px"] or 0, r["center_distance_px"] or 0))
    A("  " + "-" * 74)
    A("  구간은 시행을 복원추출해 얻은 것이다. 한 세션의 배수는 상대 표준편차")
    A("  0.3~0.4 수준으로 흔들린다 — 2배와 5배는 갈라도 2배와 2.5배는 못 가린다.")

    if any(r.get("element_angle_deg") for r in results):
        A("")
        A("  시야각 환산 (다른 화면과 비교할 때 이 값을 쓸 것)")
        A("  %-12s %14s %16s" % ("조건", "요소 크기(도)", "요소 간 거리(도)"))
        for r in results:
            if r.get("element_angle_deg"):
                A("  %-12s %14.3f %16.3f" %
                  (r["geometry"], r["element_angle_deg"],
                   r["separation_angle_deg"] or 0))
    else:
        A("")
        A("  ! 시야각을 계산하지 못했다. 실험 때 --viewing-distance 와")
        A("    --screen-diagonal 을 넣지 않으면 다른 화면의 자료와 비교할 수 없다.")

    ratios = [r["ratio"] for r in results
              if r["geometry"] != REFERENCE and r["ratio"]]
    if ratios:
        A("")
        A("  조건별 비율이 %.2f ~ %.2f 로 퍼져 있다." % (min(ratios), max(ratios)))
        if max(ratios) / max(min(ratios), 1e-9) > 1.5:
            A("  -> 하나의 배수로 환산할 수 없다는 뜻이다. 조건별로 따로 써야 한다.")
        else:
            A("  -> 조건 간 차이가 크지 않다. 다만 조건 수와 시행 수가 적으면")
            A("     우연히 비슷해 보일 수 있으니 반복 측정으로 확인할 것.")
    A("")
    A("  주의: 이 비율은 여기서 쓴 기하 조건에 한정된다. 실제 이미지의 크기·거리")
    A("        분포가 이 조건과 다르면 그대로 옮겨 쓸 수 없다.")
    A("=" * 78)
    return "\n".join(L)


def make_plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["geometry"] for r in results]
    th = [r["threshold"] for r in results]
    lo = [r["threshold"] - (r["ci"][0] if r["ci"] else r["threshold"]) for r in results]
    hi = [(r["ci"][1] if r["ci"] else r["threshold"]) - r["threshold"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.bar(range(len(names)), th, color="0.45")
    ax.errorbar(range(len(names)), th, yerr=[lo, hi], fmt="none",
                ecolor="0.15", capsize=4, lw=1.2)
    ref = next((r["threshold"] for r in results if r["geometry"] == REFERENCE), None)
    if ref:
        ax.axhline(ref, ls="--", c="#1A6E8E", lw=1.2, label="reference")
        ax.legend(fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("threshold  ΔE00 (simulated)")
    ax.set_title("threshold by geometry")

    ax = axes[1]
    xs, ys = [], []
    for r in results:
        w = r.get("element_angle_deg") or r.get("min_width_px")
        if w and r["ratio"]:
            xs.append(w)
            ys.append(r["ratio"])
    if xs:
        ax.scatter(xs, ys, s=44, color="#1A6E8E", zorder=3)
        for r, x, y in zip([r for r in results if r["ratio"]], xs, ys):
            ax.annotate(r["geometry"], (x, y), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.axhline(1.0, ls="--", c="0.5", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("element size  (visual angle deg, or px)")
        ax.set_ylabel("threshold ratio vs reference")
        ax.set_title("smaller elements need larger colour difference")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="기하 조건별 역치 분석")
    p.add_argument("csv")
    p.add_argument("--type", required=True)
    p.add_argument("--severity", type=float, required=True)
    p.add_argument("--plot", nargs="?", const="auto", default=None)
    p.add_argument("--json", nargs="?", const="auto", default=None)
    p.add_argument("--bootstrap", type=int, default=200,
                   help="비율 구간을 구할 부트스트랩 반복 수 (0이면 생략)")
    args = p.parse_args(argv)

    rows = load(args.csv)
    results, joint = analyze(rows, args.type, args.severity,
                             n_bootstrap=args.bootstrap)
    print(report(results, joint, args.type, args.severity))

    stem = os.path.splitext(args.csv)[0].replace("_trials", "")
    if args.plot is not None:
        path = stem + "_geometry.png" if args.plot == "auto" else args.plot
        make_plot(results, path)
        print("그림 저장: %s" % path)
    if args.json is not None:
        path = stem + "_geometry.json" if args.json == "auto" else args.json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"cvd_type": args.type, "severity": args.severity,
                       "reference": REFERENCE, "joint_fit": joint,
                       "conditions": results},
                      f, ensure_ascii=False, indent=2)
        print("저장: %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
