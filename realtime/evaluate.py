"""보고서용 평가 E1~E4. 정답을 아는 선 자극 장면에서 방법들을 같은 조건으로 비교한다.

    python evaluate.py --profile profiles/user.json --display profiles/display.json \
        --student models/student_v1/student.npz --scenes 200 --teacher-scenes 20 --out ../results/realtime/eval

방법
  none          보정 없음
  base_full     기준 모델(고전 달토니제이션 또는 --base-lut의 Ebelin LUT)을 화면 전체에
  base_gated    같은 기준 모델을 혼동 판단된 색 근처에만 (기존 모델 변형 1)
  student       증류 MLP 후보 재검사 + C80(s)
  student_fixed 위와 같지만 굵기 무시(모든 굵기에 C80_ref) — ablation A3
  teacher       쌍 단위 근사 탐색기(앞쪽 --teacher-scenes 장면만)
지표(장면별)
  resolved      정답 혼동 쌍 중 보정 후 Machado ΔE00 ≥ C80(가는 쪽 굵기)인 비율
  reconfused    원래 구별되던 쌍 중 보정 후 C80 미만이 된 수
  changed_px    값이 바뀐 픽셀 비율
  mean_de       원본 대비 평균 ΔE00 (표본 2만 픽셀)
  bystander_de  혼동 쌍에 속하지 않은 선의 평균 색 변화(과잉 보정)
  analysis_ms / apply_ms  분석(화면이 바뀔 때)·적용(매 프레임) 시간
  width_abs_err 굵기 추정 절대오차(px)
결과는 '계산상 결과'다. 사람의 구별 향상 근거가 아니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import common as c
from base_models import GRID, STEP, load_lut
from pipeline import RealtimeCorrector
from scene_gen import gt_pairs, make_scene
from student import Student

BLUE, GRAY, INK, INK2 = "#2a78d6", "#b9b8b2", "#0b0b0b", "#52514e"


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def corrected_color(rc: RealtimeCorrector, col) -> np.ndarray:
    col = np.asarray(col)
    cell = int(col[0] // STEP) * GRID * GRID + int(col[1] // STEP) * GRID + int(col[2] // STEP)
    return np.clip(col + np.rint(rc.delta).reshape(-1, 3)[cell], 0, 255)


def run_method(name, profile, display, frame, meta, student, base, rng):
    eval_profile = profile
    use_profile = profile.with_fixed_threshold() if name == "student_fixed" else profile
    mode = {"student_fixed": "student", "student_aa": "student", "base_gated_aa": "base_gated"}.get(name, name)
    from pipeline import Settings
    settings = Settings(aa_unmix=name.endswith("_aa"))
    rc = RealtimeCorrector(use_profile, display, mode=mode, student=student, base_delta=base, settings=settings)
    t0 = time.perf_counter()
    out, t = rc.process(frame, instant=True)
    analysis_ms = (time.perf_counter() - t0) * 1000 - t.get("apply_ms", 0)
    reps = []
    for _ in range(5):
        t1 = time.perf_counter()
        rc.process(frame)
        reps.append((time.perf_counter() - t1) * 1000)
    lines = meta["lines"]
    new_cols = [corrected_color(rc, l["color"]) for l in lines]
    before = gt_pairs(meta, eval_profile, display)  # 평가는 항상 실제 C80(s) 기준
    after = gt_pairs(meta, eval_profile, display, colors=new_cols)
    conf = [k for k, q in enumerate(before) if q["sim"] < q["need"] and q["orig"] >= eval_profile.original_min]
    fine = [k for k, q in enumerate(before) if q["sim"] >= q["need"]]
    involved = {before[k]["i"] for k in conf} | {before[k]["j"] for k in conf}
    resolved = [after[k]["sim"] >= after[k]["need"] for k in conf]
    w_px = [l["width_px"] for l in lines]
    in_range = [k for k in conf if not eval_profile.out_of_range(float(display.px_to_deg(min(w_px[before[k]["i"]], w_px[before[k]["j"]]))))]
    resolved_in = [after[k]["sim"] >= after[k]["need"] for k in in_range]
    bystander = [float(c.original_delta_e(np.array(l["color"], float), new_cols[i])) for i, l in enumerate(lines) if i not in involved]
    idx = rng.integers(0, frame.shape[0] * frame.shape[1], 20000)
    f0, f1 = frame.reshape(-1, 3)[idx].astype(float), out.reshape(-1, 3)[idx].astype(float)
    row = {
        "method": name,
        "seed": meta["seed"],
        "n_lines": len(lines),
        "gt_confused": len(conf),
        "resolved": float(np.mean(resolved)) if conf else np.nan,
        "resolved_in_range": float(np.mean(resolved_in)) if in_range else np.nan,
        "reconfused": int(sum(after[k]["sim"] < after[k]["need"] for k in fine)),
        "changed_px": float(np.any(out != frame, axis=2).mean()),
        "mean_de": float(np.mean(c.original_delta_e(f0, f1))),
        "bystander_de": float(np.mean(bystander)) if bystander else np.nan,
        "analysis_ms": analysis_ms,
        "apply_ms": float(np.median(reps)),
        "width_abs_err": np.nan,
    }
    if rc.analysis is not None:
        cols_ok = c.srgb_to_oklab(rc.analysis.colors)
        errs = []
        for l in lines:
            d = np.linalg.norm(cols_ok - c.srgb_to_oklab(np.array(l["color"], float)), axis=1)
            k = int(np.argmin(d))
            if d[k] < 0.03 and np.isfinite(rc.analysis.widths_px[k]):
                same = [m["width_px"] for m in lines if m["color"] == l["color"]]
                errs.append(abs(rc.analysis.widths_px[k] - min(same)))
        row["width_abs_err"] = float(np.mean(errs)) if errs else np.nan
    return row


def bar(ax, labels, values, highlight, title, fmt):
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.55, color=[BLUE if l in highlight else GRAY for l in labels])
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=9, color=INK)
    ax.set_yticks(y, labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    ax.margins(x=0.25)


def plot(summary: dict, order: list[str], out: Path) -> None:
    import logging

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    order = [m for m in order if m != "none"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    hl = {"student"}
    bar(axes[0], order, [summary[m]["resolved_in_range"] * 100 for m in order], hl, "Resolved pairs, measured widths (%)", "{:.0f}")
    bar(axes[1], order, [summary[m]["bystander_de"] for m in order], hl, "Color change of unrelated lines (dE00)", "{:.2f}")
    bar(axes[2], order, [summary[m]["apply_ms"] for m in order], hl, "Per-frame apply time (ms)", "{:.1f}")
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=200)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True)
    p.add_argument("--display", required=True)
    p.add_argument("--student", default="models/student_v1/student.npz")
    p.add_argument("--base-lut", default=None)
    p.add_argument("--scenes", type=int, default=200)
    p.add_argument("--teacher-scenes", type=int, default=0)
    p.add_argument("--aa", action="store_true", help="본 연구 + 안티앨리어싱 혼합색 처리(얇은 선 복원) 방법 추가")
    p.add_argument("--seed0", type=int, default=50000)
    p.add_argument("--out", default="../results/realtime/eval")
    p.add_argument("--replot", action="store_true", help="계산 없이 summary.json으로 그림만 다시 그림")
    args = p.parse_args()
    if args.scenes <= 0 or not 0 <= args.teacher_scenes <= args.scenes:
        p.error('scenes must be positive and teacher-scenes must be between 0 and scenes')
    if args.replot:
        data = json.loads((Path(args.out) / "summary.json").read_text(encoding="utf-8"))
        plot(data["summary"], list(data["summary"]), Path(args.out))
        return

    profile = c.UserProfile.load(args.profile)
    display = c.DisplayGeometry(**json.loads(Path(args.display).read_text()))
    student = Student(args.student)
    base = load_lut(args.base_lut) if args.base_lut else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    methods = ["none", "base_full", "base_gated", "student", "student_fixed"] + (["student_aa"] if getattr(args, "aa", False) else [])
    rows = []
    for s in range(args.scenes):
        frame, meta = make_scene(args.seed0 + s, profile, display)
        for m in methods + (["teacher"] if s < args.teacher_scenes else []):
            rows.append(run_method(m, profile, display, frame, meta, student, base, rng))
        if s % 20 == 0:
            print(f"scene {s}/{args.scenes}", flush=True)

    with (out / "per_scene.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    order = methods + (["teacher"] if args.teacher_scenes else [])
    summary = {}
    for m in order:
        r = [x for x in rows if x["method"] == m]
        summary[m] = {k: (float(np.nanmean(v)) if np.isfinite(v := np.array([x[k] for x in r], float)).any() else float("nan")) for k in
                      ("resolved", "resolved_in_range", "reconfused", "changed_px", "mean_de", "bystander_de", "analysis_ms", "apply_ms", "width_abs_err")}
        summary[m]["n_scenes"] = len(r)
    meta = {"profile": profile.__dict__, "display": display.__dict__, "scenes": args.scenes,
            "base": args.base_lut or "classic_daltonization", "git_commit": c.git_commit(),
            "note": "계산상 결과. 임시 프로파일이면 실험 결과가 아님."}
    (out / "summary.json").write_text(json.dumps(json_safe({"meta": meta, "summary": summary}), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    lines = ["| 방법 | 해결률 | 해결률(측정 굵기 범위) | 재혼동/장면 | 바뀐 픽셀 | 평균 ΔE00 | 무관한 선 변화 | 분석 ms | 적용 ms |", "|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        s_ = summary[m]
        lines.append(f"| {m} | {s_['resolved']:.1%} | {s_['resolved_in_range']:.1%} | {s_['reconfused']:.2f} | {s_['changed_px']:.1%} | {s_['mean_de']:.2f} | {s_['bystander_de']:.2f} | {s_['analysis_ms']:.0f} | {s_['apply_ms']:.1f} |")
    (out / "summary.md").write_text("\n".join(lines) + f"\n\n굵기 추정 평균 절대오차: {summary['student']['width_abs_err']:.2f}px\n", encoding="utf-8")
    print("\n".join(lines))

    plot(summary, order, out)
    print("saved", out)


if __name__ == "__main__":
    main()
