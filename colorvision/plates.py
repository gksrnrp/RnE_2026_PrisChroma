"""유사등색 검사판(이시하라식) 생성과 보정.

왜 직접 만드나
--------------
이시하라 원판은 저작권이 있다. 그리고 직접 만들면 세 가지가 좋다.

1. 정답을 우리가 안다.
2. 숨긴 숫자와 배경을 **그 사람의 혼동축 위에** 정확히 놓을 수 있다.
3. 몇 장이든 만들 수 있고, 색차를 원하는 값으로 지정할 수 있다.

유사등색(pseudoisochromatic)이란
--------------------------------
점마다 **밝기를 무작위로 바꿔** 밝기로는 숫자를 읽을 수 없게 만든 판이다.
색이 유일한 단서가 되므로 색각을 재는 도구가 된다. 밝기를 흔들지 않으면
색약자도 명암으로 숫자를 읽어버려 검사가 성립하지 않는다.

무엇에 쓰나
-----------
보정 전후로 같은 판을 읽게 해서 **몇 개나 더 읽히는지** 센다. 선호도 같은
주관적 평가와 달리 **정답이 있는 객관적 측정**이다. 조경선 2015 와
Zhu 2022 가 모두 이 방식을 쓴다 (docs/related-work.md 참고).

severity 를 바꿔가며 보정한 판을 함께 만들면 Zhu 2022 의 singular degree
(구별력이 뛰는 지점) 도 같은 실험에서 얻을 수 있다.

실행
    python -m colorvision.plates --params data/colorvision/PO3_..._params.json
"""

import argparse
import json
import os
import sys

import numpy as np

from . import machado as mc
from .colorspace import (delta_e00_linear, hex_of, linear_to_srgb, srgb_to_lab,
                         srgb_to_linear, to_bytes)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:                                  # pragma: no cover
    Image = None


# ---------------------------------------------------------------- 점 배치

def pack_dots(size, rng, r_min=7, r_max=17, tries=9000, margin=6):
    """원판 안에 겹치지 않는 점들을 채운다.

    큰 점부터 놓고 실패하면 작은 점을 시도하는 단순한 방식이다. 점 크기가
    제각각이어야 크기나 배열로 숫자를 읽는 것을 막을 수 있다.
    """
    R = size / 2.0 - margin
    cx = cy = size / 2.0
    dots = []
    for _ in range(tries):
        r = rng.uniform(r_min, r_max)
        a = rng.uniform(0, 2 * np.pi)
        d = np.sqrt(rng.uniform(0, 1)) * (R - r)
        x, y = cx + d * np.cos(a), cy + d * np.sin(a)
        ok = True
        for (px, py, pr) in dots:
            if (x - px) ** 2 + (y - py) ** 2 < (r + pr + 1.2) ** 2:
                ok = False
                break
        if ok:
            dots.append((x, y, r))
    return dots


def digit_mask(text, size, font_size_frac=0.62):
    """숫자 모양 마스크. 점 중심이 이 안에 있으면 그림(figure) 점이다."""
    img = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(img)
    fs = int(size * font_size_frac)
    font = None
    for path in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\ariblk.ttf",
                 r"C:\Windows\Fonts\malgunbd.ttf"):
        if os.path.exists(path):
            font = ImageFont.truetype(path, fs)
            break
    if font is None:
        font = ImageFont.load_default()
    box = dr.textbbox((0, 0), text, font=font)
    dr.text(((size - (box[2] - box[0])) / 2 - box[0],
             (size - (box[3] - box[1])) / 2 - box[1]), text, fill=255, font=font)
    return np.asarray(img)


# ---------------------------------------------------------------- 색

def jitter_lightness(lin_rgb, rng, spread=0.22):
    """색조·채도는 두고 밝기만 흔든다. 유사등색 판의 핵심 장치."""
    f = float(np.exp(rng.uniform(-spread, spread)))
    return np.clip(np.asarray(lin_rgb) * f, 0.0, 1.0)


def confusion_pair(base_srgb, cvd_type, target_de, severity_for_pair=1.0,
                   t_hi=0.5):
    """정상 색각자에게는 target_de 만큼 다르고, 그 유형에게는 거의 같은 색쌍.

    혼동축 방향으로 벌리므로 시뮬레이션 후 색차는 거의 0 이 된다.
    """
    v, _ = mc.confusion_direction(cvd_type)
    base = srgb_to_linear(np.asarray(base_srgb))
    lo, hi = 0.0, t_hi
    for _ in range(60):
        m = (lo + hi) / 2
        a, b = base - (m / 2) * v, base + (m / 2) * v
        if np.any(a < 0) or np.any(a > 1) or np.any(b < 0) or np.any(b > 1):
            hi = m
            continue
        if float(delta_e00_linear(a, b)) < target_de:
            lo = m
        else:
            hi = m
    a, b = base - (lo / 2) * v, base + (lo / 2) * v
    return np.clip(a, 0, 1), np.clip(b, 0, 1)


# ---------------------------------------------------------------- 판 그리기

def render_plate(text, fig_lin, bg_lin, size=520, seed=0, bg_page=(0.97, 0.97, 0.97)):
    """유사등색 판 한 장."""
    if Image is None:
        raise RuntimeError("Pillow 가 필요하다: pip install pillow")
    rng = np.random.default_rng(seed)
    mask = digit_mask(text, size)
    img = Image.new("RGB", (size, size),
                    tuple(int(v) for v in to_bytes(np.array(bg_page))))
    dr = ImageDraw.Draw(img)
    for (x, y, r) in pack_dots(size, rng):
        inside = mask[int(np.clip(y, 0, size - 1)), int(np.clip(x, 0, size - 1))] > 127
        col = jitter_lightness(fig_lin if inside else bg_lin, rng)
        dr.ellipse([x - r, y - r, x + r, y + r],
                   fill=tuple(int(v) for v in to_bytes(np.clip(linear_to_srgb(col), 0, 1))))
    return img


def simulate_image(img, cvd_type, severity):
    """이미지 전체를 그 사람이 보는 대로 바꾼다 (확인용)."""
    arr = np.asarray(img, dtype=np.float64) / 255.0
    out = mc.simulate_srgb(arr.reshape(-1, 3), cvd_type, severity).reshape(arr.shape)
    return Image.fromarray(to_bytes(out))


# ---------------------------------------------------------------- 보정

def correct_pair(fig_lin, bg_lin, cvd_type, severity, thresholds, src_path,
                 gate=True):
    """창민 파트로 두 대표색을 보정한다.

    판 전체가 두 색으로만 이루어져 있으므로(밝기만 흔들림), 대표색 한 쌍을
    보정하고 그 결과를 각 무리에 그대로 쓰면 된다.

    gate=False 로 두면 **혼동 판정을 건너뛰고 무조건 보정한다.**

    비교 실험에서는 반드시 gate=False 를 쓴다. 판정을 켜두면 severity 가
    바뀔 때 보정이 아예 실행되지 않는 경우가 생겨(실제로 6판 중 3판이 그랬다),
    결과가 나빠도 **severity 추정이 틀린 것인지 파이프라인이 안 켜진 것인지
    구분할 수 없다.** 비교하려는 변수는 severity 하나뿐이어야 한다.
    """
    sys.path.insert(0, src_path)
    A = tuple(int(v) for v in to_bytes(np.clip(linear_to_srgb(fig_lin), 0, 1)))
    B = tuple(int(v) for v in to_bytes(np.clip(linear_to_srgb(bg_lin), 0, 1)))

    if gate:
        from baseline import run_baseline
        from config import ConfusionThresholds
        th = ConfusionThresholds(original_min=thresholds["original_min"],
                                 simulated_max=thresholds["simulated_max"],
                                 target=thresholds["target"])
        r = run_baseline(A, B, cvd_type, severity, th)
        ca, cb = r["corrected_A"], r["corrected_B"]
    else:
        from correction import find_minimum_correction
        from distance import color_distance
        from machado import simulate_color_float
        c = find_minimum_correction(A, B, cvd_type, severity,
                                    thresholds["target"])
        ca, cb = c["corrected_A"], c["corrected_B"]
        sa = simulate_color_float(ca, cvd_type, severity)
        sb = simulate_color_float(cb, cvd_type, severity)
        sd = color_distance(sa, sb, method="ciede2000")
        r = {"status": "corrected" if c["success"] else "search_failed",
             "corrected_A": ca, "corrected_B": cb, "correction": c,
             "after": {"simulated_distance": sd,
                       "target_met": sd >= thresholds["target"]}}

    return (srgb_to_linear(np.array(ca) / 255.0),
            srgb_to_linear(np.array(cb) / 255.0), r)


# ---------------------------------------------------------------- 판 세트

DEFAULT_PLATES = [
    ("74", (0.62, 0.47, 0.36)),
    ("29", (0.55, 0.52, 0.38)),
    ("6",  (0.58, 0.44, 0.42)),
    ("5",  (0.50, 0.50, 0.34)),
    ("45", (0.64, 0.50, 0.40)),
    ("8",  (0.46, 0.46, 0.40)),
]


def build_set(params, thresholds, src_path, outdir, severities, size=520,
              target_de=24.0, seed=0):
    """원본 판 + severity 별 보정 판을 만든다."""
    os.makedirs(outdir, exist_ok=True)
    cvd_type = params["type"]
    manifest = {"cvd_type": cvd_type, "severity_estimated": params["severity"],
                "thresholds": thresholds, "target_delta_e_original": target_de,
                "plates": []}

    for i, (text, base) in enumerate(DEFAULT_PLATES):
        fig, bg = confusion_pair(base, cvd_type, target_de)
        rec = {"index": i, "answer": text, "base_srgb": list(base),
               "figure_hex": hex_of(linear_to_srgb(fig)),
               "background_hex": hex_of(linear_to_srgb(bg)),
               "delta_e_original": round(float(delta_e00_linear(fig, bg)), 2),
               "files": {}}

        img = render_plate(text, fig, bg, size=size, seed=seed + i)
        p = os.path.join(outdir, "plate%02d_original.png" % i)
        img.save(p)
        rec["files"]["original"] = os.path.basename(p)

        for s in severities:
            f2, b2, r = correct_pair(fig, bg, cvd_type, s, thresholds, src_path)
            img2 = render_plate(text, f2, b2, size=size, seed=seed + i)
            p2 = os.path.join(outdir, "plate%02d_s%02d.png" % (i, int(round(s * 100))))
            img2.save(p2)
            rec["files"]["s%.2f" % s] = os.path.basename(p2)
            rec.setdefault("corrected", {})["s%.2f" % s] = {
                "status": r["status"],
                "figure_hex": hex_of(linear_to_srgb(f2)),
                "background_hex": hex_of(linear_to_srgb(b2)),
                "simulated_delta_e_after": round(
                    float(r["after"]["simulated_distance"]), 2),
            }
        manifest["plates"].append(rec)

    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def build_target_sweep(params, src_path, outdir, targets, size=520,
                       target_de=24.0, seed=0, simulated_max=0.95, gate=True):
    """보정 목표값을 올려가며 판을 만든다 — 이미지 조건의 역치를 직접 잰다.

    왜 이렇게 하나: 실험실에서 잰 역치(큰 패치 기준)로 정한 목표값 1.83 으로
    보정한 판이 읽히지 않았다. 그러면 "얼마면 읽히는가" 를 직접 물어보면 된다.
    목표값을 계단처럼 올려가며 처음 읽히는 지점을 찾는다.

    그 값이 **검사판 조건에서의 구별 역치**다. 기하 조건을 흉내내는 대신
    실제 과제로 재는 것이므로 더 직접적이다.
    """
    os.makedirs(outdir, exist_ok=True)
    cvd_type = params["type"]
    severity = float(params["severity"])
    manifest = {"mode": "target_sweep", "cvd_type": cvd_type,
                "severity": severity, "targets": list(targets),
                "target_delta_e_original": target_de, "plates": []}

    for i, (text, base) in enumerate(DEFAULT_PLATES):
        fig, bg = confusion_pair(base, cvd_type, target_de)
        rec = {"index": i, "answer": text,
               "figure_hex": hex_of(linear_to_srgb(fig)),
               "background_hex": hex_of(linear_to_srgb(bg)),
               "simulated_delta_e_original": round(float(delta_e00_linear(
                   mc.simulate_linear(fig, cvd_type, severity),
                   mc.simulate_linear(bg, cvd_type, severity))), 2),
               "files": {}, "achieved": {}}

        render_plate(text, fig, bg, size=size, seed=seed + i).save(
            os.path.join(outdir, "plate%02d_t000.png" % i))
        rec["files"]["original"] = "plate%02d_t000.png" % i

        for tgt in targets:
            th = {"original_min": max(3.0, simulated_max),
                  "simulated_max": simulated_max,
                  "target": max(float(tgt), simulated_max)}
            f2, b2, r = correct_pair(fig, bg, cvd_type, severity, th, src_path,
                                     gate=gate)
            fn = "plate%02d_t%03d.png" % (i, int(round(tgt * 10)))
            render_plate(text, f2, b2, size=size, seed=seed + i).save(
                os.path.join(outdir, fn))
            rec["files"]["t%.1f" % tgt] = fn
            rec["achieved"]["t%.1f" % tgt] = {
                "status": r["status"],
                "simulated_delta_e": round(float(r["after"]["simulated_distance"]), 2),
                "figure_hex": hex_of(linear_to_srgb(f2)),
                "background_hex": hex_of(linear_to_srgb(b2)),
            }
        manifest["plates"].append(rec)

    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def contact_sheet(outdir, manifest, severities, path):
    """한 장에 모아 보기. 행=판, 열=원본과 severity 들."""
    return contact_sheet_cols(outdir, manifest,
                              ["original"] + ["s%.2f" % s for s in severities], path)


def contact_sheet_cols(outdir, manifest, cols, path):
    cell = 190
    W = cell * len(cols)
    H = cell * len(manifest["plates"]) + 30
    sheet = Image.new("RGB", (W, H), (250, 250, 250))
    dr = ImageDraw.Draw(sheet)
    for j, c in enumerate(cols):
        dr.text((j * cell + 8, 8), "원본" if c == "original" else c, fill=(40, 40, 40))
    for i, rec in enumerate(manifest["plates"]):
        for j, c in enumerate(cols):
            fn = rec["files"].get(c)
            if not fn:
                continue
            im = Image.open(os.path.join(outdir, fn)).resize((cell - 8, cell - 8))
            sheet.paste(im, (j * cell + 4, 30 + i * cell + 4))
    sheet.save(path)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description="유사등색 검사판 생성과 보정")
    p.add_argument("--params", required=True, help="estimate.py 의 *_params.json")
    p.add_argument("--src", default="src", help="창민 파트 src 폴더 경로")
    p.add_argument("--outdir", default="data/colorvision/plates")
    p.add_argument("--severities", default="0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--target-sweep", default=None,
                   help="보정 목표값을 올려가며 판을 만든다. 예: 2,3,5,8,12,18")
    p.add_argument("--no-gate", dest="gate", action="store_false",
                   help="혼동 판정을 건너뛰고 무조건 보정한다. 비교 실험에서는 "
                        "반드시 켤 것 — 판정이 켜져 있으면 severity 효과와 "
                        "게이트 효과가 섞인다")
    p.add_argument("--simulated-max", type=float, default=None,
                   help="혼동 판정 임계값을 직접 지정. 목표값 계단 실험에서는 "
                        "모든 판이 보정되도록 원본 시뮬 색차보다 크게 잡아야 한다")
    p.add_argument("--target-de", type=float, default=24.0,
                   help="정상 색각 기준 그림-배경 색차 (ΔE00)")
    p.add_argument("--size", type=int, default=520)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--simulate", action="store_true",
                   help="그 사람이 보는 모습도 함께 저장")
    p.set_defaults(gate=True)
    args = p.parse_args(argv)

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)
    from .handoff import baseline_thresholds
    th = baseline_thresholds(params)
    sev = [float(x) for x in args.severities.split(",")]

    print("유형 %s, 추정 severity %.2f" % (params["type"], params["severity"]))
    print("임계값 original_min=%.2f simulated_max=%.2f target=%.2f"
          % (th["original_min"], th["simulated_max"], th["target"]))
    print("보정 severity: %s\n" % sev)

    if args.target_sweep:
        targets = [float(x) for x in args.target_sweep.split(",")]
        man = build_target_sweep(params, args.src, args.outdir, targets,
                                 size=args.size, target_de=args.target_de,
                                 seed=args.seed,
                                 simulated_max=(args.simulated_max
                                                if args.simulated_max
                                                else th["simulated_max"]),
                                 gate=args.gate)
        print("%-4s %-6s %10s %s" % ("판", "정답", "원본시뮬", "목표별 실제 시뮬 ΔE00"))
        for rec in man["plates"]:
            got = "  ".join("%.1f->%.2f" % (t, rec["achieved"]["t%.1f" % t]["simulated_delta_e"])
                            for t in targets)
            print("%-4d %-6s %10.2f %s" %
                  (rec["index"], rec["answer"], rec["simulated_delta_e_original"], got))
        cols = ["original"] + ["t%.1f" % t for t in targets]
        sheet = contact_sheet_cols(args.outdir, man, cols,
                                   os.path.join(args.outdir, "target_sweep.png"))
        print("")
        print("모아보기: %s" % sheet)
        print("")
        print("읽는 법: 왼쪽(원본)부터 오른쪽으로 가며 처음 숫자가 보이는 열을 찾는다.")
        print("그 열의 목표값이 검사판 조건에서의 구별 역치다.")
        return 0

    man = build_set(params, th, args.src, args.outdir, sev,
                    size=args.size, target_de=args.target_de, seed=args.seed)

    print("%-6s %-6s %-9s %-9s %10s" % ("판", "정답", "그림색", "배경색", "원본ΔE00"))
    for rec in man["plates"]:
        print("%-6d %-6s %-9s %-9s %10.1f" %
              (rec["index"], rec["answer"], rec["figure_hex"],
               rec["background_hex"], rec["delta_e_original"]))

    sheet = contact_sheet(args.outdir, man, sev,
                          os.path.join(args.outdir, "contact_sheet.png"))
    print("\n%d 판 x (원본 + severity %d) 저장: %s" %
          (len(man["plates"]), len(sev), args.outdir))
    print("모아보기: %s" % sheet)

    if args.simulate:
        for rec in man["plates"]:
            for key, fn in rec["files"].items():
                im = Image.open(os.path.join(args.outdir, fn))
                simulate_image(im, params["type"], params["severity"]).save(
                    os.path.join(args.outdir,
                                 fn.replace(".png", "_assim.png")))
        print("시뮬레이션 이미지도 저장했다 (*_assim.png)")

    print("\n읽는 법: 원본을 먼저 읽고 답을 적은 뒤, severity 순서대로 읽는다.")
    print("정답은 manifest.json 에 있으니 채점 전에는 보지 말 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
