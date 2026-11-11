"""정답을 아는 '선 자극 장면' 생성기 (차트·노선도 아님).

배경 위에 여러 색·여러 굵기의 직선을 안티앨리어싱으로 그린다. 일부 색 쌍은
주어진 사용자 프로파일에서 일부러 혼동되도록 고른다. 반환하는 meta에는
각 선의 실제 색·굵기(px)가 들어 있어 평가의 정답으로 쓴다.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

import common as c

WIDTHS_PX = (1, 2, 3, 4, 6, 8, 12, 16, 20)
BACKGROUNDS = ((255, 255, 255), (128, 128, 128), (32, 32, 32))


def _confusing_partner(rng, a, profile: c.UserProfile, target: float, tries: int = 4000):
    """a와 시뮬 후 ΔE00 < target 이면서 원본 ΔE00 ≥ original_min 인 색을 찾는다."""
    a_ok = c.srgb_to_oklab(np.asarray(a, float))
    step = rng.normal(size=(tries, 3))
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    cand_ok = a_ok + step * rng.uniform(0.03, 0.20, (tries, 1))
    cand_ok = cand_ok[c.in_gamut_oklab(cand_ok)]
    cand = np.rint(c.oklab_to_srgb(cand_ok)).clip(0, 255)
    sim = c.simulated_delta_e(np.broadcast_to(a, cand.shape), cand, profile.cvd_type, profile.severity)
    orig = c.original_delta_e(np.broadcast_to(a, cand.shape), cand)
    ok = np.flatnonzero((sim < target) & (orig >= profile.original_min))
    return None if not len(ok) else tuple(int(v) for v in cand[rng.choice(ok)])


def make_scene(seed: int, profile: c.UserProfile, display: c.DisplayGeometry,
               height: int = 720, width: int = 1280, n_lines: int | None = None, ss: int = 4):
    rng = np.random.default_rng(seed)
    bg = BACKGROUNDS[seed % len(BACKGROUNDS)]
    n_lines = n_lines or int(rng.integers(3, 7))
    colors: list[tuple[int, int, int]] = []
    while len(colors) < n_lines:
        base = tuple(int(v) for v in rng.integers(30, 226, 3))
        if c.original_delta_e(np.array(base, float), np.array(bg, float)) < 20:
            continue  # 배경과 너무 비슷한 선은 제외(배경 대비는 이번 범위 밖)
        colors.append(base)
        if len(colors) < n_lines and rng.random() < 0.7:
            w_ref = float(display.px_to_deg(4))
            partner = _confusing_partner(rng, base, profile, float(profile.c80(w_ref)) * rng.uniform(0.5, 1.3))
            if partner and c.original_delta_e(np.array(partner, float), np.array(bg, float)) >= 20:
                colors.append(partner)
    widths = [int(rng.choice(WIDTHS_PX)) for _ in colors]
    img = Image.new("RGB", (width * ss, height * ss), bg)
    draw = ImageDraw.Draw(img)
    gap = height / (len(colors) + 1)
    lines = []
    for i, (col, w) in enumerate(zip(colors, widths)):
        y = (i + 1) * gap + rng.uniform(-gap * 0.2, gap * 0.2)
        x0, x1 = width * rng.uniform(0.05, 0.2), width * rng.uniform(0.75, 0.95)
        draw.line([(x0 * ss, y * ss), (x1 * ss, y * ss)], fill=col, width=w * ss)
        lines.append({"color": list(col), "width_px": w, "y": float(y), "x0": float(x0), "x1": float(x1)})
    frame = np.asarray(img.resize((width, height), Image.Resampling.BOX), dtype=np.uint8)
    return frame, {"seed": seed, "background": list(bg), "lines": lines}


def gt_pairs(meta: dict, profile: c.UserProfile, display: c.DisplayGeometry, colors=None):
    """정답 선 색(또는 보정 후 색 colors)으로 모든 선 쌍의 판정을 계산한다."""
    lines = meta["lines"]
    cols = np.array([l["color"] for l in lines], float) if colors is None else np.asarray(colors, float)
    w_deg = display.px_to_deg(np.array([l["width_px"] for l in lines], float))
    out = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if lines[i]["color"] == lines[j]["color"]:
                continue
            sim = float(c.simulated_delta_e(cols[i], cols[j], profile.cvd_type, profile.severity))
            orig0 = float(c.original_delta_e(np.array(lines[i]["color"], float), np.array(lines[j]["color"], float)))
            need = float(profile.c80(min(w_deg[i], w_deg[j])))
            out.append({"i": i, "j": j, "sim": sim, "need": need, "orig": orig0})
    return out
