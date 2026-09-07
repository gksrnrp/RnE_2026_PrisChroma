"""선 자극 생성.

과제는 2x2 로 배치된 선 네 개 중 색이 다른 하나를 고르는 4지선다다.
굵기만 조건으로 바꾸고 길이·방향·끝모양·배경·색 변화 방향은 고정한다.

색쌍 만드는 법
--------------
CIELAB 공간에서 **사전에 정한 방향 u** 를 따라, 같은 중심색에서 양쪽으로
절반씩 벌린다.

    c_a = c0 − (d/2)·u ,   c_b = c0 + (d/2)·u

한쪽만 움직이면 그 색의 밝기나 채도 자체가 단서가 된다. 양쪽으로 벌리면
중심색이 같으므로 그런 단서가 줄어든다.

자극의 세기 x
-------------
심리측정 함수의 x 는 **Machado 시뮬레이션 후 ΔE00** 다. Lab 상의 이동량 d 와
시뮬 후 색차는 비례하지 않으므로, 원하는 x 를 주면 이분 탐색으로 d 를 찾는다.

⚠️ 방향을 그 사람의 혼동축으로 잡으면 안 된다. 아무리 벌려도 시뮬 후 색차가
0 근처에 머물러 C80 을 잴 수 없다. `max_reachable` 로 여유를 먼저 확인한다.

안티앨리어싱
------------
쓰지 않는다. 3 px 선에서 가장자리를 부드럽게 하면 배경색이 섞여 선의 실제
색이 달라진다. 정수 좌표에 축 정렬 사각형으로 그린다.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, __file__.rsplit("experiment", 1)[0])   # 창민 파트 src/

from color_spaces import srgb_to_lab                        # noqa: E402
from distance import delta_e_2000                           # noqa: E402
from machado import simulate_color_float                    # noqa: E402


# ---------------------------------------------------------------- 색공간

def _lab_to_srgb(lab):
    """CIELAB -> sRGB(0~255 실수). 색역 밖이면 None."""
    L, a, b = [float(v) for v in lab]
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    d = 6.0 / 29.0

    def finv(t):
        return t ** 3 if t > d else 3 * d * d * (t - 4.0 / 29.0)

    X, Y, Z = finv(fx) * 0.95047, finv(fy), finv(fz) * 1.08883
    M = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    lin = M @ np.array([X, Y, Z])
    if np.any(lin < -1e-6) or np.any(lin > 1 + 1e-6):
        return None
    lin = np.clip(lin, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return srgb * 255.0


def _round_rgb(v):
    return tuple(int(min(255, max(0, round(c)))) for c in v)


def simulated_delta_e(rgb_a, rgb_b, cvd_type, severity):
    """두 색이 그 사람 눈에 도달했을 때의 ΔE00. 심리측정 함수의 x."""
    sa = simulate_color_float(rgb_a, cvd_type, severity)
    sb = simulate_color_float(rgb_b, cvd_type, severity)
    return float(delta_e_2000(srgb_to_lab(sa), srgb_to_lab(sb)))


def original_delta_e(rgb_a, rgb_b):
    """정상 색각 기준 ΔE00. 기록용."""
    return float(delta_e_2000(srgb_to_lab(rgb_a), srgb_to_lab(rgb_b)))


# ---------------------------------------------------------------- 색 방향

@dataclass
class ColorAxis:
    """색쌍을 만들 중심색과 방향. 실험 전에 고정하고 바꾸지 않는다."""
    center_lab: tuple = (55.0, 6.0, 14.0)
    direction_lab: tuple = (0.0, 1.0, 0.0)     # a* 축 (적록)
    name: str = "lab_a"

    def __post_init__(self):
        u = np.asarray(self.direction_lab, dtype=float)
        n = float(np.linalg.norm(u))
        if n == 0:
            raise ValueError("방향 벡터가 0 이다")
        self._u = u / n

    def pair_at_lab_distance(self, d):
        """Lab 상 거리 d 만큼 벌린 두 색. 색역 밖이면 (None, None)."""
        c0 = np.asarray(self.center_lab, dtype=float)
        a = _lab_to_srgb(c0 - (d / 2.0) * self._u)
        b = _lab_to_srgb(c0 + (d / 2.0) * self._u)
        if a is None or b is None:
            return None, None
        return _round_rgb(a), _round_rgb(b)

    def max_lab_distance(self, hi=200.0, steps=400):
        """색역 안에 남는 최대 d."""
        best = 0.0
        for d in np.linspace(0, hi, steps):
            if self.pair_at_lab_distance(d)[0] is None:
                break
            best = float(d)
        return best

    def max_reachable(self, cvd_type, severity):
        """이 축이 낼 수 있는 최대 시뮬 ΔE00.

        C80 의 2 배쯤은 나와야 심리측정 함수의 윗부분까지 잴 수 있다.
        """
        d = self.max_lab_distance()
        a, b = self.pair_at_lab_distance(d)
        if a is None:
            return 0.0
        return simulated_delta_e(a, b, cvd_type, severity)

    def pair_for_simulated(self, x, cvd_type, severity, tol=0.02, iters=60):
        """시뮬 후 ΔE00 가 x 가 되는 색쌍. 도달 불가면 (None, None, 실제값)."""
        lo, hi = 0.0, self.max_lab_distance()
        top_a, top_b = self.pair_at_lab_distance(hi)
        if top_a is None:
            return None, None, 0.0
        top = simulated_delta_e(top_a, top_b, cvd_type, severity)
        if top < x - tol:
            return None, None, top          # 화면이 낼 수 있는 한계를 넘음
        for _ in range(iters):
            m = (lo + hi) / 2.0
            a, b = self.pair_at_lab_distance(m)
            if a is None:
                hi = m
                continue
            if simulated_delta_e(a, b, cvd_type, severity) < x:
                lo = m
            else:
                hi = m
        a, b = self.pair_at_lab_distance((lo + hi) / 2.0)
        return a, b, simulated_delta_e(a, b, cvd_type, severity)


# ---------------------------------------------------------------- 굵기

def visual_angle_deg(px, screen_width_px, screen_width_cm, distance_cm):
    """px -> 시야각(도). px 는 그 자체로 의미가 없으므로 항상 함께 기록한다."""
    if not (screen_width_px and screen_width_cm and distance_cm):
        return None
    cm = px * float(screen_width_cm) / float(screen_width_px)
    return float(math.degrees(2 * math.atan((cm / 2.0) / float(distance_cm))))


def screen_width_cm(diagonal_inch, w_px=16, h_px=9):
    if not diagonal_inch:
        return None
    return float(diagonal_inch) * 2.54 * w_px / math.hypot(w_px, h_px)


# ---------------------------------------------------------------- 배치

@dataclass
class LineLayout:
    """2x2 로 놓인 선 네 개. 굵기만 바뀌고 나머지는 고정."""
    length_px: int = 300
    gap_x_px: int = 220
    gap_y_px: int = 120
    y_offset_frac: float = 0.02

    def rects(self, screen_w, screen_h, stroke_px):
        """[(x, y, w, h), ...] 4개. 순서가 1/2/3/4 번호다."""
        w = int(stroke_px)
        h = int(self.length_px)
        span_x = w + self.gap_x_px
        span_y = h + self.gap_y_px
        x0 = (screen_w - (span_x + w)) // 2
        y0 = (screen_h - (span_y + h)) // 2 + int(screen_h * self.y_offset_frac)
        return [(x0 + (i % 2) * span_x, y0 + (i // 2) * span_y, w, h)
                for i in range(4)]


# ---------------------------------------------------------------- 시행

@dataclass
class Trial:
    stroke_width_px: int
    stroke_width_deg: float
    level_requested: float          # 요청한 시뮬 ΔE00
    rgb_a: tuple
    rgb_b: tuple
    delta_e00_original: float
    delta_e00_simulated: float      # 실제로 만들어진 값. 분석은 이 값을 쓴다
    odd_is_b: bool
    correct_position: int
    in_gamut: bool = True


def make_trial(level, stroke_px, axis, cvd_type, severity, rng,
              screen=None):
    """자극 하나. 색역 밖이면 in_gamut=False 로 표시하고 제외한다."""
    a, b, got = axis.pair_for_simulated(level, cvd_type, severity)
    if a is None:
        return Trial(int(stroke_px), None, float(level), None, None,
                     0.0, float(got), False, 0, in_gamut=False)
    deg = None
    if screen:
        deg = visual_angle_deg(stroke_px, screen["width_px"],
                               screen["width_cm"], screen["distance_cm"])
    return Trial(
        stroke_width_px=int(stroke_px),
        stroke_width_deg=deg,
        level_requested=float(level),
        rgb_a=a, rgb_b=b,
        delta_e00_original=original_delta_e(a, b),
        delta_e00_simulated=float(got),
        odd_is_b=bool(rng.random() < 0.5),
        correct_position=int(rng.integers(0, 4)),
    )


def check_axis(axis, cvd_type, severity, expected_c80):
    """실험 전 확인: 이 축으로 C80 의 2 배까지 잴 수 있는가."""
    top = axis.max_reachable(cvd_type, severity)
    need = expected_c80 * 2.0
    return {
        "axis": axis.name,
        "max_simulated_delta_e": top,
        "needed": need,
        "ok": top >= need,
        "message": ("여유 충분" if top >= need else
                    "이 축은 최대 %.2f 까지만 낼 수 있다. C80 의 2 배(%.2f)에 "
                    "못 미치므로 중심색이나 방향을 바꿔야 한다." % (top, need)),
    }


__all__ = ["ColorAxis", "LineLayout", "Trial", "make_trial", "check_axis",
           "simulated_delta_e", "original_delta_e", "visual_angle_deg",
           "screen_width_cm"]
