"""화면 기하 조건 — 크기, 모양, 거리, 배경.

왜 필요한가
-----------
기존 실험의 역치(ΔE00 약 1)는 **큰 네모 두 개를 붙여놓고 시간 무제한으로**
잰 값이다. 실제 노선도의 얇은 선이나 멀리 떨어진 범례에서는 같은 색차라도
훨씬 구별하기 어렵다. 그 차이가 얼마인지를 재는 것이 이 모듈의 목적이다.

⚠️ **하나의 배수로 환산할 수 있다고 가정하지 않는다.**
얇은 선에서의 배수와 멀리 떨어진 영역에서의 배수가 다를 수 있다. 그래서
조건마다 따로 역치를 재고 조건별 비율을 낸다. 자료가 충분히 쌓이면 그때
크기·거리를 변수로 하는 혼동 확률 모델로 묶는다.

여기 값은 전부 잠정값이다
-------------------------
아래 프리셋의 크기와 거리는 **자리를 잡아두려고 넣은 임시값**이다.
정인교 파트가 실제 이미지에서 뽑아줄 통계로 교체해야 한다.

    영역 면적 분포 / 선 두께 / 영역 간 중심 거리 / 배경색 / 화면 해상도

교체는 JSON 파일로 한다. `--geometries my.json` 으로 넘기면 그 값을 쓴다.
"""

import json

import numpy as np

from .stimuli import BACKGROUND

# 참조 조건의 패치 크기 (화면 짧은 변에 대한 비율).
# 기존 실험(experiment.py)의 값과 같아야 비교가 성립한다.
REF_SIDE_FRAC = 0.26
REF_GAP_FRAC = 0.30          # 패치 한 변 대비 간격


class Geometry:
    """자극 하나의 화면 배치 규칙.

    kind
        "square"  정사각형 4개를 2x2 로
        "bar"     세로 막대 4개를 가로로 나란히 (그래프 선, 노선도 선을 흉내)
    size_px
        정사각형이면 한 변, 막대면 폭. None 이면 참조 크기(화면 비례)를 쓴다.
    gap_px
        요소 사이 빈 공간. None 이면 참조 간격.
    length_px
        막대 길이 (kind="bar" 일 때만).
    background
        배경 sRGB. None 이면 기본 회색.
    """

    def __init__(self, name, kind="square", size_px=None, gap_px=None,
                 length_px=None, background=None, note=""):
        self.name = name
        self.kind = kind
        self.size_px = size_px
        self.gap_px = gap_px
        self.length_px = length_px
        self.background = tuple(background) if background else tuple(BACKGROUND)
        self.note = note

    # ------------------------------------------------------------ 배치

    def layout(self, W, H, rng):
        """[(x, y, w, h), ...] 4개. 순서가 곧 1/2/3/4 번호다."""
        short = min(W, H)
        if self.kind == "bar":
            width = int(self.size_px or max(3, short * 0.02))
            length = int(self.length_px or short * 0.55)
            gap = int(self.gap_px if self.gap_px is not None else width * 6)
            total = 4 * width + 3 * gap
            x0 = (W - total) // 2
            y0 = (H - length) // 2 + int(H * 0.03)
            return [(x0 + i * (width + gap), y0, width, length) for i in range(4)]

        side = int(self.size_px or short * REF_SIDE_FRAC)
        gap = int(self.gap_px if self.gap_px is not None else side * REF_GAP_FRAC)
        span = 2 * side + gap
        x0 = (W - span) // 2
        y0 = (H - span) // 2 + int(H * 0.03)
        return [(x0 + (i % 2) * (side + gap), y0 + (i // 2) * (side + gap), side, side)
                for i in range(4)]

    # ------------------------------------------------------------ 기록용

    def measured(self, W, H):
        """실제로 화면에 그려진 크기와 거리 (기록해 둘 값)."""
        rects = self.layout(W, H, None)
        x, y, w, h = rects[0]
        x2, y2, _, _ = rects[1]
        cx1, cy1 = x + w / 2, y + h / 2
        cx2, cy2 = x2 + w / 2, y2 + h / 2
        return {
            "element_w_px": w,
            "element_h_px": h,
            "element_area_px": w * h,
            "min_width_px": min(w, h),
            "neighbor_center_distance_px": float(np.hypot(cx2 - cx1, cy2 - cy1)),
            "edge_gap_px": float(x2 - (x + w)),
        }

    def to_dict(self):
        return {"name": self.name, "kind": self.kind, "size_px": self.size_px,
                "gap_px": self.gap_px, "length_px": self.length_px,
                "background": list(self.background), "note": self.note}

    def __repr__(self):
        return "<Geometry %s kind=%s size=%s gap=%s>" % (
            self.name, self.kind, self.size_px, self.gap_px)


# ---------------------------------------------------------------- 프리셋

def default_geometries():
    """잠정 프리셋. 정인교 파트의 실측값으로 교체할 것.

    reference 는 반드시 남겨 둔다. 다른 조건의 역치를 이것과 비교해
    비율을 내기 때문이다.
    """
    return [
        Geometry("reference", "square", None, None,
                 note="기존 실험과 같은 큰 패치. 비교 기준"),
        Geometry("small", "square", size_px=48, gap_px=24,
                 note="TODO 인교: 실제 영역 면적 하위 25% 값으로 교체"),
        Geometry("tiny", "square", size_px=20, gap_px=12,
                 note="TODO 인교: 작은 UI 요소·글씨 크기로 교체"),
        Geometry("thin_line", "bar", size_px=6, length_px=340, gap_px=60,
                 note="TODO 인교: 실제 그래프 선·노선 두께로 교체"),
        Geometry("far", "square", size_px=110, gap_px=520,
                 note="TODO 인교: 범례-그래프 중심 거리로 교체"),
        Geometry("light_bg", "square", size_px=110, gap_px=60,
                 background=(0.93, 0.93, 0.93),
                 note="TODO 인교: 실제 배경색 분포로 교체"),
    ]


def load_geometries(path=None):
    if not path:
        return default_geometries()
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    items = raw["geometries"] if isinstance(raw, dict) else raw
    geoms = [Geometry(**g) for g in items]
    if not any(g.name == "reference" for g in geoms):
        raise ValueError("기하 조건 목록에 'reference' 가 반드시 있어야 한다. "
                         "다른 조건의 비율을 그것에 대해 계산하기 때문이다.")
    return geoms


def save_geometries(geoms, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"geometries": [g.to_dict() for g in geoms]}, f,
                  ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------- 시야각

def visual_angle_deg(size_px, screen_width_px, screen_width_cm, distance_cm):
    """화면상 크기(px) -> 시야각(도).

    px 는 그 자체로 의미가 없다. 같은 6px 선도 화면 크기와 보는 거리에 따라
    망막에 맺히는 크기가 달라진다. 다른 화면에서 잰 자료와 비교하려면
    반드시 시야각으로 바꿔야 한다.
    """
    if not (screen_width_px and screen_width_cm and distance_cm):
        return None
    cm_per_px = float(screen_width_cm) / float(screen_width_px)
    return float(np.degrees(2 * np.arctan((size_px * cm_per_px / 2.0)
                                          / float(distance_cm))))


def screen_width_cm_from_diagonal(diagonal_inch, w_px, h_px):
    """대각선 인치와 해상도로 가로 물리 길이(cm) 를 구한다."""
    if not diagonal_inch:
        return None
    ratio = float(w_px) / float(np.hypot(w_px, h_px))
    return float(diagonal_inch) * 2.54 * ratio


if __name__ == "__main__":
    print("기하 조건 프리셋 (전부 잠정값 — 인교 실측으로 교체할 것)")
    print("-" * 74)
    W, H = 1920, 1080
    print("%-12s %-7s %9s %9s %10s  %s" %
          ("이름", "모양", "요소크기", "면적px", "중심거리", "비고"))
    for g in default_geometries():
        m = g.measured(W, H)
        print("%-12s %-7s %5dx%-3d %9d %10.0f  %s" %
              (g.name, g.kind, m["element_w_px"], m["element_h_px"],
               m["element_area_px"], m["neighbor_center_distance_px"], g.note))
    print("-" * 74)
    print("(%dx%d 화면 기준)" % (W, H))
