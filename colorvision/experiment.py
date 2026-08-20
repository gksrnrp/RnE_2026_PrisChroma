"""색쌍 구별 실험 프로그램 (pygame, 4지선다 odd-one-out).

한 시행: 같은 색 패치 3개와 다른 색 패치 1개를 2x2 로 보여주고
다른 하나를 고르게 한다. 두 색의 차이 t 는 조건별 계단법이 조절한다.

조작
    마우스 클릭 또는 숫자키 1/2/3/4 (좌상, 우상, 좌하, 우하)
    ESC  중단 (여기까지의 데이터는 저장된다)
    스페이스  안내 화면 넘기기

실행
    python -m colorvision.experiment --participant P01
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pygame

from . import machado as mc
from .colorspace import hex_of, to_bytes
from .geometry import screen_width_cm_from_diagonal, visual_angle_deg
from .stimuli import (BACKGROUND, JITTER_MAX, build_conditions, make_staircases,
                      probe_directions, run_session)

# 화면상 achromatic 요소 색 (색 판단에 개입하지 않도록 무채색만 쓴다)
FG = (235, 235, 235)
FG_DIM = (150, 150, 150)

POS_NAMES = ("좌상", "우상", "좌하", "우하")
KEY_TO_POS = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3,
              pygame.K_KP1: 0, pygame.K_KP2: 1, pygame.K_KP3: 2, pygame.K_KP4: 3}

CSV_FIELDS = [
    "trial_index", "phase", "condition_id", "base_hex", "dir_name", "angle_deg",
    "t", "delta_e_original",
    "color_a_hex", "color_b_hex",
    "a_lin_r", "a_lin_g", "a_lin_b", "b_lin_r", "b_lin_g", "b_lin_b",
    "odd_color", "correct_pos", "response_pos", "correct", "rt_ms",
    "is_catch", "jitter_r", "jitter_g", "jitter_b", "n_reversals", "timestamp",
]


# ---------------------------------------------------------------- 글꼴

def load_font(size, bold=False):
    """한글이 나오는 글꼴을 찾는다."""
    for path in (r"C:\Windows\Fonts\malgunbd.ttf" if bold else r"C:\Windows\Fonts\malgun.ttf",
                 r"C:\Windows\Fonts\malgun.ttf",
                 r"C:\Windows\Fonts\gulim.ttc"):
        if os.path.exists(path):
            return pygame.font.Font(path, size)
    return pygame.font.SysFont("malgungothic,arial", size, bold=bold)


# ---------------------------------------------------------------- 패치 그리기

SHAPES = ("rect", "roundrect", "ellipse")


def shape_mask(w, h, shape, radius):
    """패치 모양 마스크. 계단 현상 완화(안티에일리어싱)를 일부러 쓰지 않는다.

    가장자리를 부드럽게 만들면 배경색이 섞여 들어가 그 픽셀의 색이 달라진다.
    이 실험은 그 색 자체를 재는 것이므로, 경계는 딱 떨어지게 둔다.
    """
    x = np.arange(w)[:, None]
    y = np.arange(h)[None, :]
    if shape == "ellipse":
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        return (((x - cx) / (w / 2.0)) ** 2 + ((y - cy) / (h / 2.0)) ** 2) <= 1.0
    if shape == "roundrect":
        r = int(max(2, min(radius, min(w, h) // 2 - 1)))
        m = np.ones((w, h), dtype=bool)
        for sx in (0, 1):
            for sy in (0, 1):
                cx = (r - 0.5) if sx == 0 else (w - r - 0.5)
                cy = (r - 0.5) if sy == 0 else (h - r - 0.5)
                in_corner = ((x < r) if sx == 0 else (x >= w - r)) &                             ((y < r) if sy == 0 else (y >= h - r))
                outside = ((x - cx) ** 2 + (y - cy) ** 2) > r * r
                m &= ~(in_corner & outside)
        return m
    return np.ones((w, h), dtype=bool)


def make_patch(size, srgb, rng, shape="rect", radius=0):
    """단색 패치 표면을 만든다. 8비트로 떨어지지 않는 값은 디더링으로 표현한다.

    왜 필요한가: 이 실험의 역치는 8비트 계조 한두 단계 수준까지 내려간다.
    그냥 반올림하면 t 가 작을 때 두 패치가 **같은 픽셀값**이 되어, 색각과
    무관하게 아무도 못 맞히는 시행이 생긴다. 픽셀마다 아래/위 계조를
    소수부 비율대로 무작위 배정하면 평균값이 의도한 색과 일치한다.
    공간 잡음의 진폭은 1계조라 역치보다 낮아 단서가 되지 않는다.
    """
    w, h = size
    v = np.clip(np.asarray(srgb, dtype=np.float64), 0.0, 1.0) * 255.0
    lo = np.floor(v)
    frac = v - lo
    arr = lo[None, None, :] + (rng.random((w, h, 3)) < frac[None, None, :])
    surf = pygame.surfarray.make_surface(arr.astype(np.uint8))
    if shape == "rect":
        return surf
    surf = surf.convert_alpha()
    alpha = pygame.surfarray.pixels_alpha(surf)
    alpha[:, :] = (shape_mask(w, h, shape, radius) * 255).astype(np.uint8)
    del alpha          # 표면 잠금 해제
    return surf


class Experiment:
    def __init__(self, args):
        self.args = args
        self.rng = random.Random(args.seed)
        self.nprng = np.random.default_rng(args.seed)
        self.conditions = build_conditions(n_bases=args.bases, seed=args.seed)
        self.cond_by_id = {c.cid: c for c in self.conditions}
        self.stairs = make_staircases(self.conditions, args.max_trials)
        self.rows = []
        self.trial_index = 0
        self.aborted = False

        pygame.init()
        pygame.display.set_caption("색쌍 구별 테스트")
        flags = pygame.FULLSCREEN if args.fullscreen else 0
        self.screen = pygame.display.set_mode(
            (0, 0) if args.fullscreen else (args.width, args.height), flags)
        self.W, self.H = self.screen.get_size()
        # 전체화면은 물리 패널에 맞춰 늘어나므로, 프레임버퍼 크기가 패널
        # 해상도와 달라도 '화면 대비 비율'로 계산하면 물리 크기가 맞는다.
        self.screen_width_cm = screen_width_cm_from_diagonal(
            args.screen_diagonal, 16, 9)
        self.bg = tuple(int(v) for v in to_bytes(np.array(BACKGROUND)))
        self.f_big = load_font(38, bold=True)
        self.f_mid = load_font(24)
        self.f_small = load_font(19)
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------ 그리기

    def patch_rects(self):
        """고정 배치: 화면 가운데 2x2 정사각형."""
        side = int(min(self.W, self.H) * 0.26)
        gap = int(side * 0.30)
        x0 = (self.W - (2 * side + gap)) // 2
        y0 = (self.H - (2 * side + gap)) // 2 + int(self.H * 0.03)
        return [pygame.Rect(x0 + (i % 2) * (side + gap),
                            y0 + (i // 2) * (side + gap), side, side)
                for i in range(4)]

    def sample_layout(self):
        """매 시행 새로 뽑는 배치. 위치, 크기, 모양이 다 달라진다.

        같은 그림이 200번 반복되면 지루하기도 하지만, 눈이 그 배치에 적응해
        가장자리 비교 같은 요령이 생긴다. 겹치지 않게 하려고 화면을 2x2 칸으로
        나누고 각 칸 안에서만 흔든다. 어느 칸이 몇 번인지도 매번 섞는다.

        반환: [(사각형, 모양, 모서리반지름), ...] 4개. 순서가 곧 1/2/3/4 번호다.
        """
        if not self.args.varied_layout:
            return [(r, "rect", 0) for r in self.patch_rects()]

        rng = self.rng
        span = int(min(self.W, self.H) * 0.78)
        cell = span // 2
        ox = (self.W - span) // 2
        oy = (self.H - span) // 2 + int(self.H * 0.03)

        cells = [(ox + (i % 2) * cell, oy + (i // 2) * cell) for i in range(4)]
        rng.shuffle(cells)

        # 크기와 모양은 한 시행 안에서 네 패치가 모두 같아야 한다.
        # 정답 패치만 작거나 원이면 판단 난이도가 달라져 4지선다가 공정하지 않다.
        # 시행이 바뀔 때마다 새로 뽑으므로 화면은 매번 달라진다.
        side = int(cell * rng.uniform(0.52, 0.72))
        shape = rng.choice(SHAPES)
        radius = int(side * rng.uniform(0.12, 0.35))
        slack = cell - side

        out = []
        for cx, cy in cells:
            # 칸 안에서 남는 여백만큼만 움직인다 -> 서로 절대 겹치지 않는다
            x = cx + int(rng.uniform(0.15, 0.85) * slack)
            y = cy + int(rng.uniform(0.15, 0.85) * slack)
            out.append((pygame.Rect(x, y, side, side), shape, radius))
        return out

    def draw_text(self, text, font, y, color=FG, center_x=None):
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(center_x or self.W // 2, y))
        self.screen.blit(surf, rect)
        return rect

    def draw_trial(self, colors_srgb, progress, layout):
        self.screen.fill(self.bg)
        for i, (r, shape, radius) in enumerate(layout):
            if self.args.dither or shape != "rect":
                self.screen.blit(
                    make_patch(r.size, colors_srgb[i], self.nprng, shape, radius), r)
            else:
                self.screen.fill(tuple(int(v) for v in to_bytes(colors_srgb[i])), r)
            # 번호는 패치 밖 여백에 그린다 (패치 색을 건드리지 않기 위해)
            lab = self.f_small.render(str(i + 1), True, FG_DIM)
            self.screen.blit(lab, lab.get_rect(center=(r.centerx, r.bottom + 16)))
        self.draw_text("다른 하나를 고르세요", self.f_mid, int(self.H * 0.08))
        self.draw_text(progress, self.f_small, self.H - 26, FG_DIM)
        pygame.display.flip()

    def message_screen(self, lines, wait_key=True, hold_ms=0):
        self.screen.fill(self.bg)
        y = self.H // 2 - len(lines) * 24
        for i, ln in enumerate(lines):
            font = self.f_big if i == 0 else self.f_mid
            self.draw_text(ln, font, y, FG if i == 0 else FG_DIM)
            y += 52 if i == 0 else 36
        pygame.display.flip()
        if hold_ms:
            pygame.time.wait(hold_ms)
            return True
        if not wait_key:
            return True
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return False
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        return False
                    if e.key in (pygame.K_SPACE, pygame.K_RETURN):
                        return True
                if e.type == pygame.MOUSEBUTTONDOWN:
                    return True
            self.clock.tick(60)

    # ------------------------------------------------------------ 한 시행

    def run_trial(self, cond, t, is_catch, phase, jitter=None):
        if jitter is None:
            jitter = np.zeros(3)
        srgb_a, srgb_b = cond.pair_srgb(t, jitter)

        odd_is_b = self.rng.random() < 0.5
        odd_srgb, common_srgb = (srgb_b, srgb_a) if odd_is_b else (srgb_a, srgb_b)
        correct_pos = self.rng.randrange(4)
        colors = [common_srgb] * 4
        colors[correct_pos] = odd_srgb

        # 시행 간 간격: 잔상이 다음 시행에 남지 않게 배경만 보여준다
        self.screen.fill(self.bg)
        pygame.display.flip()
        pygame.time.wait(self.args.isi_ms)
        pygame.event.clear()

        layout = self.sample_layout()
        progress = "%d / 약 %d" % (self.trial_index + 1, self.estimated_total)
        self.draw_trial(colors, progress, layout)

        rects = [r for r, _, _ in layout]
        t0 = time.perf_counter()
        response = None
        while response is None:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return None
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        return None
                    if e.key in KEY_TO_POS:
                        response = KEY_TO_POS[e.key]
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    for i, r in enumerate(rects):
                        if r.collidepoint(e.pos):
                            response = i
            self.clock.tick(120)
        rt_ms = (time.perf_counter() - t0) * 1000.0
        correct = (response == correct_pos)

        if self.args.feedback:
            self.screen.fill(self.bg)
            self.draw_text("O" if correct else "X", self.f_big, self.H // 2,
                           FG if correct else FG_DIM)
            pygame.display.flip()
            pygame.time.wait(250)

        lin_a, lin_b = cond.pair(t, jitter)
        self.rows.append({
            "trial_index": self.trial_index,
            "phase": phase,
            "condition_id": cond.cid,
            "base_hex": cond.base_hex,
            "dir_name": cond.dir_name,
            "angle_deg": "" if cond.angle_deg is None else round(cond.angle_deg, 3),
            "t": round(float(t), 6),
            "delta_e_original": round(cond.delta_e_original(t, jitter), 4),
            "color_a_hex": hex_of(srgb_a),
            "color_b_hex": hex_of(srgb_b),
            "a_lin_r": lin_a[0], "a_lin_g": lin_a[1], "a_lin_b": lin_a[2],
            "b_lin_r": lin_b[0], "b_lin_g": lin_b[1], "b_lin_b": lin_b[2],
            "odd_color": "b" if odd_is_b else "a",
            "correct_pos": correct_pos,
            "response_pos": response,
            "correct": int(correct),
            "rt_ms": round(rt_ms, 1),
            "is_catch": int(is_catch),
            "jitter_r": jitter[0], "jitter_g": jitter[1], "jitter_b": jitter[2],
            # 확인용 시행은 계단법이 없다 (조건 자체가 별도이므로)
            "n_reversals": (len(self.stairs[cond.cid].reversals)
                            if cond.cid in self.stairs else ""),
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        })
        self.trial_index += 1
        return correct

    # ------------------------------------------------------------ 전체 흐름

    @property
    def estimated_total(self):
        return len(self.conditions) * self.args.max_trials

    def run_practice(self):
        if self.args.practice <= 0:
            return True
        if not self.message_screen([
                "연습",
                "네 칸 중 색이 다른 하나를 고르세요.",
                "마우스로 클릭하거나 숫자키 1 2 3 4 를 누릅니다.",
                "확실하지 않아도 반드시 하나를 고르세요.",
                "",
                "스페이스를 누르면 시작합니다."]):
            return False
        for _ in range(self.args.practice):
            cond = self.rng.choice(self.conditions)
            j = cond.sample_jitter(self.nprng, cond.t_max, j_max=self.args.jitter)
            r = self.run_trial(cond, cond.t_max, is_catch=False, phase="practice",
                               jitter=j)
            if r is None:
                return False
        return True

    def run_main(self):
        if not self.message_screen([
                "본 시행",
                "이제부터는 색 차이가 점점 작아집니다.",
                "잘 모르겠으면 짐작으로라도 하나를 고르세요.",
                "중간에 쉬는 시간이 있습니다.",
                "",
                "스페이스를 누르면 시작합니다."]):
            return False

        def respond(cond, t, is_catch, jitter):
            return self.run_trial(cond, t, is_catch=is_catch, phase="main",
                                  jitter=jitter)

        def on_break(active, stairs):
            remaining = sum(max(0, self.args.max_trials - stairs[c].n_trials)
                            for c in active)
            return self.message_screen([
                "잠깐 쉬어가세요",
                "남은 시행 약 %d 회" % remaining,
                "",
                "준비되면 스페이스를 누르세요."])

        _, aborted = run_session(
            self.conditions, respond, stairs=self.stairs,
            max_trials=self.args.max_trials, catch_rate=self.args.catch_rate,
            seed=self.args.seed, break_every=self.args.break_every,
            on_break=on_break, jitter_max=self.args.jitter)
        self.aborted = aborted
        return not aborted

    def run(self):
        try:
            if not self.run_practice():
                self.aborted = True
                return
            self.run_main()
            if not self.aborted:
                self.message_screen(["끝났습니다", "참여해 주셔서 감사합니다.", "",
                                     "스페이스를 누르면 종료됩니다."])
        finally:
            pygame.quit()

    def viewing_info(self):
        """자극이 눈에 얼마나 크게 보였는지. px 는 그 자체로 의미가 없다.

        같은 232 px 도 화면 크기와 보는 거리에 따라 망막에 맺히는 크기가
        달라진다. 다른 화면이나 다른 연구의 수치와 비교하려면 시야각이 있어야
        한다. 1차 실험 초기 세션에는 이 값이 없어서 소급 계산해야 했다.
        """
        side = int(min(self.W, self.H) * 0.26)
        gap = int(side * 0.30)
        info = {
            "screen_diagonal_inch": self.args.screen_diagonal,
            "screen_width_cm": self.screen_width_cm,
            "viewing_distance_cm": self.args.viewing_distance,
            "patch_side_px": side,
            "patch_gap_px": gap,
            "patch_center_distance_px": side + gap,
        }
        d = self.args.viewing_distance
        info["patch_side_deg"] = visual_angle_deg(side, self.W,
                                                  self.screen_width_cm, d)
        info["patch_center_distance_deg"] = visual_angle_deg(
            side + gap, self.W, self.screen_width_cm, d)
        return info

    # ------------------------------------------------------------ 저장

    def save(self):
        os.makedirs(self.args.outdir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = "%s_%s" % (self.args.participant, stamp)
        csv_path = os.path.join(self.args.outdir, stem + "_trials.csv")
        json_path = os.path.join(self.args.outdir, stem + "_session.json")

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(self.rows)

        dirs = probe_directions()
        meta = {
            "participant": self.args.participant,
            "started": self.rows[0]["timestamp"] if self.rows else None,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "aborted": self.aborted,
            "n_trials": len(self.rows),
            "screen": [self.W, self.H],
            "viewing": self.viewing_info(),
            "config": vars(self.args),
            "simulator": {
                "model": "Machado et al. 2009",
                "protan_confusion_dir": list(mc.confusion_direction("protan")[0]),
                "deutan_confusion_dir": list(mc.confusion_direction("deutan")[0]),
                "deutan_angle_deg": mc.deutan_angle_deg(),
            },
            "directions": {k: {"vector": list(v), "angle_deg": a}
                           for k, (v, a) in dirs.items()},
            "conditions": [
                {"cid": c.cid, "base_hex": c.base_hex,
                 "base_linear": list(c.lin_base), "dir_name": c.dir_name,
                 "angle_deg": c.angle_deg, "t_max": c.t_max,
                 "n_trials": self.stairs[c.cid].n_trials,
                 "n_reversals": len(self.stairs[c.cid].reversals),
                 "threshold_t": self.stairs[c.cid].threshold()}
                for c in self.conditions],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return csv_path, json_path


def build_parser():
    p = argparse.ArgumentParser(description="색쌍 구별 테스트")
    p.add_argument("--participant", default="anon", help="피험자 식별자 (파일명에 들어간다)")
    p.add_argument("--outdir", default="data/colorvision", help="결과 저장 폴더")
    p.add_argument("--bases", type=int, default=None,
                   help="주면 옛 휴리스틱 설계를 쓴다 (기본은 최적화된 conditions.json)")
    p.add_argument("--max-trials", type=int, default=16, help="조건당 최대 시행 수")
    p.add_argument("--practice", type=int, default=6, help="연습 시행 수")
    p.add_argument("--catch-rate", type=float, default=0.08,
                   help="아주 쉬운 확인용 시행 비율 (실수율 추정에 쓴다)")
    p.add_argument("--break-every", type=int, default=45, help="몇 시행마다 쉬는 화면을 넣을지")
    p.add_argument("--isi-ms", type=int, default=350, help="시행 간 빈 화면 시간")
    p.add_argument("--no-feedback", dest="feedback", action="store_false",
                   help="정오 표시를 끈다")
    p.add_argument("--varied-layout", dest="varied_layout", action="store_true",
                   help="패치의 위치/크기/모양도 매 시행 바꾼다 (기본은 고정 2x2)")
    p.add_argument("--viewing-distance", type=float, default=None,
                   help="화면까지의 거리 (cm). 없으면 시야각을 남길 수 없다")
    p.add_argument("--screen-diagonal", type=float, default=None,
                   help="모니터 대각선 (인치). 없으면 시야각을 남길 수 없다")
    p.add_argument("--jitter", type=float, default=JITTER_MAX,
                   help="매 시행 기준색을 흔드는 폭 (0 이면 끔). 같은 색 반복을 줄인다")
    p.add_argument("--no-dither", dest="dither", action="store_false",
                   help="디더링을 끈다 (작은 색차가 8비트로 뭉개진다)")
    p.add_argument("--windowed", dest="fullscreen", action="store_false", help="창 모드")
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(feedback=True, fullscreen=True, dither=True,
                   varied_layout=False)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.viewing_distance is None or args.screen_diagonal is None:
        print("경고: --viewing-distance 와 --screen-diagonal 이 없으면")
        print("      자극 크기를 시야각으로 남길 수 없어 다른 화면·다른 연구의")
        print("      수치와 비교하지 못한다. 대략이라도 넣는 것을 권한다.")
        print("")
    exp = Experiment(args)
    exp.run()
    if not exp.rows:
        print("기록된 시행이 없다. 저장하지 않는다.")
        return 1
    csv_path, json_path = exp.save()
    done = sum(1 for r in exp.rows if r["phase"] == "main")
    print("시행 %d (본 시행 %d)%s" % (len(exp.rows), done, " — 중단됨" if exp.aborted else ""))
    print("  " + csv_path)
    print("  " + json_path)
    print("\n파라미터 추정:")
    print("  python -m colorvision.estimate %s" % csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
