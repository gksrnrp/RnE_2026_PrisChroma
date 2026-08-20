"""기하 조건별 색 구별 역치 실험.

기존 실험(experiment.py)과 과제는 똑같다. 네 칸 중 색이 다른 하나를 고른다.
다른 점은 **화면 기하를 조건으로 함께 돌린다**는 것이다.

    (색 조건) x (기하 조건) 마다 계단법 하나

색 조건은 이미 최적화된 conditions.json 에서 몇 개를 가져다 쓴다. 유형과
severity 는 본 실험에서 이미 확정했으므로 여기서는 다시 추정하지 않고,
**역치가 기하에 따라 어떻게 변하는지만** 본다.

실행
    python -m colorvision.experiment_geometry --participant P01 \
        --type protan --severity 0.95 --viewing-distance 60 --screen-diagonal 24
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
from .colorspace import delta_e00_linear, hex_of, to_bytes
from .experiment import KEY_TO_POS, load_font, make_patch, FG, FG_DIM
from .geometry import (Geometry, load_geometries, screen_width_cm_from_diagonal,
                       visual_angle_deg)
from .stimuli import (JITTER_MAX, build_catch_condition, build_conditions,
                      make_staircases, run_session)

CSV_FIELDS = [
    "trial_index", "phase", "condition_id", "color_cid", "geometry",
    "base_hex", "dir_name", "t", "delta_e_original",
    "color_a_hex", "color_b_hex",
    "a_lin_r", "a_lin_g", "a_lin_b", "b_lin_r", "b_lin_g", "b_lin_b",
    "background_hex",
    "element_w_px", "element_h_px", "element_area_px", "min_width_px",
    "neighbor_center_distance_px", "edge_gap_px",
    "element_angle_deg", "separation_angle_deg",
    "odd_color", "correct_pos", "response_pos", "correct", "rt_ms",
    "is_catch", "jitter_r", "jitter_g", "jitter_b", "n_reversals", "timestamp",
]


def achievable_delta_e(cond, cvd_type, severity):
    """그 조건이 t_max 에서 낼 수 있는 최대 지각 ΔE00."""
    a, b = cond.pair(cond.t_max)
    return float(delta_e00_linear(mc.simulate_linear(a, cvd_type, severity),
                                  mc.simulate_linear(b, cvd_type, severity)))


def pick_color_conditions(n_colors, cvd_type, severity, min_headroom=6.0):
    """기하 실험에 쓸 색 조건을 고른다.

    본 실험과 **선택 기준이 정반대**다. 본 실험은 유형을 가르려고 혼동축
    조건(그 유형에게 거의 안 보이는 색)을 일부러 골랐다. 그런 조건은
    지각 ΔE00 이 최대로 벌려도 1 을 못 넘는다. 역치가 커지는 조건에서
    쓰면 화면이 낼 수 있는 한계를 넘어버려 아예 측정이 안 된다.

    실제로 c01_plane000 은 protan 0.95 에서 최대 0.85 밖에 안 나와서,
    배수 2.0 을 재려다 계속 과소추정되는 문제가 있었다.

    그래서 여기서는 **여유가 큰 조건**을 고른다.
    """
    conds = build_conditions()
    scored = [(achievable_delta_e(c, cvd_type, severity), c) for c in conds]
    scored.sort(key=lambda x: -x[0])
    usable = [(d, c) for d, c in scored if d >= min_headroom]
    if len(usable) < n_colors:
        usable = scored[:max(n_colors, 1)]
    return [c for _, c in usable[:n_colors]]


class GeometryExperiment:
    def __init__(self, args):
        self.args = args
        self.rng = random.Random(args.seed)
        self.nprng = np.random.default_rng(args.seed)

        self.geometries = load_geometries(args.geometries)
        self.colors = pick_color_conditions(args.colors, args.type, args.severity)
        self.catch = build_catch_condition()

        # (색 x 기하) 조합마다 계단법 하나. run_session 은 cid 로만 구분하므로
        # 합성 id 를 만들어 주면 나머지 진행 논리는 그대로 쓸 수 있다.
        self.pairs = {}
        conds = []
        for c in self.colors:
            for g in self.geometries:
                cid = "%s__%s" % (c.cid, g.name)
                clone = _clone_condition(c, cid)
                self.pairs[cid] = (c, g)
                conds.append(clone)
        self.conditions = conds
        self.stairs = make_staircases(conds, args.max_trials)

        self.rows = []
        self.trial_index = 0
        self.aborted = False

        pygame.init()
        pygame.display.set_caption("색 구별 테스트 — 기하 조건")
        flags = pygame.FULLSCREEN if args.fullscreen else 0
        self.screen = pygame.display.set_mode(
            (0, 0) if args.fullscreen else (args.width, args.height), flags)
        self.W, self.H = self.screen.get_size()
        self.screen_width_cm = screen_width_cm_from_diagonal(
            args.screen_diagonal, self.W, self.H)
        self.f_big = load_font(38, bold=True)
        self.f_mid = load_font(24)
        self.f_small = load_font(19)
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------ 그리기

    def geometry_for(self, cid):
        if cid in self.pairs:
            return self.pairs[cid][1]
        return next(g for g in self.geometries if g.name == "reference")

    def draw_trial(self, colors_srgb, rects, bg, progress):
        self.screen.fill(bg)
        for i, (x, y, w, h) in enumerate(rects):
            surf = make_patch((w, h), colors_srgb[i], self.nprng)
            self.screen.blit(surf, (x, y))
            lab = self.f_small.render(str(i + 1), True, FG_DIM)
            self.screen.blit(lab, lab.get_rect(center=(x + w // 2, y + h + 16)))
        self.draw_text("다른 하나를 고르세요", self.f_mid, int(self.H * 0.06))
        self.draw_text(progress, self.f_small, self.H - 24, FG_DIM)
        pygame.display.flip()

    def draw_text(self, text, font, y, color=FG):
        surf = font.render(text, True, color)
        self.screen.blit(surf, surf.get_rect(center=(self.W // 2, y)))

    def message_screen(self, lines, wait_key=True):
        self.screen.fill(tuple(int(v) for v in to_bytes(np.array(
            next(g for g in self.geometries if g.name == "reference").background))))
        y = self.H // 2 - len(lines) * 24
        for i, ln in enumerate(lines):
            self.draw_text(ln, self.f_big if i == 0 else self.f_mid, y,
                           FG if i == 0 else FG_DIM)
            y += 52 if i == 0 else 36
        pygame.display.flip()
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

    def run_trial(self, cond, t, is_catch, jitter, phase="main"):
        if jitter is None:
            jitter = np.zeros(3)
        color_cond, geom = (self.pairs.get(cond.cid)
                            or (cond, self.geometry_for(cond.cid)))
        srgb_a, srgb_b = cond.pair_srgb(t, jitter)

        odd_is_b = self.rng.random() < 0.5
        odd_srgb, common_srgb = ((srgb_b, srgb_a) if odd_is_b else (srgb_a, srgb_b))
        correct_pos = self.rng.randrange(4)
        colors = [common_srgb] * 4
        colors[correct_pos] = odd_srgb

        bg = tuple(int(v) for v in to_bytes(np.array(geom.background)))
        rects = geom.layout(self.W, self.H, self.rng)

        self.screen.fill(bg)
        pygame.display.flip()
        pygame.time.wait(self.args.isi_ms)
        pygame.event.clear()

        self.draw_trial(colors, rects, bg,
                        "%d / 약 %d" % (self.trial_index + 1, self.estimated_total))

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
                    for i, (x, y, w, h) in enumerate(rects):
                        if pygame.Rect(x, y, w, h).collidepoint(e.pos):
                            response = i
            self.clock.tick(120)
        rt_ms = (time.perf_counter() - t0) * 1000.0
        correct = (response == correct_pos)

        if self.args.feedback:
            self.screen.fill(bg)
            self.draw_text("O" if correct else "X", self.f_big, self.H // 2,
                           FG if correct else FG_DIM)
            pygame.display.flip()
            pygame.time.wait(220)

        m = geom.measured(self.W, self.H)
        lin_a, lin_b = cond.pair(t, jitter)
        ang = visual_angle_deg(m["min_width_px"], self.W, self.screen_width_cm,
                               self.args.viewing_distance)
        sep = visual_angle_deg(m["neighbor_center_distance_px"], self.W,
                               self.screen_width_cm, self.args.viewing_distance)

        self.rows.append({
            "trial_index": self.trial_index, "phase": phase,
            "condition_id": cond.cid, "color_cid": color_cond.cid,
            "geometry": geom.name,
            "base_hex": color_cond.base_hex, "dir_name": color_cond.dir_name,
            "t": round(float(t), 6),
            "delta_e_original": round(cond.delta_e_original(t, jitter), 4),
            "color_a_hex": hex_of(srgb_a), "color_b_hex": hex_of(srgb_b),
            "a_lin_r": lin_a[0], "a_lin_g": lin_a[1], "a_lin_b": lin_a[2],
            "b_lin_r": lin_b[0], "b_lin_g": lin_b[1], "b_lin_b": lin_b[2],
            "background_hex": hex_of(np.array(geom.background)),
            "element_w_px": m["element_w_px"], "element_h_px": m["element_h_px"],
            "element_area_px": m["element_area_px"], "min_width_px": m["min_width_px"],
            "neighbor_center_distance_px": round(m["neighbor_center_distance_px"], 1),
            "edge_gap_px": round(m["edge_gap_px"], 1),
            "element_angle_deg": "" if ang is None else round(ang, 4),
            "separation_angle_deg": "" if sep is None else round(sep, 4),
            "odd_color": "b" if odd_is_b else "a",
            "correct_pos": correct_pos, "response_pos": response,
            "correct": int(correct), "rt_ms": round(rt_ms, 1),
            "is_catch": int(is_catch),
            "jitter_r": jitter[0], "jitter_g": jitter[1], "jitter_b": jitter[2],
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

    def run(self):
        try:
            if not self.message_screen([
                    "색 구별 테스트 (2차)",
                    "네 칸 중 색이 다른 하나를 고르세요.",
                    "이번에는 칸의 크기와 간격이 조건마다 달라집니다.",
                    "작고 멀리 떨어진 조건이 어려운 것은 정상입니다.",
                    "모르겠어도 반드시 하나를 고르세요.",
                    "",
                    "스페이스를 누르면 시작합니다."]):
                self.aborted = True
                return

            for _ in range(self.args.practice):
                cond = self.rng.choice(self.conditions)
                j = cond.sample_jitter(self.nprng, cond.t_max, j_max=self.args.jitter)
                if self.run_trial(cond, cond.t_max, False, j, phase="practice") is None:
                    self.aborted = True
                    return

            def respond(cond, t, is_catch, jitter):
                return self.run_trial(cond, t, is_catch, jitter)

            def on_break(active, stairs):
                left = sum(max(0, self.args.max_trials - stairs[c].n_trials)
                           for c in active)
                return self.message_screen(["잠깐 쉬어가세요",
                                            "남은 시행 약 %d 회" % left, "",
                                            "준비되면 스페이스를 누르세요."])

            _, aborted = run_session(
                self.conditions, respond, stairs=self.stairs,
                max_trials=self.args.max_trials, catch_rate=self.args.catch_rate,
                seed=self.args.seed, break_every=self.args.break_every,
                on_break=on_break, catch_condition=self.catch,
                jitter_max=self.args.jitter)
            self.aborted = aborted
            if not aborted:
                self.message_screen(["끝났습니다", "참여해 주셔서 감사합니다.", "",
                                     "스페이스를 누르면 종료됩니다."])
        finally:
            pygame.quit()

    # ------------------------------------------------------------ 저장

    def save(self):
        os.makedirs(self.args.outdir, exist_ok=True)
        stem = "%s_%s_geom" % (self.args.participant,
                               datetime.now().strftime("%Y%m%d_%H%M%S"))
        csv_path = os.path.join(self.args.outdir, stem + "_trials.csv")
        json_path = os.path.join(self.args.outdir, stem + "_session.json")

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(self.rows)

        meta = {
            "participant": self.args.participant,
            "kind": "geometry",
            "cvd_type": mc.normalize_type(self.args.type),
            "severity": self.args.severity,
            "aborted": self.aborted,
            "n_trials": len(self.rows),
            "screen": {"width_px": self.W, "height_px": self.H,
                       "diagonal_inch": self.args.screen_diagonal,
                       "width_cm": self.screen_width_cm,
                       "viewing_distance_cm": self.args.viewing_distance},
            "config": vars(self.args),
            "geometries": [dict(g.to_dict(), **g.measured(self.W, self.H))
                           for g in self.geometries],
            "color_conditions": [{"cid": c.cid, "base_hex": c.base_hex,
                                  "dir_name": c.dir_name} for c in self.colors],
            "staircase": {cid: {"n_trials": sc.n_trials,
                                "n_reversals": len(sc.reversals),
                                "threshold_t": sc.threshold()}
                          for cid, sc in self.stairs.items()},
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return csv_path, json_path


def _clone_condition(cond, cid):
    """같은 색 조건을 다른 id 로 하나 더 만든다 (계단법을 따로 돌리려고)."""
    import copy
    c = copy.copy(cond)
    c.cid = cid
    return c


def build_parser():
    p = argparse.ArgumentParser(description="기하 조건별 색 구별 역치 실험")
    p.add_argument("--participant", default="anon")
    p.add_argument("--outdir", default="data/colorvision")
    p.add_argument("--type", required=True, help="본 실험에서 확정한 유형")
    p.add_argument("--severity", type=float, required=True,
                   help="본 실험에서 확정한 severity")
    p.add_argument("--geometries", default=None,
                   help="기하 조건 JSON (없으면 잠정 프리셋)")
    p.add_argument("--colors", type=int, default=3,
                   help="쓸 색 조건 수. 기하 조건당 시행 = colors x max-trials 이며 "
                        "40 미만이면 배수 추정이 크게 흔들린다 (색 2개로는 부족)")
    p.add_argument("--max-trials", type=int, default=14)
    p.add_argument("--practice", type=int, default=4)
    p.add_argument("--catch-rate", type=float, default=0.06)
    p.add_argument("--break-every", type=int, default=45)
    p.add_argument("--isi-ms", type=int, default=350)
    p.add_argument("--jitter", type=float, default=JITTER_MAX)
    p.add_argument("--viewing-distance", type=float, default=None,
                   help="화면까지의 거리 (cm). 시야각 환산에 쓴다")
    p.add_argument("--screen-diagonal", type=float, default=None,
                   help="모니터 대각선 (인치). 시야각 환산에 쓴다")
    p.add_argument("--no-feedback", dest="feedback", action="store_false")
    p.add_argument("--windowed", dest="fullscreen", action="store_false")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(feedback=True, fullscreen=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.viewing_distance is None or args.screen_diagonal is None:
        print("경고: --viewing-distance 와 --screen-diagonal 이 없으면")
        print("      px 를 시야각으로 바꿀 수 없어 다른 화면과 비교하지 못한다.")
        print("      대략이라도 넣는 것을 권한다.\n")

    exp = GeometryExperiment(args)
    print("조건 %d개 = 색 %d x 기하 %d" %
          (len(exp.conditions), len(exp.colors), len(exp.geometries)))
    exp.run()
    if not exp.rows:
        print("기록된 시행이 없다. 저장하지 않는다.")
        return 1
    csv_path, json_path = exp.save()
    print("시행 %d%s" % (len(exp.rows), " — 중단됨" if exp.aborted else ""))
    print("  " + csv_path)
    print("\n분석:")
    print("  python -m colorvision.analyze_geometry %s --type %s --severity %s"
          % (csv_path, args.type, args.severity))
    return 0


if __name__ == "__main__":
    sys.exit(main())
