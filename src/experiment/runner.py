"""선 굵기 색 구별 실험 실행기 (pygame).

과제: 2x2 로 놓인 선 네 개 중 색이 다른 하나를 고른다 (4지선다).

두 가지 모드
------------
    calibration   굵기 3개 x 96 시행, 고정자극. 심리측정 함수 적합용
    validation    검증 굵기 하나, 적응형 계단. 세션마다 따로 돌린다

기록
----
한 시행당 한 줄씩 **즉시** JSONL 에 덧붙인다. 중간에 끊겨도 그때까지의
자료가 남고, 다시 실행하면 남은 시행부터 이어서 한다.

원자료에 이름·학번을 저장하지 않는다. participant_id 는 P01 같은 코드다.

실행
    python -m experiment.runner --participant P01 --mode calibration \\
        --cvd-type protan --severity 0.94 --severity-source patch_session_PO3 \\
        --widths 3,7,16 --screen-diagonal 15.6 --viewing-distance 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pygame

from .psychometric import stimulus_levels
from .staircase import Staircase, default_range, default_start
from .stimulus import (ColorAxis, LineLayout, check_axis, make_trial,
                       screen_width_cm)

SCHEMA_VERSION = "1.0"
CONFIG_VERSION = "1.0"

BACKGROUND = (150, 150, 150)      # 고정 배경. 실험 내내 바꾸지 않는다
FG = (245, 245, 245)
FG_DIM = (95, 95, 95)
KEY_TO_POS = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3,
              pygame.K_KP1: 0, pygame.K_KP2: 1, pygame.K_KP3: 2, pygame.K_KP4: 3}


def load_font(size, bold=False):
    for p in ((r"C:\Windows\Fonts\malgunbd.ttf" if bold
               else r"C:\Windows\Fonts\malgun.ttf"),
              r"C:\Windows\Fonts\malgun.ttf"):
        if os.path.exists(p):
            return pygame.font.Font(p, size)
    return pygame.font.SysFont("malgungothic,arial", size, bold=bold)


# ---------------------------------------------------------------- 저장

class TrialLog:
    """JSONL 한 줄 = 한 시행. 이어하기를 위해 기존 파일을 읽는다."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.rows = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.rows.append(json.loads(line))

    def append(self, rec):
        self.rows.append(rec)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def count(self, phase=None, width=None, staircase_id=None):
        n = 0
        for r in self.rows:
            if phase and r.get("phase") != phase:
                continue
            if width is not None and r.get("stroke_width_px") != width:
                continue
            if staircase_id is not None and r.get("staircase_id") != staircase_id:
                continue
            n += 1
        return n

    def next_trial_id(self):
        return max([r.get("trial_id", -1) for r in self.rows], default=-1) + 1


# ---------------------------------------------------------------- 실행기

class Runner:
    def __init__(self, args):
        self.args = args
        self.rng_py = random.Random(args.seed)
        self.rng = np.random.default_rng(args.seed)
        self.axis = ColorAxis(tuple(args.center_lab), tuple(args.direction_lab),
                              args.axis_name)
        self.layout = LineLayout(length_px=args.line_length)
        self.widths = [int(w) for w in args.widths.split(",")]

        self.log = TrialLog(args.out)
        pygame.init()
        pygame.display.set_caption("선 굵기 색 구별 실험")
        flags = pygame.FULLSCREEN if args.fullscreen else 0
        self.screen = pygame.display.set_mode(
            (0, 0) if args.fullscreen else (args.width, args.height), flags)
        self.W, self.H = self.screen.get_size()
        self.screen_info = {
            "width_px": self.W,
            "width_cm": screen_width_cm(args.screen_diagonal),
            "distance_cm": args.viewing_distance,
        }
        self.f_big, self.f_mid, self.f_small = (load_font(34, True),
                                                load_font(22), load_font(18))
        self.clock = pygame.time.Clock()
        self.aborted = False
        # 같은 초에 두 세션을 시작해도 구분되도록 뒤에 짧은 꼬리를 붙인다.
        self.session_id = "%s_%04x" % (
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            random.Random().randrange(1 << 16))

    # ------------------------------------------------------------ 화면

    def text(self, s, font, y, color=FG):
        surf = font.render(s, True, color)
        self.screen.blit(surf, surf.get_rect(center=(self.W // 2, y)))

    def message(self, lines, wait=True):
        self.screen.fill(BACKGROUND)
        y = self.H // 2 - len(lines) * 22
        for i, ln in enumerate(lines):
            self.text(ln, self.f_big if i == 0 else self.f_mid, y,
                      FG if i == 0 else FG_DIM)
            y += 46 if i == 0 else 32
        pygame.display.flip()
        if not wait:
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

    def draw(self, colors, rects, progress):
        """안티앨리어싱 없이 정수 좌표 사각형으로 그린다."""
        self.screen.fill(BACKGROUND)
        for i, (x, y, w, h) in enumerate(rects):
            self.screen.fill(colors[i], pygame.Rect(x, y, w, h))
            lab = self.f_small.render(str(i + 1), True, FG_DIM)
            self.screen.blit(lab, lab.get_rect(center=(x + w // 2, y + h + 20)))
        self.text("다른 하나를 고르세요", self.f_mid, int(self.H * 0.07))
        self.text(progress, self.f_small, self.H - 22, FG_DIM)
        pygame.display.flip()

    # ------------------------------------------------------------ 한 시행

    def run_trial(self, trial, phase, staircase_id=None, feedback=False,
                  progress=""):
        odd, common = ((trial.rgb_b, trial.rgb_a) if trial.odd_is_b
                       else (trial.rgb_a, trial.rgb_b))
        colors = [common] * 4
        colors[trial.correct_position] = odd
        rects = self.layout.rects(self.W, self.H, trial.stroke_width_px)

        self.screen.fill(BACKGROUND)
        pygame.display.flip()
        pygame.time.wait(self.args.isi_ms)
        pygame.event.clear()
        self.draw(colors, rects, progress)

        t0 = time.perf_counter()
        resp = None
        while resp is None:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return None
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        return None
                    if e.key in KEY_TO_POS:
                        resp = KEY_TO_POS[e.key]
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    for i, (x, y, w, h) in enumerate(rects):
                        if pygame.Rect(x, y, w, h).collidepoint(e.pos):
                            resp = i
            self.clock.tick(120)
        rt = (time.perf_counter() - t0) * 1000.0
        correct = (resp == trial.correct_position)

        if feedback:
            self.screen.fill(BACKGROUND)
            self.text("O" if correct else "X", self.f_big, self.H // 2,
                      FG if correct else FG_DIM)
            pygame.display.flip()
            pygame.time.wait(220)

        a = self.args
        self.log.append({
            "schema_version": SCHEMA_VERSION,
            "config_version": CONFIG_VERSION,
            "participant_id": a.participant,
            "session_id": self.session_id,
            "staircase_id": staircase_id,
            "trial_id": self.log.next_trial_id(),
            "phase": phase,
            "stroke_width_px": trial.stroke_width_px,
            "stroke_width_deg": trial.stroke_width_deg,
            "displayed_rgb_a": list(trial.rgb_a),
            "displayed_rgb_b": list(trial.rgb_b),
            "delta_e00_original": round(trial.delta_e00_original, 4),
            "delta_e00_simulated": round(trial.delta_e00_simulated, 4),
            "correct_position": trial.correct_position,
            "response_position": int(resp),
            "is_correct": bool(correct),
            "cvd_type": a.cvd_type,
            "machado_severity": a.severity,
            "severity_source": a.severity_source,
            # 아래는 조건 재현용. 핵심 지표 아님.
            "color_axis": self.axis.name,
            "center_lab": list(trial.center_lab) if trial.center_lab else None,
            "level_requested": round(trial.level_requested, 4),
            "response_time_ms": round(rt, 1),
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        })
        return correct

    # ------------------------------------------------------------ 모드

    def run_calibration(self):
        a = self.args
        levels_by_w = {}
        for w in self.widths:
            lv = stimulus_levels(a.expected_c80, n_levels=a.n_levels)
            reps = a.trials_per_width // a.n_levels
            levels_by_w[w] = list(np.repeat(lv, reps))

        # 이어하기: 굵기별로 이미 한 시행 수만큼 앞에서 덜어낸다
        plan = []
        for w in self.widths:
            done = self.log.count(phase="main", width=w)
            plan += [(w, x) for x in levels_by_w[w][done:]]
        self.rng_py.shuffle(plan)

        if not plan:
            self.message(["이미 끝났습니다", "교정 측정이 모두 완료되었습니다.",
                          "", "스페이스를 누르면 종료됩니다."])
            return True

        if not self.message([
                "측정 전 확인",
                "모니터 sRGB 모드 / 야간모드·자동밝기 끔",
                "밝기와 조명이 지난 세션과 같은가",
                "화면까지 %.0f cm" % (a.viewing_distance or 0),
                "", "맞으면 스페이스"]):
            return False

        if not self.message([
                "연습",
                "네 선 중 색이 다른 하나를 고르세요.",
                "마우스 클릭 또는 숫자키 1 2 3 4",
                "확실하지 않아도 반드시 하나를 고르세요.",
                "", "스페이스를 누르면 시작합니다."]):
            return False

        for _ in range(a.practice):
            w = self.rng_py.choice(self.widths)
            t = make_trial(a.expected_c80 * 1.8, w, self.axis, a.cvd_type,
                           a.severity, self.rng, self.screen_info)
            if not t.in_gamut:
                continue
            if self.run_trial(t, "practice", feedback=True) is None:
                return False

        if not self.message([
                "본 시행",
                "이제부터는 정오 표시가 없습니다.",
                "남은 시행 %d 회" % len(plan),
                "중간에 쉬는 화면이 나옵니다.",
                "", "스페이스를 누르면 시작합니다."]):
            return False

        since = 0
        skipped = 0
        for k, (w, level) in enumerate(plan):
            t = make_trial(level, w, self.axis, a.cvd_type, a.severity,
                           self.rng, self.screen_info)
            if not t.in_gamut:
                skipped += 1
                continue
            r = self.run_trial(t, "main", progress="%d / %d" % (k + 1, len(plan)))
            if r is None:
                self.aborted = True
                return False
            since += 1
            if since >= a.break_every and k < len(plan) - 1:
                since = 0
                if not self.message(["잠깐 쉬어가세요",
                                     "남은 시행 %d 회" % (len(plan) - k - 1),
                                     "", "준비되면 스페이스"]):
                    self.aborted = True
                    return False
        if skipped:
            print("색역 밖으로 제외된 시행 %d" % skipped)
        self.message(["끝났습니다", "참여해 주셔서 감사합니다.", "", "스페이스"])
        return True

    def run_validation(self):
        a = self.args
        w = int(a.validation_width)
        sid = a.staircase_id
        done = self.log.count(phase="validation", staircase_id=sid)
        if done >= a.staircase_trials:
            self.message(["이미 끝났습니다",
                          "계단 %s 는 완료되었습니다." % sid, "", "스페이스"])
            return True

        top = self.axis.max_reachable(a.cvd_type, a.severity)
        lo, hi = default_range(a.expected_c80, top)
        sc = Staircase(default_start(a.expected_c80, top), lo, hi,
                       max_trials=a.staircase_trials - done)

        if not self.message([
                "검증 측정 (%s)" % sid,
                "앞서와 같은 과제입니다.",
                "선 굵기는 %d px 하나로 고정됩니다." % w,
                "정오 표시는 없습니다.",
                "", "스페이스를 누르면 시작합니다."]):
            return False

        n = 0
        while not sc.done:
            t = make_trial(sc.x, w, self.axis, a.cvd_type, a.severity,
                           self.rng, self.screen_info)
            if not t.in_gamut:
                sc.x = min(sc.x, top * 0.95)
                continue
            r = self.run_trial(t, "validation", staircase_id=sid,
                               progress="%d / %d" % (done + n + 1,
                                                     a.staircase_trials))
            if r is None:
                self.aborted = True
                return False
            sc.update(r)
            n += 1
            if n % a.break_every == 0 and not sc.done:
                if not self.message(["잠깐 쉬어가세요", "", "준비되면 스페이스"]):
                    self.aborted = True
                    return False
        self.message(["끝났습니다", "참여해 주셔서 감사합니다.", "", "스페이스"])
        return True

    def run(self):
        try:
            if self.args.mode == "calibration":
                return self.run_calibration()
            return self.run_validation()
        finally:
            pygame.quit()


# ---------------------------------------------------------------- CLI

def build_parser():
    p = argparse.ArgumentParser(description="선 굵기 색 구별 실험")
    p.add_argument("--participant", required=True, help="P01 같은 코드. 실명 금지")
    p.add_argument("--mode", choices=("calibration", "validation"),
                   default="calibration")
    p.add_argument("--out", default=None, help="JSONL 경로")

    p.add_argument("--cvd-type", required=True, choices=("protan", "deutan"))
    p.add_argument("--severity", type=float, required=True)
    p.add_argument("--severity-source", required=True,
                   help="어느 독립 자료에서 정했는지. 예: patch_session_PO3")

    p.add_argument("--widths", default="3,7,16", help="교정 굵기 px")
    p.add_argument("--validation-width", type=int, default=5)
    p.add_argument("--staircase-id", default="S1")
    p.add_argument("--staircase-trials", type=int, default=60,
                   help="계획서 기본값. 다만 60x3 으로는 모델 구분이 어렵다. "
                        "compare_models.plan_check 로 먼저 확인할 것")

    p.add_argument("--trials-per-width", type=int, default=96)
    p.add_argument("--n-levels", type=int, default=8)
    p.add_argument("--expected-c80", type=float, default=4.0,
                   help="자극 수준 배치용 사전값. 예비 계단으로 잡아둘 것")
    p.add_argument("--practice", type=int, default=8)
    p.add_argument("--break-every", type=int, default=40)
    p.add_argument("--isi-ms", type=int, default=350)

    p.add_argument("--center-lab", type=float, nargs=3, default=[55.0, 6.0, 14.0])
    p.add_argument("--direction-lab", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    p.add_argument("--axis-name", default="lab_a")
    p.add_argument("--line-length", type=int, default=300)

    p.add_argument("--screen-diagonal", type=float, default=None)
    p.add_argument("--viewing-distance", type=float, default=None)
    p.add_argument("--windowed", dest="fullscreen", action="store_false")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fullscreen=True)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.out is None:
        a.out = os.path.join("data", "lines",
                             "%s_%s.jsonl" % (a.participant, a.mode))
    if a.screen_diagonal is None or a.viewing_distance is None:
        print("경고: --screen-diagonal 과 --viewing-distance 가 없으면")
        print("      굵기를 시야각으로 남길 수 없다.")
        print("")

    runner = Runner(a)
    chk = check_axis(runner.axis, a.cvd_type, a.severity, a.expected_c80)
    print("색 방향 점검: %s" % chk["message"])
    if not chk["ok"]:
        print("중단한다. 중심색이나 방향을 바꿔라.")
        pygame.quit()
        return 1

    ok = runner.run()
    print("시행 %d 기록%s" % (len(runner.log.rows),
                            " — 중단됨" if runner.aborted else ""))
    print("  " + a.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
