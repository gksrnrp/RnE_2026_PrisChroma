"""자극 생성·기록·실행기 검사.

pygame 은 SDL 더미 드라이버로 화면 없이 돌린다. 키 입력은 event.get 을
가로채서 흉내낸다. 정답 위치는 draw() 에 들어온 색 네 개에서 알아낸다.
"""

import json
import os
import pathlib

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from experiment.runner import SCHEMA_VERSION, Runner, TrialLog, build_parser
from experiment.stimulus import (ColorAxis, LineLayout, check_axis, make_trial,
                                 screen_width_cm, simulated_delta_e,
                                 visual_angle_deg)

TYPE, SEV = "protan", 0.94
REQUIRED = ["schema_version", "config_version", "participant_id", "session_id",
            "staircase_id", "trial_id", "phase", "stroke_width_px",
            "stroke_width_deg", "displayed_rgb_a", "displayed_rgb_b",
            "delta_e00_original", "delta_e00_simulated", "correct_position",
            "response_position", "is_correct", "cvd_type", "machado_severity",
            "severity_source"]


# ----------------------------------------------------------------- 자극

def test_pair_is_symmetric_about_the_center():
    # 한쪽만 움직이면 그 색 자체가 단서가 된다. 중심이 유지되는지 본다.
    from experiment.stimulus import _lab_to_srgb
    ax = ColorAxis((55, 6, 14), (0, 1, 0), "lab_a")
    a, b = ax.pair_at_lab_distance(20.0)
    mid = [(x + y) / 2 for x, y in zip(a, b)]
    center = _lab_to_srgb((55, 6, 14))
    assert max(abs(m - c) for m, c in zip(mid, center)) < 4.0


def test_requested_level_is_met_within_quantization():
    ax = ColorAxis((55, 6, 14), (0, 1, 0), "lab_a")
    for want in (1.0, 2.0, 4.0, 6.0, 9.0):
        a, b, got = ax.pair_for_simulated(want, TYPE, SEV)
        assert a is not None
        # 8 비트 양자화 탓에 정확히 일치하지는 않는다. 기록은 실제값을 남긴다.
        assert abs(got - want) / want < 0.25
        assert simulated_delta_e(a, b, TYPE, SEV) == pytest.approx(got, abs=1e-9)


def test_analysis_uses_achieved_not_requested():
    # 요청값과 실제값이 다를 수 있으므로 Trial 이 둘 다 들고 있어야 한다.
    ax = ColorAxis((55, 6, 14), (0, 1, 0), "lab_a")
    t = make_trial(4.0, 7, ax, TYPE, SEV, np.random.default_rng(0))
    assert t.level_requested == 4.0
    assert t.delta_e00_simulated != t.level_requested
    assert t.delta_e00_original > t.delta_e00_simulated   # 시뮬 후에는 줄어든다


def test_out_of_gamut_is_flagged_not_clipped():
    # 화면이 못 내는 색을 잘라서 억지로 쓰면 기록된 색차가 거짓이 된다.
    ax = ColorAxis((55, 6, 14), (0, 1, 0), "lab_a")
    t = make_trial(9999.0, 7, ax, TYPE, SEV, np.random.default_rng(0))
    assert t.in_gamut is False
    assert t.rgb_a is None


def test_rgb_values_are_valid_bytes():
    ax = ColorAxis((55, 6, 14), (0, 1, 0), "lab_a")
    rng = np.random.default_rng(0)
    for lv in np.linspace(0.5, 12.0, 25):
        t = make_trial(float(lv), 7, ax, TYPE, SEV, rng)
        if not t.in_gamut:
            continue
        for c in list(t.rgb_a) + list(t.rgb_b):
            assert isinstance(c, int) and 0 <= c <= 255


def test_confusion_like_axis_is_rejected_before_the_experiment():
    # 혼동축을 고르면 아무리 벌려도 C80 에 못 닿는다. 미리 걸러야 한다.
    good = check_axis(ColorAxis((55, 6, 14), (0, 1, 0)), TYPE, SEV, 4.0)
    assert good["ok"] and good["max_simulated_delta_e"] > 8.0
    tight = check_axis(ColorAxis((55, 6, 14), (0, 1, 0)), TYPE, SEV, 20.0)
    assert not tight["ok"] and "바꿔야" in tight["message"]


def test_visual_angle_is_recorded_and_monotone():
    scr = {"width_px": 1920, "width_cm": screen_width_cm(15.6),
           "distance_cm": 50.0}
    ax = ColorAxis((55, 6, 14), (0, 1, 0), "lab_a")
    rng = np.random.default_rng(0)
    degs = [make_trial(4.0, w, ax, TYPE, SEV, rng, scr).stroke_width_deg
            for w in (3, 7, 16)]
    assert all(d is not None for d in degs)
    assert degs[0] < degs[1] < degs[2]
    # 시야각 정보가 없으면 None 이어야 한다. 0 이나 px 값을 넣으면 안 된다.
    assert make_trial(4.0, 7, ax, TYPE, SEV, rng).stroke_width_deg is None
    assert visual_angle_deg(7, 1920, None, 50) is None


def test_layout_gives_four_separated_rects():
    rects = LineLayout().rects(1280, 800, 7)
    assert len(rects) == 4
    assert all(w == 7 and h == 300 for _, _, w, h in rects)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not pygame.Rect(*rects[i]).colliderect(pygame.Rect(*rects[j]))
    # 굵기만 달라지고 길이는 고정이다.
    assert [h for _, _, _, h in LineLayout().rects(1280, 800, 16)] == [300] * 4


def test_layout_stays_on_screen():
    for w in (3, 7, 16, 24):
        for x, y, ww, hh in LineLayout().rects(1280, 800, w):
            assert 0 <= x and x + ww <= 1280
            assert 0 <= y and y + hh <= 800


# ----------------------------------------------------------------- 기록

def test_log_appends_and_reloads(tmp_path):
    p = str(tmp_path / "d" / "x.jsonl")
    log = TrialLog(p)
    assert log.next_trial_id() == 0
    for i in range(3):
        log.append({"trial_id": log.next_trial_id(), "phase": "main",
                    "stroke_width_px": 7, "staircase_id": None})
    again = TrialLog(p)
    assert len(again.rows) == 3
    assert again.next_trial_id() == 3
    assert again.count(phase="main", width=7) == 3
    assert again.count(phase="main", width=3) == 0


def test_log_survives_a_broken_process(tmp_path):
    # 시행마다 즉시 저장하므로, 중간에 끊겨도 그때까지가 남아야 한다.
    p = str(tmp_path / "x.jsonl")
    log = TrialLog(p)
    log.append({"trial_id": 0, "phase": "main"})
    del log
    with open(p, encoding="utf-8") as f:
        assert len([ln for ln in f if ln.strip()]) == 1


# ----------------------------------------------------------------- 실행

class FakeInput:
    """draw() 로 들어온 색에서 정답을 알아내고, 정해진 비율로 맞힌다."""

    def __init__(self, accuracy=0.9, seed=0):
        self.rng = np.random.default_rng(seed)
        self.accuracy = accuracy
        self.correct_pos = 0
        self.texts = []

    def key(self):
        pos = self.correct_pos
        if self.rng.random() >= self.accuracy:
            pos = (pos + 1 + int(self.rng.integers(0, 3))) % 4
        return [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4][pos]


@pytest.fixture
def headless(monkeypatch):
    fake = FakeInput()

    def draw(self, colors, rects, progress):
        counts = {}
        for c in colors:
            counts[c] = counts.get(c, 0) + 1
        odd = [c for c, n in counts.items() if n == 1]
        assert len(odd) == 1, "네 색 중 하나만 달라야 한다"
        fake.correct_pos = colors.index(odd[0])

    def text(self, s, font, y, color=None):
        fake.texts.append(s)

    def message(self, lines, wait=True):
        fake.texts.extend(lines)
        return True

    def get(*a, **k):
        return [pygame.event.Event(pygame.KEYDOWN, key=fake.key())]

    monkeypatch.setattr(Runner, "draw", draw)
    monkeypatch.setattr(Runner, "text", text)
    monkeypatch.setattr(Runner, "message", message)
    monkeypatch.setattr(pygame.event, "get", get)
    monkeypatch.setattr(pygame.event, "clear", lambda *a, **k: None)
    monkeypatch.setattr(pygame.time, "wait", lambda *a, **k: None)
    return fake


def _args(tmp_path, **over):
    argv = ["--participant", "T01", "--cvd-type", TYPE,
            "--severity", str(SEV), "--severity-source", "test",
            "--out", str(tmp_path / "t.jsonl"), "--windowed",
            "--isi-ms", "0", "--practice", "0", "--break-every", "10000",
            "--screen-diagonal", "15.6", "--viewing-distance", "50"]
    for k, v in over.items():
        argv += ["--" + k.replace("_", "-"), str(v)]
    return build_parser().parse_args(argv)


def test_calibration_session_writes_every_required_field(tmp_path, headless):
    a = _args(tmp_path, trials_per_width=8, n_levels=8, expected_c80=4.0,
              widths="3,7,16")
    r = Runner(a)
    assert r.run() is True
    rows = TrialLog(str(tmp_path / "t.jsonl")).rows
    assert len(rows) == 24                       # 굵기 3개 x 8 시행
    for row in rows:
        for k in REQUIRED:
            assert k in row, k
        assert row["schema_version"] == SCHEMA_VERSION
        assert row["participant_id"] == "T01"
        assert row["phase"] == "main"
        assert row["staircase_id"] is None
        assert row["cvd_type"] == TYPE
        assert row["machado_severity"] == SEV
        assert row["severity_source"] == "test"
        assert row["correct_position"] in (0, 1, 2, 3)
        assert row["response_position"] in (0, 1, 2, 3)
        assert row["is_correct"] == (row["correct_position"]
                                     == row["response_position"])
        assert row["stroke_width_px"] in (3, 7, 16)
        assert row["stroke_width_deg"] > 0
        assert len(row["displayed_rgb_a"]) == 3
        assert row["delta_e00_simulated"] > 0
    assert sorted(x["trial_id"] for x in rows) == list(range(24))
    assert {x["stroke_width_px"] for x in rows} == {3, 7, 16}


def test_raw_data_contains_no_personal_information(tmp_path, headless):
    a = _args(tmp_path, trials_per_width=8, n_levels=8, widths="7")
    Runner(a).run()
    text = open(str(tmp_path / "t.jsonl"), encoding="utf-8").read()
    for banned in ("name", "이름", "학번", "student_id", "birth", "school"):
        assert banned not in text


def test_answer_position_is_not_predictable(tmp_path, headless):
    a = _args(tmp_path, trials_per_width=200, n_levels=8, widths="7")
    Runner(a).run()
    rows = TrialLog(str(tmp_path / "t.jsonl")).rows
    pos = np.array([r["correct_position"] for r in rows])
    counts = np.bincount(pos, minlength=4)
    assert counts.min() > len(rows) * 0.15        # 한 자리에 쏠리지 않는다
    # 바로 앞 시행과 같은 자리가 나오는 비율이 1/4 근처여야 한다.
    same = float(np.mean(pos[1:] == pos[:-1]))
    assert 0.13 < same < 0.40


def test_the_same_pair_is_not_repeated_too_often(tmp_path, headless):
    # 색이 몇 가지만 돌아가면 참가자가 자극을 외운다. 어느 색쌍도 자기 몫을
    # 크게 넘기지 않아야 한다.
    a = _args(tmp_path, trials_per_width=96, n_levels=8, widths="7")
    Runner(a).run()
    rows = TrialLog(str(tmp_path / "t.jsonl")).rows
    pairs = [tuple(r["displayed_rgb_a"]) + tuple(r["displayed_rgb_b"])
             for r in rows]
    assert len(set(pairs)) == 8                   # 수준 수만큼
    assert max(pairs.count(p) for p in set(pairs)) <= len(rows) / 8 + 1


def test_session_resumes_where_it_stopped(tmp_path, headless):
    out = str(tmp_path / "t.jsonl")
    a = _args(tmp_path, trials_per_width=8, n_levels=8, widths="7")
    assert Runner(a).run() is True
    assert len(TrialLog(out).rows) == 8

    # 반쯤 한 상태를 만들고 다시 실행한다.
    rows = TrialLog(out).rows[:5]
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    r2 = Runner(_args(tmp_path, trials_per_width=8, n_levels=8, widths="7"))
    assert r2.run() is True
    again = TrialLog(out).rows
    assert len(again) == 8                        # 3 개만 더 한다
    assert sorted(x["trial_id"] for x in again[5:]) == [5, 6, 7]
    assert again[5]["session_id"] != again[0]["session_id"]


def test_finished_session_does_not_run_again(tmp_path, headless):
    a = _args(tmp_path, trials_per_width=8, n_levels=8, widths="7")
    assert Runner(a).run() is True
    assert Runner(_args(tmp_path, trials_per_width=8, n_levels=8,
                        widths="7")).run() is True
    assert len(TrialLog(str(tmp_path / "t.jsonl")).rows) == 8


def test_practice_has_feedback_and_main_does_not(tmp_path, headless):
    a = _args(tmp_path, trials_per_width=8, n_levels=8, widths="7")
    a.practice = 6
    Runner(a).run()
    rows = TrialLog(str(tmp_path / "t.jsonl")).rows
    assert sum(1 for x in rows if x["phase"] == "practice") == 6
    assert sum(1 for x in rows if x["phase"] == "main") == 8
    # 정오 표시는 연습 횟수만큼만 나온다. 본 시행에는 없다.
    assert sum(1 for t in headless.texts if t in ("O", "X")) == 6


def test_validation_uses_one_width_and_a_staircase(tmp_path, headless):
    a = _args(tmp_path, widths="7", expected_c80=4.0)
    a.mode = "validation"
    a.validation_width = 5
    a.staircase_trials = 40
    a.staircase_id = "S2"
    assert Runner(a).run() is True
    rows = TrialLog(str(tmp_path / "t.jsonl")).rows
    assert len(rows) == 40
    assert {r["phase"] for r in rows} == {"validation"}
    assert {r["staircase_id"] for r in rows} == {"S2"}
    assert {r["stroke_width_px"] for r in rows} == {5}
    # 계단이므로 색차가 한 값에 머물지 않고 움직인다.
    assert len({r["delta_e00_simulated"] for r in rows}) > 5


def test_validation_sessions_are_logged_separately(tmp_path, headless):
    for sid in ("S1", "S2", "S3"):
        a = _args(tmp_path, widths="7", expected_c80=4.0)
        a.mode, a.validation_width = "validation", 5
        a.staircase_trials, a.staircase_id = 20, sid
        Runner(a).run()
    rows = TrialLog(str(tmp_path / "t.jsonl")).rows
    assert len(rows) == 60
    for sid in ("S1", "S2", "S3"):
        assert sum(1 for r in rows if r["staircase_id"] == sid) == 20


def test_widths_are_not_hard_coded_in_the_stimulus_module():
    # 굵기 값은 인교의 통계가 나온 뒤 5 단계에서 정한다. 지금 자극 코드에
    # 박아두면 나중에 바꿨는지 확인할 수 없다.
    src = open(os.path.join(os.path.dirname(__file__), "experiment",
                            "stimulus.py"), encoding="utf-8").read()
    assert "widths" not in src
    a = _args(pathlib.Path("x"), widths="2,9,21")
    assert [int(w) for w in a.widths.split(",")] == [2, 9, 21]
