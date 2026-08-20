"""색쌍 구별 실험의 자극 설계와 계단법(staircase).

설계 요지
---------
색 두 개를 아무렇게나 고르면 protan/deutan을 구별할 수 없다. 대신 Machado
시뮬레이터에서 직접 유도한 **혼동 방향**을 축으로 삼는다.

    기준색 c 에서 방향 d 로 ±t/2 만큼 벌린 두 색을 보여준다.
      d = protan 혼동 방향 -> protan 인 사람만 거의 못 본다
      d = deutan 혼동 방향 -> deutan 인 사람만 거의 못 본다
      d = 대조 방향        -> 두 유형 모두 잘 본다

방향 각도는 protan 혼동 방향을 0도로 하는 평면 좌표계로 표기한다
(deutan 은 약 14.6도). 두 방향이 가깝지만 null 이 날카로워서, 같은 t 에서
한쪽은 ΔE00 ≈ 0, 다른 쪽은 ΔE00 ≈ 3 으로 갈린다.

t 는 계단법으로 조절해 각 (기준색, 방향) 조건의 구별 역치를 찾는다.
"""

import numpy as np

from . import machado as mc
from .colorspace import (srgb_to_linear, linear_to_srgb, delta_e00_linear,
                         hex_of, to_bytes)

# 배경/여백 색 (선형이 아니라 sRGB 기준 중간 회색)
BACKGROUND = (0.46, 0.46, 0.46)

# 계단법이 훑는 t 범위 (선형 RGB 단위)
T_MIN = 0.0025
T_MAX = 0.20

# 매 시행 기준색을 흔드는 폭 (선형 RGB 단위, 측정 방향에 직교).
# 같은 색이 계속 나오면 눈이 적응하고 피험자가 패턴을 외운다.
#
# 가상 관찰자로 측정한 값 (유형별 4회 x 6조건):
#   0.000 -> 색 다양성 30%, 유형적중 100%, severity 오차 0.067
#   0.035 -> 색 다양성 99%, 유형적중 100%, severity 오차 0.065
#   0.070 -> 색 다양성 100%, 유형적중 100%, severity 오차 0.054   <- 채택
#   0.120 -> 색 다양성 100%, 유형적중  96%, severity 오차 0.050
# 0.07 까지는 정확도 손해가 없고 오히려 severity 추정이 나아진다
# (색 공간을 더 넓게 훑기 때문). 그 이상은 유형 판정이 흔들리기 시작한다.
# 0 으로 두면 흔들지 않는다.
JITTER_MAX = 0.07


# ---------------------------------------------------------------- 방향

def probe_directions():
    """실험에 쓸 방향들. {이름: (단위벡터, 평면상 각도(도) 또는 None)}"""
    e1, e2 = mc.direction_basis()
    a_d = np.radians(mc.deutan_angle_deg())

    def plane(deg):
        r = np.radians(deg)
        return np.cos(r) * e1 + np.sin(r) * e2

    return {
        "protan_null": (e1, 0.0),
        "deutan_null": (np.cos(a_d) * e1 + np.sin(a_d) * e2, mc.deutan_angle_deg()),
        "mid45": (plane(45.0), 45.0),
        "control": (mc.control_direction(), None),
    }


# ---------------------------------------------------------------- 기준색 선택

def gamut_headroom(lin_base, direction, t_hi=T_MAX):
    """base ± t/2·d 가 [0,1]^3 안에 남는 최대 t."""
    d = np.asarray(direction, dtype=np.float64)
    lo = np.full(3, np.inf)
    for k in range(3):
        if abs(d[k]) < 1e-12:
            continue
        # base ± (t/2)d_k 가 0..1 안에 있으려면
        lo[k] = 2.0 * min(lin_base[k], 1.0 - lin_base[k]) / abs(d[k])
    return float(min(t_hi, lo.min()))


def diagnostic_score(lin_base, directions, t=0.06):
    """이 기준색이 protan/deutan 을 얼마나 잘 갈라주는가.

    각 진단 방향에서 두 유형의 시뮬 ΔE00 차이를 합산한 값. 클수록 좋다.
    """
    total = 0.0
    for name in ("protan_null", "deutan_null"):
        d = directions[name][0]
        c1, c2 = lin_base - (t / 2) * d, lin_base + (t / 2) * d
        dP = float(delta_e00_linear(mc.simulate_linear(c1, "protan", 1.0),
                                    mc.simulate_linear(c2, "protan", 1.0)))
        dD = float(delta_e00_linear(mc.simulate_linear(c1, "deutan", 1.0),
                                    mc.simulate_linear(c2, "deutan", 1.0)))
        total += abs(dP - dD)
    return total


def min_original_delta_e(lin_base, directions, t=0.06):
    """모든 방향에서 정상 색각자가 보는 색차의 최솟값.

    이 값이 너무 작으면 색각과 무관하게 모두가 어려워하는 조건이 되어
    정보량이 떨어진다. 기준색을 거를 때 하한으로 쓴다.
    """
    vals = []
    for vec, _ in directions.values():
        c1, c2 = lin_base - (t / 2) * vec, lin_base + (t / 2) * vec
        vals.append(float(delta_e00_linear(c1, c2)))
    return min(vals)


def choose_base_colors(n=3, seed=0, min_headroom=0.13, min_orig_de=6.0,
                       n_candidates=20000):
    """진단력이 높으면서 서로 충분히 다른 기준색 n 개를 고른다.

    1) sRGB 공간에서 후보를 무작위로 뽑고
    2) 모든 방향에 대해 여유 공간(headroom)이 부족한 후보를 버리고
    3) 정상 색각자에게도 색차가 너무 작은 후보를 버리고
    4) 진단 점수 상위 후보 중에서 서로 멀리 떨어지도록 탐욕적으로 고른다.

    반환: 선형 RGB 기준색 배열 (n, 3)
    """
    rng = np.random.default_rng(seed)
    dirs = probe_directions()
    cand = srgb_to_linear(rng.uniform(0.10, 0.90, size=(n_candidates, 3)))

    keep, score = [], []
    for c in cand:
        if min(gamut_headroom(c, d) for d, _ in dirs.values()) < min_headroom:
            continue
        if min_original_delta_e(c, dirs) < min_orig_de:
            continue
        keep.append(c)
        score.append(diagnostic_score(c, dirs))
    if not keep:
        raise RuntimeError(
            "조건을 만족하는 기준색 후보가 없다. min_headroom 또는 min_orig_de 를 낮춰라.")

    keep = np.array(keep)
    score = np.array(score)
    top = keep[np.argsort(-score)[:max(n * 30, 60)]]

    chosen = [top[0]]
    while len(chosen) < n and len(chosen) < len(top):
        best, best_d = None, -1.0
        for c in top:
            dmin = min(np.linalg.norm(c - x) for x in chosen)
            if dmin > best_d:
                best, best_d = c, dmin
        chosen.append(best)
        top = np.array([c for c in top if not np.allclose(c, best)])
    return np.array(chosen[:n])


# ---------------------------------------------------------------- 조건

class Condition:
    """(기준색, 방향) 하나 = 계단법 하나."""

    def __init__(self, cid, lin_base, dir_name, direction, angle_deg):
        self.cid = cid
        self.lin_base = np.asarray(lin_base, dtype=np.float64)
        self.dir_name = dir_name
        self.direction = np.asarray(direction, dtype=np.float64)
        self.angle_deg = angle_deg
        self.t_max = gamut_headroom(self.lin_base, self.direction)

    @property
    def base_hex(self):
        return hex_of(linear_to_srgb(self.lin_base))

    def pair(self, t, jitter=None):
        """t 만큼 벌어진 두 색 (선형 RGB).

        jitter 를 주면 기준색을 그만큼 옮긴 뒤 벌린다. 재는 양(방향 d 로의
        색차)은 그대로이면서 매 시행 다른 색이 나오게 하는 장치다.
        """
        t = float(np.clip(t, T_MIN, self.t_max))
        h = (t / 2) * self.direction
        base = self.lin_base if jitter is None else self.lin_base + np.asarray(jitter)
        return base - h, base + h

    def pair_srgb(self, t, jitter=None):
        a, b = self.pair(t, jitter)
        return (np.clip(linear_to_srgb(a), 0, 1), np.clip(linear_to_srgb(b), 0, 1))

    def delta_e_original(self, t, jitter=None):
        a, b = self.pair(t, jitter)
        return float(delta_e00_linear(a, b))

    def sample_jitter(self, rng, t, j_max=JITTER_MAX, tries=16):
        """이 조건에서 쓸 기준색 흔들림을 하나 뽑는다.

        측정 방향 d 에 **직교**하는 방향으로만 흔든다. d 방향으로 흔들면
        두 색의 간격 자체가 바뀌어 계단법이 재는 값이 달라지기 때문이다.
        색역을 벗어나면 다시 뽑고, 끝내 안 되면 흔들지 않는다.
        """
        if j_max <= 0:
            return np.zeros(3)
        d = self.direction
        t = float(np.clip(t, T_MIN, self.t_max))
        h = (t / 2) * d
        for _ in range(tries):
            v = rng.normal(size=3)
            v = v - np.dot(v, d) * d
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            j = (j_max * rng.random()) * (v / n)
            a, b = self.lin_base + j - h, self.lin_base + j + h
            if np.all(a >= 0) and np.all(a <= 1) and np.all(b >= 0) and np.all(b <= 1):
                return j
        return np.zeros(3)

    def __repr__(self):
        return "<Condition %s base=%s dir=%s t_max=%.3f>" % (
            self.cid, self.base_hex, self.dir_name, self.t_max)


def build_conditions_heuristic(n_bases=3, seed=0, directions=None):
    """기준색 몇 개 x 이름 붙은 방향 4종. 단순하고 읽기 쉽지만 식별력이 낮다.

    protan 과 deutan 의 역치 패턴이 severity 조정만으로 서로를 흉내낼 수 있어
    유형 판정이 자주 뒤집힌다 (design.py 문서 참고). 최적화 설계와 비교하는
    대조군으로만 남겨 둔다.
    """
    dirs = probe_directions()
    if directions is not None:
        dirs = {k: v for k, v in dirs.items() if k in directions}
    bases = choose_base_colors(n=n_bases, seed=seed)
    conds = []
    for bi, base in enumerate(bases):
        for name, (vec, ang) in dirs.items():
            conds.append(Condition("b%d_%s" % (bi, name), base, name, vec, ang))
    return conds


def build_catch_condition(design_path=None):
    """확인용 시행에 쓸 조건. 모든 유형이 잘 구별하는 색쌍이다.

    각 조건의 최대 색차를 확인용으로 쓰면, 혼동축 조건에서는 해당 유형에게
    여전히 불가능해서 실수율 대신 색각이상을 재게 된다. 그래서 별도 조건을 쓴다.
    없으면 None 을 돌려주고, 호출 쪽이 옛 방식으로 되돌아간다.
    """
    from .design import load_design
    design = load_design(design_path) if design_path else load_design()
    if not design or "catch_condition" not in design:
        return None
    c = design["catch_condition"]
    cond = Condition(c["cid"], np.array(c["base_linear"]), c["dir_name"],
                     np.array(c["direction"]), None)
    # 확인용 시행은 계단법 대상이 아니므로 T_MAX 상한을 적용하지 않는다.
    # 설계 단계에서 이미 색역 안에 있음을 확인한 t 를 그대로 쓴다.
    cond.t_max = float(c["t_max"])
    cond.catch_t = float(c["t"])
    return cond


def build_conditions(n_bases=None, seed=0, directions=None, design_path=None):
    """실험에 쓸 조건 목록.

    기본은 design.py 가 미리 최적화해 저장한 conditions.json 을 읽는다.
    n_bases 를 주면 옛 휴리스틱 설계를 쓴다 (대조군 비교용).
    """
    if n_bases is not None:
        return build_conditions_heuristic(n_bases=n_bases, seed=seed,
                                          directions=directions)
    from .design import load_design
    design = load_design(design_path) if design_path else load_design()
    if design is None:
        raise RuntimeError(
            "조건 설계 파일이 없다. 먼저 `python -m colorvision.design` 을 실행하라.")
    conds = []
    for c in design["conditions"]:
        cond = Condition(c["cid"], np.array(c["base_linear"]), c["dir_name"],
                         np.array(c["direction"]), None)
        cond.t_max = min(cond.t_max, float(c["t_max"]))
        conds.append(cond)
    return conds


# ---------------------------------------------------------------- 계단법

class Staircase:
    """1-up / 2-down 계단법 (약 70.7% 정답 지점으로 수렴).

    log10(t) 위에서 움직인다. 반전(reversal)이 2회 쌓이면 보폭을 줄인다.
    """

    def __init__(self, t_start, t_max, max_trials=16, max_reversals=8,
                 step_big=0.25, step_small=0.12):
        self.t = float(min(t_start, t_max))
        self.t_max = float(t_max)
        self.max_trials = max_trials
        self.max_reversals = max_reversals
        self.step_big = step_big
        self.step_small = step_small
        self.n_correct_run = 0
        self.reversals = []
        self.last_dir = 0          # -1 내려감, +1 올라감
        self.n_trials = 0
        self.history = []

    @property
    def step(self):
        return self.step_big if len(self.reversals) < 2 else self.step_small

    @property
    def done(self):
        return self.n_trials >= self.max_trials or len(self.reversals) >= self.max_reversals

    def update(self, correct):
        self.n_trials += 1
        self.history.append((self.t, bool(correct)))
        if correct:
            self.n_correct_run += 1
            if self.n_correct_run < 2:
                return
            self.n_correct_run = 0
            new_dir = -1                      # 두 번 맞히면 더 어렵게(작게)
        else:
            self.n_correct_run = 0
            new_dir = +1                      # 틀리면 쉽게(크게)

        if self.last_dir != 0 and new_dir != self.last_dir:
            self.reversals.append(self.t)
        self.last_dir = new_dir

        lo = np.log10(self.t) + new_dir * self.step
        self.t = float(np.clip(10 ** lo, T_MIN, self.t_max))

    def threshold(self, n_last=6):
        """마지막 반전점들의 기하평균. 반전이 부족하면 None."""
        if len(self.reversals) < 2:
            return None
        r = self.reversals[-n_last:]
        return float(10 ** np.mean(np.log10(r)))


# ---------------------------------------------------------------- 세션 진행

def make_staircases(conditions, max_trials=16):
    """조건마다 계단법 하나씩. 시작 t 는 각 조건의 여유 범위 절반에서 출발한다."""
    return {c.cid: Staircase(t_start=min(0.5 * c.t_max, 0.10),
                             t_max=c.t_max, max_trials=max_trials)
            for c in conditions}


def run_session(conditions, respond, max_trials=16, catch_rate=0.08, seed=0,
                break_every=None, on_break=None, stairs=None,
                catch_condition=None, jitter_max=JITTER_MAX):
    """조건들을 무작위로 섞어가며 계단법이 끝날 때까지 시행을 돌린다.

    화면 표시 방식과 무관한 진행 논리만 담는다. 실제 실험(pygame)과
    가상 관찰자 검증이 같은 코드를 쓰게 하기 위한 것이다.

    respond(cond, t, is_catch, jitter) -> True/False, 또는 중단하려면 None
    on_break(active_cids, stairs) -> 계속하려면 True
    stairs 를 직접 넘기면 호출자가 진행 상태를 실시간으로 들여다볼 수 있다.
    catch_condition 을 주면 확인용 시행에 그 조건을 쓴다 (없으면 자동으로 찾는다).

    반환: (조건별 Staircase 딕셔너리, 중단되었는가)
    """
    import random as _random
    rng = _random.Random(seed)
    jrng = np.random.default_rng(seed + 991)
    if stairs is None:
        stairs = make_staircases(conditions, max_trials)
    if catch_condition is None:
        catch_condition = build_catch_condition()
    by_id = {c.cid: c for c in conditions}
    active = [c.cid for c in conditions]
    since_break = 0

    while active:
        cid = rng.choice(active)
        cond, sc = by_id[cid], stairs[cid]
        is_catch = rng.random() < catch_rate
        if is_catch:
            # 확인용 시행은 색각이 아니라 집중도를 잰다. 지금 조건이 아니라
            # 모든 유형이 잘 보는 별도 색쌍을 쓴다.
            trial_cond = catch_condition or cond
            t = getattr(trial_cond, "catch_t", trial_cond.t_max)
        else:
            trial_cond, t = cond, sc.t

        jitter = trial_cond.sample_jitter(jrng, t, j_max=jitter_max)
        result = respond(trial_cond, t, is_catch, jitter)
        if result is None:
            return stairs, True
        if not is_catch:
            sc.update(result)
            if sc.done:
                active.remove(cid)

        since_break += 1
        if break_every and since_break >= break_every and active:
            since_break = 0
            if on_break is not None and not on_break(active, stairs):
                return stairs, True

    return stairs, False
