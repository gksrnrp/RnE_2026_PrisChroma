"""실험에 쓸 조건 집합(기준색 x 방향)을 최적화해서 골라 저장한다.

왜 최적화가 필요한가
--------------------
색쌍을 적당히 고르면 protan 과 deutan 을 **구별할 수 없다**. 두 유형의
혼동 방향이 14.6도밖에 안 떨어져 있어서, 낮은 severity 에서는 두 모델이
예측하는 역치 패턴이 전체 배율만 다르고 모양이 거의 같아진다. 그러면
"protan 이고 severity 0.68" 과 "deutan 이고 severity 0.60" 이 같은 응답을
설명해 버린다 (실제로 초기 설계에서 이 일이 일어났다).

그래서 조건을 다음 기준으로 고른다.

    두 유형의 로그역치 패턴이, **severity 를 어떻게 맞춰 끼워도**
    서로를 흉내내지 못하도록 모양 차이를 최대화한다.

전체 배율은 역치 θ 로 흡수되므로 평균을 뺀 뒤 비교하고, severity 는
상대편이 가장 유리한 값을 고르게 두는 최소최대(minimax) 방식으로 평가한다.

실행
    python -m colorvision.design            # 최적화 후 conditions.json 저장
    python -m colorvision.design --report   # 저장된 설계 점검만
"""

import argparse
import json
import os
import sys

import numpy as np

from . import machado as mc
from .colorspace import (delta_e00_linear, hex_of, linear_to_srgb, srgb_to_linear)

DESIGN_PATH = os.path.join(os.path.dirname(__file__), "conditions.json")

# 식별력을 평가할 severity 격자.
# 0.2 이하는 두 유형 모두 정상 색각에 가까워 원리적으로 구별되지 않는다.
# 그 구간을 목적함수에 넣으면 어떤 조건을 골라도 점수가 같아져(=상수 바닥)
# 선택에 아무 정보를 주지 못하므로 제외한다. 대신 그 한계는 결과로 보고한다.
SEV_GRID = np.round(np.arange(0.3, 0.95, 0.1), 3)
# 역치를 정의할 기준 색차 (ΔE00). 개인의 실제 θ 와 달라도 패턴 비교에는 영향이 작다.
TARGET_DE = 2.5


# ---------------------------------------------------------------- 후보 풀

def direction_pool(step_deg=10):
    """혼동 평면을 훑는 방향들 + 대조축 + 둘을 섞은 방향."""
    e1, e2 = mc.direction_basis()
    ctrl = mc.control_direction()
    out = []
    for deg in range(0, 180, step_deg):
        r = np.radians(deg)
        out.append(("plane%03d" % deg, np.cos(r) * e1 + np.sin(r) * e2))
    out.append(("control", ctrl))
    for deg in range(0, 180, 30):
        r = np.radians(deg)
        v = (np.cos(r) * e1 + np.sin(r) * e2) * 0.7 + ctrl * 0.7
        out.append(("mix%03d" % deg, v / np.linalg.norm(v)))
    return out


def headroom(lin_base, direction, t_hi):
    d = np.asarray(direction)
    vals = [t_hi]
    for k in range(3):
        if abs(d[k]) > 1e-12:
            vals.append(2.0 * min(lin_base[k], 1.0 - lin_base[k]) / abs(d[k]))
    return float(min(vals))


def build_candidates(n_bases=800, t_hi=0.25, min_tmax=0.05, min_visible_de=9.0,
                     seed=3):
    """(기준색, 방향) 후보들.

    min_visible_de: t_max 에서 정상 색각자가 보는 ΔE00 의 하한.
        이 값이 작으면 색각과 무관하게 아무도 못 맞히는 조건이 되어 쓸모가 없다.
    """
    rng = np.random.default_rng(seed)
    bases = srgb_to_linear(rng.uniform(0.04, 0.96, size=(n_bases, 3)))
    dirs = direction_pool()
    cands = []
    for b in bases:
        for name, d in dirs:
            tm = headroom(b, d, t_hi)
            if tm < min_tmax:
                continue
            de_max = float(delta_e00_linear(b - (tm / 2) * d, b + (tm / 2) * d))
            if de_max < min_visible_de:
                continue
            cands.append({"base": b, "dir_name": name, "dir": d, "t_max": tm,
                          "de_at_tmax": de_max})
    return cands


# ---------------------------------------------------------------- 역치 예측

def log_thresholds(cands, cvd_type, severity, n_grid=48):
    """각 후보에서 지각 ΔE00 가 TARGET_DE 에 닿는 t 의 로그값.

    t_max 에서도 닿지 않으면 '그 위 어딘가'라는 뜻이므로 log(t_max)+0.5 로
    절단해 둔다 (검열된 관측값). 실제 추정은 이 값을 쓰지 않고 시행별
    가능도를 쓰므로, 여기서의 근사는 조건 선택에만 영향을 준다.
    """
    out = np.empty(len(cands))
    for i, c in enumerate(cands):
        ts = np.geomspace(0.002, c["t_max"], n_grid)
        half = np.outer(ts / 2, c["dir"])
        a = mc.simulate_linear(c["base"] - half, cvd_type, severity)
        b = mc.simulate_linear(c["base"] + half, cvd_type, severity)
        de = delta_e00_linear(a, b)
        hit = np.searchsorted(de, TARGET_DE)      # de 는 t 에 대해 단조 증가
        if hit >= n_grid:
            out[i] = np.log(c["t_max"]) + 0.5
        elif hit == 0:
            out[i] = np.log(ts[0])
        else:
            # 로그-로그 선형 보간
            x0, x1 = np.log(ts[hit - 1]), np.log(ts[hit])
            y0, y1 = de[hit - 1], de[hit]
            out[i] = x0 + (x1 - x0) * (TARGET_DE - y0) / max(y1 - y0, 1e-9)
    return out


def threshold_table(cands):
    """(유형, severity, 후보) 로그역치 표."""
    return {t: np.array([log_thresholds(cands, t, s) for s in SEV_GRID])
            for t in mc.TYPES}


# ---------------------------------------------------------------- 식별력 점수

def _centered(v):
    return v - v.mean(axis=-1, keepdims=True)


def _mimicry_rms(table, idx):
    """(참 유형, 참 severity) 마다 상대 유형이 가장 잘 흉내냈을 때 남는 차이.

    반환: (protan 이 참일 때의 severity 별 값, deutan 이 참일 때의 값)
    전체 배율은 역치 θ 가 흡수하므로 평균을 뺀 뒤 비교한다.
    """
    P = _centered(table["protan"][:, idx])     # (S, k)
    D = _centered(table["deutan"][:, idx])
    diff = P[:, None, :] - D[None, :, :]       # (S_P, S_D, k)
    rms = np.sqrt((diff ** 2).mean(axis=-1))   # (S_P, S_D)
    return rms.min(axis=1), rms.min(axis=0)


def identifiability(table, idx):
    """조건 집합 idx 의 유형 식별력 (평균).

    severity 별 최솟값들의 평균을 쓴다. 최솟값 하나만 보면 낮은 severity 의
    원리적 한계가 점수를 지배해 상수가 되어버려, 조건 선택에 쓸 수 없다.
    """
    a, b = _mimicry_rms(table, idx)
    return float(np.mean(np.concatenate([a, b])))


def identifiability_by_severity(table, idx):
    """severity 별 식별력. 어느 정도부터 유형 판정이 가능한지 보여준다."""
    a, b = _mimicry_rms(table, idx)
    return {float(s): {"protan_true": float(a[i]), "deutan_true": float(b[i])}
            for i, s in enumerate(SEV_GRID)}


def severity_sensitivity(table, idx):
    """severity 가 변할 때 로그역치가 얼마나 움직이는가 (severity 추정 가능성)."""
    vals = []
    for t in mc.TYPES:
        m = table[t][:, idx].mean(axis=1)      # severity 별 평균 로그역치
        vals.append(float(m.max() - m.min()))
    return float(np.mean(vals))


# 고른 기준색끼리 최소한 이만큼은 달라야 한다 (ΔE00).
#
# 제약이 없으면 식별력만 보고 고르느라 거의 같은 빨강을 5개씩 뽑는다
# (실제로 첫 설계에서 ΔE00 1.3 짜리 빨강 쌍이 들어갔다). 그러면 피험자
# 입장에서는 몇 가지 색이 계속 반복되는 지루한 실험이 되고, 같은 색에
# 눈이 적응하기도 한다.
#
# 후보 5847개 기준, 제약값별 식별력과 실제 복원 성능 (참값당 8회):
#
#   제약   식별력   색 최소ΔE   s=0.3 유형적중   s>=0.6 유형적중
#      0   0.492        1.3      100% / 100%        100%
#     12   0.457       12.4       88% /  88%        100%   <- 채택
#     20   0.407       20.2       75% /  62%        100%
#     25   조건 12개를 채우지 못함
#
# 12 면 색이 확실히 달라 보이면서 (문제였던 건 ΔE00 1.3 짜리 빨강 쌍이다)
# 복원 성능 손해는 경증(s=0.3) 구간에만 조금 남는다. 그 구간은 원래
# 원리적 한계에 걸려 있고, 추정기가 ΔLL 로 스스로 불확실하다고 보고한다.
MIN_COLOR_SEPARATION = 12.0


def _base_labs(cands):
    from .colorspace import srgb_to_lab
    return np.array([srgb_to_lab(linear_to_srgb(c["base"])) for c in cands])


def greedy_select(cands, table, k=12, alpha=0.15,
                  min_color_sep=MIN_COLOR_SEPARATION, verbose_select=False):
    """식별력을 탐욕적으로 최대화하며 조건 k 개를 고른다.

    alpha 는 severity 민감도에 주는 가중치. 유형만 잘 가르고 severity 에
    둔감한 집합이 되지 않도록 약하게 섞는다.

    min_color_sep 은 고른 기준색끼리의 최소 색차. 식별력을 조금 내주고
    피험자가 보는 색의 다양성을 산다.
    """
    from .colorspace import delta_e00
    n = len(cands)
    labs = _base_labs(cands)
    # 시작점: 두 유형 차이가 가장 큰 단일 후보
    gap = np.abs(table["protan"] - table["deutan"]).max(axis=0)
    chosen = [int(np.argmax(gap))]

    # 고른 색과 너무 가까운 후보는 미리 걸러 둔다. ΔE00 을 후보마다 하나씩
    # 부르면 수십만 번이 되어 느리므로, 전체 후보에 대해 한 번에 계산한다.
    allowed = np.ones(n, dtype=bool)

    def block_near(c):
        if min_color_sep <= 0:
            return
        d = delta_e00(labs, labs[c][None, :])
        allowed[d < min_color_sep] = False

    block_near(chosen[0])

    while len(chosen) < k:
        best, best_score = None, -np.inf
        for j in np.flatnonzero(allowed):
            j = int(j)
            idx = chosen + [j]
            sc = identifiability(table, idx) + alpha * severity_sensitivity(table, idx)
            if sc > best_score:
                best_score, best = sc, j
        if best is None:
            raise RuntimeError(
                "색 최소간격 %.0f 로는 조건 %d 개를 채울 수 없다. 값을 낮춰라."
                % (min_color_sep, k))
        chosen.append(best)
        block_near(best)
        if verbose_select:
            print("  %2d/%d  %s  (남은 후보 %d)"
                  % (len(chosen), k, hex_of(linear_to_srgb(cands[best]["base"])),
                     int(allowed.sum())), flush=True)
    return chosen


# ---------------------------------------------------------------- 확인용 시행

# 확인용 시행이 만족해야 할 조건에서 쓰는 모델들.
# 정상 색각부터 이색형까지 전부 잘 보여야 '누구나 보이는' 자극이다.
CATCH_MODELS = [("normal", 0.0), ("protan", 1.0), ("deutan", 1.0),
                ("protan", 0.6), ("deutan", 0.6)]


# 확인용 시행이 나머지 시행과 비슷해 보이도록 두는 제약.
# 밝기가 크게 차이 나면 피험자가 "이건 확인용이구나"를 알아채고,
# 거의 검정에 가까운 패치는 8비트 표현이 가장 거친 구간이라 피한다.
CATCH_L_RANGE = (35.0, 72.0)     # 두 색 모두 이 L* 범위 안
CATCH_MAX_DL = 12.0              # 두 색의 L* 차이 상한 (색상 차이로 승부하게)


def choose_catch_condition(target_min_de=18.0, n_bases=2500, seed=11):
    """모든 유형이 쉽게 구별하되, 나머지 시행과 비슷해 보이는 색쌍 하나.

    왜 따로 고르나: 확인용 시행은 '지금 집중하고 있는가'를 재는 것이지
    색각을 재는 것이 아니다. 각 조건의 최대 색차를 쓰면, 혼동축 조건의
    확인용 시행은 해당 유형에게 여전히 불가능하다. 그러면 실수율 λ 에
    색각이상이 섞여 들어가 λ 가 부풀려진다.

    그래서 protan/deutan 이 아무리 심해도 보이는 방향의 색쌍을 쓰되,
    밝기 제약을 걸어 다른 시행 사이에 섞여 들어가게 한다.
    """
    from .colorspace import srgb_to_lab
    rng = np.random.default_rng(seed)
    dirs = direction_pool()
    bases = srgb_to_linear(rng.uniform(0.06, 0.94, size=(n_bases, 3)))

    def scan(base, d, ts):
        """t 격자 전체에 대해 (모델별 최소 ΔE00, 밝기 조건 통과 여부) 를 한 번에."""
        half = np.outer(ts / 2.0, d)                 # (T, 3)
        c1, c2 = base - half, base + half
        de = None
        for ty, sv in CATCH_MODELS:
            v = delta_e00_linear(mc.simulate_linear(c1, ty, sv),
                                 mc.simulate_linear(c2, ty, sv))
            de = v if de is None else np.minimum(de, v)
        L1 = srgb_to_lab(linear_to_srgb(c1))[:, 0]
        L2 = srgb_to_lab(linear_to_srgb(c2))[:, 0]
        lo, hi = CATCH_L_RANGE
        ok = ((np.minimum(L1, L2) >= lo) & (np.maximum(L1, L2) <= hi)
              & (np.abs(L1 - L2) <= CATCH_MAX_DL))
        return de, ok

    best = None
    for b in bases:
        for name, d in dirs:
            tm = headroom(b, d, 0.30)
            if tm < 0.12:
                continue
            # 목표 색차에 닿는 가장 작은 t 를 쓴다. 지나치게 튀는 자극을 피하려는 것.
            ts = np.linspace(0.02, tm * 0.95, 24)
            de, ok = scan(b, d, ts)
            hits = np.flatnonzero((de >= target_min_de) & ok)
            if len(hits) == 0:
                continue
            i = int(hits[0])
            # 목표를 넘긴 것 중에서 여유가 가장 큰 것
            if best is None or de[i] > best[0]:
                best = (float(de[i]), b, name, d, float(ts[i]))

    if best is None:
        raise RuntimeError("확인용 시행 후보를 찾지 못했다. target_min_de 를 낮춰라.")
    score, b, name, d, t = best
    return {
        "cid": "catch_%s" % name,
        "base_linear": [float(v) for v in b],
        "base_hex": hex_of(linear_to_srgb(b)),
        "dir_name": name,
        "direction": [float(v) for v in d],
        "t": float(t),
        "t_max": float(t),
        "min_perceived_de": float(score),
        "per_model_de": {"%s_%.1f" % (ty, s):
                         float(delta_e00_linear(
                             mc.simulate_linear(b - (t / 2) * d, ty, s),
                             mc.simulate_linear(b + (t / 2) * d, ty, s)))
                         for ty, s in CATCH_MODELS},
    }


# ---------------------------------------------------------------- 저장/적재

def make_design(k=12, n_bases=800, seed=3, alpha=0.15, verbose=True,
                min_color_sep=MIN_COLOR_SEPARATION, version=2,
                recompute_catch=False):
    if verbose:
        print("후보 생성 중...")
    cands = build_candidates(n_bases=n_bases, seed=seed)
    if verbose:
        print("  후보 조건 %d 개" % len(cands))
        print("역치 표 계산 중 (유형 2 x severity %d)..." % len(SEV_GRID))
    table = threshold_table(cands)
    if verbose:
        print("조건 %d 개 선택 중..." % k)
    idx = greedy_select(cands, table, k=k, alpha=alpha,
                        min_color_sep=min_color_sep, verbose_select=verbose)

    labs = _base_labs(cands)
    from .colorspace import delta_e00
    seps = [float(delta_e00(labs[a], labs[b]))
            for ii, a in enumerate(idx) for b in idx[ii + 1:]]

    design = {
        "version": version,
        "min_color_separation": min_color_sep,
        "actual_min_color_separation": min(seps),
        "target_delta_e": TARGET_DE,
        "sev_grid": [float(s) for s in SEV_GRID],
        "identifiability": identifiability(table, idx),
        "identifiability_by_severity": identifiability_by_severity(table, idx),
        "severity_sensitivity": severity_sensitivity(table, idx),
        "seed": seed,
        "catch_condition": _catch_condition(recompute_catch, verbose),
        "conditions": [],
    }
    for rank, j in enumerate(idx):
        c = cands[j]
        design["conditions"].append({
            "cid": "c%02d_%s" % (rank, c["dir_name"]),
            "base_linear": [float(v) for v in c["base"]],
            "base_hex": hex_of(linear_to_srgb(c["base"])),
            "dir_name": c["dir_name"],
            "direction": [float(v) for v in c["dir"]],
            "t_max": float(c["t_max"]),
            "de_at_tmax": float(c["de_at_tmax"]),
            "log_thr_protan": [float(v) for v in table["protan"][:, j]],
            "log_thr_deutan": [float(v) for v in table["deutan"][:, j]],
        })
    return design


def _catch_condition(recompute, verbose=True):
    """확인용 시행 조건.

    조건 12개와 독립이므로 기본은 기존 파일에서 그대로 가져온다.
    다시 고르는 데 몇 분씩 걸리는데 결과가 같으니 낭비다.
    """
    if not recompute:
        old = load_design()
        if old and "catch_condition" in old:
            if verbose:
                print("확인용 시행: 기존 것을 재사용 (%s)"
                      % old["catch_condition"]["base_hex"])
            return old["catch_condition"]
    if verbose:
        print("확인용 시행 새로 고르는 중... (몇 분 걸린다)")
    return choose_catch_condition()


def save_design(design, path=DESIGN_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(design, f, ensure_ascii=False, indent=2)
    return path


def load_design(path=DESIGN_PATH):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def report_design(design):
    print("=" * 74)
    print("조건 설계  (조건 %d 개)" % len(design["conditions"]))
    print("  유형 식별력      %.3f  (로그역치 패턴 차이 평균, 클수록 좋다)"
          % design["identifiability"])
    print("  severity 민감도  %.3f" % design["severity_sensitivity"])
    if "actual_min_color_separation" in design:
        print("  기준색 최소 색차 %.1f  (설계 v%s)"
              % (design["actual_min_color_separation"], design.get("version", 1)))
    if "identifiability_by_severity" in design:
        print("  severity 별 식별력 (계단법 측정오차는 로그 0.35 안팎)")
        for k, v in sorted(design["identifiability_by_severity"].items(),
                           key=lambda kv: float(kv[0])):
            print("    s=%.1f   protan이 참일 때 %.2f   deutan이 참일 때 %.2f"
                  % (float(k), v["protan_true"], v["deutan_true"]))
    print("-" * 74)
    print("%-18s %-9s %-11s %7s %8s %8s" %
          ("조건", "기준색", "방향", "t_max", "ΔE@max", "logP-logD"))
    sg = design["sev_grid"]
    mid = int(np.argmin(np.abs(np.array(sg) - 0.6)))
    for c in design["conditions"]:
        gap = c["log_thr_protan"][mid] - c["log_thr_deutan"][mid]
        print("%-18s %-9s %-11s %7.3f %8.1f %+8.2f" %
              (c["cid"], c["base_hex"], c["dir_name"], c["t_max"],
               c["de_at_tmax"], gap))
    print("-" * 74)
    cc = design.get("catch_condition")
    if cc:
        print("확인용 시행  %s  기준색 %s  방향 %s  t=%.3f" %
              (cc["cid"], cc["base_hex"], cc["dir_name"], cc["t"]))
        print("  모델별 지각 ΔE00  " +
              "  ".join("%s=%.1f" % (k, v) for k, v in cc["per_model_de"].items()))
        print("-" * 74)
    print("  logP-logD: severity %.1f 에서 'protan이라면' 과 'deutan이라면' 의" % sg[mid])
    print("  로그역치 차이. 부호가 섞여 있어야 두 유형이 구별된다.")
    print("=" * 74)


def main(argv=None):
    p = argparse.ArgumentParser(description="실험 조건 설계 최적화")
    p.add_argument("--report", action="store_true", help="저장된 설계만 출력")
    p.add_argument("-k", type=int, default=12, help="고를 조건 수")
    p.add_argument("--bases", type=int, default=800, help="후보 기준색 수")
    p.add_argument("--alpha", type=float, default=0.15, help="severity 민감도 가중치")
    p.add_argument("--min-color-sep", type=float, default=MIN_COLOR_SEPARATION,
                   help="고른 기준색끼리의 최소 색차 (ΔE00)")
    p.add_argument("--recompute-catch", action="store_true",
                   help="확인용 시행도 다시 고른다 (기본은 기존 것 재사용)")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default=DESIGN_PATH)
    args = p.parse_args(argv)

    if args.report:
        d = load_design(args.out)
        if d is None:
            print("설계 파일이 없다: %s" % args.out)
            return 1
        report_design(d)
        return 0

    d = make_design(k=args.k, n_bases=args.bases, seed=args.seed, alpha=args.alpha,
                    min_color_sep=args.min_color_sep,
                    recompute_catch=args.recompute_catch)
    save_design(d, args.out)
    report_design(d)
    print("저장: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
