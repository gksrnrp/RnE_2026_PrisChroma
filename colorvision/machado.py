"""Machado(2009) 색각이상 시뮬레이터 — 최소 구현.

출처:
    Machado, G. M., Oliveira, M. M., Fernandes, L. A. F.
    "A Physiologically-based Model for Simulation of Color Vision Deficiency"
    IEEE Transactions on Visualization and Computer Graphics, 15(6), 1291-1298, 2009.
    행렬 표: https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html
    (논문 Eq.17, 18에는 오타가 있어 저자 페이지의 수정판 표를 쓴다.)

주의 1. 이 행렬들은 **선형 RGB**에 곱한다. 감마 인코딩된 sRGB에 그대로 곱하면 안 된다.
주의 2. 이 파일은 파라미터 추정에 필요한 최소 기능만 담은 것이다.
        팀의 정본 시뮬레이터(이창민 파트)가 완성되면 그쪽으로 교체하고,
        먼저 `python -m colorvision.machado` 로 두 구현의 값이 같은지 확인할 것.
"""

import numpy as np

# severity 0.0 ~ 1.0, 0.1 간격. 각 행이 3x3 행렬 하나(행 우선).
_PROTAN = np.array([
    [1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 1.000000],
    [0.856167, 0.182038, -0.038205, 0.029342, 0.955115, 0.015544, -0.002880, -0.001563, 1.004443],
    [0.734766, 0.334872, -0.069637, 0.051840, 0.919198, 0.028963, -0.004928, -0.004209, 1.009137],
    [0.630323, 0.465641, -0.095964, 0.069181, 0.890046, 0.040773, -0.006308, -0.007724, 1.014032],
    [0.539009, 0.579343, -0.118352, 0.082546, 0.866121, 0.051332, -0.007136, -0.011959, 1.019095],
    [0.458064, 0.679578, -0.137642, 0.092785, 0.846313, 0.060902, -0.007494, -0.016807, 1.024301],
    [0.385450, 0.769005, -0.154455, 0.100526, 0.829802, 0.069673, -0.007442, -0.022190, 1.029632],
    [0.319627, 0.849633, -0.169261, 0.106241, 0.815969, 0.077790, -0.007025, -0.028051, 1.035076],
    [0.259411, 0.923008, -0.182420, 0.110296, 0.804340, 0.085364, -0.006276, -0.034346, 1.040622],
    [0.203876, 0.990338, -0.194214, 0.112975, 0.794542, 0.092483, -0.005222, -0.041043, 1.046265],
    [0.152286, 1.052583, -0.204868, 0.114503, 0.786281, 0.099216, -0.003882, -0.048116, 1.051998],
]).reshape(11, 3, 3)

_DEUTAN = np.array([
    [1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 1.000000],
    [0.866435, 0.177704, -0.044139, 0.049567, 0.939063, 0.011370, -0.003453, 0.007233, 0.996220],
    [0.760729, 0.319078, -0.079807, 0.090568, 0.889315, 0.020117, -0.006027, 0.013325, 0.992702],
    [0.675425, 0.433850, -0.109275, 0.125303, 0.847755, 0.026942, -0.007950, 0.018572, 0.989378],
    [0.605511, 0.528560, -0.134071, 0.155318, 0.812366, 0.032316, -0.009376, 0.023176, 0.986200],
    [0.547494, 0.607765, -0.155259, 0.181692, 0.781742, 0.036566, -0.010410, 0.027275, 0.983136],
    [0.498864, 0.674741, -0.173604, 0.205199, 0.754872, 0.039929, -0.011131, 0.030969, 0.980162],
    [0.457771, 0.731899, -0.189670, 0.226409, 0.731012, 0.042579, -0.011595, 0.034333, 0.977261],
    [0.422823, 0.781057, -0.203881, 0.245752, 0.709602, 0.044646, -0.011843, 0.037423, 0.974421],
    [0.392952, 0.823610, -0.216562, 0.263559, 0.690210, 0.046232, -0.011910, 0.040281, 0.971630],
    [0.367322, 0.860646, -0.227968, 0.280085, 0.672501, 0.047413, -0.011820, 0.042940, 0.968881],
]).reshape(11, 3, 3)

TABLES = {"protan": _PROTAN, "deutan": _DEUTAN}
TYPES = ("protan", "deutan")


def normalize_type(t):
    """'p', 'P', 'protanomaly', 'protan' 등을 'protan'으로 정규화."""
    s = str(t).strip().lower()
    if s.startswith("p"):
        return "protan"
    if s.startswith("d"):
        return "deutan"
    if s in ("normal", "none", "n"):
        return "normal"
    raise ValueError("알 수 없는 유형: %r" % (t,))


def matrix(cvd_type, severity):
    """(유형, severity) -> 3x3 선형 RGB 변환 행렬.

    표에 없는 severity는 인접한 두 행렬의 선형 보간으로 만든다.
    (저자가 0.1 간격 표에서 이 근사가 충분히 정확하다고 명시.)
    """
    t = normalize_type(cvd_type)
    if t == "normal":
        return np.eye(3)
    s = float(np.clip(severity, 0.0, 1.0))
    table = TABLES[t]
    pos = s * 10.0
    lo = int(np.floor(pos))
    if lo >= 10:
        return table[10].copy()
    w = pos - lo
    return (1 - w) * table[lo] + w * table[lo + 1]


def simulate_linear(lin_rgb, cvd_type, severity):
    """선형 RGB -> 해당 색각이상자가 지각하는 선형 RGB."""
    M = matrix(cvd_type, severity)
    return np.asarray(lin_rgb, dtype=np.float64) @ M.T


def simulate_srgb(rgb, cvd_type, severity, clip=True):
    """sRGB [0,1] -> 시뮬레이션된 sRGB [0,1]."""
    from .colorspace import srgb_to_linear, linear_to_srgb
    out = linear_to_srgb(simulate_linear(srgb_to_linear(rgb), cvd_type, severity))
    return np.clip(out, 0.0, 1.0) if clip else out


# ---------------------------------------------------------------- 혼동 방향

def confusion_direction(cvd_type):
    """해당 유형의 **혼동선 방향**을 선형 RGB 단위벡터로 반환.

    severity 1.0 행렬 M은 (거의) 특이행렬이다. 그 최소 특이값에 대응하는
    우특이벡터 v는 M v ≈ 0 을 만족하므로, 색 c에서 v 방향으로 조금 움직여도
    시뮬레이션 결과가 거의 변하지 않는다. 즉 그 사람에게 구별되지 않는 방향이다.

    반환: (방향 v(단위벡터), 최소 특이값) — 특이값이 작을수록 순수한 혼동 방향.
    """
    M = matrix(cvd_type, 1.0)
    _, sv, vt = np.linalg.svd(M)
    v = vt[-1]
    if v[0] < 0:          # 부호를 고정해 두면 로그가 읽기 쉬워진다
        v = -v
    return v / np.linalg.norm(v), float(sv[-1])


def control_direction():
    """protan/deutan 두 혼동 방향 모두에 직교하는 방향.

    두 유형 모두 비교적 잘 구별하는 방향이므로 실험에서 대조 조건으로 쓴다.
    이 방향의 역치까지 함께 높으면 색각이상이 아니라 전반적 민감도 저하다.
    """
    p, _ = confusion_direction("protan")
    d, _ = confusion_direction("deutan")
    v = np.cross(p, d)
    return v / np.linalg.norm(v)


def direction_basis():
    """혼동 방향 탐색용 정규직교 기저 (e1, e2).

    e1 = protan 혼동 방향, e2 = 같은 평면 안에서 e1에 직교하며 deutan 쪽을 향하는 벡터.
    두 혼동선이 이루는 평면을 각도로 훑을 때 쓴다.
    """
    p, _ = confusion_direction("protan")
    d, _ = confusion_direction("deutan")
    e1 = p
    e2 = d - np.dot(d, e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2


def deutan_angle_deg():
    """protan 혼동 방향(0도) 기준으로 deutan 혼동 방향이 놓인 각도."""
    e1, e2 = direction_basis()
    d, _ = confusion_direction("deutan")
    return float(np.degrees(np.arctan2(np.dot(d, e2), np.dot(d, e1))))


# ---------------------------------------------------------------- 자체 점검

def _self_check():
    from .colorspace import delta_e00_linear, srgb_to_linear
    state = {"ok": True}

    def chk(name, cond, extra=""):
        state["ok"] = state["ok"] and bool(cond)
        print("  [%s] %s %s" % ("OK  " if cond else "FAIL", name, extra))

    for t in TYPES:
        chk("%s: severity 0.0 = 항등행렬" % t, np.allclose(matrix(t, 0.0), np.eye(3)))
        dev = float(np.abs(TABLES[t].sum(axis=2) - 1).max())
        chk("%s: 행합 = 1 (백색 보존)" % t, dev < 2e-3, "최대편차 %.5f" % dev)
        chk("%s: severity 0.55 보간 = (0.5,0.6) 평균" % t,
            np.allclose(matrix(t, 0.55), (TABLES[t][5] + TABLES[t][6]) / 2))
        v, sv = confusion_direction(t)
        chk("%s: severity 1.0 행렬이 거의 특이" % t, sv < 0.02, "최소특이값 %.5f" % sv)

    g = np.array([0.5, 0.5, 0.5])
    for t in TYPES:
        for s in (0.3, 0.7, 1.0):
            out = simulate_srgb(g, t, s)
            chk("%s s=%.1f: 회색 보존" % (t, s), np.allclose(out, g, atol=3e-3),
                "-> %s" % np.round(out, 4))

    base = srgb_to_linear(np.array([0.55, 0.45, 0.45]))
    for t in TYPES:
        v, _ = confusion_direction(t)
        c1, c2 = base - 0.06 * v, base + 0.06 * v
        d_orig = float(delta_e00_linear(c1, c2))
        d_sim = float(delta_e00_linear(simulate_linear(c1, t, 1.0),
                                       simulate_linear(c2, t, 1.0)))
        chk("%s: 혼동 방향에서 ΔE00 붕괴" % t, d_sim < 0.15 * d_orig,
            "원본 %.2f -> 시뮬 %.2f" % (d_orig, d_sim))

    print("")
    print("  protan dir = %s" % np.round(confusion_direction("protan")[0], 4))
    print("  deutan dir = %s" % np.round(confusion_direction("deutan")[0], 4))
    print("  control    = %s" % np.round(control_direction(), 4))
    print("  deutan 혼동 방향은 protan 기준 %.1f도" % deutan_angle_deg())
    return state["ok"]


if __name__ == "__main__":
    import sys
    print("Machado 시뮬레이터 자체 점검")
    sys.exit(0 if _self_check() else 1)
