"""참가자별 전체 실험 합산 굵기 모형.
p = 0.25 + (0.75 - lapse_e) * sigmoid( s_g * (ln dE - m_{session,color} - rho_w) )
 - rho_w: 굵기 w에서 80% 정답에 필요한 색차가 4 px의 exp(rho_w)배 (모든 실험 공유, rho_4 = 0)
 - 본실험: dE = Machado 시뮬레이션 ΔE00, lapse = 기록된 fixed_lapse, 기울기 s_main
 - 예비실험(v3/v4/v5): dE = 원본 sRGB ΔE00, lapse 0, 세션·색 경로별 기준 m, 공통 기울기 s_pilot
본실험 단위 C80(w) = exp(m_main + rho_w + logit(q)/s_main), q = (0.8-0.25)/(0.75-lapse)
"""
import sys, json, csv, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, r'C:\Users\user\RnE-2026-prischroma\src')
PILOT = Path(r'C:\Users\user\RnE-pilot\tools\common_color_pilot'); sys.path.insert(0, str(PILOT))
import fit_width_prediction as f
from path_correction import find_path_constrained_correction as fpc

OUT = Path(sys.argv[1]); NBOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 200
W = [1, 2, 3, 4, 6, 8, 12, 16]; RW = [w for w in W if w != 4]
BK = Path(r'C:\Users\user\RnE-backups'); PB = BK / 'pilot-measurements-20261109-raw'
MAIN_CSV = list(csv.DictReader(open(r'C:\Users\user\RnE-pilot\docs\results-20261110\보정_계산_P01_P03.csv', encoding='utf-8-sig')))
SOURCES = {
    'P01': {'main': BK / 'P01-complete-20261105/experiment/raw/P01-calibration.jsonl',
            'pilot': {'v3': PB / 'v3/aliasB/responses.jsonl', 'v4': PB / 'v4/aliasB/responses.jsonl', 'v5': PILOT / 'raw/v5/P01/responses.jsonl'},
            'cvd': ('protan', 0.94)},
    'P03': {'main': BK / 'P03-complete-20261110/experiment/raw/P03-calibration.jsonl',
            'pilot': {'v5': PILOT / 'raw/v5/P03/responses.jsonl'}, 'cvd': ('deutan', 0.88)},
}
rd = lambda p: [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]

def build(pid):
    s = SOURCES[pid]; T = []  # (group_index, is_main, lnDE, width, lapse, correct)
    groups = ['main']; counts = {}
    main = [r for r in rd(s['main']) if r.get('phase') == 'calibration']
    lam = float(main[0]['fixed_lapse'])
    for r in main:
        T.append((0, 1, math.log(r['delta_e00_simulated']), r['stroke_width_px'], lam, int(r['is_correct'])))
    counts['본실험'] = len(main)
    for ver, p in s['pilot'].items():
        rows = rd(p); counts[ver] = len(rows)
        for r in rows:
            g = f'{ver}:{r["trial"]["condition_id"]}'
            if g not in groups: groups.append(g)
            T.append((groups.index(g), 0, math.log(f.nominal_delta(r['trial'])), r['trial']['width_px'], 0.0, int(r['correct'])))
    return np.array(T, dtype=float), groups, lam, counts

def unpack(th, G):
    m = th[:G]; s_main, s_pilot = th[G], th[G + 1]; rho = dict(zip(RW, th[G + 2:])); rho[4] = 0.0
    return m, s_main, s_pilot, rho

def nll(th, T, G, y):
    m, sm, sp, rho = unpack(th, G)
    s = np.where(T[:, 1] == 1, sm, sp)
    r = np.array([rho[int(w)] for w in T[:, 3]])
    eta = s * (T[:, 2] - m[T[:, 0].astype(int)] - r)
    p = np.clip(.25 + (.75 - T[:, 4]) / (1 + np.exp(-np.clip(eta, -60, 60))), 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p)).sum()

def fit(T, G, y, x0=None):
    x0 = np.r_[np.zeros(G), 2.0, 2.0, np.zeros(len(RW))] if x0 is None else x0
    b = [(-5, 5)] * G + [(0.2, 15), (0.2, 15)] + [(-4, 4)] * len(RW)
    r = minimize(nll, x0, args=(T, G, y), method='L-BFGS-B', bounds=b)
    return r.x, r.fun

def c80(th, G, lam, w):
    m, sm, sp, rho = unpack(th, G); q = (.8 - .25) / (.75 - lam)
    return math.exp(m[0] + rho[w] + math.log(q / (1 - q)) / sm)

rng = np.random.default_rng(20261110); res = {'note': __doc__}; lines = []
for pid in ['P01', 'P03']:
    T, groups, lam, counts = build(pid); G = len(groups); y = T[:, 5]
    th, fun = fit(T, G, y)
    # 참고: 굵기 효과 없음 모형 (rho 모두 0) 대비 AIC
    th0 = minimize(lambda t: nll(np.r_[t, np.zeros(len(RW))], T, G, y), th[:G + 2], method='L-BFGS-B',
                   bounds=[(-5, 5)] * G + [(0.2, 15), (0.2, 15)]).fun
    aic_w, aic_0 = 2 * (G + 2 + len(RW)) + 2 * fun, 2 * (G + 2) + 2 * th0
    ratio = {w: math.exp(unpack(th, G)[3][w]) for w in W}
    C = {w: c80(th, G, lam, w) for w in W}
    boots = []
    for _ in range(NBOOT):
        m, sm, sp, rho = unpack(th, G)
        s = np.where(T[:, 1] == 1, sm, sp); r = np.array([rho[int(w)] for w in T[:, 3]])
        p = .25 + (.75 - T[:, 4]) / (1 + np.exp(-s * (T[:, 2] - m[T[:, 0].astype(int)] - r)))
        yb = (rng.random(len(p)) < p).astype(float)
        tb, _ = fit(T, G, yb, th)
        boots.append({w: c80(tb, G, lam, w) for w in W} | {f'r{w}': math.exp(unpack(tb, G)[3][w]) for w in W})
    ci = {w: [float(np.percentile([b[w] for b in boots], q)) for q in (2.5, 97.5)] for w in W}
    rci = {w: [float(np.percentile([b[f'r{w}'] for b in boots], q)) for q in (2.5, 97.5)] for w in W}
    n_w = {w: int((T[:, 3] == w).sum()) for w in W}
    lines.append(f'{pid} trials={len(T)} {counts} groups={groups} s_main={th[G]:.2f} s_pilot={th[G+1]:.2f} AIC width={aic_w:.1f} none={aic_0:.1f}')
    for w in W:
        lines.append(f'  {w:>2}px n={n_w[w]:>3} ratio x{ratio[w]:.2f} [{rci[w][0]:.2f},{rci[w][1]:.2f}]  C80={C[w]:.2f} [{ci[w][0]:.2f},{ci[w][1]:.2f}]')
    cvd, sev = SOURCES[pid]['cvd']; corr = []
    for r0 in [r for r in MAIN_CSV if r['participant'] == pid and r['width_px'] == '4']:
        a = json.loads(r0['source_rgb_a']); b = json.loads(r0['source_rgb_b'])
        for w in W:
            row = {'width_px': w, 'source_sim_dE': float(r0['source_sim_dE']), 'c80_target': C[w], 'c80_ci95': ci[w],
                   'ratio': ratio[w], 'ratio_ci95': rci[w], 'n_trials_width': n_w[w], 'source_rgb_a': a, 'source_rgb_b': b}
            try:
                x = fpc((65, 0, 0), (0, 1, 0), a, b, C[w], cvd, sev)
                row |= {'corrected_rgb_a': list(x['corrected_rgb_a']), 'corrected_rgb_b': list(x['corrected_rgb_b']),
                        'achieved_sim_dE': float(x['achieved_simulated_delta_e00']), 'shift': float(x['total_color_change_oklab']), 'status': 'corrected'}
            except ValueError as e:
                row |= {'status': 'unreachable', 'error': str(e)}
            corr.append(row)
        lines.append(f'  src={float(r0["source_sim_dE"]):.2f} shift: ' + ' '.join(f'{c["width_px"]}:{c["shift"]*100:.2f}' if 'shift' in c else f'{c["width_px"]}:불가' for c in corr[-len(W):]))
    res[pid] = {'cvd': f'{cvd} {sev}', 'trials': len(T), 'counts': counts, 'n_by_width': n_w, 'fixed_lapse_main': lam,
                'aic_width': aic_w, 'aic_no_width': aic_0, 'ratio': ratio, 'ratio_ci95': rci, 'c80': C, 'c80_ci95': ci, 'corrections': corr}
OUT.write_text('\n'.join(lines) + '\n@@JSON@@\n' + json.dumps(res, ensure_ascii=False), encoding='utf-8')
