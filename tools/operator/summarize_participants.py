"""참가자별 결과를 같은 형식으로 정리한다 (참가자 1명 = 1줄).

입력 위치는 participants_sources.json. 원자료는 읽기만 한다.
출력: participants_summary.csv (고정 열), participants_summary.md (표 + 참가자 카드).
사용: python summarize_participants.py --out <폴더>
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent/'common_color_pilot'
MAIN_SRC = Path('C:/Users/user/RnE-2026-prischroma/src')
V3_W, V5_W, V2_W, V6_W = [1, 2, 3, 4, 8], [1, 2, 3, 4, 6, 8, 12, 16], [2, 4, 8, 16], [1, 2, 3, 4, 8]
DASH = '—'


def jl(path):
    return [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]


def pct(k, n):
    return f'{k}/{n} ({100*k/n:.0f}%)' if n else DASH


def acc_by(rows, key, widths):
    out = {}
    for w in widths:
        sub = [r for r in rows if key(r) == w]
        out[w] = (sum(bool(r['correct']) for r in sub), len(sub))
    return out


def find_pilot(cfg, pid, version):
    alias = cfg['pilot_alias'].get(pid, {}).get(version)
    if not alias:
        return None
    for root in cfg['pilot_roots']:
        p = Path(root)/version/alias/'responses.jsonl'
        if p.is_file():
            return p
    return None


def colorvision(cfg, pid):
    for d in cfg['colorvision_dirs']:
        hits = sorted(glob.glob(os.path.join(d, f'{pid}_*_params.json')))
        if hits:
            return json.load(open(hits[-1], encoding='utf-8'))
    return None


def main_block(arch):
    a = Path(arch)
    info = {'status': '없음'}
    cfgp = a/'experiment/private/experiment_config.json'
    if not cfgp.is_file():
        return info, None
    cfg = json.load(open(cfgp, encoding='utf-8'))
    o = cfg['observer_profile']
    info.update(cvd_type=o['cvd_type'], severity=o['severity'], lapse=o['fixed_lapse'], severity_session=o['severity_session_id'])
    raw = glob.glob(str(a/'experiment/raw/*-calibration.jsonl'))
    rows = jl(raw[0]) if raw else []
    done = sum(r['trial_status'] == 'completed' for r in rows)
    info['trials'] = f'{done}/384'
    info['status'] = '완료' if done == 384 else ('진행 중' if done else '없음')
    priv = a/'experiment/private'
    flags = []
    if (priv/'runtime-os-deviation-log.jsonl').is_file(): flags.append('OS 빌드 예외')
    if (priv/'session-date-deviation-notes.md').is_file(): flags.append('같은 날 블록 진행')
    if glob.glob(str(priv/'*session-limit-deviation.json')): flags.append('30분 상한 후 이어 수집')
    info['deviations'] = ', '.join(flags) or '없음'
    rec = glob.glob(str(a/'results/parameter_recovery*.json'))
    if rec:
        r = json.load(open(rec[0], encoding='utf-8'))
        info['recovery'] = '통과' if r.get('passed') else '불합격(' + ', '.join(x.split(':')[0] for x in r.get('reasons', [])) + ')'
    info['collection_exception'] = '있음' if (priv/'calibration-collection-deviation.json').is_file() else '없음'
    fits = a/'results/calibration_model_fits.json'
    if not fits.is_file():
        info['analysis'] = '분석 전' if done == 384 else '자료 부족'
        return info, None
    p = json.load(open(fits, encoding='utf-8'))['participants'][0]
    d = [m for m in p['models'] if m['model'] == 'D384'][0]
    for t in d['training_range_checks']:
        w = round(t['stroke_width_deg']/0.1114*8)
        info[f'c80_{w}'] = f"{t['c80_sim']:.2f} ({t['ci_low']:.2f}~{t['ci_high']:.2f})"
    iv = {i['parameter']: i for i in d['bootstrap']['parameter_intervals']}
    b = d['fit']['parameters']['beta']
    info['beta'] = f"{b:+.3f} ({iv['beta']['ci_low']:+.2f}~{iv['beta']['ci_high']:+.2f})"
    lo, hi = iv['beta']['ci_low'], iv['beta']['ci_high']
    info['beta_verdict'] = '미확인(CI가 0 포함)' if lo <= 0 <= hi else ('굵을수록 C80 감소' if hi < 0 else '굵을수록 C80 증가')
    acc = [x['accuracy'] for x in p['calibration']['session_block_quality']]
    info['block_acc'] = f'{100*min(acc):.0f}~{100*max(acc):.0f}%'
    info['analysis'] = '완료'
    return info, (cfg, d)


def correction(cfg, d, source=1.0):
    sys.path.insert(0, str(MAIN_SRC))
    from stimulus_design import generate_symmetric_pair
    from path_correction import find_path_constrained_correction
    o = cfg['observer_profile']; cd = cfg['color_design']
    s = generate_symmetric_pair(cd['midpoint_lab'], cd['direction_lab'], source, o['cvd_type'], o['severity'], max_target_error=0.05)
    ref_deg = 0.1114
    out = {}
    for w, deg in [(4, 0.0557), (8, 0.1114), (16, 0.2228)]:
        c80 = d['fit']['parameters']['c80_ref']*(deg/ref_deg)**d['fit']['parameters']['beta']
        r = find_path_constrained_correction(cd['midpoint_lab'], cd['direction_lab'], s['rgb_a'], s['rgb_b'], c80, o['cvd_type'], o['severity'])
        out[w] = r['total_color_change_oklab']*100
    return out


def v5_models(rows):
    sys.path.insert(0, str(PILOT)); argv = sys.argv; sys.argv = ['x']
    import fit_width_prediction as F
    sys.argv = argv
    colors = sorted({r['trial']['condition_id'] for r in rows})
    cells = []
    for ci, c in enumerate(colors):
        for lv in sorted({r['trial']['level'] for r in rows}):
            for w in V5_W:
                sub = [r for r in rows if r['trial']['condition_id'] == c and r['trial']['level'] == lv and r['trial']['width_px'] == w]
                if sub:
                    cells.append({'color_index': ci, 'condition_id': c, 'level': lv, 'width': w, 'delta_e00': F.nominal_delta(sub[0]['trial']),
                                  'n': len(sub), 'k': sum(bool(r['correct']) for r in sub)})
    fits = {m: F.fit(cells, m)['aic'] for m in F.MODELS}
    best = min(fits, key=fits.get); gap = sorted(fits.values())[1]-fits[best]
    return fits, best, gap


COLUMNS = ['participant', 'cvd_type', 'severity', 'fixed_lapse',
           'main_status', 'main_trials', 'main_deviations', 'recovery', 'collection_exception',
           'C80_4px', 'C80_8px', 'C80_16px', 'beta', 'beta_verdict', 'block_accuracy_range',
           'corr_shift_4px', 'corr_shift_8px', 'corr_shift_16px',
           'v2_2px', 'v2_4px', 'v2_8px', 'v2_16px',
           'v3_1px', 'v3_2px', 'v3_3px', 'v3_4px', 'v3_8px', 'v4_accuracy',
           *[f'v5_{w}px' for w in V5_W], 'v5_thin_1to3', 'v5_thick_4to16', 'v5_best_model', 'v5_aic_gap',
           'v6_iso', 'v6_orig', 'v6_boost']


def summarize(cfg, pid):
    row = {c: DASH for c in COLUMNS}; row['participant'] = pid
    arch = cfg['main_archives'].get(pid)
    info, fit = main_block(arch) if arch else ({'status': '없음'}, None)
    cv = colorvision(cfg, pid)
    row.update(cvd_type=info.get('cvd_type', cv['type'] if cv else DASH), severity=info.get('severity', cv['severity'] if cv else DASH),
               fixed_lapse=info.get('lapse', cv['lapse'] if cv else DASH), main_status=info['status'], main_trials=info.get('trials', DASH),
               main_deviations=info.get('deviations', DASH), recovery=info.get('recovery', DASH), collection_exception=info.get('collection_exception', DASH),
               C80_4px=info.get('c80_4', DASH), C80_8px=info.get('c80_8', DASH), C80_16px=info.get('c80_16', DASH),
               beta=info.get('beta', DASH), beta_verdict=info.get('beta_verdict', DASH), block_accuracy_range=info.get('block_acc', DASH))
    if fit:
        sh = correction(*fit)
        row.update(corr_shift_4px=f'{sh[4]:.2f}', corr_shift_8px=f'{sh[8]:.2f}', corr_shift_16px=f'{sh[16]:.2f}')
    p = find_pilot(cfg, pid, 'v2')
    if p:
        a = acc_by(jl(p), lambda r: r['trial']['width_px'], V2_W); row.update({f'v2_{w}px': pct(*a[w]) for w in V2_W})
    p = find_pilot(cfg, pid, 'v3')
    if p:
        a = acc_by(jl(p), lambda r: r['trial']['width_px'], V3_W); row.update({f'v3_{w}px': pct(*a[w]) for w in V3_W})
    p = find_pilot(cfg, pid, 'v4')
    if p:
        r4 = jl(p); row['v4_accuracy'] = pct(sum(bool(r['correct']) for r in r4), len(r4))
    p = find_pilot(cfg, pid, 'v5')
    if p:
        r5 = jl(p); a = acc_by(r5, lambda r: r['trial']['width_px'], V5_W)
        row.update({f'v5_{w}px': pct(*a[w]) for w in V5_W})
        thin = [sum(a[w][i] for w in (1, 2, 3)) for i in (0, 1)]; thick = [sum(a[w][i] for w in (4, 6, 8, 12, 16)) for i in (0, 1)]
        row.update(v5_thin_1to3=pct(*thin), v5_thick_4to16=pct(*thick))
        if len(r5) == 384:
            fits, best, gap = v5_models(r5); row.update(v5_best_model=best, v5_aic_gap=f'{gap:.2f}')
    p = find_pilot(cfg, pid, 'v6')
    if p:
        r6 = jl(p)
        for c in ('iso', 'orig', 'boost'):
            sub = [r for r in r6 if r['trial']['lum_condition'] == c]
            row[f'v6_{c}'] = pct(sum(bool(r['correct']) for r in sub), len(sub))
    return row


SECTIONS = [
    ('참가자 정보', ['participant', 'cvd_type', 'severity', 'fixed_lapse']),
    ('본실험 (C80, Machado 후 ΔE00; 괄호는 95% CI)', ['participant', 'main_status', 'main_trials', 'C80_4px', 'C80_8px', 'C80_16px', 'beta', 'beta_verdict']),
    ('본실험 품질·절차', ['participant', 'recovery', 'collection_exception', 'main_deviations', 'block_accuracy_range']),
    ('최소 보정 계산 (원래 시뮬레이션 색차 1.0 색쌍, OKLab 이동 합 ×100)', ['participant', 'corr_shift_4px', 'corr_shift_8px', 'corr_shift_16px']),
    ('예비실험 v2·v3·v4 (공통색, 정답률)', ['participant', 'v2_2px', 'v2_4px', 'v2_8px', 'v2_16px', 'v3_1px', 'v3_2px', 'v3_3px', 'v3_4px', 'v3_8px', 'v4_accuracy']),
    ('후속 실험 v5 (같은 색쌍, 굵기 8개)', ['participant', *[f'v5_{w}px' for w in V5_W], 'v5_thin_1to3', 'v5_thick_4to16', 'v5_best_model', 'v5_aic_gap']),
    ('밝기 실험 v6 (적색약만)', ['participant', 'v6_iso', 'v6_orig', 'v6_boost']),
]


def table(rows, cols):
    head = '| ' + ' | '.join(cols) + ' |\n|' + '---|'*len(cols) + '\n'
    return head + ''.join('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n' for r in rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--sources', default=str(HERE/'participants_sources.json'))
    args = ap.parse_args()
    cfg = json.load(open(args.sources, encoding='utf-8'))
    pids = sorted(set(cfg['main_archives']) | set(cfg['pilot_alias']))
    rows = [summarize(cfg, pid) for pid in pids]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out/'participants_summary.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); [w.writerow(r) for r in rows]
    md = ['# 참가자별 결과 요약 (자동 생성)', '', f'참가자 {len(rows)}명. `—`는 해당 실험을 하지 않았거나 자료가 없음.', '']
    for title, cols in SECTIONS:
        md += [f'## {title}', '', table(rows, cols)]
    if cfg.get('excluded'):
        md += ['## 분석 제외', ''] + [f'- {k}: {v}' for k, v in cfg['excluded'].items()] + ['']
    md += ['## 읽는 법', '',
           '- C80: 정답률 80%가 되는 시뮬레이션 후 색차. 작을수록 잘 구별한다.',
           '- β: 굵기에 따른 C80 변화 기울기. 95% CI가 0을 포함하면 굵기 효과 미확인.',
           '- recovery 불합격·수집 예외가 있으면 신뢰구간은 참고용이며 확정 결론에 쓰지 않는다.',
           '- v5 thin/thick: 1~3 px 합계와 4~16 px 합계 정답률. best_model은 AIC 최소 모형, gap이 2 미만이면 구분 약함.',
           '- 예비실험 v2~v4의 참가자 대응은 participants_sources.json의 pilot_alias를 따른다.']
    (out/'participants_summary.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print('participants', len(rows), '->', out)


if __name__ == '__main__':
    main()
