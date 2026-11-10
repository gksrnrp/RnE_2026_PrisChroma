import json, sys, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

S = Path(sys.argv[1])
R = Path(r'C:\Users\user\RnE-pilot\docs\results-20261110')
D = Path(r'C:\Users\user\OneDrive\바탕 화면\RnE-2026-prischroma-main\팀전달_20261109')
data = json.loads(S.read_text(encoding='utf-8').split('@@JSON@@\n')[1]); note = data.pop('note')
W = [1, 2, 3, 4, 6, 8, 12, 16]; K = lambda d, w: d[str(w)]

# ---- CSV
cols = ['participant', 'cvd', 'width_px', 'n_trials_all_experiments', 'ratio_vs_4px', 'ratio_ci95_low', 'ratio_ci95_high', 'c80_target', 'c80_ci95_low', 'c80_ci95_high',
        'source_sim_dE', 'source_rgb_a', 'source_rgb_b', 'corrected_rgb_a', 'corrected_rgb_b', 'achieved_sim_dE', 'oklab_shift_total', 'status']
for out in [R, D]:
    for old in out.glob('보정_계산_v5_8굵기.csv'): old.unlink()
    with open(out / '보정_계산_전체합산.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh); w.writerow(cols)
        for pid, d in data.items():
            for c in d['corrections']:
                w.writerow([pid, d['cvd'], c['width_px'], c['n_trials_width'], round(c['ratio'], 3), round(c['ratio_ci95'][0], 3), round(c['ratio_ci95'][1], 3),
                            round(c['c80_target'], 3), round(c['c80_ci95'][0], 3), round(c['c80_ci95'][1], 3), round(c['source_sim_dE'], 3),
                            str(c['source_rgb_a']), str(c['source_rgb_b']), str(c.get('corrected_rgb_a', '')), str(c.get('corrected_rgb_b', '')),
                            round(c['achieved_sim_dE'], 3) if 'achieved_sim_dE' in c else '', round(c['shift'], 4) if 'shift' in c else '', c['status']])

# ---- Figure 5
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
x = np.arange(len(W)); bw = .38; TOP = 10
for off, pid, col, name in [(-bw / 2, 'P01', '#c0392b', 'P01 (적색약): 본실험+v3+v4+v5, 1,128회'), (bw / 2, 'P03', '#2e86c1', 'P03 (녹색약): 본실험+v5, 768회')]:
    cs = {c['width_px']: c for c in data[pid]['corrections'] if abs(c['source_sim_dE'] - 1.0) < .1}
    for i, w in enumerate(W):
        c = cs[w]
        if 'shift' in c:
            y = c['shift'] * 100
            ax.bar(x[i] + off, y, bw, color=col, label=name if i == 3 else None)
            ax.text(x[i] + off, y + .15, f'{y:.1f}', ha='center', fontsize=8.5)
        else:
            ax.bar(x[i] + off, TOP - .6, bw, color='white', edgecolor=col, hatch='xx', lw=1)
            ax.text(x[i] + off, TOP - .45, '불가', ha='center', fontsize=8.5, color=col)
ax.set_xticks(x); ax.set_xticklabels([f'{w} px' for w in W])
ax.set_ylabel('색을 바꿔야 하는 양'); ax.set_ylim(0, TOP)
ax.set_title('선 굵기별로 색을 얼마나 바꿔야 구별되나 (참가자별 전체 실험 합산)')
ax.legend(loc='upper right', fontsize=9); ax.grid(axis='y', alpha=.3)
fig.text(.01, .01, '원래 색차 1.0 색쌍 기준. "불가"는 필요한 색차가 모니터 색 범위를 넘음. P03의 1~3 px는 각 48회뿐이라 값이 불안정함.', fontsize=8, color='#555')
fig.tight_layout(rect=(0, .04, 1, 1))
for out in [R, D]: fig.savefig(out / '그림5_굵기별_최소보정량.png')

# ---- Doc section 6-2
doc = R / '팀전달_결과정리.md'; t = doc.read_text(encoding='utf-8')
start = t.index('### 6-2.'); end = t.index('예시(P03, 원래 색차 1.0)')
L = ['### 6-2. 전체 실험 합산 기준 (1~16 px, 8개 굵기)', '',
     '참가자별로 그 사람이 한 실험을 모두 합쳐 굵기별 C80을 다시 구했다. P01: 본실험 384 + v3 240 + v4 120 + v5 384 = 1,128회. P03: 본실험 384 + v5 384 = 768회.', '',
     '모형: `p = 0.25 + (0.75 − lapse) × sigmoid(s × (ln ΔE − m − ρ_w))`.',
     '- ρ_w(굵기별 "4 px 대비 필요한 색차 배율"의 로그)는 **모든 실험이 공유**한다. 굵기 모양(로그·포화 등)을 가정하지 않는다.',
     '- 실험마다 색쌍·단위가 달라 기준 m은 실험(예비실험은 세션×색 경로)별, 기울기 s는 본실험/예비실험 따로 둔다.',
     '- 본실험은 Machado 시뮬레이션 ΔE00와 기록된 고정 lapse, 예비실험은 원본 sRGB ΔE00와 lapse 0.',
     '- 보정 목표 = 합산 모형의 **본실험 단위 C80**. 95% 구간은 parametric bootstrap 200회.', '',
     '| 참가자 | 항목 | ' + ' | '.join(f'{w} px' for w in W) + ' |', '|---|---|' + '---:|' * len(W)]
for pid in ['P01', 'P03']:
    d = data[pid]
    L.append(f'| {pid} | 시행 수 | ' + ' | '.join(str(K(d['n_by_width'], w)) for w in W) + ' |')
    L.append(f'| {pid} | 4 px 대비 배율 | ' + ' | '.join(f"×{K(d['ratio'], w):.2f}" for w in W) + ' |')
    L.append(f'| {pid} | C80 | ' + ' | '.join(f"{K(d['c80'], w):.2f}" for w in W) + ' |')
    L.append(f'| {pid} | C80 95% | ' + ' | '.join(f"{K(d['c80_ci95'], w)[0]:.2f}–{K(d['c80_ci95'], w)[1]:.2f}" for w in W) + ' |')
    for s in [0.5, 1.0, 1.5]:
        cs = {c['width_px']: c for c in d['corrections'] if abs(c['source_sim_dE'] - s) < .1}
        L.append(f'| {pid} | 보정량 (원래 {s}) | ' + ' | '.join((f"{cs[w]['shift']*100:.2f}" if 'shift' in cs[w] else '불가') for w in W) + ' |')
p1, p3 = data['P01'], data['P03']
L += ['',
      f"- **P01: 굵기 효과가 뚜렷하다.** 굵기 공유 모형 AIC {p1['aic_width']:.1f} vs 굵기 효과 없음 {p1['aic_no_width']:.1f}(차이 {p1['aic_no_width']-p1['aic_width']:.1f}). "
      'C80이 1 → 2 → 3 → 4 px로 갈수록 줄고(5.51 → 4.03 → 3.31 → 2.44), 6~16 px는 1.7~2.0으로 거의 일정하다. 1·2 px 배율의 95% 구간은 1을 넘는다.',
      f"- **P03: 합산해도 굵기 효과를 확인할 수 없다.** AIC {p3['aic_width']:.1f} vs {p3['aic_no_width']:.1f}(차이 {p3['aic_no_width']-p3['aic_width']:.1f}). "
      '1~3 px는 v5 각 48회뿐이고 정답률이 찍기 수준에 가까워 배율이 탐색 경계(×54.6)까지 퍼진다. 3 px의 "불가"도 이 불안정성 때문이다.',
      '- 6-1(본실험만)과 값이 다르다. 예: P01 4 px C80 2.01(본실험만) vs 2.44(합산). 예비실험에서 4 px 정답률이 6 px 이상보다 낮았던 것이 반영됐다.',
      '- 가정: 굵기 배율이 실험(색쌍·단위)과 날짜에 상관없이 같다. 세션 간 차이는 기준 m으로만 흡수했다. 검증하지 않은 가정이다.',
      '- 전체 결과: `보정_계산_전체합산.csv`.', '', '']
t = t[:start] + '\n'.join(L) + t[end:]
t = t.replace('`보정_계산_v5_8굵기.csv`(v5 기준 1~16 px), 그림은 `그림5_굵기별_최소보정량.png`(v5 8개 굵기 + 본실험 4·8·16 px).',
              '`보정_계산_전체합산.csv`(참가자별 전체 실험 합산, 1~16 px), 그림은 `그림5_굵기별_최소보정량.png`(전체 합산 기준).')
t = t.replace('  **v5 기준으로는 4 px보다 가는 선에서 보정량이 커진다**(원래 색차 1.0 기준 1 px 5.2, 2 px 6.4, 3 px 2.8 vs 6~16 px 1.1).',
              '  **전체 실험을 합산하면 4 px보다 가는 선에서 보정량이 커진다**(원래 색차 1.0 기준 1 px 7.7, 2 px 5.2, 3 px 4.1, 4 px 2.3 vs 6~16 px 1.7~2.1).')
t = t.replace('- 6-2는 v5 측정값을 본실험 단위로 환산한 값이다(단위 간 같은 배율 가정). 보정 색쌍의 재판별 실험은 후속 연구로 남긴다.',
              '- 6-2는 모든 실험을 합산한 값이다(실험 간 같은 굵기 배율 가정). 보정 색쌍의 재판별 실험은 후속 연구로 남긴다.')
doc.write_text(t, encoding='utf-8', newline='\n')
print('ok', t.count('v5_8굵기'), t.count('v5 기준으로는'))
