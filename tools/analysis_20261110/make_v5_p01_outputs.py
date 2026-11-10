import sys, json, csv
from pathlib import Path
ROOT = Path(r'C:\Users\user\RnE-pilot\tools\common_color_pilot')
sys.path.insert(0, str(ROOT))
import fit_width_prediction as f
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(r'C:\Users\user\RnE-pilot\docs\results-20261110')
V3 = Path(r'C:\Users\user\RnE-backups\pilot-measurements-20261109-raw\v3\aliasB\responses.jsonl')
W5 = [1, 2, 3, 4, 6, 8, 12, 16]
W3 = [1, 2, 3, 4, 8]

def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]

p01 = rows(ROOT / 'raw/v5/P01/responses.jsonl')
p03 = rows(ROOT / 'raw/v5/P03/responses.jsonl')
v3 = rows(V3)
assert len(p01) == 384 and len(p03) == 384 and len(v3) == 240

def per_width(rs, widths):
    out = []
    for w in widths:
        sub = [r for r in rs if r['trial']['width_px'] == w]
        out.append((w, sum(r['correct'] for r in sub), len(sub)))
    return out

a1, a3, av3 = per_width(p01, W5), per_width(p03, W5), per_width(v3, W3)

# CSV: combined per-width
with open(OUT / 'v5_굵기별_정답_P01_P03.csv', 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.writer(fh); w.writerow(['participant', 'width_px', 'correct', 'n', 'accuracy'])
    for pid, data in [('P01', a1), ('P03', a3)]:
        for wd, k, n in data: w.writerow([pid, wd, k, n, round(k / n, 4)])

# CSV: per level cell for P01
with open(OUT / 'P01_v5_색차수준별_원자료.csv', 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.writer(fh); w.writerow(['condition_id', 'level', 'width_px', 'rgb_a', 'rgb_b', 'nominal_delta_e00', 'n', 'correct'])
    for c in sorted({r['trial']['condition_id'] for r in p01}):
        for lv in [1, 2, 3]:
            for wd in W5:
                sub = [r for r in p01 if r['trial']['condition_id'] == c and r['trial']['level'] == lv and r['trial']['width_px'] == wd]
                t = sub[0]['trial']
                w.writerow([c, lv, wd, str(t['rgb_a']), str(t['rgb_b']), round(f.nominal_delta(t), 3), len(sub), sum(r['correct'] for r in sub)])

# CSV: trial-level P01
with open(OUT / 'P01_v5_시행별_원자료.csv', 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.writer(fh); w.writerow(['trial_id', 'condition_id', 'level', 'width_px', 'correct_position', 'response_position', 'correct', 'response_time_ms', 'utc'])
    for r in sorted(p01, key=lambda r: r['trial']['trial_id']):
        t = r['trial']
        w.writerow([t['trial_id'], t['condition_id'], t['level'], t['width_px'], t['correct_position'], r['response_position'], r['correct'], round(r['response_time_ms'], 1), r['utc']])

# Figure 4
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.plot([x[0] for x in a1], [x[1] / x[2] for x in a1], 'o-', color='#c0392b', lw=2, label='P01 v5 (적색약 0.94, 각 48회)')
ax.plot([x[0] for x in av3], [x[1] / x[2] for x in av3], 's--', color='#e59866', lw=1.5, label='P01 v3 (같은 색쌍, 각 48회)')
ax.plot([x[0] for x in a3], [x[1] / x[2] for x in a3], 'o-', color='#2e86c1', lw=2, label='P03 v5 (녹색약 0.88, 각 48회)')
ax.axhline(0.25, color='gray', ls=':', lw=1); ax.text(16.5, 0.255, '찍기 25%', color='gray', fontsize=9, ha='right', va='bottom')
ax.set_xscale('log', base=2); ax.set_xticks(W5); ax.set_xticklabels([str(x) for x in W5])
ax.set_xlabel('선 굵기 (px)'); ax.set_ylabel('정답률'); ax.set_ylim(0, 1)
ax.set_title('같은 공통 색쌍에서 굵기별 정답률 (v3·v5, 탐색)')
ax.grid(alpha=.3); ax.legend(loc='upper left', fontsize=9)
fig.tight_layout(); fig.savefig(OUT / '그림4_P03_v5_굵기별.png')
print('v3 P01:', av3)
print('done')
