import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

SRC = Path(r'C:\Users\user\RnE-pilot\tools\common_color_pilot\raw\v5\P01\responses.jsonl')
OUT = Path(r'C:\Users\user\RnE-pilot\docs\results-20261110\그림6_P01_v5_굵기별.png')
W = [1, 2, 3, 4, 6, 8, 12, 16]
rows = [json.loads(l) for l in SRC.read_text(encoding='utf-8').splitlines() if l.strip()]
assert len(rows) == 384

font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
styles = {1: ('#f5b7b1', '색차 1단계 (작음, 각 16회)'), 2: ('#e6735f', '색차 2단계 (각 16회)'), 3: ('#922b21', '색차 3단계 (큼, 각 16회)')}
for lv, (col, lab) in styles.items():
    acc = [sum(r['correct'] for r in rows if r['trial']['level'] == lv and r['trial']['width_px'] == w) / 16 for w in W]
    ax.plot(W, acc, 'o-', color=col, lw=1.5, ms=5, label=lab)
tot = [sum(r['correct'] for r in rows if r['trial']['width_px'] == w) / 48 for w in W]
ax.plot(W, tot, 'D-', color='black', lw=2.5, ms=6, label='전체 (각 48회)')
ax.axhline(0.25, color='gray', ls=':', lw=1); ax.text(16.5, 0.255, '찍기 25%', color='gray', fontsize=9, ha='right', va='bottom')
ax.axvspan(0.9, 3.5, color='#d6eaf8', alpha=.5, lw=0); ax.text(1.8, 0.04, '1~3 px: 39.6%', ha='center', fontsize=9)
ax.text(8, 0.04, '4~16 px: 68.3%', ha='center', fontsize=9)
ax.set_xscale('log', base=2); ax.set_xticks(W); ax.set_xticklabels([str(x) for x in W]); ax.set_xlim(0.9, 18)
ax.set_xlabel('선 굵기 (px)'); ax.set_ylabel('정답률'); ax.set_ylim(0, 1.05)
ax.set_title('P01 (적색약 0.94) v5 굵기별 정답률 — 384회, 탐색')
ax.grid(alpha=.3); ax.legend(loc='upper left', fontsize=9)
fig.tight_layout(); fig.savefig(OUT)
print('ok')
