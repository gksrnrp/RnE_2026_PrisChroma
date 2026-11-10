import sys, json, hashlib
from pathlib import Path
ROOT = Path(r'C:\Users\user\RnE-pilot\tools\common_color_pilot')
sys.path.insert(0, str(ROOT))
import fit_width_prediction as f

OUT = Path(sys.argv[1])
lines = []
WIDTHS = [1, 2, 3, 4, 6, 8, 12, 16]

def load(pid):
    p = ROOT / 'raw' / 'v5' / pid / 'responses.jsonl'
    rows = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    return rows, hashlib.sha256(p.read_bytes()).hexdigest()

for pid in ['P03', 'P01']:
    rows, sha = load(pid)
    ids = [r['trial']['trial_id'] for r in rows]
    lines.append(f'== {pid}: rows={len(rows)} unique_ids={len(set(ids))} sha={sha[:16]} config={rows[0].get("config_sha256","")[:16]}')
    colors = sorted({r['trial']['condition_id'] for r in rows})
    cells = []
    for ci, c in enumerate(colors):
        for lv in [1, 2, 3]:
            for w in WIDTHS:
                sub = [r for r in rows if r['trial']['condition_id'] == c and r['trial']['level'] == lv and r['trial']['width_px'] == w]
                cells.append({'color_index': ci, 'condition_id': c, 'level': lv, 'width': w,
                              'delta_e00': f.nominal_delta(sub[0]['trial']), 'n': len(sub), 'k': sum(r['correct'] for r in sub)})
    tot = sum(r['correct'] for r in rows)
    lines.append(f'total {tot}/{len(rows)} = {tot/len(rows):.1%}')
    lines.append('width ' + ' '.join(f'{w}:{sum(c["k"] for c in cells if c["width"]==w)}' for w in WIDTHS) + '  (each /48)')
    for c in colors:
        for lv in [1, 2, 3]:
            sub = [x for x in cells if x['condition_id'] == c and x['level'] == lv]
            lines.append(f'  {c} L{lv} dE={sub[0]["delta_e00"]:.3f} ' + ' '.join(str(x['k']) for x in sub))
    for c in colors:
        k = sum(x['k'] for x in cells if x['condition_id'] == c)
        lines.append(f'  color {c}: {k}/192')
    thin = sum(x['k'] for x in cells if x['width'] <= 3); thick = sum(x['k'] for x in cells if x['width'] >= 4)
    lines.append(f'1-3px {thin}/144={thin/144:.1%}  4-16px {thick}/240={thick/240:.1%}')
    for m in f.MODELS:
        b = f.fit(cells, m)
        lines.append(f'  {m}: AIC={b["aic"]:.2f} gamma={b["gamma"]} slope={b["slope"]} lapse={b["lapse"]} alpha={b["alpha"]} boundary={b["on_grid_boundary"]}')
    ev = ROOT / 'raw' / 'v5' / pid / 'events.jsonl'
    lines.append('events: ' + ' | '.join(l[:140] for l in ev.read_text(encoding='utf-8').splitlines()))
    OUT.write_text('\n'.join(lines), encoding='utf-8')
