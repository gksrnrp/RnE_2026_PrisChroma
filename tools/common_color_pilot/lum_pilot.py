"""v6 luminance-cue pilot: same chromatic paths, luminance changed.

For each v3 color path, the largest v3 pair is shown in three luminance versions
computed in the explicitly supplied Machado model's simulated CIELAB L*:
  iso   - the second color is scaled in linear RGB so simulated L* matches the first
  orig  - the original v3 pair
  boost - the second color is scaled so the simulated L* difference is doubled
Scaling linear RGB keeps chromaticity, so only luminance changes. Values are
model-based; the observer's real luminance function is not measured.
Exploratory follow-up of a post-hoc hypothesis; see LUMINANCE-PILOT.md.
"""
import argparse
from collections import Counter
import ctypes
import json
import math
from pathlib import Path
import random
import re
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw, ImageTk
import pilot

ROOT = pilot.ROOT
V3_PLAN_PATH = ROOT/'raw'/'_design_v3'/'design.json'
PLAN_PATH = ROOT/'raw'/'_design_v6'/'design.json'
WIDTHS = [1, 2, 3, 4, 8]
LUM = ['iso', 'orig', 'boost']
REPEATS = 8
BLOCK_SIZE = 2*len(LUM)*len(WIDTHS)  # 30
TOTAL = BLOCK_SIZE*REPEATS           # 240
SEED = 20261112
DATA_DIR = 'v6'
REQUIRED_SCREEN = (3840, 2160)
SRC = Path(__file__).resolve().parents[2]/'src'


def _models():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from machado import simulate_color_float
    from color_spaces import srgb_to_lab
    from distance import delta_e_2000
    return simulate_color_float, srgb_to_lab, delta_e_2000


def _lin(c):
    c = np.asarray(c, float)/255
    return np.where(c <= .04045, c/12.92, ((c+.055)/1.055)**2.4)


def _enc(lin):
    lin = np.clip(lin, 0, 1)
    v = np.where(lin <= .0031308, lin*12.92, 1.055*lin**(1/2.4)-.055)
    return [int(x) for x in np.rint(v*255)]


def describe(a, b, observer):
    sim, lab, de = _models()
    sa, sb = sim(a, observer['cvd_type'], observer['severity']), sim(b, observer['cvd_type'], observer['severity'])
    return {'normal_delta_e00': float(de(lab(a), lab(b))), 'normal_delta_L': float(lab(b)[0]-lab(a)[0]),
            'sim_delta_e00': float(de(lab(sa), lab(sb))), 'sim_delta_L': float(lab(sb)[0]-lab(sa)[0])}


def scaled_pair(a, b, target_delta_l, observer):
    """Scale b in linear RGB (chromaticity fixed) to hit a simulated L* difference."""
    sim, lab, _ = _models()
    la = lab(sim(a, observer['cvd_type'], observer['severity']))[0]
    best = None
    for k in np.linspace(0.80, 1.20, 4001):
        b2 = _enc(_lin(b)*k)
        if b2 == list(a):
            continue
        err = abs((lab(sim(b2, observer['cvd_type'], observer['severity']))[0]-la)-target_delta_l)
        if best is None or err < best[0]:
            best = (err, b2, float(k))
    return best


def load_observer(path):
    """Require an explicit model and source; no participant measurements are bundled."""
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict) or value.get('cvd_type') not in ('protan', 'deutan'):
        raise ValueError('cvd_type은 protan 또는 deutan이어야 합니다.')
    severity = value.get('severity')
    if isinstance(severity, bool) or not isinstance(severity, (int, float)) or not math.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError('severity를 0~1의 유한한 수로 명시하세요.')
    source = value.get('source')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('모델 설정의 source를 명시하세요.')
    return {'cvd_type': value['cvd_type'], 'severity': float(severity), 'source': source.strip()}


def build_plan(v3, observer):
    if len(v3.get('conditions', [])) != 2 or any(not c.get('pairs') for c in v3['conditions']):
        raise ValueError('색 경로 2개와 각 경로의 색쌍이 있는 v3 계획이 필요합니다.')
    conditions = []
    for c in v3['conditions']:
        base = c['pairs'][-1]  # largest v3 separation
        a, b = list(base['rgb_a']), list(base['rgb_b'])
        orig = describe(a, b, observer)
        pairs = []
        for name in LUM:
            if name == 'orig':
                b2, k, err = b, 1.0, 0.0
            else:
                target = 0.0 if name == 'iso' else 2*orig['sim_delta_L']
                err, b2, k = scaled_pair(a, b, target, observer)
            pairs.append({'lum_condition': name, 'rgb_a': a, 'rgb_b': b2, 'linear_scale_of_b': k,
                          'target_error_L': err, **describe(a, b2, observer),
                          'separation_t': base['separation_t'], 'fraction': base['fraction'],
                          'source_level': base['source_level']})
        conditions.append({'condition_id': c['condition_id'], 'pairs': pairs})
    return {'pilot_version': 6, 'mode': 'exploratory-luminance-cue-test-not-confirmatory',
            'source_v3_design_sha256': pilot.sha(V3_PLAN_PATH),
            'observer_model': dict(observer),
            'luminance_rule': 'scale second color in linear RGB; iso: simulated dL*=0, boost: 2x original simulated dL*',
            'conditions': conditions, 'luminance_conditions': LUM, 'widths': WIDTHS, 'repeats': REPEATS,
            'block_size': BLOCK_SIZE, 'seed': SEED, 'total': TOTAL,
            'line_length_px': pilot.LENGTH, 'background_rgb': list(pilot.BG),
            'calibrated_colorimetry': False, 'observer_luminance_measured': False}


def prepare(observer_path):
    if not V3_PLAN_PATH.exists():
        raise FileNotFoundError(f'v3 색쌍 계획이 필요합니다: {V3_PLAN_PATH}')
    plan = json.loads(json.dumps(build_plan(json.loads(V3_PLAN_PATH.read_text(encoding='utf-8')), load_observer(observer_path))))
    if PLAN_PATH.exists():
        if json.loads(PLAN_PATH.read_text(encoding='utf-8')) != plan:
            raise ValueError('이미 준비된 v6 계획과 다릅니다. 덮어쓰지 않습니다.')
    else:
        PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PLAN_PATH.open('x', encoding='utf-8') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)
    for c in plan['conditions']:
        for p in c['pairs']:
            print(c['condition_id'], p['lum_condition'], p['rgb_a'], p['rgb_b'],
                  f"sim dL*={p['sim_delta_L']:+.2f} sim dE={p['sim_delta_e00']:.2f}")


def plan():
    return json.loads(PLAN_PATH.read_text(encoding='utf-8'))


def schedule():
    p = plan(); rng = random.Random(p['seed']); cells = []
    for c in p['conditions']:
        for width in p['widths']:
            for level, pair in enumerate(c['pairs'], 1):
                combos = [(pos, odd) for pos in range(1, 5) for odd in [False, True]]
                rng.shuffle(combos)
                cells.append((c['condition_id'], width, level, pair, combos))
    trials = []
    for rep in range(p['repeats']):
        block = []
        for cid, width, level, pair, combos in cells:
            pos, odd = combos[rep]
            block.append({'condition_id': cid, 'width_px': width, 'level': level,
                          'lum_condition': pair['lum_condition'],
                          'rgb_a': pair['rgb_a'], 'rgb_b': pair['rgb_b'], 'length_px': pilot.LENGTH,
                          'correct_position': pos, 'odd_is_b': odd, 'block': rep+1, 'repeat': rep+1})
        rng.shuffle(block)
        for trial in block:
            trial['trial_id'] = len(trials)+1; trials.append(trial)
    return trials


class App(pilot.App):
    def __init__(self):
        super().__init__()
        self.root.title(f'색과 선 — 밝기 단서 v6 · {TOTAL}문항')

    def home(self):
        self.clear(); self.phase = 'home'
        self.label(f'v6 · 밝기 단서 실험\n2색 × 밝기 3조건 × 1·2·3·4·8px × 8회 = {TOTAL}문항\n'
                   f'연습 1문항 후 시작 / {BLOCK_SIZE}문항마다 휴식 / 순서 무작위\n'
                   '세 선은 같고 하나만 다릅니다. 모두 같아 보여도 가장 다르게 보이는 하나를 고르세요.\n'
                   '이미 동의한 참가자만 진행하세요. 실명 대신 가명 코드를 입력하세요.')
        self.subject = self.tk.Entry(self.body, font=('맑은 고딕', 16)); self.subject.pack(pady=10)
        self.subject.focus_set()
        self.button('v6 새로 시작 / 이어하기', self.start)
        self.label('1=좌상 / 2=우상 / 3=좌하 / 4=우하\nEsc: 저장 유지·일시정지 / Space: 시작·재개')

    def start(self):
        screen = (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        if screen != REQUIRED_SCREEN:
            return self.messagebox.showerror('화면 해상도', f'프로그램이 인식한 화면이 {screen[0]}x{screen[1]}입니다.\n'
                                             '모니터를 3840x2160으로 맞춘 뒤 이 창을 닫고 다시 켜세요.')
        subject = self.subject.get().strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', subject):
            return self.messagebox.showerror('가명 코드', '영문·숫자·밑줄·하이픈 1~32자를 입력하세요.')
        folder = ROOT/'raw'/DATA_DIR/subject; folder.mkdir(parents=True, exist_ok=True)
        kernel = ctypes.windll.kernel32; kernel.CreateMutexW.restype = ctypes.c_void_p
        self.mutex = kernel.CreateMutexW(None, False, 'Local\\RNELumPilot'+pilot.hashlib.sha256(
            str(folder.resolve()).casefold().encode()).hexdigest())
        if not self.mutex or kernel.GetLastError() == 183:
            if self.mutex: kernel.CloseHandle(ctypes.c_void_p(self.mutex))
            self.mutex = None
            return self.messagebox.showerror('중복 실행', '같은 가명 코드의 v6 창을 먼저 닫으세요.')
        self.journal = folder/'responses.jsonl'; self.events = folder/'events.jsonl'
        config = {'pilot_version': 6, 'design_sha256': pilot.sha(PLAN_PATH), 'code_sha256': pilot.sha(__file__),
                  'shared_module_sha256': pilot.sha(pilot.__file__), 'seed': SEED, 'widths': WIDTHS,
                  'luminance_conditions': LUM, 'repeats': REPEATS, 'total': TOTAL,
                  'line_length_px': pilot.LENGTH, 'background_rgb': list(pilot.BG),
                  'nominal_srgb_not_colorimeter_verified': True}
        try:
            self.trials = schedule(); cp = folder/'config.json'
            if cp.exists():
                if json.loads(cp.read_text(encoding='utf-8')) != config:
                    raise ValueError('이전 v6 설정과 다릅니다. 기존 자료는 수정하지 않았습니다.')
            else:
                if self.journal.exists(): raise ValueError('설정 없는 응답이 있습니다. 덮어쓰지 않습니다.')
                with cp.open('x', encoding='utf-8') as f: json.dump(config, f, ensure_ascii=False, indent=2)
            rows = pilot.read_rows(self.journal); pilot.verify_rows(rows, self.trials); self.index = len(rows)
            self.row_config_sha = pilot.sha(cp)
            if any(r.get('config_sha256') != self.row_config_sha for r in rows):
                raise ValueError('응답에 결합된 설정 해시가 다릅니다.')
            pilot.append(self.events, {'event': 'open_or_resume', 'utc': pilot.now(), 'next_trial': self.index+1,
                                       'screen': list(screen), 'config_sha256': self.row_config_sha})
        except Exception as e:
            kernel.CloseHandle(ctypes.c_void_p(self.mutex)); self.mutex = None
            return self.messagebox.showerror('저장 자료 확인', str(e))
        self.running = True; self.root.attributes('-fullscreen', True)
        self.pause('준비됐으면 시작하세요. 연습 1문항 후 본문을 시작합니다.')

    def pause(self, message='일시정지 — 완료 응답은 저장됐습니다.'):
        self.clear(); self.phase = 'pause'; self.label(message)
        self.label(f'저장 완료 {self.index}/{TOTAL}문항\n같은 가명 코드로 이어 할 수 있습니다.')
        if self.index < len(self.trials): self.button('계속 / 시작 (Space)', self.next)
        else: self.label(f'{TOTAL}문항 완료. 감사합니다.')
        self.button('저장 유지하고 닫기', self.close)

    def next(self):
        if self.index >= len(self.trials): return self.pause('v6 완료')
        if not self.practice_done: return self.practice()
        self.clear(); self.phase = 'fixation'; current = self.token
        self.top.config(text=f'밝기 단서 v6 · {self.index+1}/{TOTAL} · 블록 {self.index//BLOCK_SIZE+1}/{REPEATS}')
        self.panel = self.tk.Label(self.body, bg='#757575'); self.panel.pack(expand=True)
        im = Image.new('RGB', (960, 640), pilot.BG); ImageDraw.Draw(im).text((477, 312), '+', fill='white')
        self.photo = ImageTk.PhotoImage(im); self.panel.config(image=self.photo)

        def show():
            if current != self.token: return
            self.photo = ImageTk.PhotoImage(pilot.scene(self.trials[self.index])); self.panel.config(image=self.photo)
            self.root.update_idletasks(); self.started = time.perf_counter(); self.phase = 'trial'; self.active = True
        self.root.after(500, show)

    def key(self, event):
        if not self.running or event.keysym in self.held: return
        self.held.add(event.keysym)
        if event.keysym == 'Escape':
            if self.active:
                pilot.append(self.events, {'event': 'interrupted_display', 'utc': pilot.now(), 'trial_id': self.index+1})
            self.pause(); return
        if event.keysym == 'space' and not self.active:
            if self.phase == 'pause': self.next()
            return
        if not self.active or not event.char or event.char not in '1234': return
        response = int(event.char); self.active = False
        if self.phase == 'practice':
            self.practice_done = True
            pilot.append(self.events, {'event': 'practice_answer', 'utc': pilot.now(), 'response_position': response,
                                       'correct': response == pilot.practice_trial()['correct_position']})
            self.pause(('맞았습니다.' if response == 3 else '연습 정답은 3번(좌하)이었습니다.')+' 본문을 시작합니다.')
            return
        row = {'utc': pilot.now(), 'trial': self.trials[self.index], 'response_position': response,
               'correct': response == self.trials[self.index]['correct_position'],
               'response_time_ms': (time.perf_counter()-self.started)*1000, 'config_sha256': self.row_config_sha}
        try: pilot.append(self.journal, row)
        except Exception as e:
            self.clear(); self.running = False
            self.label('저장 실패로 중단했습니다.'); self.messagebox.showerror('저장 실패', str(e)); return
        self.index += 1; self.clear(); self.phase = 'iti'
        if self.index % BLOCK_SIZE == 0: self.pause(f'{BLOCK_SIZE}문항 완료 — 저장됐습니다. 필요하면 쉬세요.')
        else:
            current = self.token; self.root.after(300, lambda: self.next() if current == self.token else None)


def check_schedule(trials):
    assert len(trials) == TOTAL and trials == schedule()
    cells = Counter((t['condition_id'], t['width_px'], t['lum_condition']) for t in trials)
    assert len(cells) == 2*len(WIDTHS)*len(LUM) and set(cells.values()) == {REPEATS}
    assert Counter(t['correct_position'] for t in trials) == {p: TOTAL//4 for p in range(1, 5)}
    for b in range(1, REPEATS+1):
        assert len([t for t in trials if t['block'] == b]) == BLOCK_SIZE
    for key in cells:
        sub = [t for t in trials if (t['condition_id'], t['width_px'], t['lum_condition']) == key]
        assert set((t['correct_position'], t['odd_is_b']) for t in sub) == {(q, o) for q in range(1, 5) for o in [True, False]}
    for t in trials:
        pixels = {color: n for n, color in pilot.scene(t).getcolors()}
        odd = tuple(t['rgb_b'] if t['odd_is_b'] else t['rgb_a']); common = tuple(t['rgb_a'] if t['odd_is_b'] else t['rgb_b'])
        assert odd != common and pixels[odd] == pilot.LENGTH*t['width_px'] and pixels[common] == 3*pilot.LENGTH*t['width_px']
    for c in plan()['conditions']:
        by = {p['lum_condition']: p for p in c['pairs']}
        assert abs(by['iso']['sim_delta_L']) < 0.15
        assert abs(by['boost']['sim_delta_L']) > abs(by['orig']['sim_delta_L']) > abs(by['iso']['sim_delta_L'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--observer-config', type=Path, help='비공개 JSON: cvd_type, severity, source')
    args = parser.parse_args()
    if args.prepare:
        if args.observer_config is None:
            parser.error('--prepare에는 --observer-config가 필요합니다.')
        prepare(args.observer_config)
    elif args.check:
        check_schedule(schedule()); print(f'PASS: {TOTAL} trials, balance, rasters, luminance ordering iso<orig<boost')
    else: App().run()
