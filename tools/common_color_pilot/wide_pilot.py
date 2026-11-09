"""v5 wide-width follow-up: v3 color pairs, 8 widths, 384 trials.

Only the width set changes from v3. Color paths, the three color pairs,
line length, background and 2x2 4AFC stay exactly as frozen in the v3 plan.
Exploratory follow-up; the 4px plateau hypothesis came from viewed pilot data.
"""
import argparse
from collections import Counter
import ctypes
import json
from pathlib import Path
import random
import re
import tempfile
import time

from PIL import Image, ImageDraw, ImageTk
import pilot

ROOT = pilot.ROOT
V3_PLAN_PATH = ROOT/'raw'/'_design_v3'/'design.json'
PLAN_PATH = ROOT/'raw'/'_design_v5'/'design.json'
WIDTHS = [1, 2, 3, 4, 6, 8, 12, 16]
REPEATS = 8
COLORS = 2
LEVELS = 3
BLOCK_SIZE = COLORS*len(WIDTHS)*LEVELS  # 48: every color x width x level once
TOTAL = BLOCK_SIZE*REPEATS              # 384
SEED = 20261111
DATA_DIR = 'v5'
REQUIRED_SCREEN = (3840, 2160)


def build_plan(v3):
    if len(v3.get('conditions', [])) != COLORS:
        raise ValueError('v3 계획에 색 경로가 2개가 아닙니다.')
    for c in v3['conditions']:
        if len(c.get('pairs', [])) != LEVELS:
            raise ValueError('v3 계획의 색차 단계가 3개가 아닙니다.')
    return {'pilot_version': 5,
        'mode': 'exploratory-wide-width-follow-up-not-confirmatory',
        'source_v3_design_sha256': pilot.sha(V3_PLAN_PATH),
        'changed_from_v3': 'widths only: 1,2,3,4,8 -> 1,2,3,4,6,8,12,16; repeats stay 8',
        'design_rule': 'simulation of v3 vs wider designs on 2026-11-09 chose 8 widths x 8 repeats = 384; '
                       'no new human response used for this choice',
        'conditions': v3['conditions'], 'widths': WIDTHS, 'repeats': REPEATS,
        'block_size': BLOCK_SIZE, 'seed': SEED, 'total': TOTAL,
        'line_length_px': pilot.LENGTH, 'background_rgb': list(pilot.BG),
        'calibrated_colorimetry': False, 'physical_pixel_mapping_verified': False,
        'claim_allowed': 'descriptive follow-up only; 4px plateau is a post-hoc hypothesis'}


def prepare():
    if not V3_PLAN_PATH.exists():
        raise FileNotFoundError(f'v3 색쌍 계획이 필요합니다: {V3_PLAN_PATH}')
    plan = json.loads(json.dumps(build_plan(json.loads(V3_PLAN_PATH.read_text(encoding='utf-8')))))
    if PLAN_PATH.exists():
        if json.loads(PLAN_PATH.read_text(encoding='utf-8')) != plan:
            raise ValueError('이미 준비된 v5 계획과 다릅니다. 덮어쓰지 않습니다.')
    else:
        PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PLAN_PATH.open('x', encoding='utf-8') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)
    print(f'v5 계획 준비: 2색 x {len(WIDTHS)}굵기 x 3색차 x {REPEATS}회 = {TOTAL}문항')
    print('색 경로:', ','.join(c['condition_id'] for c in plan['conditions']))


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
                'rgb_a': pair['rgb_a'], 'rgb_b': pair['rgb_b'],
                'separation_t': pair['separation_t'], 'fraction': pair['fraction'],
                'source_level': pair['source_level'], 'length_px': pilot.LENGTH,
                'correct_position': pos, 'odd_is_b': odd, 'block': rep+1, 'repeat': rep+1})
        rng.shuffle(block)
        for trial in block:
            trial['trial_id'] = len(trials)+1; trials.append(trial)
    return trials


class App(pilot.App):
    def __init__(self):
        super().__init__()
        self.root.title(f'색과 선 — 넓은 굵기 v5 · {TOTAL}문항')

    def home(self):
        self.clear(); self.phase = 'home'
        self.label(f'v5 · 같은 색쌍, 굵기 8가지\n2색 조건 × 1·2·3·4·6·8·12·16px × 3색차 × 8회 = {TOTAL}문항\n'
                   f'연습 1문항 후 시작 / {BLOCK_SIZE}문항마다 휴식 / 순서 무작위\n'
                   '세 선은 같고 하나만 다릅니다. 모두 같아 보여도 가장 다르게 보이는 하나를 고르세요.\n'
                   '이미 동의한 참가자만 진행하세요. 실명 대신 가명 코드를 입력하세요.')
        self.subject = self.tk.Entry(self.body, font=('맑은 고딕', 16)); self.subject.pack(pady=10)
        self.subject.focus_set()
        self.button('v5 새로 시작 / v5 이어하기', self.start)
        self.label('1=좌상 / 2=우상 / 3=좌하 / 4=우하\nEsc: 저장 유지·일시정지 / Space: 시작·재개\n'
                   '화면 배율·거리·조명은 v3와 같게 유지하세요. 정답 피드백은 연습에서만 제공합니다.')

    def start(self):
        # v3와 같은 1:1 4K 출력에서만 진행한다. 복제 모드 2560x1440이면 모니터가 확대해 선이 번진다.
        screen = (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        if screen != REQUIRED_SCREEN:
            return self.messagebox.showerror(
                '화면 해상도', f'현재 {screen[0]}x{screen[1]}입니다. Windows 디스플레이 설정에서 '
                f'AOC 모니터를 {REQUIRED_SCREEN[0]}x{REQUIRED_SCREEN[1]}로 맞춘 뒤 다시 시작하세요.\n'
                'Win+P에서 "두 번째 화면만"을 쓰면 안전합니다.')
        subject = self.subject.get().strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', subject):
            return self.messagebox.showerror('가명 코드', '영문·숫자·밑줄·하이픈 1~32자를 입력하세요.')
        folder = ROOT/'raw'/DATA_DIR/subject; folder.mkdir(parents=True, exist_ok=True)
        kernel = ctypes.windll.kernel32; kernel.CreateMutexW.restype = ctypes.c_void_p
        self.mutex = kernel.CreateMutexW(None, False, 'Local\\RNEWidePilot'+pilot.hashlib.sha256(
            str(folder.resolve()).casefold().encode()).hexdigest())
        if not self.mutex or kernel.GetLastError() == 183:
            if self.mutex: kernel.CloseHandle(ctypes.c_void_p(self.mutex))
            self.mutex = None
            return self.messagebox.showerror('중복 실행', '같은 가명 코드의 v5 창을 먼저 닫으세요.')
        self.journal = folder/'responses.jsonl'; self.events = folder/'events.jsonl'
        config = {'pilot_version': 5, 'mode': 'exploratory-wide-width-follow-up-not-confirmatory',
            'design_sha256': pilot.sha(PLAN_PATH), 'code_sha256': pilot.sha(__file__),
            'shared_module_sha256': pilot.sha(pilot.__file__),
            'seed': SEED, 'widths': WIDTHS, 'repeats': REPEATS, 'total': TOTAL,
            'presentation': f'randomized-within-balanced-{BLOCK_SIZE}-trial-blocks',
            'practice': 'one-blue-yellow-not-scored', 'line_length_px': pilot.LENGTH,
            'background_rgb': list(pilot.BG), 'nominal_srgb_not_colorimeter_verified': True}
        try:
            self.trials = schedule(); cp = folder/'config.json'
            if cp.exists():
                if json.loads(cp.read_text(encoding='utf-8')) != config:
                    raise ValueError('이전 v5 설정과 다릅니다. 기존 자료는 수정하지 않았습니다.')
            else:
                if self.journal.exists(): raise ValueError('설정 없는 응답이 있습니다. 덮어쓰지 않습니다.')
                with cp.open('x', encoding='utf-8') as f: json.dump(config, f, ensure_ascii=False, indent=2)
            rows = pilot.read_rows(self.journal); pilot.verify_rows(rows, self.trials); self.index = len(rows)
            self.row_config_sha = pilot.sha(cp)
            if any(r.get('config_sha256') != self.row_config_sha for r in rows):
                raise ValueError('응답에 결합된 설정 해시가 다릅니다. 원자료를 수정하지 않습니다.')
            pilot.append(self.events, {'event': 'open_or_resume', 'utc': pilot.now(), 'next_trial': self.index+1,
                'screen': [self.root.winfo_screenwidth(), self.root.winfo_screenheight()],
                'tk_scaling': float(self.root.tk.call('tk', 'scaling')), 'config_sha256': self.row_config_sha})
        except Exception as e:
            kernel.CloseHandle(ctypes.c_void_p(self.mutex)); self.mutex = None
            return self.messagebox.showerror('저장 자료 확인', str(e))
        self.running = True; self.root.attributes('-fullscreen', True)
        self.pause('준비됐으면 시작하세요. 연습 1문항 후 본문을 시작합니다.')

    def pause(self, message='일시정지 — 완료 응답은 저장됐습니다.'):
        self.clear(); self.phase = 'pause'; self.label(message)
        self.label(f'저장 완료 {self.index}/{TOTAL}문항\n필요하면 쉬거나 닫아도 됩니다. 같은 가명 코드로 이어 할 수 있습니다.')
        if self.index < len(self.trials): self.button('계속 / 시작 (Space)', self.next)
        else: self.label(f'{TOTAL}문항 완료. 감사합니다. raw/{DATA_DIR} 폴더에 저장됐습니다.')
        self.button('저장 유지하고 닫기', self.close)

    def next(self):
        if self.index >= len(self.trials): return self.pause('v5 완료')
        if not self.practice_done: return self.practice()
        self.clear(); self.phase = 'fixation'; current = self.token
        self.top.config(text=f'넓은 굵기 v5 · {self.index+1}/{TOTAL} · 블록 {self.index//BLOCK_SIZE+1}/{REPEATS}')
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
                pilot.append(self.events, {'event': 'interrupted_display', 'utc': pilot.now(),
                                           'phase': self.phase, 'trial_id': self.index+1})
            self.pause(); return
        if event.keysym == 'space' and not self.active:
            if self.phase == 'pause': self.next()
            return
        if not self.active or not event.char or event.char not in '1234': return
        response = int(event.char); self.active = False
        if self.phase == 'practice':
            self.practice_done = True
            pilot.append(self.events, {'event': 'practice_answer', 'utc': pilot.now(), 'response_position': response,
                'correct': response == pilot.practice_trial()['correct_position'], 'excluded_from_measurement': True})
            feedback = '맞았습니다.' if response == 3 else '연습 정답은 3번(좌하)이었습니다.'
            self.pause(feedback+' 본문은 색차와 굵기가 섞여 나옵니다. 같아 보여도 하나를 선택하세요.'); return
        row = {'utc': pilot.now(), 'trial': self.trials[self.index], 'response_position': response,
            'correct': response == self.trials[self.index]['correct_position'],
            'response_time_ms': (time.perf_counter()-self.started)*1000, 'config_sha256': self.row_config_sha}
        try: pilot.append(self.journal, row)
        except Exception as e:
            self.clear(); self.running = False; self.phase = 'save_error'
            self.label('저장 실패로 중단했습니다. 응답 파일을 임의 수정하지 마세요.')
            self.messagebox.showerror('저장 실패', str(e)); return
        self.index += 1; self.clear(); self.phase = 'iti'
        if self.index % BLOCK_SIZE == 0: self.pause(f'{BLOCK_SIZE}문항 완료 — 저장됐습니다. 필요하면 쉬세요.')
        else:
            current = self.token; self.root.after(300, lambda: self.next() if current == self.token else None)


def check_schedule(trials):
    assert len(trials) == TOTAL and trials == schedule()
    cells = Counter((t['condition_id'], t['width_px'], t['level']) for t in trials)
    assert len(cells) == COLORS*len(WIDTHS)*LEVELS and set(cells.values()) == {REPEATS}
    assert Counter(t['correct_position'] for t in trials) == {p: TOTAL//4 for p in range(1, 5)}
    for b in range(1, REPEATS+1):
        block = [t for t in trials if t['block'] == b]
        assert len(block) == BLOCK_SIZE
        assert Counter(t['level'] for t in block) == {lv: BLOCK_SIZE//LEVELS for lv in range(1, LEVELS+1)}
        assert Counter(t['width_px'] for t in block) == {w: COLORS*LEVELS for w in WIDTHS}
    p = plan(); v3_pairs = {c['condition_id']: c['pairs'] for c in p['conditions']}
    for cid, width, level in cells:
        sub = [t for t in trials if (t['condition_id'], t['width_px'], t['level']) == (cid, width, level)]
        assert set((t['correct_position'], t['odd_is_b']) for t in sub) == {(q, o) for q in range(1, 5) for o in [True, False]}
        pair = v3_pairs[cid][level-1]
        assert all(t['rgb_a'] == pair['rgb_a'] and t['rgb_b'] == pair['rgb_b'] for t in sub)
    for t in trials:
        pixels = {color: n for n, color in pilot.scene(t).getcolors()}
        odd = tuple(t['rgb_b'] if t['odd_is_b'] else t['rgb_a'])
        common = tuple(t['rgb_a'] if t['odd_is_b'] else t['rgb_b'])
        assert odd != common
        assert pixels[odd] == pilot.LENGTH*t['width_px']
        assert pixels[common] == 3*pilot.LENGTH*t['width_px']


def self_test():
    """Synthetic v3 plan in a temporary root; never touches raw/ or human data."""
    global ROOT, V3_PLAN_PATH, PLAN_PATH
    saved = (ROOT, V3_PLAN_PATH, PLAN_PATH)
    with tempfile.TemporaryDirectory(prefix='_synthetic-', dir=pilot.ROOT) as tmp:
        ROOT = Path(tmp); V3_PLAN_PATH = ROOT/'raw/_design_v3/design.json'; PLAN_PATH = ROOT/'raw/_design_v5/design.json'
        try:
            conditions = []
            for c in pilot.load_design()['conditions'][:2]:
                pairs = []
                for lv, frac in enumerate(pilot.FRACTIONS[2:5], 3):
                    t = frac*c['t_max']
                    a = pilot.encode_rgb([b-t*d/2 for b, d in zip(c['base_linear'], c['direction'])])
                    b2 = pilot.encode_rgb([b+t*d/2 for b, d in zip(c['base_linear'], c['direction'])])
                    pairs.append({'source_level': lv, 'fraction': frac, 'separation_t': t, 'rgb_a': a, 'rgb_b': b2})
                conditions.append({'condition_id': c['cid'], 'base_hex': c['base_hex'], 'pairs': pairs})
            V3_PLAN_PATH.parent.mkdir(parents=True)
            V3_PLAN_PATH.write_text(json.dumps({'pilot_version': 3, 'conditions': conditions}), encoding='utf-8')
            prepare(); prepare()  # 같은 계획 재준비는 허용, 덮어쓰기 없음
            trials = schedule(); check_schedule(trials)
            assert plan()['conditions'] == conditions
            V3_PLAN_PATH.write_text(json.dumps({'pilot_version': 3, 'conditions': conditions[::-1]}), encoding='utf-8')
            try: prepare(); raise AssertionError('바뀐 v3 계획으로 v5를 덮어쓰면 안 됨')
            except ValueError: pass
            file = ROOT/'synthetic.jsonl'
            for t in trials[:49]: pilot.append(file, {'trial': t, 'response_position': t['correct_position'], 'correct': True})
            pilot.verify_rows(pilot.read_rows(file), trials); assert len(pilot.read_rows(file)) == 49
        finally:
            ROOT, V3_PLAN_PATH, PLAN_PATH = saved
    print(f'PASS: {TOTAL} trials; v3 pairs reused unchanged; {len(WIDTHS)} widths; 8 repeats/cell; '
          'position/polarity balance; exact rasters 1-16px; no overwrite; append+resume')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.self_test: self_test()
    else: App().run()
