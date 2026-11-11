"""v3 repeated common-color screening pilot, separate from v1/v2 and frozen study.

Selection uses pooled color-separation responses only, never width differences.
Every selected pair is identical across widths and repeated eight times.
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

import numpy as np
from PIL import Image, ImageDraw, ImageTk
import pilot

ROOT=pilot.ROOT
PLAN_PATH=ROOT/'raw'/'_design_v3'/'design.json'
WIDTHS=[1,2,3,4,8]
REPEATS=8
BLOCK_SIZE=30
TOTAL=240
SEED=20261109

def prepare():
    summary=json.loads((ROOT/'raw/_analysis/complete-v2-analysis.json').read_text(encoding='utf-8'))
    source=ROOT/summary['source_relative']
    if pilot.sha(source)!=summary['source_sha256']:
        raise ValueError('선별 근거 원자료 해시가 바뀌었습니다.')
    rows=pilot.read_rows(source); pilot.verify_rows(rows,pilot.schedule())
    if len(rows)!=288: raise ValueError('선별용 288회 자료가 아닙니다.')
    colors=pilot.load_design()['conditions']
    ranked=[]
    for c in colors:
        by_level={l:[r for r in rows if r['trial']['condition_id']==c['cid'] and r['trial']['level']==l] for l in range(1,7)}
        rates={l:sum(r['correct'] for r in sub)/len(sub) for l,sub in by_level.items()}
        low=(rates[1]+rates[2])/2; high=(rates[5]+rates[6])/2
        if high>=.75 and rates[6]>=.75:
            ranked.append((high-low,c['cid'],c,rates))
    ranked.sort(key=lambda x:(-x[0],x[1]))
    if len(ranked)<2: raise ValueError('색차 난도 변화가 있는 후보가 두 개 미만입니다.')
    selected=[]
    for score,cid,c,rates in ranked[:2]:
        # Screening-only transition target, not a fitted threshold or C80.
        center=next((l for l in range(2,6) if rates[l]>=.65),None)
        if center is None: raise ValueError('내부 전이 수준을 찾지 못했습니다.')
        pairs=[]
        for l in [center-1,center,center+1]:
            fraction=pilot.FRACTIONS[l-1]; t=fraction*c['t_max']
            base=np.asarray(c['base_linear']); direction=np.asarray(c['direction'])
            a=pilot.encode_rgb(base-t*direction/2); b=pilot.encode_rgb(base+t*direction/2)
            if a==b: raise ValueError('양자화 후 동일한 색쌍')
            pairs.append({'source_level':l,'fraction':fraction,'separation_t':t,'rgb_a':a,'rgb_b':b})
        selected.append({'condition_id':cid,'base_hex':c['base_hex'],'selection_score':score,
            'selection_pooled_level_rates':rates,'pairs':pairs})
    plan={'pilot_version':3,'mode':'exploratory-repeated-pairs-not-confirmatory',
        'selection_source_sha256':summary['source_sha256'],
        'common_conditions_sha256':pilot.sha(ROOT/'common_conditions.json'),
        'selection_rule':'rank by pooled high(level5,6)-low(level1,2) accuracy; high>=.75, level6>=.75; no width contrast used; top2',
        'level_rule':'first internal pooled level >=.65 and its immediate neighbors; screening only, not C80',
        'selection_note':'Chosen using exploratory v2 responses; subsequent data are new but no preregistered population claim.',
        'conditions':selected,'widths':WIDTHS,'repeats':REPEATS,'block_size':BLOCK_SIZE,
        'seed':SEED,'total':TOTAL,'line_length_px':pilot.LENGTH,'background_rgb':list(pilot.BG),
        'calibrated_colorimetry':False,'physical_pixel_mapping_verified':False,
        'claim_allowed':'descriptive pilot only; no clinical diagnosis, equivalence, calibrated C80 or correction efficacy'}
    if PLAN_PATH.exists():
        existing=json.loads(PLAN_PATH.read_text(encoding='utf-8'))
        if existing!=json.loads(json.dumps(plan)): raise ValueError('준비된 v3 계획을 덮어쓰지 않습니다.')
    else:
        PLAN_PATH.parent.mkdir(parents=True,exist_ok=True)
        with PLAN_PATH.open('x',encoding='utf-8') as stream: json.dump(plan,stream,ensure_ascii=False,indent=2)
    print('Plan ready: 2 colors x 5 widths x 3 separations x 8 repeats = 240')
    print('Selected common color paths:',','.join(c['condition_id'] for c in selected))

def plan():
    return json.loads(PLAN_PATH.read_text(encoding='utf-8'))

def schedule():
    p=plan(); rng=random.Random(p['seed']); cells=[]
    for c in p['conditions']:
        for width in p['widths']:
            for level,pair in enumerate(c['pairs'],1):
                combos=[(pos,odd) for pos in range(1,5) for odd in [False,True]]
                rng.shuffle(combos)
                cells.append((c['condition_id'],width,level,pair,combos))
    trials=[]
    for rep in range(REPEATS):
        block=[]
        for cid,width,level,pair,combos in cells:
            pos,odd=combos[rep]
            block.append({'condition_id':cid,'width_px':width,'level':level,
                'rgb_a':pair['rgb_a'],'rgb_b':pair['rgb_b'],
                'separation_t':pair['separation_t'],'fraction':pair['fraction'],
                'source_level':pair['source_level'],'length_px':pilot.LENGTH,
                'correct_position':pos,'odd_is_b':odd,'block':rep+1,'repeat':rep+1})
        rng.shuffle(block)
        for trial in block:
            trial['trial_id']=len(trials)+1; trials.append(trial)
    return trials

class App(pilot.App):
    def __init__(self):
        super().__init__()
        self.root.title('색과 선 — 반복 비교 v3 · 240문항')

    def home(self):
        self.clear(); self.phase='home'
        self.label('v3 · 같은 색쌍 반복 비교\n2색 조건 × 1·2·3·4·8px × 3색차 × 8회 = 240문항\n연습 1문항 후 시작 / 30문항마다 휴식 / 순서 무작위\n세 선은 같고 하나만 다릅니다. 모두 같아 보여도 가장 다르게 보이는 하나를 고르세요.\n기존 288회와 분리된 예비실험입니다. 색차가 계속 줄어드는 순서가 아닙니다.\n이미 동의한 참가자만 진행하세요. 실명 대신 가명 코드를 입력하세요.')
        self.subject=self.tk.Entry(self.body,font=('맑은 고딕',16)); self.subject.pack(pady=10)
        self.subject.focus_set()
        self.button('v3 새로 시작 / v3 이어하기',self.start)
        self.label('1=좌상 / 2=우상 / 3=좌하 / 4=우하\nEsc: 저장 유지·일시정지 / Space: 시작·재개\n화면 배율·거리·조명은 유지하세요. 정답 피드백은 연습에서만 제공합니다.')

    def start(self):
        subject=self.subject.get().strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}',subject):
            return self.messagebox.showerror('가명 코드','영문·숫자·밑줄·하이픈 1~32자를 입력하세요.')
        folder=ROOT/'raw'/'v3'/subject; folder.mkdir(parents=True,exist_ok=True)
        kernel=ctypes.windll.kernel32; kernel.CreateMutexW.restype=ctypes.c_void_p
        self.mutex=kernel.CreateMutexW(None,False,'Local\\RNERepeatPilot'+pilot.hashlib.sha256(str(folder.resolve()).casefold().encode()).hexdigest())
        if not self.mutex or kernel.GetLastError()==183:
            if self.mutex: kernel.CloseHandle(ctypes.c_void_p(self.mutex))
            self.mutex=None
            return self.messagebox.showerror('중복 실행','같은 가명 코드의 v3 창을 먼저 닫으세요.')
        self.journal=folder/'responses.jsonl'; self.events=folder/'events.jsonl'
        config={'pilot_version':3,'mode':'exploratory-repeated-pairs-not-confirmatory',
            'design_sha256':pilot.sha(PLAN_PATH),'code_sha256':pilot.sha(__file__),
            'shared_module_sha256':pilot.sha(pilot.__file__),
            'seed':SEED,'widths':WIDTHS,'repeats':REPEATS,'total':TOTAL,
            'presentation':'randomized-within-balanced-30-trial-blocks',
            'practice':'one-blue-yellow-not-scored','line_length_px':pilot.LENGTH,
            'background_rgb':list(pilot.BG),'nominal_srgb_not_colorimeter_verified':True}
        try:
            self.trials=schedule(); cp=folder/'config.json'
            if cp.exists():
                if json.loads(cp.read_text(encoding='utf-8'))!=config: raise ValueError('이전 v3 설정과 다릅니다. 기존 자료는 수정하지 않았습니다.')
            else:
                if self.journal.exists(): raise ValueError('설정 없는 응답이 있습니다. 덮어쓰지 않습니다.')
                with cp.open('x',encoding='utf-8') as f: json.dump(config,f,ensure_ascii=False,indent=2)
            rows=pilot.read_rows(self.journal); pilot.verify_rows(rows,self.trials); self.index=len(rows)
            self.row_config_sha=pilot.sha(cp)
            if any(r.get('config_sha256')!=self.row_config_sha for r in rows):
                raise ValueError('응답에 결합된 설정 해시가 다릅니다. 원자료를 수정하지 않습니다.')
            append_event={'event':'open_or_resume','utc':pilot.now(),'next_trial':self.index+1,
                'screen':[self.root.winfo_screenwidth(),self.root.winfo_screenheight()],
                'tk_scaling':float(self.root.tk.call('tk','scaling')),'config_sha256':self.row_config_sha}
            pilot.append(self.events,append_event)
        except Exception as e:
            kernel.CloseHandle(ctypes.c_void_p(self.mutex)); self.mutex=None
            return self.messagebox.showerror('저장 자료 확인',str(e))
        self.running=True; self.root.attributes('-fullscreen',True)
        self.pause('준비됐으면 시작하세요. 연습 1문항 후 같은 색쌍 반복 비교를 시작합니다.')

    def pause(self,message='일시정지 — 완료 응답은 저장됐습니다.'):
        self.clear(); self.phase='pause'; self.label(message)
        self.label(f'저장 완료 {self.index}/{TOTAL}문항\n필요하면 쉬거나 닫아도 됩니다. 같은 가명 코드로 이어 할 수 있습니다.')
        if self.index<len(self.trials): self.button('계속 / 시작 (Space)',self.next)
        else: self.label('240문항 완료. 감사합니다. raw/v3 폴더에 저장됐습니다.')
        self.button('저장 유지하고 닫기',self.close)

    def next(self):
        if self.index>=len(self.trials): return self.pause('반복 비교 완료')
        if not self.practice_done:return self.practice()
        self.clear(); self.phase='fixation'; current=self.token
        self.top.config(text=f'반복 비교 v3 · {self.index+1}/{TOTAL} · 블록 {self.index//BLOCK_SIZE+1}/8')
        self.panel=self.tk.Label(self.body,bg='#757575'); self.panel.pack(expand=True)
        im=Image.new('RGB',(960,640),pilot.BG); ImageDraw.Draw(im).text((477,312),'+',fill='white')
        self.photo=ImageTk.PhotoImage(im); self.panel.config(image=self.photo)
        def show():
            if current!=self.token:return
            self.photo=ImageTk.PhotoImage(pilot.scene(self.trials[self.index])); self.panel.config(image=self.photo)
            self.root.update_idletasks(); self.started=time.perf_counter(); self.phase='trial'; self.active=True
        self.root.after(500,show)

    def key(self,event):
        if not self.running or event.keysym in self.held:return
        self.held.add(event.keysym)
        if event.keysym=='Escape':
            if self.active:pilot.append(self.events,{'event':'interrupted_display','utc':pilot.now(),'phase':self.phase,'trial_id':self.index+1})
            self.pause(); return
        if event.keysym=='space' and not self.active:
            if self.phase=='pause':self.next()
            return
        if not self.active or not event.char or event.char not in '1234':return
        response=int(event.char); self.active=False
        if self.phase=='practice':
            self.practice_done=True
            pilot.append(self.events,{'event':'practice_answer','utc':pilot.now(),'response_position':response,
                'correct':response==pilot.practice_trial()['correct_position'],'excluded_from_measurement':True})
            feedback='맞았습니다.' if response==3 else '연습 정답은 3번(좌하)이었습니다.'
            self.pause(feedback+' 본문은 색차와 굵기가 섞여 나옵니다. 같아 보여도 하나를 선택하세요.'); return
        row={'utc':pilot.now(),'trial':self.trials[self.index],'response_position':response,
            'correct':response==self.trials[self.index]['correct_position'],
            'response_time_ms':(time.perf_counter()-self.started)*1000,
            'config_sha256':self.row_config_sha}
        try:pilot.append(self.journal,row)
        except Exception as e:
            self.clear(); self.running=False; self.phase='save_error'
            self.label('저장 실패로 중단했습니다. 응답 파일을 임의 수정하지 마세요.')
            self.messagebox.showerror('저장 실패',str(e)); return
        self.index+=1; self.clear(); self.phase='iti'
        if self.index%BLOCK_SIZE==0:self.pause('30문항 완료 — 저장됐습니다. 필요하면 쉬세요.')
        else:
            current=self.token; self.root.after(300,lambda:self.next() if current==self.token else None)

def self_test():
    trials=schedule(); assert len(trials)==TOTAL and trials==schedule()
    cells=Counter((t['condition_id'],t['width_px'],t['level']) for t in trials)
    assert len(cells)==30 and set(cells.values())=={8}
    assert Counter(t['correct_position'] for t in trials)=={1:60,2:60,3:60,4:60}
    for b in range(1,9):
        block=[t for t in trials if t['block']==b]
        assert len(block)==30
        assert Counter(t['level'] for t in block)=={1:10,2:10,3:10}
        assert Counter(t['width_px'] for t in block)=={w:6 for w in WIDTHS}
    for cid,width,level in cells:
        sub=[t for t in trials if (t['condition_id'],t['width_px'],t['level'])==(cid,width,level)]
        assert set((t['correct_position'],t['odd_is_b']) for t in sub)=={(p,o) for p in range(1,5) for o in [True,False]}
        same=[t for t in trials if t['condition_id']==cid and t['level']==level]
        assert len({(tuple(t['rgb_a']),tuple(t['rgb_b'])) for t in same})==1
    for t in trials:
        pixels={color:n for n,color in pilot.scene(t).getcolors()}
        odd=tuple(t['rgb_b'] if t['odd_is_b'] else t['rgb_a'])
        common=tuple(t['rgb_a'] if t['odd_is_b'] else t['rgb_b'])
        assert odd!=common
        assert pixels[odd]==pilot.LENGTH*t['width_px']
        assert pixels[common]==3*pilot.LENGTH*t['width_px']
    with tempfile.TemporaryDirectory(prefix='_repeat-test-',dir=ROOT) as tmp:
        file=Path(tmp)/'synthetic.jsonl'
        for t in trials[:31]:pilot.append(file,{'trial':t,'response_position':t['correct_position'],'correct':True})
        pilot.verify_rows(pilot.read_rows(file),trials)
        assert len(pilot.read_rows(file))==31
    print('PASS: 240 trials; 8 repeats/cell; identical pairs across widths; position/polarity balance; exact 1/2/3/4/8px rasters; append+resume')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.self_test:self_test()
    else:App().run()
