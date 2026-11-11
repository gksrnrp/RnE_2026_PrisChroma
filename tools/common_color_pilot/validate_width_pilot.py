"""Fresh 6px follow-up for the exploratory v3 models, not frozen-study validation.

Locks all model predictions before collecting new responses. Separate journal;
never re-fits or overwrites the training model using these responses.
"""
import argparse
from collections import Counter
import ctypes
import json
import math
from pathlib import Path
import random
import re
import tempfile
import time
import numpy as np
from PIL import Image,ImageDraw,ImageTk
import pilot
import repeat_pilot as repeat
import fit_width_prediction as prediction

ROOT=pilot.ROOT
DESIGN=ROOT/'raw/_design_v4'
PLAN=DESIGN/'validation_plan.json'
SNAPSHOT=DESIGN/'model_snapshot.json'
TOTAL=120
BLOCK=30
SEED=20261110

def load_plan():
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    if pilot.sha(SNAPSHOT)!=plan['model_snapshot_sha256']:raise ValueError('고정 모델 스냅샷 해시가 다릅니다.')
    return plan

def prepare():
    if PLAN.exists():
        load_plan();print('Existing locked 6px plan reused; no overwrite');return
    if SNAPSHOT.exists():raise ValueError('미완료 준비 파일이 있습니다. 자동 덮어쓰기하지 않습니다.')
    model=json.loads(prediction.OUT.read_text(encoding='utf-8'))
    if pilot.sha(ROOT/model['source_relative'])!=model['source_sha256']:raise ValueError('학습 원자료 해시가 바뀌었습니다.')
    common={c['cid']:c for c in pilot.load_design()['conditions']}
    selected=[]
    for ci,color in enumerate(model['colors']):
        c=common[color]; roots=[]
        for m in model['models']:
            f=m['full_fit'];q=(.8-.25)/(.75-f['lapse'])
            root=math.exp((math.log(q/(1-q))-f['alpha'][ci]-f['gamma']*float(prediction.width_feature(6,f['model'])))/f['slope'])
            roots.append(root)
        lower,upper=min(roots),max(roots)
        targets=[.60*lower,.85*lower,(lower+upper)/2,upper,1.25*upper]
        # Enumerate a common linear-RGB path, quantize first, then compute nominal dE.
        ts=np.linspace(0,float(c['t_max']),16385)
        base=np.asarray(c['base_linear']);direction=np.asarray(c['direction'])
        aa=np.asarray(pilot.encode_rgb(base[None,:]-ts[:,None]*direction/2))
        bb=np.asarray(pilot.encode_rgb(base[None,:]+ts[:,None]*direction/2))
        des=np.asarray(prediction.delta_e00_linear(prediction.srgb_to_linear(aa/255),prediction.srgb_to_linear(bb/255)))
        if max(targets)>float(des.max()):raise ValueError('예측 주변 색차가 원래 경로 색역 범위를 넘습니다. 임의로 색을 바꾸지 않습니다.')
        pairs=[];used=set()
        for li,target in enumerate(targets,1):
            idx=int(np.argmin(np.abs(des-target))); a=aa[idx].tolist();b=bb[idx].tolist()
            if a==b or (tuple(a),tuple(b)) in used:raise ValueError('중복 또는 동일한 검증 색쌍')
            used.add((tuple(a),tuple(b)))
            if abs(float(des[idx])-target)>.10:raise ValueError('목표 색차에 충분히 가까운 양자화 후보가 없습니다.')
            pairs.append({'level':li,'target_nominal_delta_e00':target,'nominal_delta_e00':float(des[idx]),
                'rgb_a':a,'rgb_b':b,'separation_t':float(ts[idx]),
                'frozen_accuracy_predictions':{m['model']:prediction.predict(m['full_fit'],ci,6,float(des[idx])) for m in model['models']}})
        selected.append({'condition_id':color,'model_implied_6px_C80_for_design_only':dict(zip([m['model'] for m in model['models']],roots)),
            'pairs':pairs})
    DESIGN.mkdir(parents=True,exist_ok=True)
    with SNAPSHOT.open('xb') as f:f.write(prediction.OUT.read_bytes())
    plan={'version':4,'phase':'fresh-6px-exploratory-followup-not-confirmatory-study',
        'width_px':6,'colors':selected,'repeats':12,'total':TOTAL,'seed':SEED,'block_size':BLOCK,
        'model_snapshot_sha256':pilot.sha(SNAPSHOT),'training_source_sha256':model['source_sha256'],
        'prepared_utc':pilot.now(),'analysis_rule':'Compare all three locked models using fresh-response Brier/logloss, overall and per color; no refit or model tuning on followup responses.',
        'level_rule':'same levels for all models: .60*minC80, .85*minC80, midpoint(min,max), maxC80,1.25*maxC80; quantize along original path then nominal original sRGB deltaE00',
        'C80_claim_rule':'Do not treat model-implied or sparse observed 80% crossings as validated C80; independent precision/coverage not established.',
        'same_observer_and_spectacles_required':True,'original_frozen_protocol_replaced':False}
    with PLAN.open('x',encoding='utf-8') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
    print('Locked: 6px, 2 colors x 5 separations x 12 repeats = 120; all model predictions fixed')

def schedule():
    p=load_plan();rng=random.Random(p['seed']);cells=[]
    for color in p['colors']:
        for pair in color['pairs']:
            # Twelve repeats: 3 per position, 6 per odd color; all 8 joint cells occur.
            extras=[True,True,False,False];rng.shuffle(extras)
            combos=[(pos,odd) for pos in range(1,5) for odd in [False,True]]
            combos += [(pos,extras[pos-1]) for pos in range(1,5)]
            rng.shuffle(combos);cells.append((color['condition_id'],pair,combos))
    trials=[]
    for rep in range(12):
        items=[]
        for color,pair,combos in cells:
            pos,odd=combos[rep]
            items.append({'condition_id':color,'width_px':6,'length_px':pilot.LENGTH,
                'rgb_a':pair['rgb_a'],'rgb_b':pair['rgb_b'],'level':pair['level'],
                'nominal_delta_e00':pair['nominal_delta_e00'],'separation_t':pair['separation_t'],
                'repeat':rep+1,'correct_position':pos,'odd_is_b':odd})
        rng.shuffle(items)
        for t in items:
            t['trial_id']=len(trials)+1;t['block']=len(trials)//BLOCK+1;trials.append(t)
    return trials

class App(repeat.App):
    def __init__(self):
        super().__init__();self.root.title('색과 선 — 6px 예측 확인 · 120문항')

    def home(self):
        self.clear();self.phase='home'
        self.label('새 굵기 6px 예측 확인 · 120문항\n연습 1문항 → 본문 / 30문항마다 쉬거나 종료 가능\n방금 240문항을 한 같은 분이, 같은 안경·모니터·거리로 계속해 주세요.\n기존 가명 코드로 시작하면 이번 6px 자료만 별도로 저장합니다.\n세 선은 같고 하나만 다릅니다. 모두 같아 보여도 하나를 선택하세요.\n기존 동결 본실험이 아닌 별도 탐색 후속검사입니다.')
        self.subject=self.tk.Entry(self.body,font=('맑은 고딕',16));self.subject.pack(pady=10)
        model=json.loads(SNAPSHOT.read_text(encoding='utf-8'))
        self.subject.insert(0,Path(model['source_relative']).parent.name)
        self.button('같은 참가자·안경으로 6px 시작 / 이어하기',self.start)
        self.label('1=좌상 / 2=우상 / 3=좌하 / 4=우하\nEsc: 저장 유지·일시정지 / Space: 시작·재개\n이미 동의한 참가자만 진행하세요. 예측값은 새 응답을 보기 전에 고정했습니다.')

    def start(self):
        subject=self.subject.get().strip();model=json.loads(SNAPSHOT.read_text(encoding='utf-8'))
        expected=Path(model['source_relative']).parent.name
        if subject!=expected:return self.messagebox.showerror('회차 연결','방금 240문항에 사용한 같은 가명 코드로 진행하세요. 다른 참가자의 검증에 적용하지 않습니다.')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}',subject):return
        folder=ROOT/'raw/v4'/subject;folder.mkdir(parents=True,exist_ok=True)
        kernel=ctypes.windll.kernel32;kernel.CreateMutexW.restype=ctypes.c_void_p
        self.mutex=kernel.CreateMutexW(None,False,'Local\\RNE6pxPilot'+pilot.hashlib.sha256(str(folder.resolve()).casefold().encode()).hexdigest())
        if not self.mutex or kernel.GetLastError()==183:
            if self.mutex:kernel.CloseHandle(ctypes.c_void_p(self.mutex))
            self.mutex=None;return self.messagebox.showerror('중복 실행','같은 6px 검사 창을 먼저 닫으세요.')
        self.journal=folder/'responses.jsonl';self.events=folder/'events.jsonl'
        config={'version':4,'mode':'fresh-6px-exploratory-followup','plan_sha256':pilot.sha(PLAN),
            'model_snapshot_sha256':pilot.sha(SNAPSHOT),'code_sha256':pilot.sha(__file__),
            'repeat_module_sha256':pilot.sha(repeat.__file__),'pilot_module_sha256':pilot.sha(pilot.__file__),
            'total':TOTAL,'seed':SEED,'same_observer_spectacles':'start-button-reported; not independently verified',
            'physical_pixel_colorimetry_verified':False}
        try:
            self.trials=schedule();cp=folder/'config.json'
            if cp.exists():
                if json.loads(cp.read_text(encoding='utf-8'))!=config:raise ValueError('저장된 검사 설정과 다릅니다. 자동 변경하지 않습니다.')
            else:
                if self.journal.exists():raise ValueError('설정 없는 응답 파일은 덮어쓰지 않습니다.')
                with cp.open('x',encoding='utf-8') as f:json.dump(config,f,ensure_ascii=False,indent=2)
            rows=pilot.read_rows(self.journal);pilot.verify_rows(rows,self.trials)
            self.index=len(rows);self.row_config_sha=pilot.sha(cp)
            if any(r.get('config_sha256')!=self.row_config_sha for r in rows):raise ValueError('응답 설정 해시가 다릅니다.')
            pilot.append(self.events,{'event':'open_or_resume','utc':pilot.now(),'next_trial':self.index+1,
                'screen':[self.root.winfo_screenwidth(),self.root.winfo_screenheight()],
                'tk_scaling':float(self.root.tk.call('tk','scaling')),'config_sha256':self.row_config_sha})
        except Exception as e:
            kernel.CloseHandle(ctypes.c_void_p(self.mutex));self.mutex=None
            return self.messagebox.showerror('저장 확인',str(e))
        self.running=True;self.root.attributes('-fullscreen',True)
        self.pause('준비됐으면 시작하세요. 새 6px 응답으로 기존 예측을 확인합니다.')

    def pause(self,message='일시정지 — 응답 저장 완료'):
        message=message.replace('색차와 굵기가 섞여','색차가 섞여')
        self.clear();self.phase='pause';self.label(message)
        self.label(f'저장 완료 {self.index}/{TOTAL}문항\n필요하면 쉬거나 닫으세요. 같은 가명 코드로 다음 문항부터 이어집니다.')
        if self.index<len(self.trials):self.button('계속 / 시작 (Space)',self.next)
        else:self.label('120문항 완료. 기존 모델을 바꾸지 않고 예측을 평가할 준비가 됐습니다.')
        self.button('저장 유지하고 닫기',self.close)

    def next(self):
        if self.index>=len(self.trials):return self.pause('6px 검사 완료')
        if not self.practice_done:return self.practice()
        self.clear();self.phase='fixation';current=self.token
        self.top.config(text=f'새 굵기 6px · {self.index+1}/{TOTAL} · 블록 {self.index//BLOCK+1}/4')
        self.panel=self.tk.Label(self.body,bg='#757575');self.panel.pack(expand=True)
        im=Image.new('RGB',(960,640),pilot.BG);ImageDraw.Draw(im).text((477,312),'+',fill='white')
        self.photo=ImageTk.PhotoImage(im);self.panel.config(image=self.photo)
        def show():
            if current!=self.token:return
            self.photo=ImageTk.PhotoImage(pilot.scene(self.trials[self.index]));self.panel.config(image=self.photo)
            self.root.update_idletasks();self.started=time.perf_counter();self.phase='trial';self.active=True
        self.root.after(500,show)

def self_test():
    p=load_plan();trials=schedule();assert len(trials)==120 and trials==schedule()
    assert set(t['width_px'] for t in trials)=={6}
    assert set(Counter((t['condition_id'],t['level']) for t in trials).values())=={12}
    assert Counter(t['correct_position'] for t in trials)=={pos:30 for pos in range(1,5)}
    for cid,level in {(t['condition_id'],t['level']) for t in trials}:
        sub=[t for t in trials if (t['condition_id'],t['level'])==(cid,level)]
        assert len({(tuple(t['rgb_a']),tuple(t['rgb_b'])) for t in sub})==1
        assert Counter(t['correct_position'] for t in sub)=={pos:3 for pos in range(1,5)}
        assert Counter(t['odd_is_b'] for t in sub)=={True:6,False:6}
    for b in range(1,5):
        block=[t for t in trials if t['block']==b];assert len(block)==30
        assert set(Counter((t['condition_id'],t['level']) for t in block).values())=={3}
    for t in trials:
        pixels={color:n for n,color in pilot.scene(t).getcolors()}
        assert pixels[tuple(t['rgb_b'] if t['odd_is_b'] else t['rgb_a'])]==240*6
        assert pixels[tuple(t['rgb_a'] if t['odd_is_b'] else t['rgb_b'])]==3*240*6
        assert abs(prediction.nominal_delta(t)-t['nominal_delta_e00'])<1e-10
    print('PASS: frozen predictions, 120 unique trials, matched repeats, 6px raster, balanced positions/polarity, four 30-trial blocks')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.self_test:self_test()
    else:App().run()
