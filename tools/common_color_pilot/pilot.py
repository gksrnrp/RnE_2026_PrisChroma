"""Common-color line screening pilot. Not the frozen confirmatory experiment.

The stimulus definitions come from the original public conditions_v1.json, not
participant responses. No Machado severity/lapse or personal threshold is used.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'source_conditions_v1.json'
WIDTHS = [2, 4, 8, 16]
FRACTIONS = [1/32, 1/16, 1/8, 1/4, 1/2, 1]
LENGTH = 240
BG = (117, 117, 117)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def encode_rgb(linear):
    linear = np.asarray(linear, dtype=float)
    if np.any(linear < -1e-9) or np.any(linear > 1+1e-9):
        raise ValueError('선형 RGB 색역 밖 자극')
    linear = np.clip(linear, 0, 1)
    v = np.where(linear <= .0031308, linear*12.92,
                 1.055*np.power(linear, 1/2.4)-.055)
    return np.rint(v*255).astype(int).tolist()

def prepare():
    """Generate a common design asset only; never read participant responses."""
    doc = json.loads(SOURCE.read_text(encoding='utf-8-sig'))
    conditions = []
    for c in doc['conditions']:
        conditions.append({k:c[k] for k in ('cid','base_linear','base_hex','direction','t_max')})
    target = ROOT/'common_conditions.json'
    if target.exists():
        raise FileExistsError('이미 준비된 공통 자극 파일은 덮어쓰지 않습니다.')
    target.write_text(json.dumps({'source_sha256':sha(SOURCE),
        'source_name':'colorvision/conditions_v1.json','conditions':conditions},
        ensure_ascii=False, indent=2), encoding='utf-8')

def load_design():
    return json.loads((ROOT/'common_conditions.json').read_text(encoding='utf-8'))

def schedule(seed=20261108):
    """Six blocks, each containing every color x width once.

    Version 2 starts with each path's largest separation, then halves it per
    block. Order is randomized inside blocks, not across levels. This is a
    screening sequence, not an unbiased threshold experiment (time confound).
    """
    rng = random.Random(seed)
    conditions = load_design()['conditions']
    trials=[]
    for block in range(6):
        items=[]
        for ci,c in enumerate(conditions):
            for wi,width in enumerate(WIDTHS):
                level=5-block
                t=float(c['t_max'])*FRACTIONS[level]
                base=np.asarray(c['base_linear']); direction=np.asarray(c['direction'])
                a=encode_rgb(base-t*direction/2); b=encode_rgb(base+t*direction/2)
                items.append({'condition_id':c['cid'],'width_px':width,
                    'length_px':LENGTH,'level':level+1,'separation_t':t,
                    'rgb_a':a,'rgb_b':b,'identical_after_quantization':a==b,
                    'block':block+1})
        rng.shuffle(items)
        positions=list(range(1,5))*12; rng.shuffle(positions)
        # Six A-odd and six B-odd at each answer position within every block.
        odd_by_pos={p:[False]*6+[True]*6 for p in range(1,5)}
        for values in odd_by_pos.values(): rng.shuffle(values)
        for item,pos in zip(items,positions):
            item.update(trial_id=len(trials)+1,correct_position=pos,
                        odd_is_b=odd_by_pos[pos].pop())
            trials.append(item)
    return trials

def scene(trial, canvas=(960,640)):
    im=Image.new('RGB',canvas,BG); draw=ImageDraw.Draw(im)
    cx,cy=canvas[0]//2,canvas[1]//2
    for pos,(dx,dy) in enumerate([(-160,-130),(160,-130),(-160,130),(160,130)],1):
        is_b=trial['odd_is_b'] if pos==trial['correct_position'] else not trial['odd_is_b']
        color=trial['rgb_b'] if is_b else trial['rgb_a']
        x=cx+dx-LENGTH//2; y=cy+dy-trial['width_px']//2
        draw.rectangle((x,y,x+LENGTH-1,y+trial['width_px']-1),fill=tuple(color))
    return im

def practice_trial():
    # Deliberately easy blue/yellow instruction example, outside measured paths.
    return {'width_px':16,'rgb_a':[45,100,220],'rgb_b':[245,215,45],
            'correct_position':3,'odd_is_b':True}

def append(path, row):
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())

def read_rows(path):
    if not path.exists(): return []
    # Fail explicitly on a torn record instead of silently deleting raw data.
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def verify_rows(rows,trials):
    if len(rows)>len(trials): raise ValueError('응답 수가 시행표보다 많습니다.')
    for i,r in enumerate(rows):
        if r['trial']!=trials[i] or r['response_position'] not in [1,2,3,4]:
            raise ValueError('저장된 응답과 시행표가 일치하지 않습니다.')
        if r['correct'] != (r['response_position']==trials[i]['correct_position']):
            raise ValueError('채점 기록 불일치')

def preview():
    trials=schedule(); conditions=load_design()['conditions']
    im=Image.new('RGB',(1240,120+len(conditions)*100),BG); d=ImageDraw.Draw(im)
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    d.text((20,12),'공통 12색 조건 / 색쌍 A·B / 2·4·8·16px 미리보기 (연구 응답 아님)',font=font,fill='white')
    d.text((20,45),'예시 색차: 각 경로 최대 간격의 1/2. 실험에서는 6가지 간격 사용.',font=font,fill='white')
    for ci,c in enumerate(conditions):
        y=110+ci*100
        d.text((15,y),c['cid'],font=font,fill='white')
        for wi,width in enumerate(WIDTHS):
            t=next(t for t in trials if t['condition_id']==c['cid'] and t['width_px']==width and t['level']==5)
            x=210+wi*255
            d.text((x,y-16),str(width)+'px',font=font,fill='white')
            for off,color in [(15,t['rgb_a']),(48,t['rgb_b'])]:
                d.rectangle((x,y+off,x+LENGTH-1,y+off+width-1),fill=tuple(color))
    im.save(ROOT/'preview.png')

def self_test():
    from collections import Counter
    import tempfile
    trials=schedule()
    assert len(trials)==288 and schedule()==trials
    assert len({t['condition_id'] for t in trials})==12
    assert all(v==6 for v in Counter((t['condition_id'],t['width_px']) for t in trials).values())
    assert all(v==1 for v in Counter((t['condition_id'],t['width_px'],t['level']) for t in trials).values())
    assert all(t['level']==6 for t in trials[:48])
    for cid in {t['condition_id'] for t in trials}:
        for width in WIDTHS:
            assert [t['level'] for t in trials if t['condition_id']==cid and t['width_px']==width]==[6,5,4,3,2,1]
    for b in range(1,7):
        block=[t for t in trials if t['block']==b]
        assert all(v==1 for v in Counter((t['condition_id'],t['width_px']) for t in block).values())
        assert Counter(t['correct_position'] for t in block)=={1:12,2:12,3:12,4:12}
    for t in trials:
        assert all(0<=v<=255 for c in ['rgb_a','rgb_b'] for v in t[c])
        im=scene(t)
        # Pixel exact rectangle dimensions and solid interiors, no edge blending.
        x=480-160-LENGTH//2; y=320-130-t['width_px']//2
        expected=t['rgb_b'] if (t['odd_is_b'] if t['correct_position']==1 else not t['odd_is_b']) else t['rgb_a']
        assert im.getpixel((x,y))==tuple(expected)
        assert im.getpixel((x+LENGTH-1,y+t['width_px']-1))==tuple(expected)
        assert im.getpixel((x,y-1))==BG
        pixels=dict((color,n) for n,color in im.getcolors())
        assert pixels[tuple(t['rgb_b'] if t['odd_is_b'] else t['rgb_a'])]==LENGTH*t['width_px']
        assert pixels[tuple(t['rgb_a'] if t['odd_is_b'] else t['rgb_b'])]==3*LENGTH*t['width_px']
    p=practice_trial(); pixels={color:n for n,color in scene(p).getcolors()}
    assert pixels[tuple(p['rgb_b'])]==LENGTH*16
    assert pixels[tuple(p['rgb_a'])]==3*LENGTH*16
    with tempfile.TemporaryDirectory(prefix='_selftest-',dir=ROOT) as temp:
        p=Path(temp)/'responses.jsonl'
        for i in range(32): append(p,{'trial':trials[i],'response_position':trials[i]['correct_position'],'correct':True})
        rows=read_rows(p); verify_rows(rows,trials); assert len(rows)==32
        append(p,{'trial':trials[32],'response_position':trials[32]['correct_position'],'correct':True})
        verify_rows(read_rows(p),trials); assert len(read_rows(p))==33
    print('PASS: schedule, gamut, scoring, 288 rasters, append+resume; no human responses generated')
    print('identical quantized pairs:',sum(t['identical_after_quantization'] for t in trials))

class App:
    def __init__(self):
        import tkinter as tk
        from tkinter import messagebox
        self.tk=tk; self.messagebox=messagebox
        try: ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError,OSError): pass
        self.root=tk.Tk(); self.root.title('색과 선 — 예비실험 v2 (연습 후 큰 색차부터)')
        self.root.geometry('1000x760'); self.root.configure(bg='#757575')
        self.root.minsize(960,700)
        self.active=False; self.token=0; self.running=False; self.index=0; self.held=set(); self.mutex=None
        self.phase='home'; self.practice_done=False
        self.top=tk.Label(self.root,text='공통색 예비실험 · 기존 본실험과 별도',bg='#757575',fg='white',font=('맑은 고딕',16))
        self.top.pack(pady=12)
        self.body=tk.Frame(self.root,bg='#757575'); self.body.pack(fill='both',expand=True)
        self.root.bind('<KeyPress>',self.key)
        self.root.bind('<KeyRelease>',lambda e:self.held.discard(e.keysym))
        self.root.bind('<FocusOut>',lambda e:self.root.after(100,self.check_focus))
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.home()

    def clear(self):
        self.token+=1; self.active=False
        for w in self.body.winfo_children(): w.destroy()

    def label(self,text):
        self.tk.Label(self.body,text=text,bg='#757575',fg='white',font=('맑은 고딕',14),justify='left',wraplength=900).pack(pady=10)

    def button(self,text,command):
        self.tk.Button(self.body,text=text,command=command,font=('맑은 고딕',14),padx=20,pady=8).pack(pady=8)

    def home(self):
        self.clear()
        self.label('v2: 확실히 다른 색의 연습 1문항 → 큰 색차부터 시작\n공통 12색 조건 × 굵기 2·4·8·16px\n본문 288문항, 48문항씩 6블록 · 중단/재개 가능\n세 선은 같고 하나만 다릅니다. 같아 보여도 가장 다르게 보이는 하나를 고르세요.\n새 예비실험이며 C80 확정 검사가 아닙니다. 기존 v1 응답과 분리 저장합니다.\n이미 동의한 참가자만 진행하세요. 이름 대신 가명 코드를 입력하세요.')
        self.subject=self.tk.Entry(self.body,font=('맑은 고딕',16)); self.subject.pack(pady=12)
        self.subject.focus_set()
        self.button('새 예비실험 / 저장된 실험 이어하기',self.start)
        self.label('대략적인 시간은 반응 속도에 따라 달라집니다.\n1~4키: 좌상·우상·좌하·우하 / Esc: 저장 유지하고 일시정지\n문항별 정답 피드백은 없습니다. 화면·배율·거리·조명은 같은 상태로 유지하세요.')

    def start(self):
        subject=self.subject.get().strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}',subject):
            return self.messagebox.showerror('가명 코드','영문·숫자·밑줄·하이픈 1~32자로 입력하세요.')
        folder=ROOT/'raw'/'v2'/subject
        folder.mkdir(parents=True,exist_ok=True)
        mutex_name='Local\\RNEColorPilot'+hashlib.sha256(str(folder).encode()).hexdigest()
        kernel=ctypes.windll.kernel32
        kernel.CreateMutexW.restype=ctypes.c_void_p
        self.mutex=kernel.CreateMutexW(None,False,mutex_name)
        if not self.mutex or kernel.GetLastError()==183:
            if self.mutex: kernel.CloseHandle(ctypes.c_void_p(self.mutex))
            self.mutex=None
            return self.messagebox.showerror('중복 실행','같은 가명 코드로 열린 실험 창을 먼저 닫으세요.')
        self.journal=folder/'responses.jsonl'; self.events=folder/'events.jsonl'
        config={'pilot_version':2,'mode':'exploratory-common-colors-not-confirmatory',
            'level_order':'largest-to-smallest-by-block','practice':'one-blue-yellow-not-scored',
            'design_sha256':sha(ROOT/'common_conditions.json'),'code_sha256':sha(__file__),
            'seed':20261108,'widths':WIDTHS,'fractions':FRACTIONS,'line_length':LENGTH,
            'background':list(BG),'nominal_srgb_not_colorimeter_verified':True}
        cp=folder/'config.json'
        try:
            if cp.exists():
                if json.loads(cp.read_text(encoding='utf-8'))!=config:
                    raise ValueError('기존 저장 설정과 현재 코드가 다릅니다. 새 가명 코드를 사용하세요.')
            else:
                if self.journal.exists(): raise ValueError('설정 없는 응답 파일이 있습니다. 덮어쓰지 않습니다.')
                with cp.open('x',encoding='utf-8') as f: json.dump(config,f,ensure_ascii=False,indent=2)
            self.trials=schedule(); rows=read_rows(self.journal); verify_rows(rows,self.trials)
            self.index=len(rows)
            append(self.events,{'event':'open_or_resume','utc':now(),'next_trial':self.index+1,
                'screen':[self.root.winfo_screenwidth(),self.root.winfo_screenheight()]})
        except Exception as e:
            kernel.CloseHandle(ctypes.c_void_p(self.mutex)); self.mutex=None
            return self.messagebox.showerror('저장 자료 확인',str(e))
        self.running=True; self.root.attributes('-fullscreen',True)
        self.pause('준비됐으면 시작하세요. 색이 다른 선 하나를 1~4키로 고르세요.')

    def pause(self,message='일시정지 — 답한 문항은 모두 저장되어 있습니다.'):
        self.clear(); self.phase='pause'; self.label(message)
        self.label(f'저장 완료: {self.index}/288문항\n다음 문항부터 이어갑니다. 자세를 편하게 하고 필요하면 쉬세요.')
        if self.index<len(self.trials): self.button('계속 / 시작 (Space)',self.next)
        else: self.label('예비실험을 마쳤습니다. 감사합니다. 응답은 raw 폴더에 저장됐습니다.')
        self.button('저장 유지하고 닫기',self.close)

    def next(self):
        if self.index>=len(self.trials): return self.pause('예비실험 완료')
        if not self.practice_done: return self.practice()
        self.clear(); self.phase='fixation'; current=self.token
        self.top.config(text=f'공통색 예비실험 · {self.index+1}/288 · 블록 {self.index//48+1}/6')
        self.panel=self.tk.Label(self.body,bg='#757575'); self.panel.pack(expand=True)
        fixation=Image.new('RGB',(960,640),BG); ImageDraw.Draw(fixation).text((477,312),'+',fill='white')
        self.photo=ImageTk.PhotoImage(fixation); self.panel.config(image=self.photo)
        def show():
            if current!=self.token:return
            self.photo=ImageTk.PhotoImage(scene(self.trials[self.index]))
            self.panel.config(image=self.photo)
            self.root.update_idletasks(); self.started=time.perf_counter(); self.phase='trial'; self.active=True
        self.root.after(500,show)

    def practice(self):
        self.clear(); self.phase='practice'
        self.top.config(text='연습 1문항 · 다른 색의 선을 1~4키로 선택 · 본문 점수에 포함 안 됨')
        self.panel=self.tk.Label(self.body,bg='#757575'); self.panel.pack(expand=True)
        self.photo=ImageTk.PhotoImage(scene(practice_trial())); self.panel.config(image=self.photo)
        self.root.update_idletasks(); self.active=True

    def key(self,event):
        if not self.running:return
        if event.keysym in self.held:return
        self.held.add(event.keysym)
        if event.keysym=='Escape':
            if self.active: append(self.events,{'event':'interrupted_display','utc':now(),'trial_id':self.index+1})
            self.pause(); return
        if event.keysym=='space' and not self.active:
            if self.phase=='pause': self.next()
            return
        if not self.active or event.char not in '1234' or not event.char:return
        if self.phase=='practice':
            self.active=False; self.practice_done=True
            response=int(event.char)
            append(self.events,{'event':'practice_answer','utc':now(),
                'response_position':response,'correct':response==practice_trial()['correct_position'],
                'excluded_from_measurement':True})
            feedback='맞았습니다.' if response==practice_trial()['correct_position'] else '연습 정답은 3번(왼쪽 아래)이었습니다.'
            self.pause(feedback+' 이제 본문은 각 색 경로의 큰 색차부터 시작합니다.\n나중에는 네 선이 같아 보일 수 있습니다. 그때도 가장 다르게 보이는 하나를 고르세요.')
            return
        self.active=False; response=int(event.char); trial=self.trials[self.index]
        row={'utc':now(),'trial':trial,'response_position':response,
            'correct':response==trial['correct_position'],
            'response_time_ms':(time.perf_counter()-self.started)*1000}
        try: append(self.journal,row)
        except Exception as e:
            self.clear(); self.running=False
            self.label('저장 실패로 중단했습니다. 응답 파일을 임의 수정하지 마세요.')
            self.messagebox.showerror('저장 실패',str(e)); return
        self.index+=1; self.clear(); self.phase='iti'
        if self.index%48==0 or self.index==len(self.trials): self.pause('블록 완료 — 응답 저장 완료')
        else:
            current=self.token
            self.root.after(300,lambda:self.next() if current==self.token else None)

    def check_focus(self):
        if self.running and self.root.focus_displayof() is None:
            self.held.clear()
            if self.active: append(self.events,{'event':'focus_interruption','utc':now(),'trial_id':self.index+1})
            self.pause('창을 벗어나 일시정지했습니다. 같은 문항을 다시 표시합니다.')

    def close(self):
        if self.running:
            append(self.events,{'event':'close','utc':now(),'saved_responses':self.index,
                'current_display_interrupted':self.active})
        if self.mutex: ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self.mutex))
        self.root.destroy()

    def run(self): self.root.mainloop()

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--self-test',action='store_true'); parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare()
    elif args.self_test: self_test()
    elif args.preview: preview()
    else: App().run()
