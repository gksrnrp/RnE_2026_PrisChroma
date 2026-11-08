"""Descriptive v3 analysis; preserves all responses and does not force C80 fits."""
from collections import Counter
import argparse
import json
import math
from statistics import median
import pilot
import repeat_pilot as repeat

def summary(rows):
    n=len(rows); k=sum(r['correct'] for r in rows)
    z=1.959963984540054
    if n:
        den=1+z*z/n; center=(k/n+z*z/(2*n))/den
        half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
        interval=[center-half,center+half]
    else:interval=None
    return {'n':n,'correct':k,'accuracy':k/n if n else None,
            'wilson95_descriptive':interval,
            'median_rt_ms':median(r['response_time_ms'] for r in rows) if n else None}

def main(same_glasses_reported=False,fringing_reported=False):
    trials=repeat.schedule(); journals=[]
    for p in (pilot.ROOT/'raw/v3').glob('*/responses.jsonl'):
        r=pilot.read_rows(p); pilot.verify_rows(r,trials)
        if len(r)==240:journals.append((p,r))
    if len(journals)!=1:raise ValueError('완료 회차를 명시해야 합니다.')
    source,rows=journals[0]; before=pilot.sha(source)
    config_path=source.parent/'config.json'; config=json.loads(config_path.read_text(encoding='utf-8'))
    if not all(r['config_sha256']==pilot.sha(config_path) for r in rows):raise ValueError('설정 결합 불일치')
    if config['design_sha256']!=pilot.sha(repeat.PLAN_PATH):raise ValueError('계획 해시 불일치')
    events=pilot.read_rows(source.parent/'events.jsonl')
    shown_again=sorted({e['trial_id'] for e in events if e['event'] in ['focus_interruption','interrupted_display'] and e.get('trial_id')})
    fast=[r['trial']['trial_id'] for r in rows if r['response_time_ms']<200]
    cells=[]
    for color in repeat.plan()['conditions']:
        for l in [1,2,3]:
            pair=color['pairs'][l-1]
            cells.append({'condition_id':color['condition_id'],'base_hex':color['base_hex'],'level':l,
                'fraction':pair['fraction'],'rgb_a':pair['rgb_a'],'rgb_b':pair['rgb_b'],
                'by_width':{str(w):summary([r for r in rows if r['trial']['condition_id']==color['condition_id'] and r['trial']['level']==l and r['trial']['width_px']==w]) for w in repeat.WIDTHS}})
    omit=set(shown_again+fast); sensitive=[r for r in rows if r['trial']['trial_id'] not in omit]
    report={'analysis_utc':pilot.now(),'analysis_code_sha256':pilot.sha(__file__),
        'source_relative':source.relative_to(pilot.ROOT).as_posix(),'source_sha256':before,
        'config_sha256':pilot.sha(config_path),'design_sha256':pilot.sha(repeat.PLAN_PATH),
        'participant_identity_not_inferred':True,'other_sessions_not_pooled':True,
        'self_report':{'source':'explicit operator CLI annotation; not independently verified',
            'same_spectacles_throughout':True if same_glasses_reported else None,
            'observation':'Color-dependent apparent spreading/fringing reported.' if fringing_reported else None,
            'cause_confirmed':False,'retrospective_annotation_not_a_change_to_frozen_config':True},
        'overall':summary(rows),'by_width':{str(w):summary([r for r in rows if r['trial']['width_px']==w]) for w in repeat.WIDTHS},
        'cells':cells,'by_block':{str(b):summary([r for r in rows if r['trial']['block']==b]) for b in range(1,9)},
        'by_position':{str(p):summary([r for r in rows if r['trial']['correct_position']==p]) for p in range(1,5)},
        'reexposed_trial_ids':shown_again,'rt_under200ms_trial_ids':fast,
        'automatic_primary_exclusions':0,
        'sensitivity_only_omit_reexposure_and_fast':{'omitted_count':len(omit),'by_width':{str(w):summary([r for r in sensitive if r['trial']['width_px']==w]) for w in repeat.WIDTHS}},
        'C80_estimated':False,'confirmatory_claim_allowed':False,
        'limitations':['Selected two color paths using prior exploratory responses; not all colors.',
            'One participant/session; repeated responses are not independent participants.',
            'Eight answers per exact color pair x width; substantial uncertainty.',
            'Spectacles, ocular optics and screen characteristics are not isolated from color discrimination.',
            'No calibrated physical colorimetry or physical 1:1 display validation.',
            'Descriptive Wilson intervals are not a repeated-measures or population-effect analysis.',
            'No C80 estimated; bracketing and precision require separate assessment.']}
    assert pilot.sha(source)==before
    out=pilot.ROOT/'raw/_analysis'; out.mkdir(exist_ok=True)
    (out/'repeat-v3-analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 동일 색쌍 반복 예비실험 v3 — 240회', '',
        '## 저장 및 분석 범위', '',
        '한 완료 회차 240개 고유 시행을 시행표와 대조했다. 정답 위치·채점·설정/계획 해시 연결이 일치했다. '
        '기존 288회 및 이전 부분 회차와 합산하지 않았다. 참가자 동일성/색각 유형을 파일 이름으로 추정하지 않는다.', '',
        f"전체 {report['overall']['correct']}/240 정답 ({report['overall']['accuracy']:.1%}). "
        '전체 정답률은 본래 난도가 섞인 수치이며 개인의 능력 점수나 C80이 아니다.', '',
        '## 굵기별 (동일한 여섯 색쌍이 각 8회씩)', '',
        '| 굵기 | 정답/48 | 정답률 |','|---|---:|---:|']
    for w,s in report['by_width'].items():lines.append(f"| {w}px | {s['correct']} | {s['accuracy']:.1%} |")
    lines+=['', '## 정확히 같은 두 RGB에서 비교', '',
        '각 칸은 8회 중 정답 수다. 색 경로·색차가 같은 행 안에서 굵기만 비교한다. '
        '서로 다른 색 경로의 색차 수준 번호는 같은 절대 ΔE를 뜻하지 않는다.', '',
        '| 조건 | 색차 단계 | 1px | 2px | 3px | 4px | 8px |',
        '|---|---|---:|---:|---:|---:|---:|']
    for cell in cells:
        values=' | '.join(str(cell['by_width'][str(w)]['correct'])+'/8' for w in repeat.WIDTHS)
        lines.append(f"| {cell['condition_id']} | {cell['level']} | {values} |")
    lines+=['', '## 해석', '',
        '굵기별 기술통계는 위 표와 같다. 이 결과만으로 특정 굵기의 경계·일반적인 색각이상 특성·역치 동등성을 증명하지 않는다. '
        '8/8 정답도 실제 구별확률 100%를 확정하지 않는다. 기술적 비율 구간은 모집단 효과 검정이 아니다. '
        '3개 색차×8회만으로 정밀 C80이 확보됐다고 주장하거나 범위 밖으로 역치를 외삽하지 않는다.', '',
        '## 안경·재제시·반응시간', '',
        f"같은 안경 유지 진술 기록: {'있음' if same_glasses_reported else '미입력'}. "
        f"색별 번짐 진술 기록: {'있음' if fringing_reported else '미입력'}. "
        '이는 명시적으로 입력된 사후 주석이며 원설정 변경이 아니다. 번짐의 원인을 안경으로 확정하거나 '
        '시각적 구별 결과를 적록색각만의 효과로 분리하지 않는다.', '',
        f"재제시 관련 시행: {shown_again}. 200ms 미만 반응: {fast}. 원자료에서 자동 제외하지 않았다.", '',
        '사후 민감도 점검으로 두 종류 시행을 뺀 집계도 별도 제공한다. 주 결과를 유리하게 바꾸기 위한 제외가 아니며 원자료는 그대로다.', '',
        '| 굵기 | 민감도 정답/시행 | 정답률 |','|---|---:|---:|']
    for w,s in report['sensitivity_only_omit_reexposure_and_fast']['by_width'].items():
        lines.append(f"| {w}px | {s['correct']}/{s['n']} | {s['accuracy']:.1%} |")
    lines+=['', '원자료 SHA256: `'+before+'`', '',
        '후속은 표시 1:1 및 광학적 단서 점검, 계획 고정 후 독립 반복으로 진행한다. '
        '당장 추가 응답을 자동 시작하거나 기존 연구의 수집/추론 제한을 해제하지 않는다.', '']
    (out/'repeat-v3-analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'output':str(out/'repeat-v3-analysis.md'),'saved':len(rows),'correct':report['overall']['correct'],
        'source_unchanged':pilot.sha(source)==before,'sensitivity':report['sensitivity_only_omit_reexposure_and_fast']},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--same-glasses-reported',action='store_true')
    parser.add_argument('--color-fringing-reported',action='store_true')
    args=parser.parse_args()
    main(args.same_glasses_reported,args.color_fringing_reported)
