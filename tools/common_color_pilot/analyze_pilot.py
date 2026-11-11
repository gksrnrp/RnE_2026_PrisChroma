"""Read-only descriptive analysis of complete v2 pilot journals.

No inference of identities, clinical type, C80, or confirmatory effect sizes.
"""
import json
from collections import Counter
from statistics import median
import pilot

def summarize(rows):
    n=len(rows); k=sum(r['correct'] for r in rows)
    return {'n':n,'correct':k,'accuracy':k/n if n else None,
            'median_rt_ms':median(r['response_time_ms'] for r in rows) if n else None}

def analyze():
    journals=[]
    for path in (pilot.ROOT/'raw'/'v2').glob('*/responses.jsonl'):
        before=pilot.sha(path); rows=pilot.read_rows(path)
        pilot.verify_rows(rows,pilot.schedule())
        journals.append((path,rows,before))
    complete=[entry for entry in journals if len(entry[1])==288]
    if len(complete)!=1:
        raise ValueError('완료 회차가 정확히 한 개가 아니므로 분석 대상을 명시해야 합니다.')
    source,rows,source_hash=complete[0]
    colors=[]
    for c in pilot.load_design()['conditions']:
        sub=[r for r in rows if r['trial']['condition_id']==c['cid']]
        colors.append({'condition_id':c['cid'],'base_hex':c['base_hex'],
            **summarize(sub),'by_width':{str(w):summarize([r for r in sub if r['trial']['width_px']==w]) for w in pilot.WIDTHS},
            'by_level':{str(l):summarize([r for r in sub if r['trial']['level']==l]) for l in range(1,7)}})
    report={'analysis_time_utc':pilot.now(),'analysis_code_sha256':pilot.sha(__file__),
        'source_relative':source.relative_to(pilot.ROOT).as_posix(),'source_sha256':source_hash,
        'config_sha256':pilot.sha(source.parent/'config.json'),
        'scope':'one complete exploratory session; other journals NOT pooled',
        'other_journal_lengths':[len(r) for p,r,h in journals if p!=source],
        'overall':summarize(rows),
        'by_width':{str(w):summarize([r for r in rows if r['trial']['width_px']==w]) for w in pilot.WIDTHS},
        'by_block':{str(b):summarize([r for r in rows if r['trial']['block']==b]) for b in range(1,7)},
        'colors':colors,'trial_id_duplicates':len(rows)-len({r['trial']['trial_id'] for r in rows}),
        'fast_under_200ms':sum(r['response_time_ms']<200 for r in rows),
        'slow_over_30s':sum(r['response_time_ms']>30000 for r in rows),
        'automatic_exclusions':0,'C80_estimated':False,
        'limitations':['One trial per color x width x level; no precise color-specific C80.',
            'Level decreases by block: color separation is confounded with time/order/fatigue.',
            'A level is a fraction of that color path maximum, not a common DeltaE across colors.',
            'Similar aggregate accuracy is not proof of equivalence or absence of a width effect.',
            'Different session IDs do not prove different participants. No comparison to P01 as a person.',
            'No medical diagnosis, normal-vision comparison, generalization, or correction efficacy claim.']}
    assert pilot.sha(source)==source_hash
    out=pilot.ROOT/'raw'/'_analysis'; out.mkdir(exist_ok=True)
    (out/'complete-v2-analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 공통색 예비실험 v2: 완료 288회 기술통계', '',
        '## 요약', '',
        '완료된 한 회차만 분석했다. 다른 회차는 보존하고 합산하지 않았다. '
        '같은 사람의 재시작인지 다른 참가자인지는 파일만으로 확정하지 않는다.', '',
        f"전체 {report['overall']['correct']}/288 정답 ({report['overall']['accuracy']:.1%}). "
        f"시행표·채점 대조 통과, 중복 시행 {report['trial_id_duplicates']}개. 원응답 수정 없음.", '',
        '## 굵기별', '', '| 굵기 | 정답/시행 | 정답률 |','|---|---:|---:|']
    for w,s in report['by_width'].items():
        lines.append(f"| {w}px | {s['correct']}/{s['n']} | {s['accuracy']:.1%} |")
    lines+=['', '굵을수록 높아지는 단조 순서가 관찰되지 않았다. 비슷한 전체 정답률은 역치가 같다는 증거가 아니다. '
        '현재 자료로 색 조건별 역치나 굵기 효과의 부재를 확정하지 않는다.', '',
        '## 색차 단계별 (시간 순서와 결합됨)', '',
        '| 블록 | 각 색 경로 최대 간격의 비율 | 정답/시행 | 정답률 |','|---|---|---:|---:|']
    for b,s in report['by_block'].items():
        lines.append(f"| {b} | {pilot.FRACTIONS[6-int(b)]:g} | {s['correct']}/{s['n']} | {s['accuracy']:.1%} |")
    lines+=['', '색차를 줄이며 정답률이 감소하는 모습은 관찰되나, 항상 큰 간격부터 제시했으므로 '
        '피로/순서/학습 영향을 분리할 수 없다. 비율이 같아도 색 조건 간 절대 색차가 같지 않다.', '',
        '## 색 경로별', '',
        '색 이름 전체가 아니라 특정 기준색과 변화 방향의 결과다. 네 굵기와 여섯 색차를 합한 기술통계다.', '',
        '| 조건 | 기준색 HEX | 정답/24 | 정답률 | 2px | 4px | 8px | 16px |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in colors:
        values=' | '.join(str(c['by_width'][str(w)]['correct'])+'/6' for w in pilot.WIDTHS)
        lines.append(f"| {c['condition_id']} | {c['base_hex']} | {c['correct']} | {c['accuracy']:.1%} | {values} |")
    lines+=['', '## 해석과 다음 단계', '',
        '이번 예비검사에서 뚜렷한 굵기별 평균 정답률 차이를 확보하지 못했다. '
        '기존 주제의 효과가 입증됐다고 쓰지 않는다. 여러 색 관계의 난도 범위를 확인한 탐색자료로 보존한다.', '',
        '후속 설계는 색 경로별로 범위를 잡고 색차 수준을 무작위로 섞어 반복해야 한다. '
        '이번 자료에서 유리하게 보이는 경로만 골라 확증 결과라고 보고하지 않는다. '
        '새 계획과 측정 정밀도를 먼저 정하고 독립 자료로 확인한다. 이번 결과에 C80 숫자를 억지로 붙이지 않는다.', '',
        f"반응 200ms 미만 {report['fast_under_200ms']}건, 30초 초과 {report['slow_over_30s']}건. 자동 제외하지 않았다.", '',
        '## 재현', '', '원자료 SHA256: `'+source_hash+'`', '',
        '세부 집계 및 분석 출처는 같은 폴더의 complete-v2-analysis.json에 기록했다.', '']
    (out/'complete-v2-analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'saved':str(out),'n':288,'correct':report['overall']['correct'],
        'source_unchanged':pilot.sha(source)==source_hash},ensure_ascii=False))

if __name__=='__main__': analyze()
