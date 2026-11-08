"""Score pre-locked predictions against fresh v4 responses; never refit models."""
from collections import Counter
from datetime import datetime
import json
import math
from statistics import median
import numpy as np
import pilot
import validate_width_pilot as validation
import fit_width_prediction as prediction

def summarize(rows):
    n=len(rows);k=sum(r['correct'] for r in rows)
    return {'n':n,'correct':k,'accuracy':k/n if n else None,
        'median_rt_ms':median(r['response_time_ms'] for r in rows) if n else None}

def scores(cells,model):
    n=sum(c['n'] for c in cells);ll=0.;brier=0.
    for c in cells:
        p=c['predictions'][model];k=c['correct'];nn=c['n']
        ll-=k*math.log(p)+(nn-k)*math.log1p(-p)
        brier+=k*(1-p)**2+(nn-k)*p*p
    return {'n':n,'logloss':ll/n,'brier':brier/n}

def bootstrap_difference(cells,model,seed=20261111):
    """Post-hoc conditional within-cell bootstrap, paired prediction losses.

    Does not refit or claim population-level intervals/validated C80 coverage.
    """
    rng=np.random.default_rng(seed);draws=2000
    ll=np.zeros(draws);bb=np.zeros(draws);total=sum(c['n'] for c in cells)
    for c in cells:
        k=rng.binomial(c['n'],c['correct']/c['n'],size=draws);n=c['n']
        p=c['predictions'][model];q=c['predictions']['constant']
        ll+=k*math.log(q/p)+(n-k)*math.log((1-q)/(1-p))
        bb+=k*((1-p)**2-(1-q)**2)+(n-k)*(p*p-q*q)
    observed=scores(cells,model);baseline=scores(cells,'constant')
    return {'relative_to':'constant; negative difference favors width model',
        'brier_difference':observed['brier']-baseline['brier'],
        'logloss_difference':observed['logloss']-baseline['logloss'],
        'brier_bootstrap_percentile95':np.quantile(bb/total,[.025,.975]).tolist(),
        'logloss_bootstrap_percentile95':np.quantile(ll/total,[.025,.975]).tolist(),
        'draws':draws,'seed':seed,'interpretation':'post-hoc within-cell conditional diagnostic; not population inference'}

def main():
    plan=validation.load_plan(); snapshot_hash=pilot.sha(validation.SNAPSHOT)
    models=json.loads(validation.SNAPSHOT.read_text(encoding='utf-8'))
    paths=[]
    for path in (pilot.ROOT/'raw/v4').glob('*/responses.jsonl'):
        rows=pilot.read_rows(path)
        if len(rows)==120:paths.append((path,rows))
    if len(paths)!=1:raise ValueError('완료 회차를 명시해야 합니다.')
    source,rows=paths[0];source_hash=pilot.sha(source)
    pilot.verify_rows(rows,validation.schedule())
    cp=source.parent/'config.json';cfg=json.loads(cp.read_text(encoding='utf-8'))
    if cfg['plan_sha256']!=pilot.sha(validation.PLAN) or cfg['model_snapshot_sha256']!=snapshot_hash:raise ValueError('고정 계획/모델 연결이 다릅니다.')
    if not all(r['config_sha256']==pilot.sha(cp) for r in rows):raise ValueError('응답 설정 해시 불일치')
    prepared=datetime.fromisoformat(plan['prepared_utc'])
    if not all(datetime.fromisoformat(r['utc'])>=prepared for r in rows):raise ValueError('계획 고정 전 응답이 있습니다.')
    cells=[]
    for c in plan['colors']:
        ci=models['colors'].index(c['condition_id'])
        for pair in c['pairs']:
            sub=[r for r in rows if r['trial']['condition_id']==c['condition_id'] and r['trial']['level']==pair['level']]
            if len(sub)!=12:raise ValueError('수준별 12반복 불일치')
            for m in models['models']:
                actual=prediction.predict(m['full_fit'],ci,6,pair['nominal_delta_e00'])
                if abs(actual-pair['frozen_accuracy_predictions'][m['model']])>1e-12:raise ValueError('동결 예측값 재계산 불일치')
            cells.append({'condition_id':c['condition_id'],'level':pair['level'],
                'nominal_delta_e00':pair['nominal_delta_e00'],
                **summarize(sub),'predictions':pair['frozen_accuracy_predictions']})
    names=[m['model'] for m in models['models']]
    events=pilot.read_rows(source.parent/'events.jsonl')
    model_scores={m:scores(cells,m) for m in names}
    brier_best=min(names,key=lambda m:model_scores[m]['brier'])
    logloss_best=min(names,key=lambda m:model_scores[m]['logloss'])
    report={'analysis_utc':pilot.now(),'analysis_code_sha256':pilot.sha(__file__),
        'source_relative':source.relative_to(pilot.ROOT).as_posix(),'source_sha256':source_hash,
        'config_sha256':pilot.sha(cp),'plan_sha256':pilot.sha(validation.PLAN),'model_snapshot_sha256':snapshot_hash,
        'model_refit_performed':False,'all_responses_after_plan_lock':True,
        'scope':'fresh 6px exploratory follow-up; not original protocol validation or confirmed C80 comparison',
        'overall':summarize(rows),'cells':cells,
        'model_scores':model_scores,
        'by_color':{c['condition_id']:{'observed':summarize([r for r in rows if r['trial']['condition_id']==c['condition_id']]),
            'model_scores':{m:scores([x for x in cells if x['condition_id']==c['condition_id']],m) for m in names}} for c in plan['colors']},
        'paired_loss_sensitivity':{m:bootstrap_difference(cells,m) for m in names if m!='constant'},
        'event_counts':dict(Counter(e['event'] for e in events)),
        'reexposure_ids':[e.get('trial_id') for e in events if e['event'] in ['focus_interruption','interrupted_display']],
        'fast_under200ms':sum(r['response_time_ms']<200 for r in rows),
        'slow_over30s':sum(r['response_time_ms']>30000 for r in rows),
        'duplicates':len(rows)-len({r['trial']['trial_id'] for r in rows}),
        'C80_estimated':False,'correction_efficacy_confirmed':False,
        'numerical_best_brier':brier_best,'numerical_best_logloss':logloss_best,
        'conclusion':'Numerical rankings are descriptive; inspect both metrics and uncertainty. No automatic superiority or precise C80 claim.',
        'limitations':['Single observer and two selected color paths; no population effect or clinical diagnosis.',
            'Same observer/glasses/display condition is requested and self-reported, not independently verified.',
            'New responses test locked models but model family and color paths were selected using exploratory training data.',
            'Twelve trials per level; observed proportions have substantial uncertainty.',
            'Bootstrap is post-hoc conditional paired-loss diagnostic, not validated C80 coverage.',
            'No refit of the training model or retrospective change of success criteria.']}
    assert pilot.sha(source)==source_hash and pilot.sha(validation.SNAPSHOT)==snapshot_hash
    out=pilot.ROOT/'raw/_analysis/validation-v4-results.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 새 굵기 6px 예측 확인 — 120회', '',
        '## 결론', '',
        '새 응답을 보기 전에 고정한 세 모델을 그대로 평가했다. 새 응답으로 모델을 다시 맞추지 않았다. '
        f'수치상 최소 Brier 모델은 {brier_best}, 최소 logloss 모델은 {logloss_best}다. '
        '수치 순위만으로 통계적 우위·정밀 역치를 확정하지 않으며 아래 실제 점수와 불확실성을 함께 본다.', '',
        f"전체 {report['overall']['correct']}/120 정답 ({report['overall']['accuracy']:.1%}). "
        '정답률 자체는 제시 난도에 따라 달라지며 예측 모델 정확도의 점수와 다르다.', '',
        '## 고정 모델의 예측 점수', '',
        '둘 다 낮을수록 좋다. Brier는 확률 오차의 제곱, logloss는 확신한 오답에 더 큰 벌점을 주는 점수다.', '',
        '| 모델 | Brier | logloss |','|---|---:|---:|']
    for m,s in report['model_scores'].items():lines.append(f"| {m} | {s['brier']:.4f} | {s['logloss']:.4f} |")
    lines+=['', '유리한 지표만 골라 성공이라고 하지 않는다. 어떤 조건에서 과소/과대 예측했는지 아래에서 확인한다.', '',
        '## 색 경로별', '', '| 색 경로 | 정답/60 | constant Brier | log_width Brier | plateau4 Brier |',
        '|---|---:|---:|---:|---:|']
    for cid,c in report['by_color'].items():
        lines.append(f"| {cid} | {c['observed']['correct']} | {c['model_scores']['constant']['brier']:.4f} | {c['model_scores']['log_width']['brier']:.4f} | {c['model_scores']['plateau4']['brier']:.4f} |")
    lines+=['', '## 예측과 관측을 그대로 비교', '',
        '| 색 경로 | 명목 원본 ΔE00 | 정답/12 | 관측률 | constant | log_width | plateau4 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c in cells:
        pred=' | '.join(f"{c['predictions'][m]:.1%}" for m in names)
        lines.append(f"| {c['condition_id']} | {c['nominal_delta_e00']:.3f} | {c['correct']} | {c['accuracy']:.1%} | {pred} |")
    lines+=['', '관측 정답률 12회만으로 실제 확률을 확정하지 않는다. '
        '단일 오차 방향이나 단일 개인 배율로 문제가 해결된다고 가정하지 않는다.', '',
        '## 불확실성의 사후 보조 점검', '',
        '각 색×수준 안에서 관측 반응을 재표집한 조건부 paired-loss bootstrap 2000회다. '
        '음수면 굵기 모델이 낫다는 뜻이다. 참가자 모집단 신뢰구간이나 C80 coverage 검증이 아니다.', '',
        '| 모델-constant | Brier 차이 (보조95%구간) | logloss 차이 (보조95%구간) |',
        '|---|---|---|']
    for m,s in report['paired_loss_sensitivity'].items():
        b=s['brier_bootstrap_percentile95'];l=s['logloss_bootstrap_percentile95']
        lines.append(f"| {m} | {s['brier_difference']:.4f} ({b[0]:.4f}~{b[1]:.4f}) | {s['logloss_difference']:.4f} ({l[0]:.4f}~{l[1]:.4f}) |")
    lines+=['', '## 무결성 및 해석 범위', '',
        '120개 고유 응답, 채점/시행표/설정/모델/계획 해시 대조 통과. 모든 응답은 계획 고정 이후다. '
        f"재제시 사건 {len(report['reexposure_ids'])}건, 200ms 미만 {report['fast_under200ms']}건/30초 초과 {report['slow_over30s']}건. "
        '원응답과 모델 스냅샷 변경 없음.', '',
        'v3에서 굵기별 성적을 관찰하는 것과 새 굵기 역치/확률을 정확히 예측하는 것은 다른 주장이다. '
        '정밀 C80이나 최소 보정 효능을 선언하지 않는다. 추가 수집은 자동 시작하지 않는다.', '',
        '원응답 SHA256: `'+source_hash+'`', '',
        '고정 모델 SHA256: `'+snapshot_hash+'`', '']
    out.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'saved':120,'scores':report['model_scores'],'paired_loss':report['paired_loss_sensitivity'],
        'by_color':report['by_color'],'report':str(out.with_suffix('.md'))},ensure_ascii=False))

if __name__=='__main__':main()
