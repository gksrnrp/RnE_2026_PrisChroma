"""Exploratory width prediction from v3 only; not frozen study C80 inference.

Exact finite-grid likelihood search, numpy only. Shared psychometric slope,
width coefficient and lapse, with separate intercepts for the two color paths.
No individual's clinical or old patch parameters are injected.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import pilot
import repeat_pilot as repeat

from legacy_color_math import srgb_to_linear,delta_e00_linear

OUT=pilot.ROOT/'raw/_analysis/width-prediction-pilot.json'
MODELS=['constant','log_width','plateau4']
ALPHAS=np.round(np.arange(-8,8.001,.1),6)
SLOPES=np.arange(.5,8.001,.5)
GAMMAS=np.arange(-4,4.001,.25)
LAPSES=np.array([0,.04,.08,.12,.16])

def width_feature(w,model):
    w=np.asarray(w,dtype=float)
    if model=='constant':return np.zeros_like(w)
    if model=='log_width':return np.log(w/4)
    if model=='plateau4':return np.log(np.minimum(w,4)/4)
    raise ValueError('알 수 없는 모형')

def probability(eta,lapse):
    return .25+(.75-lapse)/(1+np.exp(-np.clip(eta,-60,60)))

def nominal_delta(pair):
    a=srgb_to_linear(np.asarray(pair['rgb_a'],dtype=float)/255)
    b=srgb_to_linear(np.asarray(pair['rgb_b'],dtype=float)/255)
    return float(delta_e00_linear(a,b))

def load_cells():
    audit=json.loads((pilot.ROOT/'raw/_analysis/repeat-v3-analysis.json').read_text(encoding='utf-8'))
    source=pilot.ROOT/audit['source_relative']
    if pilot.sha(source)!=audit['source_sha256']:raise ValueError('원자료가 분석 근거와 다릅니다.')
    rows=pilot.read_rows(source);pilot.verify_rows(rows,repeat.schedule())
    if len(rows)!=240:raise ValueError('완료된 v3 자료만 사용합니다.')
    cells=[]
    colors=sorted(c['condition_id'] for c in repeat.plan()['conditions'])
    for ci,color in enumerate(colors):
        for level in [1,2,3]:
            for width in repeat.WIDTHS:
                sub=[r for r in rows if r['trial']['condition_id']==color and r['trial']['level']==level and r['trial']['width_px']==width]
                if len(sub)!=8:raise ValueError('반복 배정 불일치')
                cells.append({'color_index':ci,'condition_id':color,'level':level,'width':width,
                    'delta_e00':nominal_delta(sub[0]['trial']),'n':8,'k':sum(r['correct'] for r in sub)})
    return cells,colors,source,audit

def fit(cells,model):
    best=None
    parts=[]
    for ci in range(2):
        sub=[c for c in cells if c['color_index']==ci]
        parts.append((np.log([c['delta_e00'] for c in sub]),
            width_feature([c['width'] for c in sub],model),
            np.asarray([c['n'] for c in sub]),np.asarray([c['k'] for c in sub])))
    for slope in SLOPES:
        for gamma in ([0.] if model=='constant' else GAMMAS):
            # Evaluate all lapse candidates and alpha candidates in one array.
            total=np.zeros(len(LAPSES)); selected=[]
            for x,fw,n,k in parts:
                eta=ALPHAS[None,:,None]+slope*x[None,None,:]+gamma*fw[None,None,:]
                p=probability(eta,LAPSES[:,None,None]);p=np.clip(p,1e-10,1-1e-10)
                nll=-(k*np.log(p)+(n-k)*np.log1p(-p)).sum(axis=2)
                indices=np.argmin(nll,axis=1)
                total+=nll[np.arange(len(LAPSES)),indices]
                selected.append(ALPHAS[indices])
            li=int(np.argmin(total)); value=float(total[li])
            if best is None or value<best['nll']-1e-10:
                best={'model':model,'alpha':[float(a[li]) for a in selected],
                    'slope':float(slope),'gamma':float(gamma),'lapse':float(LAPSES[li]),'nll':value}
    best['parameters']=4 if model=='constant' else 5
    best['aic']=2*best['parameters']+2*best['nll']
    best['on_grid_boundary']=bool(best['slope'] in [float(SLOPES[0]),float(SLOPES[-1])]
        or any(a in [float(ALPHAS[0]),float(ALPHAS[-1])] for a in best['alpha'])
        or (model!='constant' and best['gamma'] in [float(GAMMAS[0]),float(GAMMAS[-1])])
        or best['lapse']==float(LAPSES[-1]))
    return best

def predict(fit_result,ci,width,delta):
    eta=fit_result['alpha'][ci]+fit_result['slope']*math.log(delta)+fit_result['gamma']*float(width_feature(width,fit_result['model']))
    return float(probability(eta,fit_result['lapse']))

def losses(cells,fit_result):
    n=sum(c['n'] for c in cells); ll=0.;brier=0.
    predictions=[]
    for c in cells:
        p=predict(fit_result,c['color_index'],c['width'],c['delta_e00'])
        ll-=c['k']*math.log(p)+(c['n']-c['k'])*math.log1p(-p)
        brier+=c['k']*(1-p)**2+(c['n']-c['k'])*p*p
        predictions.append({'condition_id':c['condition_id'],'width':c['width'],
            'level':c['level'],'predicted_accuracy':p,'observed_correct':c['k'],'n':c['n']})
    return {'n':n,'logloss':ll/n,'brier':brier/n,'predictions':predictions}

def threshold(fitted,ci,width,cells):
    """Only expose model-implied C80 if within sampled nominal color range."""
    bounds=[c['delta_e00'] for c in cells if c['color_index']==ci]
    low,high=min(bounds),max(bounds)
    q=(.8-.25)/(.75-fitted['lapse'])
    raw=math.exp((math.log(q/(1-q))-fitted['alpha'][ci]
        -fitted['gamma']*float(width_feature(width,fitted['model'])))/fitted['slope'])
    supported=low<=raw<=high
    observed=[c for c in cells if c['color_index']==ci and c['width']==width]
    observed_crossing=bool(observed and min(c['k']/c['n'] for c in observed)<.8<=max(c['k']/c['n'] for c in observed))
    return {'width_px':width,'sampled_nominal_delta_e00_range':[low,high],
        'model_implied_C80_nominal_delta_e00':raw if supported else None,
        'status':'provisional_in_sample_range' if supported else ('above_sampled_range' if raw>high else 'below_sampled_range'),
        'raw_observed_80pct_bracket':observed_crossing,
        'unmeasured_width':width not in repeat.WIDTHS,'validated':False,'allowed_for_research_correction':False}

def run():
    cells,colors,source,audit=load_cells(); outputs=[]
    for model in MODELS:
        print('Fitting',model,flush=True)
        full=fit(cells,model); folds=[]
        for width in repeat.WIDTHS:
            train=[c for c in cells if c['width']!=width]; test=[c for c in cells if c['width']==width]
            assert all(c['width']!=width for c in train)
            fitted=fit(train,model)
            folds.append({'heldout_width':width,'kind':'extrapolation' if width in [1,8] else 'interpolation',
                'training_n':sum(c['n'] for c in train),'fit':fitted,**losses(test,fitted)})
        outputs.append({'model':model,'full_fit':full,
            'leave_one_width_out_brier':sum(f['brier']*f['n'] for f in folds)/240,
            'leave_one_width_out_logloss':sum(f['logloss']*f['n'] for f in folds)/240,
            'folds':folds,
            'thresholds':{color:[threshold(full,ci,w,cells) for w in [1,2,3,4,5,6,7,8]] for ci,color in enumerate(colors)}})
    best=min(outputs,key=lambda m:m['leave_one_width_out_logloss'])
    report={'analysis_utc':pilot.now(),'analysis_code_sha256':pilot.sha(__file__),
        'source_relative':source.relative_to(pilot.ROOT).as_posix(),'source_sha256':audit['source_sha256'],
        'design_sha256':pilot.sha(repeat.PLAN_PATH),
        'color_math_sha256':pilot.sha(Path(__file__).with_name('legacy_color_math.py')),
        'colors':colors,'cells':cells,'models':outputs,
        'exploratory_lowest_cv_logloss':best['model'],
        'probability_formula':'0.25 + (0.75-lapse)*sigmoid(alpha[color]+slope*ln(nominal_original_deltaE00)+gamma*width_feature)',
        'features':{'constant':'0','log_width':'ln(width/4)','plateau4':'ln(min(width,4)/4)'},
        'grid':{'alpha':ALPHAS.tolist(),'slope':SLOPES.tolist(),'gamma':GAMMAS.tolist(),'lapse':LAPSES.tolist()},
        'validated':False,'correction_enabled':False,'uncertainty_intervals_validated':False,
        'limitations':['Post-hoc exploratory models; plateau at 4px was motivated by the observed pilot.',
            'Leave-one-width-out reanalysis is not a new prospective independent experiment.',
            'Predicts raw behavior using nominal original sRGB deltaE00, NOT post-Machado deltaE00 or the frozen study C80.',
            'Uses two selected color paths and one glasses/display observation condition; no diagnosis or population generalization.',
            'Common slope, width effect and lapse across colors are unvalidated simplifying assumptions.',
            'Finite grid optimization; boundary solutions and model disagreement must be reported.',
            'No confidence/coverage calibration; do not use provisional thresholds to claim minimum effective correction.',
            'Out-of-range C80 is censored rather than supplied as a seemingly measured number.']}
    assert pilot.sha(source)==audit['source_sha256']
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 선 굵기 예측 모델 — 탐색용', '',
        '예측하는 것은 굵기 자체가 아니라, 주어진 굵기와 색차에서의 정답확률 및 모델상 80% 지점이다. '
        '이번 240회만 사용하고 다른 참가자/회차의 자료는 합치지 않았다.', '',
        '## 한 굵기를 제외하고 예측한 결과', '',
        '각 굵기를 한 번씩 학습에서 빼고 나머지 네 굵기로 예측했다. 평균 점수가 낮을수록 낫다. '
        '1/8px 제외는 외삽, 2/3/4px 제외는 내삽이다. 이미 본 자료로 모델을 정한 사후 분석으로 독립 확증 검증은 아니다.', '',
        '| 모델 | 평균 Brier | 평균 logloss |','|---|---:|---:|']
    for m in outputs:lines.append(f"| {m['model']} | {m['leave_one_width_out_brier']:.4f} | {m['leave_one_width_out_logloss']:.4f} |")
    lines+=['', 'constant=굵기 무관, log_width=굵기에 따라 연속 변화, plateau4=4px 이상에서 포화한다고 가정. '
        '모든 굵기 계수는 양/음 모두 허용했으며 원하는 방향을 강제하지 않았다.', '',
        f"굵기 제외 적합 {sum(f['fit']['on_grid_boundary'] for m in outputs for f in m['folds'])}개가 탐색 격자 경계에 닿았다. "
        '교차검증 점수는 해당 유한 격자에서의 잠정 비교이며 안정된 최적 적합이나 두 굵기 모형의 우열을 확정하지 않는다. '
        '정답확률 예측 점수가 좋아졌다는 것이 C80 예측 오차를 직접 검증했다는 뜻도 아니다.', '',
        '## 모델상 C80 (명목 원본 sRGB ΔE00)', '',
        '숫자가 있더라도 새로 관측한 역치나 검증된 보정값이 아니다. 범위 밖은 숫자 대신 표시한다. '
        '두 경로·이번 안경/표시 조건 이외로 일반화하지 않는다.', '',
        '| 색 경로 | 모델 | 1px | 2px | 3px | 4px | 6px | 8px |',
        '|---|---|---|---|---|---|---|---|']
    for color in colors:
        for m in outputs:
            vals=[]
            for t in m['thresholds'][color]:
                if t['width_px'] not in [1,2,3,4,6,8]:continue
                value=t['model_implied_C80_nominal_delta_e00']
                vals.append(f'{value:.2f}' if value is not None else ('측정 범위 초과' if t['status']=='above_sampled_range' else '범위 하한 미만'))
            lines.append('| '+color+' | '+m['model']+' | '+' | '.join(vals)+' |')
    lines+=['', '## 사용 한계', '',
        '모델과 임시 예측기는 구현했지만 예측 성공/정밀 C80/효과적 보정을 확정하지 않았다. '
        '특히 1px 역치가 관측 색차 범위 밖이면 추가 범위가 필요하다. '
        'lapse를 격자로 공동 추정했으며 독립적으로 측정한 개인 상수가 아니다. '
        '모델별 예측이 다르거나 경계에 걸리면 불확실성을 먼저 보고한다.', '',
        '임의의 새 색쌍·다른 화면·안경 미착용 조건에는 이 모델을 적용하지 않는다. '
        '기존 동결 프로토콜/연구 게이트와 별도 탐색 도구이며 본실험 준비도를 해제하지 않는다.', '']
    OUT.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    for m in outputs:print(json.dumps({k:m[k] for k in ['model','leave_one_width_out_brier','leave_one_width_out_logloss']},ensure_ascii=False))
    print('Saved',OUT)

def query(width,color):
    if not 1<=width<=8:raise ValueError('현재 모델의 굵기 사용 범위는 1~8px입니다.')
    data=json.loads(OUT.read_text(encoding='utf-8'))
    if pilot.sha(pilot.ROOT/data['source_relative'])!=data['source_sha256']:
        raise ValueError('원자료가 바뀌었습니다. 예측기를 다시 적합해야 합니다.')
    if pilot.sha(repeat.PLAN_PATH)!=data['design_sha256']:
        raise ValueError('색쌍 계획이 바뀌었습니다. 기존 모델을 적용하지 않습니다.')
    if color not in data['colors']:raise ValueError('측정한 두 색 경로만 지원합니다.')
    ci=data['colors'].index(color); output=[]
    for model in data['models']:
        output.append({'model':model['model'],**threshold(model['full_fit'],ci,width,data['cells'])})
    print(json.dumps({'condition_id':color,'provisional_only':True,'predictions':output},ensure_ascii=False,indent=2))

def self_test():
    synthetic=[]
    for ci in range(2):
        for width in repeat.WIDTHS:
            for delta in [.7,1.4,2.8]:
                eta=[-.5,.5][ci]+2*math.log(delta)+float(width_feature(width,'log_width'))
                p=float(probability(eta,.04))
                synthetic.append({'color_index':ci,'condition_id':'synthetic_'+str(ci),
                    'level':delta,'width':width,'delta_e00':delta,'n':200,'k':200*p})
    fitted=fit(synthetic,'log_width')
    assert abs(fitted['gamma']-1)<=.25 and abs(fitted['slope']-2)<=.5
    assert losses(synthetic,fitted)['brier']>=0
    assert float(width_feature(4,'plateau4'))==float(width_feature(8,'plateau4'))
    assert width_feature(1,'log_width')<width_feature(2,'log_width')
    assert .25<=predict(fitted,0,2,1.4)<=1
    for heldout in repeat.WIDTHS:
        train=[c for c in synthetic if c['width']!=heldout]
        assert len(train)==24 and all(c['width']!=heldout for c in train)
    extreme={**fitted,'alpha':[-8,-8]}
    assert threshold(extreme,0,1,synthetic)['model_implied_C80_nominal_delta_e00'] is None
    print('PASS: synthetic grid recovery, probability bounds, plateau feature, out-of-range C80 censorship; not human-study validation')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--width',type=float);parser.add_argument('--color')
    args=parser.parse_args()
    if args.self_test:self_test()
    elif args.width is not None:
        if not args.color:parser.error('--color 필요')
        query(args.width,args.color)
    else:run()
