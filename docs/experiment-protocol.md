# 사람 실험·분석 사전 고정 문서

연구 제목: **고정 표시 조건에서 적록색각이상 사례 참가자의 선 굵기별 C80 예측과 제약 기반 쌍 단위 최소 색이동 보정**

문서 상태: `DRAFT — 사람 자료 수집 금지`
현재 최상위 연구 기준: 이 문서
최종 수정일: 2026-09-14

이 연구는 큰 색 패치에서 얻은 한 세션의 값 0.62, 개발용 값 3.0 또는 논문의
경험적 ΔE를 얇은 선의 보편 임계값으로 사용하지 않는다. 화면·과제·색 방향을
고정하고 참가자별 행동 반응에서 C80을 직접 추정한다. 아래 `미정` 항목,
실제자료 계약, 자극·응답 프로그램, **동결 설정과 결합한** parameter recovery,
윤리 절차와 현장 dry-run이 모두 끝나기 전에는 사람 응답을 수집하지 않는다.

## 1. 왜 하는가

같은 RGB 두 색도 넓은 패치에서는 구별되지만 얇은 선에서는 구별하기 어려울 수
있다. 기존 색보정은 표시 크기와 개인차를 무시한 고정 색차 또는 자연영상 전체
변환을 주로 사용한다. 그러면 얇은 요소는 덜 보정되거나, 이미 보이는 색은
불필요하게 크게 바뀔 수 있다. 본 연구는 **고정한 자극군 안에서** 개인의 선
굵기별 80% 정답 색차를 예측하고 그만큼만 움직이는 규칙을 시험한다.

## 2. 연구 질문, 주 비교, 반증 조건

1. 고정 표시 조건과 고정 색 경로에서 선 굵기가 증가할수록 참가자별 C80이
   낮아지는가?
   - 모형 D288의 `beta`는 부호 제약 없이 적합한다.
   - 참가자별 bootstrap 95% CI의 상한이 0보다 작을 때만 음의 방향을 확인한다.
   - CI가 0을 포함하면 `현재 정밀도로 굵기 효과 미확인`으로 보고한다.
2. 한 기준 굵기 96시행만 쓴 Szafir 상대형 전이 모형 C96이, 같은 96시행만 쓴
   굵기 불변 모형 B96보다 미사용 굵기 C80을 정확히 예측하는가?
   - **주 비교:** held-out 굵기에서 `|예측 C80-관측 C80|`의 참가자별 차이.
   - 같은 모의 교정 반응으로 짝지은 B96/C96 예측 bootstrap과 held-out pooled
     C80 bootstrap을 결합한 `Δloss=|C96-O|-|B96-O|` 95% 구간으로 판정한다.
     구간이 0을 포함하면 `모델 구분 불가`, 필요한 복제값이 없으면
     `comparison_unresolvable`로 보고한다.
3. 각 모형이 예측한 C80을 같은 색쌍·같은 탐색기에 넣었을 때 요구되는 OKLab
   이동 비용과 과소·과잉 보정은 어떻게 다른가?
   - 이는 **공학적 보조 분석**이다. 사람이 보정색을 다시 판별하지 않으면
     `가독성이 향상됐다`고 쓰지 않는다.

보조 비교는 D288 대 B288(같은 288시행에서 굵기항의 이득), C96 대 B288
(측정량 절감 가능성)이다. 근거가 없는 보편 고정 모형 A는 제외한다.

## 3. 결론 범위

- 참가자는 적록색각이상 사례 2명이며 반복측정 사례연구다.
- protan 1명과 deutan 1명이 다르게 나와도 유형 차이와 개인차를 구분할 수 없다.
- 결론은 동결한 모니터, 관찰 거리, 렌더러, 4AFC 과제, 배경, 선 길이·간격,
  Lab 중점과 방향, 측정한 굵기 범위 안에만 적용한다.
- 색도계로 화면 출력을 재지 못하면 `colorimetry_policy`를
  `nominal-srgb-device-specific`으로 고정한다. 이때 ΔE00는 `저장 sRGB를 표준
  변환한 계산 좌표`이며 장치 독립적 절대 역치가 아니다. 휘도계도 없으면
  `luminance_cd_m2=null`, `luminance_measurement_method=`
  `not_measured_no_luminance_meter`로 기록하고 임의의 cd/m²를 만들지 않는다.
- 길이와 간격을 고정하고 굵기를 바꾸면 색 면적도 함께 변한다. 따라서 결과는
  순수 기하학적 폭이 아니라 `고정 길이·간격에서의 표시 굵기 효과`다.
- Machado는 조건을 표준화하는 계산 모형이지 참가자의 시각을 완벽히 복제하는
  개인 지각 모형이 아니다. CIEDE2000도 행동 확률 자체가 아니다.
- 한 Lab 중점·방향에서 얻은 scalar C80을 임의 RGB 색쌍에 일반화하지 않는다.
  보정의 연구 근거는 같은 자극군으로 제한한다. 임의 색쌍 출력은 소프트웨어
  시연일 뿐이며 별도 인간 검증 없이는 가독성 근거가 아니다.

## 4. 정확한 심리측정 함수

4AFC 우연 정답률 `gamma=0.25`, 실제 표시 RGB에서 다시 계산한 Machado 후
색차 `x>0`, C80 `C80(s)>0`, 기울기 폭 `w>0`, lapse `0<=lambda<0.20`에 대해
다음을 사용한다.

```text
p(correct | x,s)
= 0.25 + (0.75-lambda) * logistic(
    q80(lambda) + {ln(x)-ln(C80(s))}/w
  )

q80(lambda)
= logit[(0.80-0.25)/(0.75-lambda)]
```

따라서 `x=C80(s)`이면 실제 정답률이 정확히 0.80이다. Bernoulli 로그우도를
최대화하고, C80과 w는 로그 좌표에서 적합한다. lapse는 96시행에서 C80·기울기와
동시에 자유 추정하지 않는다. 각 참가자의 **독립 큰 패치 세션**에서 정한 값을
출처·세션 ID와 함께 고정한다. 독립적으로 정한 lapse의 근거가 없으면 사람
자료를 수집하지 않는다. 현재 분석기는 lapse별 민감도 분석을 구현하지 않았으므로
그 분석을 했다고 보고하지 않으며, 주 결과가 고정 lapse에 조건부라는 한계를
명시한다.

구현은 `src/psychometrics.py`, 합성 회복 검사는
`src/parameter_recovery.py`에 둔다. 실제 분석에서는 적합 수렴 여부, 음의
로그우도, 시행 수, 고정 lapse, bootstrap seed·성공률·95% CI를 모두 저장한다.

## 5. 비교 모형

- **B96:** 기준 굵기 `s_ref`의 96시행만으로 상수 C80과 공통 w를 적합한다.
- **B288:** 세 굵기의 288시행 전체로 상수 C80과 공통 w를 적합한다.
- **C96:** 기준 굵기의 96시행으로 개인 `C80_ref`와 w를 적합하고,
  `C80_C(s)=C80_ref*T_Szafir(s,u,p)/T_Szafir(s_ref,u,p)`로 예측한다.
  여기서 `u`는 동결한 Lab 방향이다. 원 식의 축별 값이 모두 `p`에 비례하므로
  이 **두 굵기의 비율에서는 p가 대수적으로 소거된다**. 구현에는 재현성을 위해
  `p=0.5`를 넣지만 이는 본 연구의 4AFC 50%나 80%가 아니며, 다른 허용 p를
  골라도 상대배율이 같음을 자동 테스트한다.
- **D288:** `C80_D(s)=C80_ref*(s/s_ref)^beta`. 288시행 전체로
  C80_ref, beta, 공통 w를 적합하며 beta 부호를 강제하지 않는다.

C96은 Szafir(2018)의 정상색각·이진 과제 절대 역치를 가져오지 않고,
0.05~0.35도에서 보고된 선 굵기별 상대 형태만 탐색적으로 전이한다. 범위 밖
외삽은 주 결과에 쓰지 않는다. B96·B288·C96·D288의 공통 `w`는 현재 시행 수에
맞춘 사전 고정 단순화 가정이다. 교정자료에서는 굵기별 C80 세 개와 공통 `w`
하나인 모형(`k=4`)을 굵기별 C80·`w`가 모두 자유인 모형(`k=6`)과 AIC로 비교한다.
둘 중 하나라도 수렴하지 않거나 자유 `w` 모형의 AIC가 더 낮으면 D288의
`beta`·굵기효과 보조 해석은 부적격으로 기록한다. 이 점검은 기준 굵기 자료만
쓰는 C96 대 B96 주 비교에는 영향을 주지 않으며, D288 예측값으로 계산 보정을
실행하는 것 자체도 금지하지 않는다. 별도로 held-out 검증자료에서도 세 세션
블록의 공통 `w` 모형과 블록별 자유 `w` 모형을 AIC로 점검한다.

## 6. 실험 전에 채우고 동결할 값

| 항목 | 현재 | 동결 근거 |
|---|---|---|
| 익명 참가자 ID 2개 | 미정 | 이름·학번과 별도 보관 |
| CVD type·Machado severity | 미정 | 독립 큰 패치 자료, 세션 ID·추정법 기록 |
| 참가자별 고정 lapse | 미정 | 독립 큰 패치 자료의 출처·세션 ID; 근거가 없으면 수집하지 않음 |
| 모니터 모델·일련 설정 | 미정 | 실험 장치 기록 |
| 화면 가로 mm·측정법·해상도 | 미정 | 발광영역 직접 측정·OS 확인 |
| OS·OS/앱 배율·Python/Tk | 미정 | native GUI가 config와 정확히 같은지 세션 직전 자동 대조 |
| 1:1 픽셀 매핑 | 미정 | native Tk/Pillow PNG의 실제 raster와 화면 캡처 확인 |
| 밝기·휘도·색관리·야간모드 | 미정 | 밝기 %는 고정한다. 계측 휘도가 없으면 null과 nominal-sRGB 장치 한정 정책을 명시하고, ACM/ICC 상태와 미측정 사유를 기록한다. |
| 관찰 거리 mm·고정 방법 | 미정 | 직접 측정, 거리표시/턱받침 등 물리적 통제 |
| 주변 조명·측정법 | 미정 | 위치·조도 또는 재현 가능한 조건 기록 |
| 배경 RGB | 미정 | 렌더링 캡처와 함께 기록 |
| 선 길이·방향·끝모양 | 미정 | 굵기 외 요소 고정 |
| 네 자극 좌표·중심 간격 | 미정 | 픽셀 좌표로 고정 |
| fixation·ITI·응답시간 | 미정 / 응답 무제한 | 응답은 반드시 무제한으로 동결한다. 시간초과 시행만 재시도하면 어려운 시행이 선택적으로 빠져 C80이 낙관 편향되기 때문이다. |
| 안티앨리어싱·renderer·pixel snapping | 미정 | 디코딩한 PNG와 1px 단위 raster로 확인 |
| Lab 중점·방향 u | 미정 | 결과 전에 하나의 색 경로 동결 |
| 교정 굵기 3개 | 미정 | 정수 px·시각도, Szafir 범위 확인 |
| 시작 색차·자극 수준 배치 | 미정 | parameter recovery로 결정 |
| 세션당 시행·휴식·중단 | 미정 | 피로와 이탈을 고려해 사전 고정 |
| feedback 정책 | 연습 `true` / 교정·검증 본 시행 `false` | 연습은 과제 이해를 돕고, 본 시행은 반복 색쌍 학습이라는 phase별 교란을 막는다. |

`src/plan_line_widths.py`로 px를 시각도로 바꾼다. 현재 FX517ZC 후보 장치의
`화면 발광영역 가로 342 mm·2560 px·관찰거리 500 mm`에서는 교정 4/8/16 px가
각각 약 0.0612/0.1225/0.2449도이고 held-out 후보 6/11 px는 약
0.0919/0.1684도다. 모두 Szafir 측정범위 안이지만, 관찰거리 통제·OS 및 앱
배율·Python/Tk 런타임·1:1 픽셀 매핑·실제 raster를 현장에서 확인한 뒤에만
최종 동결한다.
본 프로그램은 웹 브라우저가 아니라 native Tk/Pillow를 사용하므로
웹용 런타임 지표를 장치 변수로 쓰지 않는다. 가정이 하나라도 다르면 후보를
그대로 채택하지 않는다.

## 7. 자극·원자료 규칙

1. Lab 중점 `m`과 단위 방향 `u`를 고정하고 두 색을 `m±d*u/2`로 만든다.
2. 색역 밖 후보는 버린다. 8-bit sRGB 반올림 후 실제 표시된 두 RGB에 Machado
   type/severity를 적용하여 `x=post-Machado ΔE00`를 다시 계산한다.
3. 2×2 네 위치 중 홀수색 위치를 seed로 균형 무작위화한다. 기준색/홀수색의
   정체, 네 위치 RGB, 시행 순서를 저장한다.
4. 원자료에는 config version/hash, Git commit, participant/session/block/trial
   ID, phase, 실제 타임스탬프, 폭 px/도, 표시 RGB/Lab, 시뮬레이션 RGB/Lab,
   원본/시뮬레이션 ΔE00, 정답·응답·정오, 반응시간(품질 점검용), 렌더시간,
   완료/중단 사유, random seed를 저장한다.
5. type/severity 결정 자료, 모형 적합 자료, held-out 굵기 자료, 보정색 인간
   확인 자료를 섞지 않는다.
6. 원자료의 참가자 정보는 공개 저장소에 올리지 않는다. 익명 요약과 분석에
   필요한 비식별 자료 공개 여부는 동의 절차에서 따로 정한다.
7. 사람 교정 분석에는 실제 표시한 PNG 묶음의 `calibration_manifest`를, 사람
   검증 분석에는 `validation_manifest`를 반드시 함께 넣는다. 분석기는 config
   원문·canonical hash, Git commit, 참가자/CVD/severity, 시행 ID, PNG·sidecar·
   디코딩 RGB hash를 응답과 다시 대조한다. 매니페스트가 없거나 hash가 다르면
   사람 분석을 중단하며, 응답 JSONL만으로 사람 결과를 만들지 않는다.
8. 완료·중단 시도는 phase별 canonical JSONL 하나에 append-only로 남긴다.
   config의 `journal` 계약은 root를 정확히 `data/experiment/raw`로, 파일명을
   `{participant_id}-practice.jsonl`, `{participant_id}-calibration.jsonl`,
   `{participant_id}-validation.jsonl`로 고정한다. 외부 root는 지원하지 않으며
   `--journal`은 현재 phase 경로와 정확히 일치해야 한다. 파일을 바꾸거나 새
   JSONL을 골라 이전 중단·누적 시간을 숨기는 실행은 수집 전에 차단한다. GUI는
   교정과 validation 전에 공용 validator로 **전체** calibration/practice
   manifest의 288회 구조, 균형, config·renderer·PNG hash, 자극 설정을 다시
   검사한다.
9. 연습 중 중단은 연습만 다시 할 수 있다. 교정 또는 validation의 활성 자극에서
   Esc/중단이 한 번이라도 기록되면 `active_trial_abort_policy=
   invalidate-analysis-no-retry`에 따라 그 참가자 자료 전체를 확증 분석에 쓰지
   않는다. `TrialJournal`은 완료 시행과 aborted 본 시행을 terminal로 취급하고
   `remaining_trials`에서도 제외하며, GUI는 그 중단을 발견하는 즉시 실패한다.
   aborted practice만 재시도할 수 있다. 분석기는 중단 행까지 manifest와 대조한
   뒤 하나라도 있으면 실패한다.
10. 세션 최대 시간에 도달했는데 예정 block이 끝나지 않았으면
    `incomplete_block_time_limit_policy=invalidate-participant-dataset-no-retry`에
    따라 이어서 받거나 세션을 합치지 않고 참가자 자료 전체를 무효화한다.
11. 연습 1개, 교정 block 4개, validation block 3개의 `session_id`는 config에
    사전 동결하고 모두 다르게 둔다. 현재 block과 다른 ID, 같은 block을 다른
    ID로 이어 붙이는 경우, 한 ID를 다른 block에 재사용한 경우를 첫 자극 전에
    차단한다.

## 8. 시행 수와 calibration에서 분리한 held-out 검증

### 8.1 교정 단계

- 참가자별 `3 굵기 × 12 색차수준 × 4 위치 × odd 정체 2 = 288` 완료 시행이다.
- 288회는 72회씩 네 세션 블록으로 나눈다. 각 블록은 굵기당 24회,
  굵기×수준당 2회이고 전체를 합치면 모든 위치×odd 조합이 한 번씩 나온다.
- 기준 굵기는 오름차순 세 굵기의 중앙값이다. B96/C96은 이 96회만 쓰고,
  B288/D288은 288회 전부를 쓴다. B96/C96은 같은 모의 반응 난수열로 paired
  bootstrap하여 후보 굵기별 예측 C80 복제값의 대응 관계를 보존한다.
- 12개 고정 수준은 실제 참가자 응답이 아니라 동결 장치·색 경로에 결합한
  200회 가상 참가자 parameter recovery를 먼저 통과해야 한다.
- 사람 교정 모형의 모수적 bootstrap은 모형마다 **최소 1000회** 요청한다.
  각 훈련 굵기에서 C80 점추정과 bootstrap 95% CI 전체가 실제 제시 ΔE00의
  최솟값~최댓값 안에 놓이지 않으면 C80이 식별됐다고 보지 않고 분석을 중단한다.
- 교정 네 block의 시간적 비정상성은 현재 별도 모형·사전 임계값으로 자동 판정하지
  않는다. 따라서 block별 정답률 추세는 반드시 사후 품질표에 기술하고, 큰 drift를
  발견한 뒤 임계값을 새로 만들어 확증 결론을 구제하지 않는다. 이는 남은 한계다.

### 8.2 교정 후 validation plan 동결

- held-out 후보 두 개는 인접 교정 굵기의 기하평균에 가장 가까우면서 교정
  굵기와 겹치지 않는 정수 px로 **교정 응답 전에** config에 기록한다.
- 교정자료만으로 B96/C96을 적합한 뒤, 두 95% CI 사이의 보수적 로그 간격이
  가장 큰 후보를 먼저 선택한다. 그 간격이 같으면 절대 로그 C80 점추정 비가
  큰 후보, 그래도 같으면 낮은 px를 고른다. 두 후보의 보수적 간격이 모두 0이면
  점추정 비로 한 후보를 동결하되 `predicted_separability=false`와
  `not-established-before-held-out-collection`을 plan에 남긴다. 이 선택은
  불확실성을 고려해 사전 정의한 한 후보에서 검사했다는 뜻일 뿐, 실제 검증이
  두 모형을 구분한다는 보장이 아니며 다른 굵기로 일반화하지 않는다.
- B96/C96 예측값·95% CI와 w가 만드는 범위를 로그 간격 8수준으로 나눈 뒤,
  실제 8-bit RGB에서 재계산한 Machado 후 ΔE00가 엄격히 증가하고 서로 다른
  색쌍인지 확인한다. 수준 선택에는 held-out 응답을 전혀 사용하지 않는다.
- 원 experiment config는 수정하지 않는다. 원 config 원문 hash, 교정 분석파일
  hash, 후보·8수준·192 시행·seed를 별도 불변 `validation_plan`으로 묶고 자체
  SHA-256을 만든다. 이후 모든 validation 시행은 두 hash를 함께 기록한다.
- calibration은 전역 trial ID 1~288, validation은 **289~480**을 정확히 한 번씩
  사용한다. plan 생성기·compiler·GUI가 다른 시작 ID나 누락·중복 ID를 거부한다.

### 8.3 고정 블록 검증

- 선택한 한 held-out 굵기에서 `8수준 × 4위치 × odd 정체 2 = 64회` 고정
  블록을 서로 다른 세션·seed로 세 번 실시한다. 적응형 계단은 사용하지 않는다.
  이 세 블록은 calibration과 분리된 같은 참가자의 session-separated 반복
  held-out 자료이지, 서로 다른 참가자의 통계적 독립 표본이나 IID 보장이 아니다.
- 세 블록의 held-out 반응을 한 joint likelihood에 넣어 **블록별 C80 세 개와
  세션 간 공통 w 하나**를 동시에 적합한다. 모수적 bootstrap도 매 복제마다 이
  네 모수를 함께 다시 적합해 세 C80의 95% CI와 log-C80 공분산을 구한다.
- 세 C80의 `CV=SD/mean<=0.20`, 세 CI의 공통 교집합, joint fit 수렴과 bootstrap
  수렴률 0.90 이상을 모두 확인한다. 안정할 때만 joint bootstrap log-C80
  공분산을 이용한 GLS 통합 C80과 95% CI를 만든다.
- 불안정하면 세션별 값을 그대로 보고하며 단일 관측 C80이나 모형 승패를
  만들지 않는다.
- 사람 validation bootstrap은 최소 1000회다. 각 block의 C80 점추정과 95% CI,
  그리고 pooled C80 점추정과 95% CI 전체가 그 block들에 실제 제시된 ΔE00
  범위에 bracket되지 않으면 `range_bracketed=false`, `stable=false`로 두고
  `confirmed=true`를 금지한다.
- 안정한 통합 C80이 있어도 교정 B96/C96의 paired bootstrap 예측과 검증
  pooled bootstrap을 함께 전파한 `Δloss=|C96-O|-|B96-O|` 95% 구간이 0을
  포함하면 `모델 구분 불가`로 판정한다. 필요한 복제값이 없으면 승패를 만들지
  않고 `comparison_unresolvable`로 기록한다. 이는 참가자 사례 안의 예측오차
  불확실성 분석이며 모집단 유의성 검정이 아니다.
- 모수적 bootstrap은 모형과 고정 조건 아래 시행 반응이 조건부 Bernoulli 독립이라는
  가정을 쓴다. 학습·피로·순서효과·연속상관이 있으면 CI가 실제보다 좁아질 수
  있으므로 세션별 정답률 추세와 품질 메모를 함께 보고하고, bootstrap CI만으로
  이 의존성을 해결했다고 주장하지 않는다.
- `confirmed=true`는 단순 실행 플래그로 만들지 않는다. 별도 validation review
  record에 검토자 가명 ID(`R...`), 실제 검토 시각, 검토 메모, participant/config,
  raw trial, plan, stimulus manifest hash와 명시적 승인을 기록한다. 검토 시각은
  마지막 완료 시행보다 이르거나 실행 당시 UTC보다 늦을 수 없고, 분석 결과에는
  review 원문 SHA-256과 검토 내용을 함께 결박한다.

## 9. 보정 분석

한 Lab 중점·방향에서 잰 scalar C80을 임의 색 방향에 적용하지 않는다.
**연구용 주 보정**은 validation plan의 동일한 대칭 Lab 색 경로만 사용한다.

- 목적함수: `C_pair = ΔOKLab(A,A') + ΔOKLab(B,B')`.
- 제약: Machado 후 A'–B' ΔE00가 해당 모형의 held-out 예측 C80 이상.
- 실현가능성 필터는 위 `ΔE00 >= C80` 하나뿐이다. config의
  `target_absolute_tolerance_delta_e00`와 `target_relative_tolerance`는 동결
  자극 수준을 8-bit로 생성할 때만 쓰며, 보정 후보의 목표 초과량 상한이나
  C80 제약 완화로 사용하지 않는다.
- 후보: 같은 연속 Lab 중점과 방향에서 대칭인 색쌍만 허용한다. 명시한 scale
  grid를 8-bit sRGB로 양자화한 모든 유일 후보를 열거해, 제약을 만족하는 후보
  중 이동 합이 최소인 것을 고른다.
- 따라서 `minimum`은 **동결한 유한 동일경로 후보 집합 안의 정확한 최소**이며,
  연속 색공간이나 전체 팔레트의 전역 최소가 아니다.
- validation plan에 동결된 8개 원 색쌍을 모두 plan 순서대로 분석한다. 각
  수준에서 모든 모형에 같은 원 색쌍·severity·grid를 사용하고, 관측 held-out
  C80 95% CI로 `과소`, `관측범위`, `과잉`과 비용을 함께 보고한다. 사람 자료의
  단일 수준 선택과 부분 결과 저장을 금지해 사후 선택편향을 막는다.

기존 A만/B만/둘 다 radial·multi-start RGB 탐색은 임의 색쌍 소프트웨어
baseline으로 남긴다. 이는 제3색의 새 충돌 방지·기존 충돌 비악화와 영역 제안
충돌의 `unresolved` 처리를 지원하지만, 보정 후보가 측정한 색 방향과 중점에서
벗어날 수 있다. 그러므로 독립 인간 검증 없이 이 경로를 본 연구의 실제
가독성 증거로 사용하지 않는다. 실제 가독성을 주장하려면 보정 전/후 색을
**새 시행과 미사용 색쌍**으로 무작위 제시해 80% 도달을 별도로 검증해야 한다.

## 10. 분석·판정 순서

1. 중단·무효 시행 규칙에 따라 자료를 잠근다. 제외 이유와 수를 보고한다.
2. 표시 RGB에서 모든 색차와 시각도를 재계산하고 저장값과 대조한다.
3. 네 교정 session block의 session ID·완료 수·정답 수·정답률을 기술통계로
   보존한다. 학습·피로 추세의 투명한 품질 점검일 뿐 사후 제외나 유의성 검정에
   쓰지 않는다. 이어 고정 lapse로 B96, B288, C96, D288을 적합하고 수렴 실패를
   숨기지 않는다.
4. 사람 자료는 모수적 bootstrap을 1000회 이상 수행해 참가자별 파라미터·C80
   예측 95% CI를 구하고, 점추정과 CI 전체의 실제 제시범위 bracket을 확인한다.
5. 교정 굵기별 공통 `w(k=4)`와 자유 `w(k=6)` AIC를 점검한다. 비교가 수치적으로
   해결되고 공통 `w`가 유지되며 D288 적합·bootstrap이 수렴한 경우에만 D288
   beta의 점추정과 CI로 질문 1을 판정한다.
6. calibration에서 분리한 고정수준 블록 3개의 세션 간 안정성을 먼저 판정한 뒤
   held-out 관측 C80을 만든다. 세 블록은 같은 참가자의 반복 측정이다.
7. C96 대 B96의 절대 예측오차 차이를 주 결과로 제시한다. 참가자별 원값과 CI를
   모두 보이고 n=2 모집단 유의성 검정을 하지 않는다.
8. D288 대 B288과 C96 대 B288을 보조 비교로 표시한다. 교정 굵기와 held-out
   세션 각각의 공통 `w` 대 자유 `w` AIC를 가정 점검으로 보존한다. 구현·사전
   고정되지 않은 Brier score나 calibration error는 본 분석 지표로 약속하지 않는다.
9. 연구용 동일경로 보정 비용·과소·과잉 결과는 동결된 8수준 전부에서 같은
   색쌍·설정·모형 목록으로 계산한다.
10. 분석에 불리한 결과도 사전 문구대로 `미확인`, `구분 불가`, `불안정`으로
    보고한다.

## 11. 윤리·중단

- 미성년 참가자라면 보호자 동의, 참가자 본인 동의, 지도교사·학교 절차를
  먼저 확인한다.
- 참가 여부는 팀 활동·평가에 불이익을 주지 않으며 언제든 이유 없이 중단할 수
  있음을 서면으로 알린다.
- 세션 최대 시간, 정기 휴식, 눈 피로·두통·불편 시 즉시 중단 기준을 동결한다.
- 이름·연락처·학번, 원 CVD 검사 결과를 공개 Git에 넣지 않는다. 접근자,
  암호화 위치, 보관 기간, 폐기 시점을 동의서에 적는다.
- 이 과제는 의학적 진단이 아니다.

## 12. 단계별 시작 게이트

### 12.1 교정자료 수집

다음이 모두 충족되어야 `human_data_collection_allowed=true`인 참가자별 1.0
설정으로 **연습과 교정자료만** 수집할 수 있다.

1. 이 문서의 `미정`이 모두 실제값으로 채워짐.
2. 자극 화면·응답·중단·JSONL 저장 프로그램과 raster 캡처가 검증됨.
3. 실제 자료용 1.0 schema가 표시·응답·provenance 필드를 포함함.
4. 실제 동결 config를 `--experiment-config`로 결합한 200회 합성 parameter
   recovery가 사전 오차·수렴 기준을 통과함. 기본 후보 수준으로 돌린 대표
   recovery는 개발 점검일 뿐 이 게이트를 열지 못함.
5. 별도 고유 trial ID의 6~10회 연습에서 위치, 키 입력, 색역, 중단·저장
   재개가 정상임.
6. 동의·보관·중단 절차가 승인됨.
7. Git commit, config 원문 hash, renderer 구현 hash, calibration/practice
   manifest와 PNG hash를 동결함.
8. config-bound parameter recovery의 `implementation_hashes`가 분석 코드,
   schema와 잠금 환경 전체의 현재 SHA-256과 정확히 일치하고, 호환용
   `implementation_sha256`도 같은 전체 map이며, 자극 매니페스트가 기록한
   renderer hash도 현재 코드와 일치함.

### 12.2 validation 자료 수집

교정 게이트에 더해 다음이 모두 충족되어야 한다.

1. 288 완료 시행이 계약과 균형 검사를 통과하고 네 모형 분석이 수렴함.
2. held-out 응답을 쓰지 않은 별도 validation plan이 원 config와 교정 분석
   원문 hash에 결합되어 동결됨.
3. 8수준×3블록의 192 PNG가 validation plan과 renderer hash 검사를 통과함.
4. validation 응답 GUI가 plan hash를 매 시행 기록하고 dry-run을 통과함.
5. 실제 동결 validation plan의 양자화 8수준과 hash에 결합한 200회×bootstrap
   50회 validation recovery가 사전 기준을 통과함. 대표 후보 수준에서의 개발
   recovery 결과만으로는 validation 수집을 시작하지 않음.
6. validation plan과 recovery의 `implementation_hashes` 키 집합·SHA-256이
   현재 코드·schema·잠금 파일과 정확히 일치함. 하나라도 변경되면 원자료를
   보존하고 해당 산출물 생성부터 다시 수행함.

Parameter recovery의 사전 통과 기준은 simulation 200회와 대표 bootstrap
200회 이상이다. 동결한 12수준에서 유도한 low-steep/central/high-shallow
profile 각각과 전체에서 전 모형 수렴률 0.95 이상, 기준 및 held-out C80 중앙
절대상대오차 0.20 이하·90백분위 오차 0.50 이하를 요구한다. D288은 beta 부호
회복률 0.80 이상, beta 중앙 절대오차 0.25 이하·90백분위 오차 0.65 이하도
요구한다. 대표 B96·B288·C96·D288 bootstrap은 수렴률 0.80 이상이고 알려진
기준 C80(및 D288 beta)을 95% CI에 포함해야 하며, B96/C96 기준굵기 paired
bootstrap의 입력·난수열·표본 동일성 검사도 통과해야 한다. 이는
임상적 기준이 아니라 현재 시행 수로 분석기가 결론에 필요한 방향과 대략적
크기를 되찾는지 막는 **공학적 시작 게이트**다. 실제 반응을 보기 전에 고정하며,
하나라도 실패하면 시행 수·자극 수준을 다시 설계하고 사람 자료를 받지 않는다.

준비도는 `python research_readiness.py --format markdown`의
`ready_for_calibration_collection`과 `ready_for_validation_collection`을 각각
확인한다. 준비도 검사 자체가 윤리 승인이나 실제 장치 점검을 대신하지 않는다.

## 13. 허용되는 결과 문장

- `참가자 P__의 beta 95% CI가 0 아래여서 이 고정 조건에서는 굵기가
  증가할수록 C80이 낮아지는 방향을 확인했다.`
- `교정 예측과 held-out 관측의 bootstrap 불확실성을 함께 전파한 Δloss 95%
  구간이 0을 포함해 모형 우열을 구분하지 못했다.`
- `제안 보정은 계산상 목표 ΔE00 제약을 만족했으나 실제 가독성 향상은 별도
  인간 검증을 하지 않아 결론내리지 않았다.`

금지 문장: `모든 적록색각이상자에게 효과가 있다`, `FAST/ODP나 AI를 이겼다`,
`Machado 화면이 참가자가 보는 것과 같다`, `전역 최소를 찾았다`, `0.62가 실제
이미지의 혼동 기준이다`, `사람 검증 없이 가독성이 향상됐다`.

## 참고문헌

- Machado, Oliveira, & Fernandes (2009), *A Physiologically-based Model for
  Simulation of Color Vision Deficiency*. https://doi.org/10.1109/TVCG.2009.113
- Luo, Cui, & Rigg (2001), *The Development of the CIE 2000 Colour-Difference
  Formula*. https://doi.org/10.1002/col.1049
- Sharma, Wu, & Dalal (2005), *The CIEDE2000 Color-Difference Formula:
  Implementation Notes, Supplementary Test Data, and Mathematical
  Observations*. https://doi.org/10.1002/col.20070
- Szafir (2018), *Modeling Color Difference for Visualization Design*.
  https://doi.org/10.1109/TVCG.2017.2744359
- Wichmann & Hill (2001), *The psychometric function II: bootstrap-based
  confidence intervals and sampling*. https://doi.org/10.3758/BF03194545
- Prins (2012), *The psychometric function: the lapse rate revisited*.
  https://doi.org/10.1167/12.6.25
