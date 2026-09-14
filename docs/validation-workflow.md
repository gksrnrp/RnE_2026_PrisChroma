# Calibration에서 분리한 held-out 고정수준 검증 절차

이 문서는 교정 단계가 끝난 뒤 검증 반응을 보기 전에 계획을 동결하고, 서로
독립인 세 블록을 분석하는 절차를 고정한다. 적응형 staircase는 사용하지 않는다.

## 1. 계획 동결

입력은 원 `experiment_config`와 `calibration_model_fits` 두 파일뿐이다.
`validation_design.py`는 calibration-only B96/C96가 두 후보 굵기에서 예측한
C80 95% CI 사이의 보수적 로그 간격을 먼저 최대화한다. 동률이면 절대 로그 C80
점추정 비를 최대화하고, 그래도 같으면 더 낮은 px를 고른다. 두 후보의 CI 간격이
모두 0이면 점추정 비로 한 후보를 동결하지만 plan에
`predicted_separability=false`와
`pre_validation_discriminability=not-established-before-held-out-collection`을
남긴다. 이는 불확실성을 고려해 사전 정의한 한 굵기에서 검증했다는 뜻이며,
검증 후에도 두 모형이 구분되지 않을 수 있고 결과를 다른 굵기로 일반화하지 않는다.

선택 후보에서 B96/C96의 C80 95% 구간과 기울기 `w`를 사용한다. 각 모형에 대해
다음 하한·상한을 만들고, 모든 하한의 최솟값과 상한의 최댓값 사이를 8개
로그 등간격으로 나눈다.

```text
lower_m = CI_low,m  × exp(-log(9) × w_m)
upper_m = CI_high,m × exp( log(9) × w_m)
```

이는 실제 정답률 10~90%를 뜻하지 않는다. `log(9)`는 C80을 중심으로 한
**log 색차 축의 고정 폭**이다. 예를 들어 lapse=0.02일 때 점추정 C80 기준
양 끝의 예상 실제 정답률은 대략 0.44와 0.95이고, 정확한 값은 plan의 모형별
metadata에 계산해 저장한다. 이 범위는 direct-C80 곡선의 목표점 양쪽을 넓게
덮으면서 두 모형의 교정 불확실성도 포함하기 위한 고정 규칙이다. 각 목표는
실제 8-bit sRGB 색쌍으로 만든 뒤 Machado 후 ΔE00를 재계산한다. 다음 중 하나면
계획 전체를 만들지 않는다.

- 목표 오차가 동결 config의 절대·상대 허용오차보다 큼
- 서로 다른 수준이 같은 RGB 색쌍으로 양자화됨
- 실제 Machado 후 ΔE00가 수준 순서대로 엄격히 증가하지 않음

각 블록은 `8수준 × 정답 위치 4개 × odd 색 정체 A/B = 64회`의 완전 요인
배치다. 세 블록은 서로 다른 seed로 독립 무작위화하며 trial ID는 교정 288회
다음인 289부터 480까지 전역 유일하다.

```powershell
cd src
python validation_design.py `
  --config ../data/experiment/private/experiment_config.json `
  --calibration-fits ../results/calibration_model_fits.json `
  --output ../data/experiment/validation/validation-plan.json `
  --seed 20260914 `
  --first-trial-id 289
```

`plan_sha256`은 정렬된 compact JSON에서 최상위 `plan_sha256`과 각 schedule의
`validation_plan_sha256`을 뺀 canonical preimage의 SHA-256이다. 원 config raw
SHA와 calibration artifact raw SHA도 함께 저장한다. 따라서 교정 후 config를
몰래 바꾸거나 검증 수준·시행표를 바꾸면 분석이 차단된다.

계획의 canonical preimage에는 `implementation_hashes`도 들어간다. 이 map은
`validation_design.py`뿐 아니라 실제 예측·색쌍 생성에 사용된 모듈, 교정
산출물 schema와 `requirements-lock.txt`를 명시적으로 포함한다. 누락 키·예상
밖 키·현재와 다른 hash를 하나라도 발견하면, plan hash를 다시 계산했더라도
검증 계획을 거부한다. 검증 분석과 recovery도 각각 자기 계산 의존성 전체를
같은 방식으로 기록하고 다음 단계에서 fail closed한다.

## 2. 세 블록 분석

계획을 만든 직후, 사람에게 실제 표시할 192개 PNG와 매니페스트를 같은 디렉터리에
컴파일한다. `validation-manifest.json`의 각 시행, PNG, sidecar와 디코딩 RGB가
계획·config·교정 적합 hash에 결합된다.

```powershell
python stimulus_renderer.py compile-validation `
  --config ../data/experiment/private/experiment_config.json `
  --calibration-fits ../results/calibration_model_fits.json `
  --validation-plan ../data/experiment/validation/validation-plan.json `
  --manifest ../data/experiment/validation/validation-manifest.json `
  --png-dir ../data/experiment/validation
```

응답 JSONL은 완료·중단 시도를 모두 append-only로 보존한다. 중단된 시도도
config·plan·manifest 및 PNG hash와 일치해야 하며 감사 대상에서 빼지 않는다.
`analyze_validation.py`는 이 전체 기록을 검증한 뒤, 동결 계획·매니페스트의
192개 시행과 정확히 일치하는 **완료 시행만** 적합에 사용한다. 사람 자료에서
`--validation-manifest`를 빼거나 응답과 PNG hash가 다르면 분석을 거부한다.
held-out 완료 자료만 사용해 블록별 direct C80 세 개와 세션 간 공통 `w` 하나를
joint 최대우도 적합한다. calibration의 `w`를 가져오지 않는다. 모수적
bootstrap도 매 복제마다 네 모수를 모두 joint 재적합해 C80/w 95% 구간을
구한다. 세 블록은 하나의 session으로 합치지 않고 각각 고유한 `session_id`를
가져야 한다.

재현성 기준은 사전에 다음처럼 고정한다.

- 세 C80의 표본 변동계수(CV) ≤ 0.20
- 세 C80 95% 구간에 공통 교집합이 있음
- 각 적합 수렴, 각 bootstrap 수렴률 ≥ 0.90

위 세 조건을 모두 만족할 때만 joint bootstrap `log(C80)` 공분산을 사용한
일반화 최소제곱 역분산 가중 통합 C80와 95% 구간을 만든다. 하나라도 만족하지
않으면 `stable=false`, `pooled_c80=null`로 기록하고 통합값을 만들지 않는다.
보조 민감도 분석으로 block별 자유 `w` 3개(k=6)와 공통 `w`(k=4)의 AIC를
비교한다. 자유 `w`가 더 낮은 AIC를 보이면 표본 부족 때문에 모형을 바꾸지 않고
가정 경고와 `shared_w_sensitivity_warning`에 따른 `indeterminate`를 보고한다.

안정적 통합값이 있을 때 주 비교는 같은 96회 자료량의 B96 대 C96이다. 보조
비교는 D288 대 B288과 C96 대 B288이다.
각 예측의 절대 오차 `|예측 C80 - 관측 pooled C80|`를 기록한다. 다만 B96와
C96은 같은 모의 교정 반응으로 짝지은 bootstrap 예측 복제값을 보존한다. 이를
held-out joint bootstrap의 pooled C80 복제값과 결합해
`Δloss=|C96-O|-|B96-O|`의 95% 구간을 계산한다. 구간 전체가 0 아래일 때만
C96, 0 위일 때만 B96의 오차가 작다고 판정하고 0을 걸치면
우열을 확인하지 않는다. 구현 결과는 사실상 0인 사전 허용띠 안이면
`equivalent_within_tolerance`, 그 밖의 경계 중첩이면 `indeterminate`다.
복제값·pairing 표지가 없으면 점추정으로 대체하지 않고
`comparison_unresolvable`로 닫는다. 검증 반응을 본 뒤 모델, 굵기, 수준을 다시
고르지 않는다.

```powershell
python analyze_validation.py `
  --config ../data/experiment/private/experiment_config.json `
  --calibration-fits ../results/calibration_model_fits.json `
  --validation-plan ../data/experiment/validation/validation-plan.json `
  --validation-manifest ../data/experiment/validation/validation-manifest.json `
  --trials ../data/experiment/raw/P01-validation.jsonl `
  --bootstrap-replicates 1000 `
  --bootstrap-seed 20260915 `
  --review-record ../data/experiment/private/P01-validation-review.json `
  --output ../results/participant_thresholds.json
```

사람 자료의 `confirmed=true`는 파일과 수치를 사람이 검토한 뒤 별도 review
record를 제공했을 때만 만든다. review JSON에는 `schema_version=1.0`,
`review_type=human_validation_confirmation`, `R` 접두 검토자 가명 ID,
`reviewed_at_utc`, 비식별 `review_note`, `approved_for_confirmation=true`와
participant/config/raw trial/plan/manifest hash를 모두 기록한다. 검토 시각은
마지막 완료 시행 이후이며 현재 UTC보다 늦지 않아야 한다. 분석 결과에는 review
원문 SHA-256이 남는다. `--confirm-human-after-review` 단독 사용은 차단된다.
합성 자료는 `--allow-synthetic`을 주어야 분석되며, 결과에는 항상
`research_conclusion_allowed=false`가 남고 confirmed로 승격할 수 없다.

## 3. 해석 범위

계획 단계의 95% 구간 비중첩 여부는 **검증 전에 두 모형을 구분할 가능성이
있었는지**를 기록할 뿐 우월성 결론이나 검증 성공 보장이 아니다. 사전
`predicted_separability=false`인 후보를 계획대로 검증한 뒤에도 `모델 구분 불가`가
정상 결과일 수 있다. 실제 검증에서는 블록별 관측
C80의 불확실성을 포함해 B96/C96 예측 오차를 비교해야 한다. 선택된 한 굵기와
고정 Lab 중점·방향·표시 조건 밖으로 결과를 일반화하지 않는다.

## 4. 검증 설계 parameter recovery

`validation_recovery.py`는 동결 plan의 실제 양자화 8수준에 가상 반응을 만들고,
매 복제에서 block C80 세 개와 공통 `w`를 다시 joint 적합한다. 사람 자료를
대체하지 않으며 결과는 항상 `data_origin=synthetic`,
`research_conclusion_allowed=false`다.

다음 수치는 **대표 후보 수준으로 분석기만 점검한 개발 recovery**다. 대표 후보
수준 `[0.60, 0.85, 1.20, 1.70, 2.40, 3.40, 4.80, 6.80]`, lapse
0.02, 교정 B96/C96 95% 구간을 모사한 true C80 1.76~3.808 및 true `w`
0.272~0.432에서 200명×3블록, block bootstrap 20회 recovery를 실행했다.
joint fit 수렴률 100%, bootstrap 수렴률 100%, stable 비율 70.0%, pooled C80
중앙 상대오차 7.71%, stable 140명에서 95% CI coverage 92.14%였다. 사전 최소
gate(수렴 95%, stable 70%, 중앙 상대오차 20%, coverage 80%)를 모두 통과했다.
단 stable 70.0%는 기준과 정확히 같아 여유가 없다. 이 기준을 결과에 맞춰
낮추지 않으며, 최종 plan에서 다시 실패하면 validation 수집을 중단하고 시행
수를 재설계한다. 현재 저장소에는 이 대표 실행의 독립 원산출물 파일이 없으므로
위 숫자는 개발 기록이지 최종 게이트 증거가 아니다.

이 수치는 **최종 사람 plan에 결합된 결과가 아니다.** 실제 교정 적합 후 plan을
동결하면 다음 명령으로 정확한 양자화 수준과 해시에 결합해 기본 200명,
bootstrap 50회로 다시 실행해야 한다.
`research_readiness.py`도 현재 plan/config/구현 hash에 묶인 200명×bootstrap
50회 통과 파일이 없으면 validation 수집을 차단한다.

```powershell
python validation_recovery.py `
  --config ../data/experiment/private/experiment_config.json `
  --validation-plan ../data/experiment/validation/validation-plan.json `
  --repetitions 200 `
  --bootstrap-replicates 50 `
  --seed 20260916 `
  --output ../results/validation_recovery.json
```

이 최종 plan-bound recovery와 validation 매니페스트의 hash 검사가 모두 통과한
뒤 `python research_readiness.py --format markdown`에서
`ready_for_validation_collection=true`인지 확인한다. 정확한 GUI dry-run과 실제
세션 명령은 [이창민 실행서](changmin-runbook.md)를 따른다.
