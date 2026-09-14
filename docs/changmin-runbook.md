# 이창민 최종 실행서

이 문서는 현재 코드와 [사전 고정 프로토콜](experiment-protocol.md)을 기준으로,
이창민이 **준비 확인 → 교정 288회 → 검증 64회×3 → C80 분석 → 동일경로 보정**을
순서대로 실행하는 인계 문서다. 명령은 저장소의 `src/` 폴더에서 실행한다.

현재는 소프트웨어·합성 검증 단계다. 실제 장치값, 참가자별 1.0 config, 동의와
사람 응답이 아직 없으므로 아래 사람 세션을 지금 임의로 실행하거나 가짜 값을
채워 연구 결과를 만들면 안 된다.

## 1. 연구에서 이창민이 맡는 범위

1. 동결 config·시행표·PNG 매니페스트가 서로 같은지 검사한다.
2. 교정 288회 응답을 B96/B288/C96/D288로 적합한다.
3. 응답을 보지 않은 held-out 검증 계획과 192개 PNG를 만든다.
4. calibration과 분리해 서로 다른 세션·seed로 실시한 64회 반복 블록 세 개에서
   C80 세 개와 공통 `w`를 joint 적합하고, 안정할 때만
   joint bootstrap 공분산을 이용한 GLS 통합 C80을 만든다.
5. 최종 C80을 같은 Lab 중점·방향의 유한 경로 보정에 연결한다.
6. 계산상 제약 충족과 실제 사람 가독성 향상을 구분해 보고한다.

사람 모집·동의·CVD 자료 출처는 한정훈, 실제 화면 물리값과 raster 현장 확인은
정인교가 준비하되, 이창민은 파일을 받았다는 이유만으로 통과 처리하지 않고 아래
게이트를 직접 확인한다.

## 2. 시작 전에 받아야 하는 것

| 담당 | 받아야 할 것 | 통과 조건 |
|---|---|---|
| 한정훈 | 익명 참가자 ID, CVD type, Machado severity와 출처 세션, 고정 lapse와 출처 세션 | 이름·학번·연락처와 분리되고 근거가 기록됨 |
| 한정훈·지도교사 | 동의·철회·중단·보관 계획 | `privacy.approved=true`를 사실대로 쓸 수 있음 |
| 정인교 | 모니터 발광영역 가로 mm와 측정법, 실제 해상도, OS·OS/앱 배율, 운영체제·Python/Tk, native Tk/Pillow 1:1 픽셀 확인, 관찰거리와 고정법, 계측 휘도와 측정법, 색관리·야간모드, 주변 조명과 기록법, 실제 raster 캡처 | 현장에서 직접 확인한 값이며 config와 세션 런타임에서 정확히 대조 가능 |
| 공동 | 한 Lab 중점·방향, 배경, 선 길이·간격·cap, 노출/ITI, 교정 굵기 3개, held-out 후보 2개, 12수준, 연습 6~10회 | 사람 응답을 보기 전에 문서와 config에 고정 |

자동 조회를 300 mm 화면·500 mm 거리라고 가정했을 때의 5/10/20 px와 7/14 px는
후보일 뿐이다. 직접 측정값이 다르면 다시 계산한다. Szafir 측정 시각도 범위
0.05~0.35도 밖인 굵기를 주 분석에 쓰지 않는다.

고정 lapse는 독립 큰 패치 자료와 세션 ID로 근거를 남겨야 한다. 현재 분석
파이프라인에는 lapse별 민감도 분석이 없으므로 출처 없는 임의값으로 대신하지
않고, 독립 근거가 없으면 사람 자료 수집을 시작하지 않는다.

참가자별 config는 `data/experiment/private/experiment_config.json`에 두고
`schemas/human/experiment_config.schema.json` 1.0을 만족하게 만든다. 기존
`data/contracts/*.example`은 합성 software baseline용 0.1-draft이므로 버전
문자만 바꿔 사람 config로 재사용하지 않는다. `sampling.practice_trials`는
6~10 중 하나를 실험 전에 고정한다. 새 설정은
`data/contracts/human_experiment_config.template.json`의 구조에서 시작하되,
이 파일의 모든 `PLACEHOLDER`와 숫자는 형식 예시일 뿐 실제 후보값이 아니다.
`status=template`, `human_data_collection_allowed=false`, `privacy.approved=false`인
동안 준비도 검사는 반드시 차단되어야 한다.

`journal`은 다음 원칙으로 참가자 config에 함께 동결한다.

```json
{
  "storage_policy": "repository-relative-gitignored",
  "root_repository_relative": "data/experiment/raw",
  "phase_files": {
    "practice": "P01-practice.jsonl",
    "calibration": "P01-calibration.jsonl",
    "validation": "P01-validation.jsonl"
  },
  "session_ids": {
    "practice": "P01-PRACTICE-01",
    "calibration": {"1": "P01-CAL-01", "2": "P01-CAL-02", "3": "P01-CAL-03", "4": "P01-CAL-04"},
    "validation": {"V01": "P01-VAL-01", "V02": "P01-VAL-02", "V03": "P01-VAL-03"}
  }
}
```

현재 구현은 외부 보안 root를 허용하지 않는다. `--journal`은 config의 해당
phase 파일과 정확히 같아야 하며, 파일명이나 경로를 바꿔 새 이력을 시작할 수
없다. 세션 ID 여덟 개도 모두 달라야 하고 severity/lapse 세션과 겹치면 안 된다.

## 3. 두 참가자 운영 원칙

현재 준비도 검사는 canonical 파일 한 세트, 즉 한 명의
`results/calibration_model_fits.json`, `validation-plan.json`,
`participant_thresholds.json`을 기준으로 작동한다. 따라서 P01과 P02를 섞지
않고 다음처럼 **참가자 한 명씩 끝낸다**.

1. P01 config로 아래 전 과정을 완료한다.
2. 공개 가능한 비식별 산출물만 `results/participants/P01/`에 복사해 보관하고,
   비공개 원자료·config는 공개 Git 밖의 승인된 보관 위치에 둔다.
3. P02의 canonical config와 산출물을 새로 만들고 같은 전 과정을 반복한다.
4. 두 참가자 결과를 마지막 표에서 나란히 제시하되 모집단 검정이나
   protan/deutan 유형 차이로 해석하지 않는다.

## 4. 0단계 — 코드와 작업 상태 고정

먼저 저장소의 추적 파일 변경을 팀과 합쳐 commit한다. 응답 GUI는 실행 코드와
스키마가 Git에 포함되어 있고 추적 파일이 HEAD와 같을 때만 열린다. commit 뒤
코드·스키마·프로토콜을 바꾸면 이전 시행표와 PNG는 폐기하고 새 commit으로
다시 생성한다.

```powershell
cd <저장소를 복제한 경로>\src
pip install -r ../requirements-lock.txt
python -m unittest -q
git diff --quiet HEAD --
git rev-parse HEAD
```

`git diff --quiet`의 종료 코드가 0이 아니면 사람 세션을 시작하지 않는다. 이때
변경을 강제로 지우지 말고 팀과 검토·commit한 뒤 다시 검사한다.

## 5. 1단계 — config와 장치 게이트

아래 폴더를 준비한다. `private/`와 `raw/`에는 개인정보나 원응답이 있으므로
공개 저장소에 올리지 않는다.

```text
data/experiment/private/
data/experiment/raw/
data/experiment/calibration/
data/experiment/practice/
data/experiment/validation/
results/
```

실제 측정값으로 px를 시각도로 다시 확인한다.

```powershell
python plan_line_widths.py --display-width-mm <실측가로mm> --resolution-px <실제가로해상도> --viewing-distance-mm <실측거리mm> --widths-px <교정폭1> <교정폭2> <교정폭3> <후보폭1> <후보폭2>
python experiment_runner.py --check-config ../data/experiment/private/experiment_config.json
```

둘 중 하나라도 경고·차단이면 다음 단계로 가지 않는다. GUI를 켠 실제 PC에서
해상도, 배율과 화면 치수도 config와 다시 같아야 한다.

## 6. 2단계 — 대표 recovery와 최종 config-bound recovery 구분

아래 첫 명령은 후보 수준에서 분석 코드가 작동하는지 보는 **대표 개발 점검**이다.
사람 수집을 허용하지 않는다.

```powershell
python parameter_recovery.py --simulation-replicates 200 --bootstrap-replicates 200 --output ../results/parameter_recovery.synthetic.json
```

실제 교정 시작 게이트는 동결한 참가자 config의 굵기·Lab 방향·12수준·lapse에
결합한 다음 결과다.

```powershell
python parameter_recovery.py --experiment-config ../data/experiment/private/experiment_config.json --simulation-replicates 200 --bootstrap-replicates 200 --seed 20260914 --output ../results/parameter_recovery.config-bound.json
```

동결 12수준에서 유도한 low/central/high profile 각각의 수렴률·중앙오차·90백분위
오차, 대표 네 모형 bootstrap의 알려진 C80/beta CI 포함, 같은 96회를 쓰는
B96/C96 paired-bootstrap 동일성을 모두 확인한다. simulation과 bootstrap이
각각 200회 미만이면 수치가 좋아도 `passed=false`다. 실패하면 사람에게 보여주지
말고 수준·시행 설계를 다시 검토한다.

## 7. 3단계 — 교정 시행표·PNG·연습 PNG 만들기

현재 commit을 시행표에 넣고, 교정 288회와 연습 자극을 컴파일한다. 매니페스트와
PNG를 같은 디렉터리에 두면 매니페스트의 상대 경로가 안전하게 유지된다.

```powershell
$rneCommit = git rev-parse HEAD
python experiment_runner.py --make-schedule ../data/experiment/private/experiment_config.json --seed 20260914 --git-commit $rneCommit --output ../data/experiment/calibration/calibration-schedule.json
python stimulus_renderer.py compile --config ../data/experiment/private/experiment_config.json --schedule ../data/experiment/calibration/calibration-schedule.json --manifest ../data/experiment/calibration/calibration-manifest.json --png-dir ../data/experiment/calibration
python stimulus_renderer.py compile-practice --config ../data/experiment/private/experiment_config.json --calibration-manifest ../data/experiment/calibration/calibration-manifest.json --manifest ../data/experiment/practice/practice-manifest.json --png-dir ../data/experiment/practice
python research_readiness.py --format markdown
```

여기서는 `ready_for_calibration_collection=true`만 확인한다.
`ready_for_research_claims=false`는 사람 결과 전에는 정상이다.
`--strict-exit`는 모든 연구 차단까지 함께 검사하므로 교정 시작 여부만 확인할 때
사용하지 않는다.

## 8. 4단계 — GUI dry-run과 실제 연습

dry-run은 전체 PNG·sidecar·해시·화면·Git 상태를 검사하지만 GUI를 열거나 응답을
저장하지 않는다.

```powershell
python response_gui.py --config ../data/experiment/private/experiment_config.json --manifest ../data/experiment/practice/practice-manifest.json --phase practice --calibration-manifest ../data/experiment/calibration/calibration-manifest.json
python response_gui.py --config ../data/experiment/private/experiment_config.json --manifest ../data/experiment/calibration/calibration-manifest.json --phase calibration --session-block 1
```

두 보고서의 `technical_readiness_passed=true`를 확인한다. 실제 연습은 해당 세션의
동의를 다시 확인한 뒤에만 다음처럼 실행한다.

```powershell
python response_gui.py --config ../data/experiment/private/experiment_config.json --manifest ../data/experiment/practice/practice-manifest.json --phase practice --calibration-manifest ../data/experiment/calibration/calibration-manifest.json --journal ../data/experiment/raw/P01-practice.jsonl --session-id P01-PRACTICE-01 --run-gui --confirm-consent
```

위치·키 입력·색역·중단·저장을 확인한다. 연습 중단만 재시도할 수 있다. 연습 반응은 교정 288회나 분석에
합치지 않는다.

## 9. 5단계 — 교정 72회×4세션

서로 다른 날짜에 config에 미리 동결한 session code로 block 1~4를 각각
실행한다. 같은 canonical journal에 완료·중단 시도가 모두 append-only로 남고,
완료 또는 aborted 본 시행은 terminal이라 중복·재제시가 거부된다. aborted
practice만 같은 canonical practice journal에서 재시도할 수 있다.

```powershell
python response_gui.py --config ../data/experiment/private/experiment_config.json --manifest ../data/experiment/calibration/calibration-manifest.json --phase calibration --session-block 1 --journal ../data/experiment/raw/P01-calibration.jsonl --session-id P01-CAL-01 --run-gui --confirm-consent
python response_gui.py --config ../data/experiment/private/experiment_config.json --manifest ../data/experiment/calibration/calibration-manifest.json --phase calibration --session-block 2 --journal ../data/experiment/raw/P01-calibration.jsonl --session-id P01-CAL-02 --run-gui --confirm-consent
python response_gui.py --config ../data/experiment/private/experiment_config.json --manifest ../data/experiment/calibration/calibration-manifest.json --phase calibration --session-block 3 --journal ../data/experiment/raw/P01-calibration.jsonl --session-id P01-CAL-03 --run-gui --confirm-consent
python response_gui.py --config ../data/experiment/private/experiment_config.json --manifest ../data/experiment/calibration/calibration-manifest.json --phase calibration --session-block 4 --journal ../data/experiment/raw/P01-calibration.jsonl --session-id P01-CAL-04 --run-gui --confirm-consent
```

교정 활성 자극 중 Esc/중단이 한 번이라도 기록되면 재개·재시도하지 않는다.
기존 중단 행을 삭제·수정하거나 새 trial ID로 바꾸지 말고, 이 참가자 자료 전체를
확증 분석에서 제외한다. 세션 최대 시간에 block을 완료하지 못한 경우에도 이어서
합치지 않고 참가자 자료를 무효화한다. 두 규칙은 config의
`active_trial_abort_policy`와 `incomplete_block_time_limit_policy`에 사전 고정된다.

## 10. 6단계 — 교정 288회 분석

사람 분석에서는 `--calibration-manifest`가 필수다. JSONL만 받은 상태에서는
분석하지 않는다.

```powershell
python analyze_calibration.py --config ../data/experiment/private/experiment_config.json --trials ../data/experiment/raw/P01-calibration.jsonl --calibration-manifest ../data/experiment/calibration/calibration-manifest.json --bootstrap-replicates 1000 --bootstrap-seed 20260914 --output ../results/calibration_model_fits.json
```

출력은 교정 전용 중간 산출물이다. B96/C96은 중앙 굵기의 같은 96회, B288/D288은
전체 288회를 사용했는지, 네 block 균형과 bootstrap 수렴을 확인한다. B96/C96은
같은 모의 반응 난수열로 paired bootstrap하고 선택 후보별 C80 복제값을 보존한다. 아직
held-out 관측 C80이 없으므로 이 파일만으로 모델 승패나 사람 역치를 확정하지
않는다.
각 훈련 굵기에서 C80 점추정과 95% CI 전체가 실제 제시 ΔE00 범위 안에 있는지도
확인한다. 하나라도 범위 밖이면 수준 범위가 역치를 식별하지 못한 것이므로 다음
단계로 가지 않는다. 교정 block별 정답률 drift는 별도 자동모형이 없는 한계로
`calibration.session_block_quality`의 session ID·완료 72회·정답 수·정답률로
기술한다. 이 값은 학습·피로 가능성을 투명하게 보이는 기술통계일 뿐, 사후
제외 기준이나 유의성 검정으로 쓰거나 결론을 구제하는 데 쓰지 않는다.

`width_slope_sensitivity`에서 굵기별 C80+공통 `w`(`k=4`)와 굵기별 자유
C80+`w`(`k=6`)의 AIC·수렴 상태를 확인한다. `comparison_resolved=false`이거나
`free_w_preferred=true`이면 `d288_beta_interpretation_eligible=false`여야 하며,
D288의 beta·굵기효과 보조 결론은 내리지 않는다. 이는 C96 대 B96 주 비교에는
영향을 주지 않고, D288 예측 C80으로 계산 보정을 실행하는 것 자체도 막지 않는다.
이 필드는 validation 뒤 최종 `participant_thresholds.json`에도 그대로 보존된다.

## 11. 7단계 — 검증 plan·최종 recovery·192 PNG

검증 반응을 보기 전에 두 후보 중 B96/C96 예측 95% CI의 보수적 로그 간격이
가장 큰 굵기를 먼저 고르고, 동률이면 절대 로그 C80 점추정 비, 다시 동률이면
낮은 px를 고른다. 두 후보 간격이 모두 0이면 점추정으로 한 굵기를 동결하되
plan의 `predicted_separability=false`와 구분 가능성 미확인 표지를 확인한다.
이 경우에도 계획대로 검증하며, 검증 후 `모델 구분 불가`는 허용되는 결과다.
8수준과 64회×3 시행을 동결하고 validation 전역 trial ID는 289~480으로 고정한다.

```powershell
python validation_design.py --config ../data/experiment/private/experiment_config.json --calibration-fits ../results/calibration_model_fits.json --seed 20260914 --first-trial-id 289 --output ../data/experiment/validation/validation-plan.json
python validation_recovery.py --config ../data/experiment/private/experiment_config.json --validation-plan ../data/experiment/validation/validation-plan.json --repetitions 200 --bootstrap-replicates 50 --seed 20260916 --output ../results/validation_recovery.json
python stimulus_renderer.py compile-validation --config ../data/experiment/private/experiment_config.json --calibration-fits ../results/calibration_model_fits.json --validation-plan ../data/experiment/validation/validation-plan.json --manifest ../data/experiment/validation/validation-manifest.json --png-dir ../data/experiment/validation
python research_readiness.py --format markdown
```

대표 후보 수준에서 예전에 통과한 recovery는 여기서 쓸 수 없다. 방금 만든
plan의 실제 양자화 8수준과 hash에 결합된 `validation_recovery.json`이 사전
기준을 통과하고 `ready_for_validation_collection=true`여야 한다.

검증 GUI dry-run도 세 block마다 한다.

```powershell
python response_gui.py --config ../data/experiment/private/experiment_config.json --manifest ../data/experiment/validation/validation-manifest.json --phase validation --validation-block V01 --validation-plan ../data/experiment/validation/validation-plan.json --calibration-model-fits ../results/calibration_model_fits.json
python response_gui.py --config ../data/experiment/private/experiment_config.json --manifest ../data/experiment/validation/validation-manifest.json --phase validation --validation-block V02 --validation-plan ../data/experiment/validation/validation-plan.json --calibration-model-fits ../results/calibration_model_fits.json
python response_gui.py --config ../data/experiment/private/experiment_config.json --manifest ../data/experiment/validation/validation-manifest.json --phase validation --validation-block V03 --validation-plan ../data/experiment/validation/validation-plan.json --calibration-model-fits ../results/calibration_model_fits.json
```

## 12. 8단계 — calibration에서 분리한 held-out 검증 64회×3세션

```powershell
python response_gui.py --config ../data/experiment/private/experiment_config.json --manifest ../data/experiment/validation/validation-manifest.json --phase validation --validation-block V01 --validation-plan ../data/experiment/validation/validation-plan.json --calibration-model-fits ../results/calibration_model_fits.json --journal ../data/experiment/raw/P01-validation.jsonl --session-id P01-VAL-01 --run-gui --confirm-consent
python response_gui.py --config ../data/experiment/private/experiment_config.json --manifest ../data/experiment/validation/validation-manifest.json --phase validation --validation-block V02 --validation-plan ../data/experiment/validation/validation-plan.json --calibration-model-fits ../results/calibration_model_fits.json --journal ../data/experiment/raw/P01-validation.jsonl --session-id P01-VAL-02 --run-gui --confirm-consent
python response_gui.py --config ../data/experiment/private/experiment_config.json --manifest ../data/experiment/validation/validation-manifest.json --phase validation --validation-block V03 --validation-plan ../data/experiment/validation/validation-plan.json --calibration-model-fits ../results/calibration_model_fits.json --journal ../data/experiment/raw/P01-validation.jsonl --session-id P01-VAL-03 --run-gui --confirm-consent
```

세 block은 독립 세션 ID를 유지한다. 편의상 한 session으로 합치지 않는다.
validation 활성 자극 중 중단 또는 시간 제한 내 block 미완료가 생기면 재개하지
않고 참가자 자료 전체를 확증 분석에서 제외한다.

## 13. 9단계 — 검증 joint 분석

먼저 파일·시행 수·세션 메모를 사람이 검토하고 아래 구조의 별도 JSON을
`data/experiment/private/P01-validation-review.json`에 쓴다. `<...>`는 실제
값으로 바꿔야 하며, 직접 식별자는 넣지 않는다. `reviewed_at_utc`는 마지막 완료
validation 시행 뒤이면서 현재 UTC보다 늦지 않아야 한다.

```json
{
  "schema_version": "1.0",
  "review_type": "human_validation_confirmation",
  "reviewer_id": "R_AUDIT_01",
  "reviewed_at_utc": "<UTC date-time ending in Z>",
  "review_note": "<검토한 파일, block 메모, 제외 여부의 비식별 요약>",
  "approved_for_confirmation": true,
  "participant_id": "P01",
  "config_sha256": "<experiment config raw SHA-256>",
  "source_trials_sha256": "<P01-validation.jsonl raw SHA-256>",
  "validation_plan_sha256": "<validation plan canonical plan_sha256>",
  "validation_manifest_sha256": "<validation-manifest.json raw SHA-256>"
}
```

예전 `--confirm-human-after-review` 플래그만으로는 더 이상 확정할 수 없다.
분석기는 review JSON의 중복 key와 모든 hash를 검사하고 그 원문 SHA-256을
`participant_thresholds.json`에 남긴다.

```powershell
python analyze_validation.py --config ../data/experiment/private/experiment_config.json --calibration-fits ../results/calibration_model_fits.json --validation-plan ../data/experiment/validation/validation-plan.json --validation-manifest ../data/experiment/validation/validation-manifest.json --trials ../data/experiment/raw/P01-validation.jsonl --bootstrap-replicates 1000 --bootstrap-seed 20260915 --review-record ../data/experiment/private/P01-validation-review.json --output ../results/participant_thresholds.json
```

해석 순서는 다음과 같다.

1. block C80 세 개와 공통 `w`의 joint fit·bootstrap 수렴률을 확인한다.
2. `CV<=0.20`, 세 C80 CI의 공통 교집합, 수렴률 0.90 이상을 모두 확인한다.
3. 안정할 때만 GLS pooled C80과 CI를 사용한다. 불안정하면 단일 통합값과 모델
   승패를 만들지 않는다.
   각 block 및 pooled C80의 점추정과 95% CI 전체가 실제 제시 ΔE00 범위에
   bracket되지 않아도 `stable=false`이며 확정할 수 없다.
4. 같은 교정 모의 반응으로 짝지은 B96/C96 예측 bootstrap과 validation pooled
   bootstrap으로 `Δloss=|C96-O|-|B96-O|` 95% 구간을 계산한다. 구간 전체가
   0 한쪽에 있을 때만 우열을 판정하고, 0을 걸치면 `모델 구분 불가`, 복제값이
   없으면 `comparison_unresolvable`로 기록한다.
5. 공통 `w` 대 block별 자유 `w` AIC는 보조 민감도 점검이다. 구현되지 않은
   Brier score나 calibration error를 결과표에 추가하지 않는다. 두 적합 중
   하나라도 수렴하지 않으면 `free_w_preferred=false`로 통과시키지 않고
   `shared_w_sensitivity_unresolved`인 `indeterminate`로 기록한다.

세 블록은 같은 참가자 안의 session-separated 반복 held-out 자료다. 서로 다른
참가자 표본처럼 통계적으로 독립이거나 IID라고 쓰지 않는다. 모수적 bootstrap은
고정 모형 아래 반응의 조건부 Bernoulli 독립을 가정하므로, 학습·피로·연속상관이
있으면 CI가 과소 추정될 수 있다. 세션별 추세와 현장 품질 메모를 함께 확인한다.

## 14. 10단계 — 연구용 동일경로 최소 보정

검증 plan에 동결된 8개 원본 색쌍을 모두 같은 규칙으로 분석한다. 사람 모드에서
분석자가 수준을 하나 고르는 옵션은 금지되며, 8수준 중 하나라도 무결성 검사나
보정 계산에 실패하면 부분 결과를 저장하지 않는다. 사람 모드는
config·plan·threshold의 raw/canonical hash, 참가자, CVD/severity,
선택 굵기, `confirmed=true`, 세 block 안정성을 모두 다시 대조한다. 또한
`participant_thresholds.json`의 Git commit은 현재 보고서 commit이 아니라
**실제 자극을 제시한 자료수집 commit**이므로 validation plan의
`source_git_commit`과 같아야 한다. 분석 재현성은 wrapper 하나의 hash만 보지
않고 산출물의 `implementation_hashes`에 기록된 심리측정·색변환·자극 검증
모듈, 관련 schema, `requirements-lock.txt` 전체를 현재 파일과 대조한다.
하나라도 바뀌었다면 원자료는 보존한 채 그 산출물을 만드는 단계부터 다시
실행한다. 보고서나 결과 파일만 나중에 commit한 것은 자료수집 commit
불일치가 아니다.

```powershell
python run_path_correction_analysis.py --config ../data/experiment/private/experiment_config.json --validation-plan ../data/experiment/validation/validation-plan.json --participant-thresholds ../results/participant_thresholds.json --participant-id P01 --models B96 B288 C96 D288 --target-policy point --output ../results/path_correction_analysis.json
```

주 분석은 `--target-policy point`, CI 상한을 쓴 보수적 민감도 분석은 별도 파일에
`--target-policy ci_high`로 실행한다. 같은 출력 파일을 덮어쓰지 않는다.
출력의 `level_results`에는 1~8 수준이 plan 순서대로 들어가고, 각 수준 안에
요청한 모든 모델의 목표·달성 색차·OKLab 이동량이 기록되어야 한다.
config의 목표 허용오차는 자극 수준 생성 검사에만 쓰인다. 동일경로 보정은
`Machado 후 ΔE00 >= C80`인 유한 후보를 목표 초과량으로 다시 제외하지 않는다.

이 결과가 허용하는 문장은 다음뿐이다.

> 동결한 동일 Lab 경로의 유한 8-bit 후보 집합에서, 보정 결과는 해당 모형의
> Machado 후 C80 제약을 만족하면서 OKLab 이동 합이 가장 작았다.

연속 색공간·전체 팔레트의 전역 최소, 임의 색쌍 일반화, 실제 가독성 향상은
주장하지 않는다. `run_personal_correction.py`의 A만/B만/둘 다 radial 탐색은
임의 색쌍용 보조 software baseline이며 이 주 분석을 대신하지 않는다.

## 15. 멈춰야 하는 경우

| 상황 | 조치 |
|---|---|
| 실제 장치값 또는 config에 `미정`이 남음 | 사람 수집 중단 |
| config-bound calibration recovery 실패 | 수준·시행 설계 재검토 후 새 config |
| 자료수집 Git commit, config, PNG, sidecar, manifest hash 불일치 | 수집 전이면 새 commit/config에서 자극 재생성; 이미 수집했다면 원자료를 고치지 말고 해당 세션 제외·편차 보고 |
| 교정 288 완료 시행 또는 균형 불충족 | 분석·validation 계획 생성 중단 |
| plan-bound validation recovery 실패 | 192회 수집 중단, 설계 재검토 |
| 검증 세 블록 CV>0.20, CI 공통교집합 없음 또는 수렴률<0.90 | pooled C80·모델 승패 금지, 세션별 결과 보고 |
| B96/C96 Δloss bootstrap CI가 0을 포함 | `모델 구분 불가` 보고 |
| paired 교정 예측 또는 검증 bootstrap 복제값 누락 | `comparison_unresolvable`, 확정·승패 보고 중단 |
| 교정 공통/자유 `w` 비교 미수렴 또는 자유 `w` AIC가 더 낮음 | D288 beta·굵기효과 해석 금지; C96/B96 주 비교와 D288 계산 보정은 유지 |
| 검증 공통/자유 `w` 비교 미수렴 | primary Δloss를 `indeterminate`로 보고 |
| D288 beta 95% CI가 0을 포함 | `굵기 감소 방향 미확인` 보고 |
| 보정 전·후 새 사람 시험 없음 | `실제 가독성 향상` 문장 금지 |

`validation_recovery.py`는 설계 회복 게이트를 못 넘으면 결과 파일을 남기고 종료
코드 3을, `analyze_validation.py`는 세 블록이 불안정하면 결과 파일을 남기고
종료 코드 3을 반환한다. 이는 프로그램 고장으로 숨길 값이 아니라 사전 규칙에
따른 실패·불안정 결과다. 같은 출력이 이미 있을 때는 기존 파일을 조용히
덮어쓰지 말고 먼저 보관하고 변경 이유를 기록한다.

## 16. 최종 인계 산출물

| 파일 | 의미 | 공개 주의 |
|---|---|---|
| `calibration-manifest.json` | 실제 표시한 288 PNG와 provenance | 익명 ID만 |
| `P01-calibration.jsonl` | 교정 완료·중단 시도의 append-only 사람 원응답 | 비공개 |
| `calibration_model_fits.json` | B96/B288/C96/D288 교정 전용 적합 | held-out 결론 아님 |
| `validation-plan.json` | 응답 전 동결한 굵기·8수준·192 시행 | 수정 금지 |
| `validation-manifest.json` | 실제 표시한 검증 PNG와 hash | 익명 ID만 |
| `P01-validation.jsonl` | 검증 완료·중단 시도의 append-only 사람 원응답 | 비공개 |
| `P01-validation-review.json` | 입력 hash·검토자 가명 ID·시각·메모가 결박된 확정 승인 | 비공개 |
| `participant_thresholds.json` | 세 C80·공통 w·안정성·GLS 통합값 | 사람 검토 후 confirmed |
| `path_correction_analysis.json` | 동일경로 계산 보정 결과 | 가독성 증거 아님 |

최종 보고서에는 코드 commit, config/plan/manifest hash, 완료·중단·제외 시행 수,
수렴 실패, 참가자별 원값과 CI를 함께 남긴다. n=2 사례연구이므로 두 참가자의
차이를 protan/deutan 전체의 차이로 바꾸거나 FAST·ODP·AI보다 우수하다고 쓰지
않는다.
