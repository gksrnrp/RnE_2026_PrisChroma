# 코드 리뷰 및 검증 기록

대상: `src/` 의 규칙 기반 색보정 baseline
최초 리뷰 2026-08-20. Claude 리뷰 → 수정 → Codex 수정 → 재검증 순서로 진행됨.

이 문서는 **어떤 판단으로 무엇을 고쳤는지**를 남기기 위한 것이다.
Codex와 번갈아 작업할 때 같은 논의를 반복하지 않기 위해 유지한다.

---

## 1. 검증된 부분 (직접 실행 확인)

| 항목 | 결과 |
|---|---|
| CIEDE2000 정확도 | Sharma(2005) 표준 14케이스, 최대오차 `4.1e-5` |
| Machado 행렬 | severity 0 = 항등행렬, severity 1 행합 = 1.0 (백색 보존) |
| severity 연속성 | deutan `(220,50,50)` 0.0→1.0 단조 변화 |
| protan/deutan 구분 | `(103,93,48)` vs `(145,130,43)` |
| 탐색 대칭성 | `f(A,B)` = `f(B,A)` |
| numpy 중복 구현 | 스칼라 기준 구현과 최대차 `1e-13` |

### 핵심 대조 실험 (research-plan.txt 6절)

```
테스트 A: RED(220,50,50) + BLUE(50,50,220)
  orig_dE=44.85  sim_dE=61.46  confusing=False  → 두 색 모두 유지

테스트 B: 같은 RED + GREEN(60,160,60)
  orig_dE=69.68  sim_dE= 9.32  confusing=True   → 최소 이동 보정
```

같은 빨강이 상대 색에 따라 유지되거나 보정되는 **소프트웨어 동작만**
확인했다. 이 예시는 합성 색쌍 단위 기능 시험이며, 사람의 C80 변화나 실제
가독성 향상을 입증하지 않는다.

---

## 2. 발견 → 수정된 문제

### 2-1. 재혼동 검사 부재 (심각)

`is_confusing()` 은 **구조적으로 재혼동을 탐지할 수 없다.** 이 함수는
"원본에서는 달랐는가"를 함께 묻기 때문에, 보정이 만들어낸 새 충돌을
"색각이상 탓이 아니다"라는 이유로 통과시킨다.

실제로 재현된 사례:

```
팔레트에 THIRD(20,175,100) 이 있을 때
  보정 전 GREEN(60,160,60) vs THIRD : 7.65
  보정 후 B'(10,171,94)    vs THIRD : 1.34   ← 기존 충돌을 더 악화시킴
  그런데 is_confusing(B', THIRD) = False
```

**수정**: `find_minimum_correction(..., palette=[...])` 로 별도 제약을 추가했다.
원래 목표 이상 떨어진 관계는 목표 미만으로 만들지 않고, 원래부터 목표
미만이던 관계는 보정 전보다 더 가깝게 만들지 않는다. 즉 이 쌍의 보정이
새 충돌을 만들거나 기존 충돌을 악화시키는 것만 막으며, 팔레트의 모든
기존 문제까지 한 번에 해결했다고 주장하지 않는다.
시뮬레이션 후 거리만 보고 `min D_CVD(후보, C) >= target` 을 요구하며,
탐색 feasibility 마스크에 직접 AND 된다.

설계상 두 가지 판단:

1. **A·B 자신은 팔레트에서 자동 제외.** 포함하면 "원본에서 멀어져라"라는
   정반대 제약이 된다.
2. **이동하지 않은 색에는 제약을 걸지 않는다.** 이 제약이 보장하는 것은
   *"보정이 상황을 악화시키지 않는다"* 이지 *"장면 전체가 완벽히 구별된다"* 가
   아니다. 후자는 팔레트 전체 최적화가 필요하며 두 색 baseline의 범위 밖이다.

### 2-2. 임계값 논리 구멍

`ConfusionThresholds` 가 `original_min` 과 `simulated_max` 의 관계를 검사하지
않아, `original_min < simulated_max` 로 두면 원래부터 비슷한 색쌍이
혼동으로 분류됐다.

```
(120,120,120) vs (128,124,126)  orig_dE=3.36  sim_dE=2.11  → confusing=True
```

**수정**: `original_min >= simulated_max` 를 코드가 강제한다.

### 2-3. 탐색 정밀도가 "최소값"을 과대평가

기본 `step=0.02` 는 수렴값 대비 약 18% 큰 값을 "최소"로 보고했다.
**논문에서 baseline이 실제보다 약해 보이게 만드는 문제**라 기본값을 0.005로 낮췄고,
이후 Codex가 정수 RGB 지역 정밀 탐색과 multi-start를 추가해 더 개선했다.

### 2-4. 기타

| 문제 | 수정 |
|---|---|
| 탐색 실패 안내가 `max_shift` 를 늘리라고 함 (gamut이 먼저 막아 무효) | `step` / `direction_grid_radius` 안내로 교체 |
| `in_srgb_gamut` 이 하드코딩된 `True` | 반환 색을 실제로 검사 |
| `target_threshold` 과대 설정 시 색상 파괴 감지 불가 | `hue_shift_A/B_degrees` 보고 추가 |
| `refinement_performed` 가 "돌렸는가"와 "개선됐는가"를 구분 못 함 | `refinement_improved` / `refinement_gain` 추가 |
| pair RGB cube 경계에서 개선이 중단됨 | 새 해를 중심으로 최대 20회 반복 정밀화 |
| 더 좋은 both 해가 A_only/B_only 바로 옆에 있어 누락됨 | 정밀화된 단일 이동 해도 pair cube 중심으로 사용 |
| 0에 가까운 기준값에서 비표준 JSON `Infinity` 생성 가능 | 편차를 `null`로 기록하고 `allow_nan=False`로 강제 |
| stdout에 안내문이 섞여 JSON 파이프 불가 | 대화형일 때만 출력 |
| 결과 JSON이 cp949로 저장됨 | UTF-8 강제 |

---

## 3. 남아 있는 한계 (논문에 반드시 밝힐 것)

### 3-1. 근사 탐색과 step 의존 가능성

최소 색이동은 **전역 최적해가 아니다.** radial grid + 지역 RGB cube +
multi-start + 반복 정밀화 방식이다. 고정 표본에서 step 안정성이 크게
개선됐지만, 전체 8-bit 색공간에서 항상 같은 해나 전역 최적해를 찾는다는
증명은 아니다.

아래 표는 과거 `benchmark_search.py --pairs 30` 개발 실행(seed 20260820)을
기록한 것이다. 당시 JSON은 현재 구현·환경·commit provenance에 결합되지 않아
Git 근거 산출물에서 제외했다. 따라서 현재 성능 수치로 인용하지 말고, 같은
조건을 현재 commit에서 다시 실행해 provenance를 남기기 전에는 역사적 메모로만
취급한다.

| 지표 | 반복 정밀화 적용 후 |
|---|---|
| 편차 중앙값 | **0.0000 %** |
| 편차 평균 | **0.0176 %** |
| 편차 최대 | **0.5267 %** |
| 5% 초과 | **0/30** |
| 비단조 | **0/30** |
| 편차 계산 불가 | **0/30** |

과거 비단조 사례 `(81,226,75)/(226,217,80)`와 새로 발견된
`(96,214,134)/(172,196,18)`은 회귀 테스트로 고정했다.

> **색쌍 하나로 "step과 무관하다"고 일반화하지 말 것.**
> 이 프로젝트에서 두 번 반복된 실수다.

### 3-2. 목적함수 단위 불일치

실현가능성 판정은 ΔE00(CIELAB), 최소화 비용은 OKLab 유클리드 거리를 쓴다.
버그가 아니라 설계 판단이지만, **논문에서 왜 두 지표를 섞었는지 정당화해야 한다.**

### 3-3. 계산 비용

| | 값 |
|---|---|
| 탐색 1회 (팔레트 있음, 대표 색쌍) | 약 3.9 초 |
| 탐색 1회 (팔레트 없음, 대표 색쌍) | 약 7.1 초 |
| 전체 테스트 41개 | 약 1분 32초 |
| 과거 벤치마크 30쌍(현재 provenance 없음) | 약 7분 42초 |

두 색 실험에는 문제없다. **이미지 단계에서 영역 쌍마다 돌리면 병목이 된다.**
`_SINGLE_REFINEMENT_STARTS = 16` 의 비용 대비 효과는 아직 측정하지 않았다.

---

## 4. 다음 작업 후보

- [x] 비단조 사례를 회귀 테스트로 고정하고 반복 pair 정밀화 적용
- [ ] `_SINGLE_REFINEMENT_STARTS` 를 4 / 8 / 16 으로 바꿔 비용 대비 효과 측정
- [x] `_relative_spread_percent` 의 `inf` → `null`
- [ ] 결과 저장 및 시각화 (research-plan.txt 8절)
- [ ] 실제 이미지 혼동 영역과 연결

---

## 5. ②단계(탐지·측정) 구현 기록 — 정인교, 2026-08-20

`src/`에 ②단계 코드가 하나도 없는 상태에서 시작했다. 작업 계획의
1번(시뮬레이터 통합)은 지울 중복 구현 자체가 없어 자동으로 충족되었다.
`cvd_simulation.py`와 `color_utils.py`는 이 저장소에 존재한 적이 없다.
시뮬레이터는 `machado.py` 하나뿐이다.

### 5-1. 판단이 갈렸던 곳과 근거

**대표색은 평균이 아니라 중앙값을 쓴다.** 얇은 선은 안티앨리어싱으로
가장자리에 배경색이 섞이는데, 평균은 그 값에 끌려간다. 중앙값은 이상치
한두 픽셀에 흔들리지 않는다.

**유사색 병합에서 원래 대표색으로 판단하면 안 된다.** 처음에는
union-find로 ΔE00가 작은 경계를 차례로 합쳤는데, 안티앨리어싱으로 생긴
중간색 superpixel이 징검다리가 되어 선이 배경까지 이어져 사라졌다.
`merge_thresh=4`에서 577개 영역이 5개로 줄고 선 8개가 전부 없어졌다.
합칠 때마다 덩어리의 넓이 가중 평균색을 다시 계산하고 우선순위 큐로
다시 재는 방식으로 바꾸니 같은 설정에서 16개 영역이 남고 색 8개가 모두
보존되었다.

**조각 수를 세는 규칙을 바꿨다.** 작업 계획의 기준은 "대상 위치에 겹치는
영역이 8개 초과"인데, 스친 영역까지 모두 세면 길게 뻗은 선은 어떤
설정에서도 8개를 넘는다. 400 px 선이 superpixel 하나에 들어가는 일은
없기 때문이다. 세는 규칙만 "요소 면적의 90%를 덮는 데 필요한 개수"로
바꾸고 8개라는 경계값은 그대로 두었다. 색 어긋남도 조각 하나가 아니라
덮은 영역들의 겹침 비율로 가중해 잰다.

**쌍 사이 간격은 경계상자 거리로 근사했다.** 실제 픽셀 경계 사이의 최단
거리는 쌍이 수만 개일 때 비용이 크다. 근사라서 실제보다 작게 나오고
대각으로 뻗은 선에서 특히 그렇다는 것을 문서와 `conditions.md`에 적었다.

**쌍 수 상한에 걸리면 그 사실을 기록한다.** 가까운 쌍부터 남기고, 잘라낸
개수와 실제 적용된 거리 상한을 결과 JSON에 넣는다. 조용히 자르면 "전부
측정했다"로 읽힌다.

### 5-2. 민감도 분석에서 뒤집힌 예상

처음 예비 측정에서 compactness가 얇은 선의 생사를 가르는 것처럼 보였다
(n_segments=800에서 compactness 10은 팔레트 색 1/8, 3은 8/8 재현).
그러나 병합을 켠 상태로 다시 재니 compactness 1~20에서 검출률이 거의
같았다. 앞의 관찰은 compactness가 아니라 **병합을 끈 상태에서 두꺼운
요소가 조각나는 효과**를 본 것이었다. 축을 하나 더 늘려 재보지 않았으면
잘못된 결론을 보고서에 실을 뻔했다.

### 5-3. 남은 한계

1. **3 px 이하 요소는 검출되지 않는다.** n_segments를 3000까지 올려도
   0%다. 원인이 분할 해상도가 아니라 대표색 오염이므로 superpixel을 더
   잘게 쪼개는 방향으로는 풀리지 않는다. 선 픽셀만 골라내는 별도 처리가
   필요하다.
2. **`d_max`에 정답이 없다.** 200에서는 범례-선 쌍을 하나도 못 잡고,
   400에서 5쌍을 모두 잡되 전체 쌍 수가 83% 늘어난다. 거리 대신 의미
   관계를 쓰려면 다른 근거가 필요하다.
3. **한 영역이 여러 쌍에 걸린다.** `run_baseline`이 쌍마다 다른 보정색을
   제안하는 경우가 생긴다. 영역 단위로 색 하나를 정하는 규칙은 ③단계에서
   정해야 한다. 지금은 충돌 횟수만 센다.
4. **실제 이미지가 아직 없다.** `conditions.md`의 수치는 합성 이미지
   기준이다. 실제 이미지를 넣어 다시 돌려야 실험에 쓸 수 있다.

---

## 6. Claude Code CLI MCP 연결 (2026-08-27, Codex)

연구 알고리즘과 무관한 개발 도구 연결을 추가했다. 연구 방향, 임계값, baseline
구현에는 변경이 없다.

### 구성

- `.codex/config.toml`: 프로젝트 범위 STDIO MCP 서버 `claude_code`를 등록했다.
- `tools/claude-mcp/server.mjs`: `claude -p --output-format json`을 호출하는
  의존성 없는 Node.js MCP 브리지다.
- `ask_claude`: 파일 도구를 끄고 Claude의 독립 의견만 받는다.
- `review_with_claude`: `Read`, `Glob`, `Grep`만 허용해 저장소를 읽기 전용으로
  검토한다.
- 두 도구 모두 `--permission-mode plan`, `--no-session-persistence`를 사용하며,
  Codex 쪽 호출 승인 모드는 `prompt`다.

Claude CLI 내장 `claude mcp serve`도 프로토콜 초기화에는 성공했지만 Claude에게
질문하는 도구가 아니라 Claude Code의 `Read`, `Edit`, `Bash`, `Agent` 등 내부
도구 전체를 노출한다. 이번 목적에는 권한 범위가 지나치게 넓고 직접 자문 결과를
안정적으로 회수하기 어려워 사용하지 않았다.

### 검증

- `node --test tools/claude-mcp/server.test.mjs`: 1개 테스트 통과. 초기화,
  도구 목록, 두 도구의 CLI 인자, 입력 오류를 mock CLI로 확인했다.
- `node tools/claude-mcp/smoke.mjs`: 실제 로그인된 Claude CLI를 통한 MCP 왕복에서
  `CLAUDE_MCP_OK`를 확인했다.
- 로컬 샌드박스 안에서는 Claude API 네트워크 호출이 대기하므로 실제 smoke
  test에는 네트워크 권한이 필요하다. mock 단위 테스트에는 필요하지 않다.

### 남은 제약

- 현재 실행 중인 Codex 세션은 새 MCP 설정을 즉시 다시 읽지 않는다. 프로젝트를
  신뢰한 상태에서 Codex 앱/CLI의 새 세션을 열고 `/mcp`로 연결을 확인해야 한다.
- `.codex/config.toml`의 상대 경로는 Codex를 이 저장소에서 시작하는 것을
  전제로 한다.
- 실제 Claude 호출은 사용자의 Claude 인증과 사용량 정책을 그대로 사용한다.

---

## 7. 선 굵기별 C80 실험 기반 추가 (2026-09-07, Codex)

사람 대상 실험을 시작하기 전에 단위를 고정하고 선행모델을 잘못 전용하지 않도록
`src/experiment_design.py`와 `docs/experiment-protocol.md`를 추가했다.

### 구현한 것

- 화면 가로 길이·해상도·관찰 거리로 px와 시각도를 상호 변환한다.
- Szafir(2018) 선 모델 Eq. 14~16과 고정 Lab 방향의 상대 역치 배율을 계산한다.
- 원 논문의 측정 범위 0.05~0.35도를 기본으로 강제한다. 범위 밖 외삽은 코드에서
  명시적으로 요청해야만 가능하다.
- 세 교정 굵기 사이의 기하평균 검증 후보를 계산한다.
- `plan_line_widths.py`가 실제 장치값을 받아 교정·검증 굵기의 시각도와 선행모델
  범위 이탈을 JSON으로 보고한다. 화면값과 굵기에는 기본값을 두지 않았다.
- C80 정의, 자료 분리, B192/B384/C/D 비교, held-out 검증과 구분 불가 기준,
  윤리·중단 규칙을 시험 전 고정 문서에 모았다. 당시의 `독립 계단 3회` 안은
  이후 `64회 고정수준 블록 × 3세션`으로 대체되었다.

### 의도적으로 구현하지 않은 것

- 0.62나 3.0을 선 자극의 C80 기본값으로 넣지 않았다.
- Szafir의 정상색각·이진 과제 확률을 적록색각이상 4AFC의 절대 확률로 해석하지
  않았다. 현재 코드는 상대 굵기 함수만 제공한다.
- 실제 화면 크기, 관찰 거리, Lab 방향, 세 굵기는 팀이 측정·합의해야 하므로
  `미정`으로 남겼다.

다음 구현은 자극 생성·시행 기록 형식과 가상 참가자 parameter recovery다.

### 검증

- 새 시각도·Szafir·굵기 계획 테스트 12개 통과.
- `.venv`에서 이미지 분할을 포함한 전체 테스트 101개 통과(약 80초).
- Windows의 cp949 환경에서 `requirements.txt`의 특수문자 주석 때문에 설치가
  중단되는 문제를 재현했고, 패키지 버전은 유지한 채 주석만 ASCII로 바꿨다.

---

## 8. 3인 통합 데이터 계약 초안 (2026-09-07, Codex)

이창민의 통합 작업 첫 단계로 `schemas/` 5개, 형식 예제 5개,
`data_contracts.py`와 명령줄 검증기를 추가했다. 아직 팀 합의 전이므로 버전은
`0.1-draft`이고 실제 사람 자료를 구조적으로 거부한다.

스키마 모양 외에 다음 의미 규칙도 검사한다.

- 화면 물리값으로 px→시각도를 다시 계산해 저장값과 대조한다.
- 기하평균 검증 후보와 실제 정수 픽셀 굵기를 분리한다.
- Szafir 전이 모델을 쓰면 모든 굵기가 0.05~0.35도 안인지 확인한다.
- 완료 시행의 정오 값이 실제 응답 위치와 일치하는지 확인한다.
- `detection_failed`는 가짜 굵기를 채우지 않고 null/excluded로 기록한다.
- C80 점추정치가 신뢰구간 안인지, 안정성 판정과 pooled C80이 모순되지
  않는지 확인한다.
- 팔레트에 보정 대상 A/B 자신이 다시 들어가는 것을 막는다.

원자료 경로 `data/experiment/raw/`와 개인정보 경로
`data/experiment/private/`는 Git에서 제외했다. 다음 단계는 세 팀원의 필드 검토와
가짜 시행→C80→보정의 end-to-end 연결이다.

### 이창민 연결 모듈

`threshold_adapter.py`는 참가자·모델·정수 픽셀 굵기가 정확히 일치하는 C80만
읽는다. 없는 굵기를 보간하거나 0.62/3.0으로 대체하지 않는다. 기본 모드에서는
synthetic 또는 `confirmed=false` 자료를 거부하며, 초안 연결 시험은
`development_mode=True`를 읽기와 변환 양쪽에 명시해야 한다.

주 분석은 C80 점추정치를, 민감도 분석은 신뢰구간 상한을 선택할 수 있다.
`original_min`은 C80과 다른 별도 변수이므로 호출자가 명시적으로 제공해야 한다.

검증 결과: 계약·adapter 관련 검사 29개와 저장소 전체 검사 123개가 모두
통과했다.

추가 end-to-end 회귀 사례에서는 synthetic C80 2.8을 adapter로 읽고,
deutan 0.7에서 `(32,96,224)/(160,32,224)`의 시뮬레이션 색차 2.263을
기존 최소이동 보정에 전달했다. B만 이동한 결과 2.804로 목표를 만족했다.

### 검증 색쌍 일괄 실행기

`run_personal_correction.py`를 추가해 `validation_pairs`와
`participant_thresholds`를 한 번에 읽고, 정확히 일치하는 참가자·모델·정수
굵기의 C80만 기존 `run_baseline()`에 전달하도록 했다. 결과는 새
`correction_results` 계약으로 다시 검증한 뒤 UTF-8 JSON으로 원자 저장한다.
결과에는 두 입력 계약 버전과 탐색 간격·범위·방향 격자·청크 크기를 함께
기록해 같은 계산을 재현할 수 있게 했다. calibration에서 분리한 held-out
검증 여부도 각 결과 행에
보존한다.

가짜 예제에는 유지 경로와 보정 경로를 각각 한 쌍씩 넣었다. C80 2.8에서
첫 쌍은 이미 시뮬레이션 색차가 13.870이라 유지됐고, 둘째 쌍은 2.263에서
2.804로 보정됐다. 출력은 다음 불일치를 거부한다.

- target 정책(point/CI 상한)과 실제 선택값 불일치
- 결과 행과 요약 개수 불일치
- 보정 후 색차와 `target_met` 불일치
- A/B 이동량 합과 총이동량 불일치
- `corrected`인데 전략·이동·목표 달성이 없는 결과

현재 결과 계약도 `0.1-draft`와 `synthetic`에 잠겨 있다. 또한 출력에
`approximate_pairwise_not_global`을 기록해 색쌍별 근사해를 전체 화면의 전역
최적해로 오해하지 않게 했다. 실제 사람 C80과 실제 held-out 검증쌍은 팀 합의와
parameter recovery 이후 별도 1.0 계약을 만들기 전까지 이 실행기에 넣지 않는다.

검증 결과: 공통 계약 예제 6개가 모두 통과했고, 기존 기능을 포함한 전체
단위·통합 테스트 131개가 통과했다.

---

## 9. 픽셀 검증 가능한 선 자극 렌더러 (2026-09-14, Codex)

`stimulus_renderer.py`에 Pillow 12.3.0 잠금 환경용 2×2 선 자극 렌더러를
추가했다. 고수준 선 API 대신 반열린 정수 사각형을 채워 `butt` 끝,
안티앨리어싱 없음, 0/90도 축 정렬, 정수 anchor와 짝수 extent의 음의 방향
tie-break를 명시했다. 각 PNG 옆의 `*.pixels.json`에는 캔버스·배경·네 좌표·
경계상자·길이·굵기·위치별 색·RGB별 픽셀 수·PNG 및 해독 RGB SHA-256을 남긴다.
검증기는 파일을 다시 열어 전체 RGB byte 배열을 예상 raster와 대조한다.

시행표 컴파일은 `color_design`의 Lab 중점·방향과 목표 허용오차, 참가자의
Machado type/severity를 필수로 받고, 같은 목표 ΔE00는 한 번만
`generate_symmetric_pair()`로 생성한다. 실제 8-bit RGB에서 재계산한 색차가
허용오차를 넘거나 색역·기하·잠금 버전을 만족하지 못하면 부분 manifest 없이
실패한다. 허용오차는 사람용 기본값 없이 동결 설정의 절대·상대값을 모두 받아
`max(abs_tol, rel_tol*target)`으로 계산한다. 서로 다른 12수준이 같은 RGB 쌍으로
양자화되거나 실제 표시 ΔE00 순서가 뒤집혀도 전체 컴파일을 거부한다. 생성된
manifest와 PNG는 합성/실행 준비 산출물일 뿐 사람 반응 자료가 아니다.

`experiment_runner`의 게이트는 설정, 렌더러, 세션 런타임으로 분리했다.
`response_gui.py`에는 practice/calibration/validation 전체화면 제시, 키 응답,
응답·렌더 시간, 중단 사유와 append-only JSONL 기록이 구현되어 있다. 다만
config-only 점검의 `human_data_collection_allowed`는 의도적으로 항상 false다.
실제 한 세션의 허가는 `response_gui`가 manifest·화면·Git·validation plan을
다시 검사하고 운영자가 해당 세션의 동의를 명시했을 때만 true가 된다.

교정 288시행은 4개 사전 세션 블록에 72회씩 나눈다. 각 블록은 굵기당 24회,
각 굵기×수준 셀당 2회이며 네 블록을 합치면 위치×odd색 8조합이 정확히 한 번씩
나온다. 블록 내부만 seed로 섞는다. 시행 저널 고유 키는
`(participant_id, config_sha256, trial_id)`라서 session 이름만 바꾼 중복 실행을
막는다.

검증 결과: 렌더러·시행표 전용 테스트 21개 통과. PNG 변조, 메타데이터 변조,
좌표 겹침·클리핑, 모호한 렌더링 정책, 참가자 설정 불일치, 도달 불가능한 목표
색차를 모두 거부한다. 실제 모니터 출력, 색 프로파일 적용, 키 입력과 세션 흐름은
아직 검증하지 않았다.

---

## 10. 교정자료 전용 심리측정 분석기 (2026-09-14, Codex)

`analyze_calibration.py`는 파일로 받은 한 참가자의 동결 config, trial JSONL과
실제 표시한 `calibration_manifest`를 검증한 뒤 B192/B384/C192/D384, 모수적
bootstrap, 두 held-out 후보의 예측 C80을 계산한다. 사람 자료는 manifest가
없으면 거부하고, 입력이 없을 때 임의 수치를 채우는 옵션도 없다. 합성 자료는
`--allow-synthetic`을 명시해야만 실행된다.

적합 전에 config 원문 SHA-256과 모든 행의 hash, 참가자·Machado·고정 lapse
값/출처/독립 세션, stimulus family, renderer, Git commit을 대조한다. 완료
교정자료는 3굵기×96뿐 아니라 4개 session block, 12수준, 네 정답 위치와 odd
A/B가 동결 시행표대로 완전 균형인지 검사한다. 같은 trial ID를 session 이름만
바꿔 다시 넣어도 `(participant, config hash, trial ID)` 중복으로 거부한다.

C192 상대 배율은 별도 재구현하지 않고 `experiment_design.szafir_relative_scale`
에 config의 동결 Lab 방향과 확률 좌표를 전달한다. 세 교정 굵기와 두 후보 중
하나라도 Szafir 선 측정 범위 0.05~0.35도를 벗어나면 외삽하지 않고 분석 전체를
차단한다. 기준 굵기는 오름차순 세 굵기의 중앙값이며 B192/C192에는 정확히 그
96회만 들어간다.

새 `calibration_model_fits` 1.0 스키마는 이 결과를
`calibration_only_intermediate`, `contains_observed_held_out=false`,
`final_thresholds_ready=false`, `research_conclusion_allowed=false`로 고정한다.
따라서 calibration에서 분리한 held-out validation block의 관측 C80을 담는 human
`participant_thresholds` 최종 계약과 구조적으로 섞이지 않는다. 합성 288행
fixture의 전체 CLI 실행과 자료 없음, hash·lapse provenance·시행 수·중복 ID·
Szafir 범위 위반 거부 경로를 자동 테스트로 고정했다.

---

## 11. 고정수준 validation과 응답 GUI (2026-09-14, Codex)

초기의 적응형 계단 계획을 폐기하고 `validation_design.py`가 교정 반응을 보지
않은 두 held-out 후보 중 B192/C192의 절대 로그 C80 비가 큰 한 굵기를 고르게
했다. 선택 굵기에서는 8수준×4정답 위치×odd 색 정체 2의 64회 완전요인 블록을
서로 다른 seed와 session으로 세 번 만든다. 원 config hash, 교정 분석 원문 hash,
192개 trial과 renderer hash를 불변 validation plan에 묶는다.

`analyze_validation.py`는 세 블록을 따로 적합한 것으로 가장하지 않는다. held-out
반응만으로 블록 C80 세 개와 공통 `w` 하나를 joint MLE로 적합하고, 모수적
bootstrap에서도 네 모수를 함께 재적합한다. `CV<=0.20`과 세 C80 CI의 공통
교집합을 만족할 때만 joint log-C80 공분산을 사용한 GLS pooled C80을 만든다.
블록별 자유 `w`의 AIC가 더 낮으면 `shared_w_sensitivity_warning`과 함께
`indeterminate`로 남긴다. AIC 경고가 없더라도 paired 교정 예측과 validation
pooled bootstrap을 결합한 B192/C192 `Δloss` 95% 구간이 0을 포함하면 모델 우열을
`indeterminate`로 남긴다. 필요한 복제값이 없으면
`comparison_unresolvable`로 fail closed한다.

`response_gui.py`는 validation에서도 매 시행 plan hash를 기록하고, 중복 trial과
다른 participant/config가 섞인 저널을 거부한다. 자동 테스트는 프로그램의 상태
전이와 기록 규칙을 검증한 것이며, 실제 화면 1:1 raster, 색관리, 관찰거리,
키 입력과 피로·중단 절차의 현장 검증을 대신하지 않는다.

사람 검증 분석에는 실제 표시한 `validation_manifest`가 필수다. 분석기는 응답을
plan뿐 아니라 PNG·sidecar·디코딩 RGB hash와도 다시 대조한다. 대표 후보 수준의
개발 recovery와 실제 동결 plan의 200회×bootstrap 50회 recovery를 구분하며,
후자의 hash-bound 통과 파일이 없으면 validation 수집을 허용하지 않는다.

---

## 12. 연구용 동일 Lab 경로 보정 분리 (2026-09-14, Codex)

`path_correction.py`는 C80을 측정한 Lab 중점·방향의 대칭 색쌍만 받는다. 동결한
scale grid를 8-bit sRGB로 양자화해 얻은 유일 후보를 모두 검사하고, Machado 후
ΔE00가 모형의 목표 C80 이상인 후보 중 두 색의 OKLab 이동 합이 가장 작은 것을
고른다. 따라서 반환하는 최소값은 **그 유한 동일경로 후보 집합 안에서는 정확**
하지만 연속 색공간이나 팔레트 전체의 전역 최소가 아니다.

기존 A만/B만/둘 다 radial·multi-start 탐색과 palette 충돌 제약은 임의 색쌍을
다루는 보조 software baseline으로 유지한다. 측정한 색 방향과 중점 밖으로 나갈
수 있으므로 독립 인간 판별 없이 연구의 행동적 가독성 근거로 사용하지 않는다.

`run_path_correction_analysis.py`는 사람 모드에서 config·validation plan·최종
`participant_thresholds`의 raw/canonical hash, 참가자/CVD/severity, 선택 굵기,
검증 안정성·사람 확인과 분석 코드 hash를 대조한 뒤 이 동일경로 탐색을 실행한다.
사람 자료에서는 B192/B384/C192/D384 네 모형을 고정 순서로 모두 처리하고,
일부만 고르는 선택 보고를 거부한다. 보정 wrapper뿐 아니라 Machado,
CIEDE2000, 색공간 변환, 경로 생성, C80 adapter, schema와 lockfile의 SHA-256
전체를 기록하며, 이 파일들이 모두 Git에 추적되고 현재 commit과 같을 때만
계산 제약 주장을 허용한다.
또한 plan의 동결된 8개 색차 수준을 모두 plan 순서로 분석하며, 사람 모드의
단일 수준 선택 인수는 거부한다. 따라서 결과를 본 뒤 유리한 원본 수준만 골라
보고하는 선택편향 경로를 차단한다. 한 수준이라도 실패하면 부분 결과를 쓰지 않는다.
출력은 `computational_constraint_claim_allowed=true`,
`human_readability_improvement_claim_allowed=false`를 분리해 기록한다.

---

## 13. recovery와 사람 수집 게이트 구분 (2026-09-14, Codex)

기본 후보 수준의 `parameter_recovery.synthetic.json`은 분석기 개발 점검이다.
교정 시작에는 참가자별 동결 config의 실제 굵기·12수준·Lab 방향·lapse를 넣은
config-bound 200회 recovery가 별도로 필요하다. 교정 384회 뒤에는 선택된
validation plan의 실제 양자화 8수준을 넣은 200회×bootstrap 50회 plan-bound
recovery가 다시 필요하다. 앞 단계의 대표 결과는 뒤 단계 게이트를 대신하지
않는다.

현재 `research_readiness.py`는 교정과 validation 준비도를 분리해 보고한다.
운영상 canonical 산출물 한 세트를 검사하므로 n=2 사례 참가자는 한 명씩 전
과정을 완료하고 비식별 결과를 참가자별로 보관한 뒤 다음 참가자로 넘어간다.

---

## 14. 분석 의존성 전체 provenance 결합 (2026-09-14, Codex)

기존에는 교정·검증·recovery 산출물이 주로 실행 wrapper 한 파일의 SHA-256만
기록했다. 이 방식은 `psychometrics.py`, `validation_psychometrics.py`, 색변환
코드나 schema가 바뀌어도 과거 산출물이 현재 계산과 같은 것처럼 통과할 수
있었다.

`implementation_provenance.py`에 단계별 명시적 파일 집합을 두고, 각 산출물에
프로젝트 상대 경로를 키로 하는 `implementation_hashes` map을 저장하도록
바꿨다. 각 집합은 수치 계산에 참여하는 first-party 모듈, 입력·출력 schema,
`requirements-lock.txt`를 포함한다. `validate_calibration_artifact`,
`validate_plan`, `threshold_adapter`, `research_readiness`는 현재 저장소의 정확한
키 집합과 hash를 비교한다. 누락·추가·변경 중 하나라도 있으면 다음 단계로
진행하지 않는다. validation plan에서는 이 map 자체도 canonical `plan_sha256`
preimage에 들어가므로 map을 고친 뒤 plan만 다시 해시하는 우회도 거부된다.

호환용 단일 `analysis_script_sha256`/`implementation_sha256` 필드는 당분간
유지하되, 해당 wrapper의 map 항목과 같은 값인지 검사한다. 새 검사는
psychometric 의존성 변조, plan 재해시 우회, validation recovery 의존성 변조,
최종 human threshold 의존성 변조를 포함한 회귀 테스트로 고정했다.

---

## 15. 동일경로 최소의 feasible 집합·provenance 재감사 (2026-09-14, Codex)

문서는 `Machado 후 ΔE00 >= C80`인 유한 후보 전부 중 OKLab 이동 합 최소를
주장했지만, 구현은 그 집합을 만든 뒤 config의 자극 생성 허용오차를 목표
초과량 상한으로 다시 적용했다. 이 때문에 더 적게 움직이는 후보의 초과량이
상한보다 크면 진짜 최소가 배제되고, 도달 가능한 후보가 모두 상한 밖이면
보정이 잘못 실패할 수 있었다.

초과량 상한과 관련 인수를 제거했다. 이제 표시 RGB에서 재계산한 ΔE00가 목표
이상인 모든 유일 후보가 비용 비교에 참여하며, 후보가 하나도 없을 때만
도달 불가능으로 실패한다. config의 절대·상대 목표 허용오차는 동결 자극 수준
생성에만 쓰고 보정 제약을 완화하거나 후보를 제거하지 않는다. 큰 초과량의
저비용 후보도 선택되는 사례와 큰 초과량 후보만 있는 도달 가능 사례를 회귀
테스트로 고정했다.

또한 보정 provenance를 validation 분석 의존 집합 전체에서 상속하도록 바꿔
`experiment_runner.py`, `config.py`, `data_contracts.py`,
`analyze_calibration.py`, `analyze_validation.py`, `psychometrics.py`, 렌더러·실험
설계 모듈과 실제 읽는 human/development schema가 모두 hash 집합에 들어가게
했다. 이 상속 관계와 경로별 필수 파일은 회귀 테스트로 고정했다.

---

## 16. 사람 저널·중단·검토 확정 우회 차단 (2026-09-14, Codex)

기존 GUI는 `--journal`로 임의의 새 JSONL을 선택할 수 있었다. 따라서 이전
파일의 중단 기록과 최초 시행 시각이 남아 있어도 새 파일로 실행하면 이력을
보지 못했다. 사람 config 1.0에 `journal` 계약을 필수화해 root를 정확히
`data/experiment/raw`로, phase별 파일을 `{participant_id}-{phase}.jsonl`로
고정했다. config 원문 hash가 자극 manifest와 각 시행에 이미 결박되므로 GUI와
사람 분석기는 같은 participant+config+phase에서 오직 그 canonical 파일만
받는다. 외부 root, 경로 이동, 다른 참가자 파일명, phase 파일 공유와
`--journal` 교체는 fail closed한다.

연습 1개, 교정 4개 또는 6개, validation 3개의 session ID도 config에 사전 동결하고 모두
다르게 했다. GUI는 첫 자극 전에 현재 block의 정확한 ID인지, 기존 ID가 다른
block에 재사용됐는지, 같은 block 이력이 다른 ID로 이어졌는지 검사한다.
`TrialJournal`은 completed와 aborted calibration/validation을 terminal key로
관리하고 `remaining_trials`도 그 key를 제외한다. aborted practice만 재시도가
가능하며, 기존 또는 실행 중 본 시행 abort를 발견한 GUI는 더 제시하지 않는다.
JSON config/manifest/journal/review에서 중복 key도 거부한다.

사람 validation의 `confirmed=true`는 더 이상 boolean 플래그만으로 만들 수
없다. 별도 review record의 검토자 가명 ID, 검토 시각, 메모, 명시적 승인과
participant/config/raw trial/plan/manifest hash를 대조하고 review 원문 SHA-256을
최종 결과에 보존한다. 검토 시각이 마지막 완료 시행 전이거나 현재 UTC 뒤면
거부한다. downstream C80 adapter도 review의 결박 필드와 최종 산출물 provenance를
다시 대조한다.

남은 물리적 한계가 있다. 이 소프트웨어는 OS 권한 밖에서 canonical JSONL을
삭제·rollback하거나 디스크 자체를 복제한 행위를 독립적으로 탐지하지 못한다.
또한 자극 onset 뒤 append+`fsync` 전에 전원·프로세스가 강제 종료되면 마지막
노출과 그 직전 경과시간을 기록하지 못할 수 있다. 승인된 장치의 접근통제·백업과
세션 운영 기록이 필요하며, 그런 crash가 의심되면 새 파일로 재개하지 말고 해당
참가자 자료를 제외·편차 보고한다. 기존 JSONL 행은 수정하거나 삭제하지 않는다.

---

## 17. 교정 굵기별 기울기 가정 감사 (2026-09-15, Codex)

D384은 세 굵기에서 심리측정 기울기 `w`가 같다고 가정하지만, 기존 교정
산출물은 이 가정을 확인하지 않았다. 교정 384회에 굵기별 C80 세 개+공통
`w` 하나(`k=4`)와 굵기별 C80·`w` 여섯 개(`k=6`)를 적합해 AIC를 비교하고,
AIC·차이·두 적합의 수렴 상태를 산출물에 보존하도록 했다. 둘 중 하나라도
수렴하지 않으면 `free_w_preferred=false`로 우회하지 않고 비교 미해결로 둔다.

비교 미해결, 자유 `w`의 더 낮은 AIC, 또는 D384 수치 미완료 중 하나면
`d384_beta_interpretation_eligible=false`다. 이 플래그를 최종
`participant_thresholds`와 동일경로 보정 보고서까지 전파한다. 이는 D384의
beta·굵기효과 보조 해석만 막고, 기준 굵기만 쓰는 C192 대 B192 주 비교나 D384
예측값의 계산 보정은 막지 않는다. 네 교정 session block의 정답률도 학습·피로
확인용 기술통계로 저장하되 제외 기준이나 유의성 검정으로 쓰지 않는다.

검증 세 블록의 공통/자유 `w` 비교도 어느 적합이든 미수렴하면 primary
Δloss를 `indeterminate`로 만든다. 세 블록은 calibration과 분리된 같은
참가자의 session-separated 반복일 뿐 참가자 수준 독립 표본이나 IID 자료가
아니다. 모수 bootstrap의 조건부 Bernoulli 독립 가정 때문에 학습·피로·연속
상관이 있으면 CI가 과소 추정될 수 있다는 한계를 프로토콜에 명시했다.

---

## 18. 원격 작업 브랜치 호환성·개인정보 감사 (2026-09-15, Codex)

`feature/hanjeonghun-experiment`, `hanjeonghun-colorvision`, `related-work`를
읽기 전용으로 확인했다. 예전 사람 실험 브랜치의 계단법, 자유 lapse 적합,
다른 심리측정식, 중단 세션 이어하기와 큰 패치 severity 추정은 현재 고정수준
direct-C80 프로토콜 및 중단 처리와 호환되지 않는다. 일부 코드만 섞으면 분석
대상과 사전 규칙이 달라지므로 병합하지 않았다.

또한 legacy colorvision 브랜치에는 실제 참가자 자료일 가능성이 있는 CSV·그림과
식별 가능한 파일명이 있다. 해당 자료는 main에 복사하지 않았고 현재 연구의
근거로 사용하지 않는다. `.gitignore`는 앞으로의 추가만 막을 뿐 이미 원격
브랜치에 올라간 기록을 지우지 못하므로, 저장소 관리자가 참가자 동의·공개 범위를
확인한 뒤 필요하면 별도 보존 및 GitHub 기록 정리 절차를 결정해야 한다.

---

## 19. FX517ZC 무계측 색도 정책과 150% DPI 게이트 (2026-09-17, Codex)

실험 후보 장치는 ASUS FX517ZC의 2560×1440, 165 Hz 내부 화면이며 Windows
배율은 150%, 밝기는 70%, 발광영역 가로는 342 mm로 보고됐다. Windows 자동 색
관리는 켜져 있고 임의로 추가했던 일반 sRGB 프로필은 제거했다. 휘도계·색도계는
확보하지 못했으므로 절대 휘도나 실제 발광 색좌표를 꾸며 넣지 않는다.

사람 config에 `colorimetry_policy`와 `brightness_percent`를 추가했다.
`nominal-srgb-device-specific` 정책에서만 `luminance_cd_m2=null`을 허용하며,
측정법 문자열은 정확히 `not_measured_no_luminance_meter`여야 한다.
`measured-display-colorimetry` 정책에서 null을 넣거나 밝기 100%를 넘기면
수집 게이트가 차단된다. 이 경로는 장치 독립적 절대 ΔE00 주장을 허용하지 않는다.

기존 DPI 설정 함수는 Windows API가 FALSE를 반환해도 성공으로 간주했고 system
aware와 per-monitor aware를 구분하지 않았다. 이제 Tk 생성 전에 Per-Monitor V2
context를 정확히 비교하고, 실제 전체화면 HWND의 DPI가 동결 배율과 같은지도
확인한다. 150% 설정은 144 DPI이며 PNG 자체는 리샘플링하지 않는다.

---

## 20. 96/192/96회 혼합 교정 설계 전환 (2026-09-17, Codex)

P01의 lapse 0.07을 결합한 탐색에서 기존 굵기당 96회 설계는 기준 굵기만 쓰는
두 모형의 C80 회복오차 사전 기준을 넘었다. 탐색한 96·144·192·240회 가운데
192회가 단일 seed에서 여유를 두고 오차 기준을 통과한 첫 조건이었지만, 이를
`최소 시행 수`나 사람 실험 승인으로 해석하지 않는다. 따라서 후보 설계를
바깥/기준/바깥 굵기 `96/192/96`, 총 384회로 바꾸고 B192/C192과
B384/D384으로 이름을 바꿨다.

공통 `calibration_design.py`에서 시행 수, 전체 시행 수, 4개 또는 6개 세션,
모형 순서와 무작위화 계약을 한 번만 정의한다. 스케줄 생성기·PNG compiler·GUI·
교정 분석·recovery·준비도 검사와 JSON schema가 이 계약을 함께 사용한다.
4세션은 블록당 96회(24/48/24), 6세션은 블록당 64회(16/32/16)이며 자동검사로
두 배치의 균형을 고정했다. calibration trial ID는 1~384, validation은
385~576으로 이동했다.

가상 참가자 기울기 `w`가 색차 수준 범위를 좁힐수록 자동으로 작아지던
자기충족적 recovery 문제도 제거했다. low/central/high의 `w`는 수준 간격과
독립된 0.225/0.321/0.482로 사전 고정하고 산출물에
`w_source=fixed-before-range-selection`을 남긴다. 서로 다른 수준 범위에서도
세 `w`가 같다는 회귀 테스트를 추가했다.

실제 GUI 실행은 이제 기술 점검과 동의만으로 열리지 않는다. 별도 프로세스로
전체 `research_readiness`를 다시 실행해 practice/calibration에는
`ready_for_calibration_collection`, validation에는
`ready_for_validation_collection`이 true일 때만 창을 연다. 따라서 정식
config-bound recovery가 없거나 실패한 상태에서는 사람 자료를 받을 수 없다.

Python 3.13과 잠금 의존성에서 핵심 136개 및 전체 319개 테스트가 통과했다.
새 기본 합성 recovery의 200회 Monte Carlo 오차 게이트는 통과했지만 단일 대표
D384 bootstrap이 참 beta를 포함하지 못해 전체 `passed=false`였다. seed를 골라
통과시키지 않고 실패 산출물을 그대로 보존한다. 대상 노트북의 비공개 P01
config로 정식 200/200 recovery를 다시 실행해 `passed=true`가 되기 전까지 사람
실험은 승인하지 않는다.

---

## 21. 교정 세션 안의 정답위치·odd 색 교란 수정 (2026-09-27, Codex)

384회 시행표의 전체 굵기×수준×위치×odd 색 빈도와 블록별 굵기 수는 맞았지만,
블록별 위치×odd 색 조합은 맞지 않았다. 이전 4세션 배정에서는 바깥 굵기의
각 세션에 정답위치 두 곳과 odd 색 한쪽만 나타났고, 6세션에서도 바깥 굵기의
odd 색이 세션별로 고정됐다. 따라서 세션별 학습·피로 또는 위치·색 반응 편향이
굵기별 C80 추정에 섞일 수 있었다. 이는 전체 빈도만 검사하던 기존 테스트가
놓친 설계 오류다.

각 굵기×수준의 여덟 위치×odd 조합을 수준 번호에 따라 블록 사이에서
회전하도록 배정하고, 기준 굵기의 두 번째 반복은 반대쪽 블록으로 보낸다.
그 결과 4세션에서 블록별 바깥/기준 굵기의 각 조합은 3/6회, 6세션에서
2/4회다. 시행표 생성기와 PNG compiler가 이 블록별 균형을 검사하고,
4·6세션 회귀 테스트로 고정한다.

이 변경은 사람 자료 수집 전의 설계 오류 수정이다. 이전 코드로 생성한
calibration 일정표·PNG·manifest와 그 구현 hash에 묶인 recovery 결과는
새 코드의 사람 실험 승인 근거로 재사용하지 않는다. 원본은 삭제하지 말고
보존하며, 대상 노트북에서 새 commit으로 모두 재생성·정식 recovery 후
준비도 검사를 다시 통과해야 한다.

별도의 통계적 검토 과제: 현행 공학적 시작 게이트는 단일 대표 모의표본의
95% bootstrap CI가 참값을 반드시 포함하도록 요구한다. 충분히 잘 보정된
95% CI도 반복표본 중 일부에서는 참값을 놓치므로 이 단일 조건만으로
실험 설계의 적합성을 판단하기 어렵다. 현재 실패한 합성 recovery의 D384
beta CI도 참값을 놓쳤다. 이것이 단순 표본 변동인지, beta 추정 편향·CI
미보정인지는 반복 모의표본의 coverage로 따로 확인해야 한다. 사람 자료를
보기 전 팀에서 게이트 변경 여부와 기준을 명시적으로 결정해야 하며, 지금은
기존 `passed=false`를 임의로 뒤집지 않는다.

## 2026-10-29: AOC 준비와 calibration recovery의 반복 coverage 게이트

- 대표 1개 95% CI의 참값 미포함을 전체 설계 실패로 처리하던 판정을 수정했다.
  그 미포함 결과는 산출물 `truth_coverage`에 계속 보존하고 `diagnostic_only`로
  명시한다. 반복 표본에서 CI 성능을 평가해야 하며 모든 CI의 참값 포함을
  요구하는 것은 명목 95% CI의 뜻과 다르다.
- `recovery_coverage.py`: 새 고정 난수열, 세 stress profile 각 20개 이상,
  총 60개 이상 × bootstrap 200회 이상. 전체 포함률 ≥0.90, 단측 exact 95%
  하한 ≥0.80, profile별 ≥0.80의 공학적 과소포함 방지 기준을 적용한다.
  정확한 95% coverage 입증이나 행동 효과의 증거라고 해석하지 않는다.
- 기존 200회 point recovery의 중앙 상대오차·90백분위·수렴·beta 방향 기준과
  B192/C192 paired-reference 검사는 유지했다. 짧은 smoke run은 계속 실패한다.
  readiness가 반복별 기록에서 집계를 다시 계산하고 새 모듈도 provenance에 묶는다.
- 이전 실패와 중앙 profile 40회 진단을 새 게이트의 통과 자료로 재사용하지 않는다.
  새 검증 결과가 나오기 전에 규칙을 코드/문서에 기록했다. 통과 seed 탐색 금지.
- `/output/`, `/tmp/`는 문서 변환·미리보기의 로컬 생성물로 Git에서 제외한다.
  원래 파일은 삭제/이동하지 않으며 코드·schema의 미커밋 차단은 그대로 유지한다.
- 장비 sRGB 모드는 측정 교정이 아니다. 개인 config·원응답·비공개 파라미터는
  공개 기록에 복사하지 않는다. 최종 승인 정보·화면 검증·동결 config-bound
  recovery가 없으면 수집 불가 상태를 유지한다. 이 메모는 사람 실험 승인 아님.
- 이 변경 상태에서 `.venv` Python으로 전체 `python -m unittest -q`를 실행해
  **328개, 294.929초, OK**를 확인했다. 이전의 322개 통과 기록을 재사용한 값이 아니다.
  독립 반복 coverage의 실제 결과는 별도 비공개 JSON에 보존하고 최종 결과를
  확인하기 전에는 `passed=true`나 본실험 준비 완료라고 쓰지 않는다.
- 새 고정 난수열의 **60자료 × bootstrap 200회 검증 완료**: B192/C192 기준 C80
  56/60, B384 55/60, D384 beta 54/60은 새 공학 기준을 통과했다. 그러나
  D384 기준 C80은 **50/60(0.8333), 단측 95% 하한 0.7337**, low-steep
  profile 15/20으로 실패했다. 따라서 최종 `passed=false`를 유지한다.
  전체 elapsed 506.215초. 200회 point recovery는 모두 통과했지만 이것이 CI
  coverage 실패를 상쇄하지 않는다. 단일 대표 CI 문제만 고치면 실험을 시작할
  수 있다는 예상은 이 결과로 반박됐다. 현재 percentile bootstrap 구현에서
  즉시 확인되는 수식·인덱스 오류는 발견하지 못했으며 통계 방법/설계의 추가
  검증이 필요하다. seed 재시도·기준 완화·D384 조용한 제외는 하지 않는다.
  개발 결과는 비공개 `data/experiment/private/`에 보존했다. 최종 config-bound
  성공도 아니고 인간 수집 승인도 아니다.

## 2026-10-29: 범위 유지, D384 CI의 편향·비대칭 보정 개발

- 사용자 결정은 D384를 제외하지 않고 기존 연구 질문/96·192·96회 설계를 유지하는
  것이다. 합격 기준, 신뢰수준, 고정 lapse, MLE 점추정은 변경하지 않았다.
- 실패 60자료의 D384 MLE를 SciPy L-BFGS-B 세 시작점과 독립 대조했다.
  최대 NLL 개선은 1.18e-7, 1e-5 이상 개선은 0/60이었다. 최적화기 국소해
  오류의 근거는 없었다. low-steep에서 평균 log-C80 오차 -0.0703, 추정 w/참 w
  평균 0.758로 유한 표본 편향을 관찰했다. 이 원인 진단은 사람 결과가 아니다.
- 같은 실패 60자료를 개발에만 재사용한 log-basic(역백분위) 구간은 C80 53/60,
  beta 55/60 포함이었다. C80이 기준에 미달해 최종 방법으로 채택하지 않았다.
  개발/실패 기록을 보존하며 이를 독립 검증이라고 부르지 않는다.
- Wichmann & Hill(2001, II), doi:10.3758/BF03194545의 심리측정 CI 편향·왜도와
  BCa 제안을 근거로 D384에 `bca-log`를 구현했다. B192/B384/C192는 percentile
  그대로다. 양수 모수는 log, beta는 원 좌표. delete-one jackknife는 동일한
  (x,y,width) 삭제만 캐시하고 전체 시행 빈도를 정확히 복원한다. 퇴화/미수렴은
  실패 처리하고 더 쉬운 구간으로 자동 fallback하지 않는다.
- 변환된 미사용 굵기 C80도 paired 모수·jackknife를 같이 변환한다. 산출물에
  interval_method와 jackknife C80 복제를 남기고 validator가 동일 방식으로
  CI를 재계산한다. percentile 재검사를 유지해 새 CI를 잘못 거부하던 연결도 수정.
- `recovery_coverage`에 선택적 프로세스 병렬 실행을 추가했다. 각 6자료 chunk의
  seed/profile 순서를 그대로 보존하며 직렬/병렬 결과 완전 일치 회귀검사 통과.
  자동 seed 탐색은 없다. 후보 모형을 골라 재시도하는 기능도 없다.
- 새 방법을 정한 뒤 과거 개발 60자료와 겹치지 않는 seed20261030, point200,
  반복자료120×bootstrap400으로 별도 검증 계획을 비공개 저장하고 실행했다.
  최종 결과 전에는 BCa 통과·본실험 준비 완료라고 주장하지 않는다.
- 선행 근거: https://pubmed.ncbi.nlm.nih.gov/11800459/ 및
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html

### 실제 독립 검증 결과와 수집/추론 분리 결정

- BCa 적용 상태의 새 독립 검증 120자료×400bootstrap 완료, elapsed 917.058초.
  D384 C80 113/120, beta 108/120 및 B384 114/120은 기존 기준 통과.
  B192/C192는 같은 기준굵기 반응/우도를 공유하며 각각 106/120으로 기준 90%
  미달. 두 결과를 240개의 독립 사례로 합치지 않는다. 전체 passed=false를 보존.
- 저장·재검산 연결 수정까지 반영한 전체 테스트 **335개,449.611초,OK**.
  수정 도중 실패했던 테스트를 통과 결과로 쓰지 않았다.
- 사용자에게 미달을 설명한 뒤 사용자가 실험 시작을 요청했다. 연구 질문·모형·
  384회 설계·원래 통과 기준을 유지하면서 교정자료 수집과 확정 추론을 분리했다.
  실패한 recovery를 passed=true로 바꾸지 않으며 기본 차단은 유지한다.
- 최종 config와 실제 통계 입력/구현이 완전히 같은 기존 recovery 원문에 대해서만,
  로컬 운영자가 실패 원문 SHA-256/config SHA-256에 결박한 수집 예외를 명시적으로
  승인할 수 있다. 새 재계산을 했다고 쓰지 않는다. 예외는 coverage 실패에만
  적용하며 point회복/수렴/paired무결성/동의/실제표시/Git·PNG 검사는 면제하지 않는다.
- 이때 parameter_recovery는 blocked 상태를 유지하지만 calibration 수집 차단만
  해제하며 validation 수집과 연구 주장 차단은 유지한다. 원래 validation 전용
  차단 목록에도 parameter_recovery를 추가해 교정 예외가 후속 단계로 번지지 않는다.
- 이 분리와 승인없음/해시변조 차단을 포함한 readiness **15개,16.023초,OK**.
  이는 335개 전체 검사 이후 추가된 readiness 변경의 별도 검사이며 전체336개를
  다시 실행했다고 주장하지 않는다. 실제 final config/승인/PNG는 현장 확인 전이다.

### 2026-10-29 자극 생성 전 현장 결정 반영

운영자가 제공한 원본 3840×2160 캡처에서 네 240×8px 선의 주변 영역이 기준 PNG와
픽셀 단위로 동일함을 확인했고 비공개 원본 근거를 보존했다. 이 범위는 8px
native preview이며 색도 계측이나 모든 조건의 인간 검증이 아니다. 운영자가
별도 64회 시간·피로 pilot을 거부한 결정을 본 시행 전에 기록했다. 6×64 배정,
휴식과 중단 규칙은 유지하고 pilot 미실시를 명시한다. 기관·보호자 승인이나
정밀 계측을 새로 받은 것처럼 기록하지 않는다. 이미 확인한 사용자 진술의
동의/보관 및 collection-only 결정을 최종 설정에 연결하는 단계다.

### 4K 자극 생성 병목 수정 (같은 날, 실제 파일 생성 중)

384장 생성이 느린 원인은 `_pixel_counts`가 8,294,400픽셀을 Python generator와
Counter로 매번 세고, 생성/저장후/재검증 때 반복하는 것이었다. 동일한 전체
픽셀 histogram을 Pillow native `getcolors(maxcolors=width*height)`로 계산한다.
RGB bytes 대조·해시·전체 픽셀 수 검사는 유지한다. 실제 4K 기준 이미지에서
기존/새 결과 정확히 일치; 집계 1.874초→0.061초, 약30.7배. 전체 한 장 생성·
검사는 0.723초, 추가 재검사는0.159초 실측. 전체 속도30배라는 주장은 하지 않는다.
관련 renderer/readiness32개19.364초OK. 중단된 생성물은 보존 후 새 버전으로 재생성.
회복 결과는 변경하지 않는다. 교정수집 예외에서만 통계 의존성 동일성, 이전
renderer Git 원문 hash, `_pixel_counts` 외 AST 동일성 및 새 함수의 정확한
AST hash whitelist를 확인하는 좁은 호환 경로를 둔다. 다른 변경은 거부하고
validation/추론 차단을 유지한다. 새 통계 검증을 했다고 기록하지 않는다.

### 2026-11-08 별도 공통색·얇은 선 탐색 도구 인계

`tools/common_color_pilot/`에 별도 탐색 프로그램과 분석 도구를 추가했다.
기존 `src/` 본실험·동결 프로토콜·교정/검증 게이트는 변경하지 않았다.

- v2: 기존 공통색 설계의 12경로에서 2/4/8/16px·6간격을 검사하는 예비도구.
  큰 간격부터 제시하므로 순서/피로와 얽힌 탐색자료이며 확증 역치로 해석하지 않는다.
- v3: 색차 난도 변화로 선별한 2색 경로, 1/2/3/4/8px, 3색차, 8반복.
  동일 RGB 쌍을 모든 굵기에 반복하고 순서/위치/홀수색을 균형화한다.
  선별은 굵기별 성적 차이가 아니라 굵기를 합산한 색차 반응으로 한다.
- 예측: 명목 원본 sRGB ΔE00 기반 확률 모형을 굵기 무관/로그 굵기/4px포화로
  비교한다. 유한 격자이며 일부 적합의 경계·공통 기울기/lapse 가정을 기록한다.
  굵기 제외 확률 점수는 C80 예측 오차·독립 확증을 대체하지 않는다.
- v4: 새6px 후속 응답 전에 모델/예측/색차 계획을 스냅샷으로 고정한다.
  새 응답으로 재적합하지 않고 세 모델 모두의 Brier/logloss를 비교한다.
- 사람 원응답/설정/선별 계획/모델/개인별 분석 결과는 Git 제외 `raw/`에만 둔다.
  이 커밋에는 코드·공통 설계·문서·합성 검사만 포함한다. 수치 결과를 공개하지 않는다.
- 실제 시행 때의 작업공간 원본 프로그램/원응답은 보존했다. 이식용 코드는 개인 PC
  경로를 없앤 별도 사본이므로 기존 config 해시를 고쳐 강제로 이어 쓰면 안 된다.
  다른 자료를 넣어도 이전 참가자의 결과 문구나 안경 진술을 자동 출력하지 않게 했다.

공통 설계/수학 코드는 같은 저장소 `hanjeonghun-colorvision`의
`95dc87999a50281f3bcbc16eac990da76ae13f4d`에서 바이트 보존하여 포함하고
출처/hash를 도구 README에 적었다. 비공개 데이터 없이 실행한 합성 전용 검사
6개가 통과했다(스케줄/픽셀/저장재개, 격자 합성 회복, v3→v4 고정 모델 평가,
명시적 사후 주석, 손상 자료 거부, 공통 소스 hash). 물리 화면/색도계·임상·모집단
검증이나 기존 `src/` 전체 회귀 재실행을 완료한 것처럼 기록하지 않는다.
