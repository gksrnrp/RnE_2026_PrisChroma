# 고정 표시 조건에서 적록색각이상 사례 참가자의 선 굵기별 C80 예측과 제약 기반 쌍 단위 최소 색이동 보정

고정한 화면·과제·색 방향에서 선 굵기에 따른 개인별 80% 정답 색차(C80)를
측정·예측하고, 그 값을 같은 Lab 경로의 유한 후보 집합 안에서 정확한 최소
색이동 제약으로 연결하는 R&E 2026 사례연구입니다. AI는 이번 연구 범위가
아닙니다.

현재 상태는 **소프트웨어와 합성자료 검증 단계**입니다. 실제 참가자 C80,
굵기 효과, 모델 우위, 가독성 향상은 아직 관측되지 않았으므로 연구 결과로
주장하지 않습니다. 사람 실험 시작 가능 여부는 아래 준비도 검사로 확인합니다.

## 폴더 구조

| 폴더 | 용도 |
|---|---|
| `src/` | 분석·실험 스크립트 (.py) |
| `data/` | 입력 데이터 (큰 파일은 올리지 말 것 — 아래 주의사항 참고) |
| `results/` | 그래프, 표, 출력 결과 |
| `docs/` | 회의록, 참고자료 정리, 보고서 초안 |

---

## 코드 실행

**모든 명령은 `src/` 안에서 실행합니다.** 모듈들이 서로를 직접 import하기 때문에
다른 폴더에서 실행하면 `ModuleNotFoundError`가 납니다.

```bash
cd src
pip install -r ../requirements.txt
```

논문 결과를 재현할 때는 검증된 Python 3.13.15/Tk 8.6.15 환경의 정확한 버전 묶음인
`requirements-lock.txt`를 사용합니다. `requirements.txt`는 새 개발환경용
최소 조건이고, 잠금 파일은 결과 재현용입니다.

임의 두 색 보조 소프트웨어 예시(아래 숫자는 연구 임계값이 아님):

```bash
python two_color_demo.py --color-a 220 50 50 --color-b 60 160 60     --type deutan --severity 0.8     --original-threshold 20 --confusion-threshold 15 --target-threshold 15
```

주변 색과의 재혼동까지 막으려면 `--palette` 에 R G B 를 3개씩 이어서 넣습니다.

자동 테스트 (전체 검사는 현재 환경에서 수 분 걸릴 수 있음):

```bash
python -m unittest -q
```

이미지 탐지·측정 탐색 파이프라인(주 심리측정 연구와 별도인 보조 도구):

```bash
python -c "from generate_synthetic import generate_dataset; generate_dataset('../data/synthetic')"
python -c "from measure_pairs import measure_directory; measure_directory('../data/real/images', '../results/measurements_real.json', n_segments=800, merge_thresh=1.0, sources_csv='../data/real/sources.csv')"
python -c "from export_tables import export_all; export_all('../results/measurements_real.json', '../results')"
```

탐색 안정성 벤치마크 (현재 환경에서 약 8분, 결과는 `results/`에 보관):

```bash
python benchmark_search.py --pairs 30 --output ../results/search_benchmark.json
```

선 굵기 실험 전 px→시각도 확인(숫자는 실제 실험 장치를 직접 재서 입력):

```bash
python plan_line_widths.py --display-width-mm <화면가로mm> --resolution-px <가로해상도> --viewing-distance-mm <관찰거리mm> --widths-px <얇음> <중간> <두꺼움>
```

출력의 `inside_szafir_measured_range`가 `false`인 굵기는 Szafir 선행모델의
측정 범위 밖이다. 경고를 무시하고 본 실험을 시작하지 않는다.

보조 소프트웨어용 공통 데이터 형식 초안 검사(가짜 데이터만 허용):

```bash
python validate_contracts.py --all-examples
python validate_contracts.py --contract participant_thresholds --data ../data/contracts/participant_thresholds.example.json
```

`data/contracts/`의 `schema_version=0.1-draft` 예제는 임의 색쌍 소프트웨어의
합성 연결 시험용이며 사람 자료 수집용이 아니다. 사람 실험용 1.0 계약은
`schemas/human/`에 별도로 있다. 다만 계약 파일이 존재한다는 사실만으로 수집할
수 있는 것은 아니며, 참가자별 실제 config·매니페스트·장치 점검·윤리 절차와
config/plan-bound recovery를 모두 완료해야 한다.

확정한 4AFC direct-C80 식의 단위검사와 합성 parameter recovery:

```bash
python -m unittest -q test_psychometrics test_parameter_recovery
python parameter_recovery.py --simulation-replicates 200 --bootstrap-replicates 200 --output ../results/parameter_recovery.synthetic.json
```

합성 출력은 분석 코드가 알려진 모수를 되찾는지 보는 점검 자료이며 사람의
역치나 연구 결론이 아닙니다. 위 기본 출력은 아직 **후보 고정 색차 수준**을
검사한 것입니다. 실제 `experiment_config.json`을 동결한 뒤에는 반드시 다음처럼
그 파일의 굵기·Lab 방향·12개 수준·고정 lapse로 다시 통과해야 합니다.

Recovery는 12개 수준과 두 굵기 모형의 배율에서 가능한 기준-C80 범위를 계산해
low-steep/central/high-shallow 진실값을 결정한다. 세 profile 각각과 전체에서
수렴률, 중앙오차와 90백분위 오차를 통과해야 한다. 별도의 대표 자료에서는
B192·B384·C192·D384 bootstrap이 알려진 C80(및 D384 beta)을 95% CI에 포함해야
하며, 같은 기준 굵기 192회를 쓰는 B192/C192의 bootstrap 표본이 정확히 같아야
한다. 200회보다 적은 simulation 또는 bootstrap은 개발 점검은 가능하지만
`passed=false`이므로 사람 수집 게이트를 열지 못한다.

```bash
python parameter_recovery.py --experiment-config ../data/experiment/private/experiment_config.json --simulation-replicates 200 --bootstrap-replicates 200 --output ../results/parameter_recovery.config-bound.json
```

동결된 교정 시행 JSONL에서 참가자별 네 모형과 held-out 후보 예측을 만드는
분석기는 실제 입력 파일을 반드시 요구합니다.

```bash
python analyze_calibration.py --config ../data/experiment/private/experiment_config.json --trials ../data/experiment/raw/P01-calibration.jsonl --calibration-manifest ../data/experiment/calibration/calibration-manifest.json --bootstrap-replicates 1000 --bootstrap-seed 20260914 --output ../results/calibration_model_fits.json
```

이 분석기는 config 원문 hash, 참가자 profile과 고정 lapse 출처, 네 session
block의 완전 균형, 굵기별 96/192/96 완료 시행을 모두 대조합니다. C192은 config에
동결된 Lab 방향·Szafir 확률 좌표만 사용하고 0.05~0.35도 밖을 외삽하지
않습니다. 출력은 독립 validation block을 관측하지 않은
`calibration_only_intermediate`이며 `participant_thresholds` 최종 계약이
아닙니다. 합성 fixture는 `--allow-synthetic`을 명시해야 하고 언제나 연구 결론
불가로 표시됩니다.

교정 적합, 검증 계획, 검증 분석, validation recovery 산출물은 실행 wrapper 한
개의 hash만 저장하지 않습니다. 실제 계산에 쓰인 심리측정·색변환·자극 검증
모듈, JSON schema, `requirements-lock.txt`의 프로젝트 상대 경로별 SHA-256을
`implementation_hashes`에 저장합니다. 다음 단계는 이 키 집합과 현재 파일
hash를 정확히 다시 비교하므로, 의존 파일 하나라도 바뀌면 원자료를 보존한 채
해당 단계부터 다시 실행해야 합니다.

사람 실험·연구 결론 준비도 검사:

```bash
python research_readiness.py --format markdown
```

교정 전에는 `ready_for_calibration_collection`, 검증 전에는
`ready_for_validation_collection`을 각각 확인한다. `ready_for_human_data`는
교정 준비도의 이전 이름을 유지한 호환 필드일 뿐이다. 해당 단계 값이 `true`가
되기 전에는 사람 응답을 수집하지 않고,
`ready_for_research_claims=true`가 되기 전에는 효과나 우위를 결론내리지 않습니다.

사람 검증 분석까지 끝난 뒤의 **연구용 주 보정**은 측정한 Lab 중점·방향의
대칭 경로만 탐색한다. 사람 자료에서는 validation plan에 미리 동결된 원본
색쌍 8수준을 전부 같은 규칙으로 계산한다. 결과를 본 뒤 한 수준을 고르는
선택은 허용하지 않는다.

```bash
python run_path_correction_analysis.py --config ../data/experiment/private/experiment_config.json --validation-plan ../data/experiment/validation/validation-plan.json --participant-thresholds ../results/participant_thresholds.json --participant-id P01 --models B192 B384 C192 D384 --target-policy point --output ../results/path_correction_analysis.json
```

사람 모드에서는 config·plan·최종 C80 파일의 참가자, CVD 설정, hash, 선택 굵기,
`confirmed=true`와 세 블록 안정성을 모두 대조한다. 출력이 보장하는 것은
`동결한 유한 동일경로 후보 안의 계산상 최소`뿐이다. 보정 전·후를 새 사람
시행으로 비교하지 않았다면 실제 가독성 향상을 주장하지 않는다.
config의 목표 허용오차는 8-bit 자극 수준 생성용이며 보정 후보를 목표 초과량으로
제외하지 않는다. 보정에서는 표시 RGB의 Machado 후 ΔE00가 C80 이상인 유한
후보 전부가 최소화에 참여한다.
네 모형은 전부 고정 순서로 실행하며, 보정 계산의 모든 핵심 의존 파일 hash와
깨끗한 Git commit을 결과에 남긴다. 일부 모형 선택이나 commit되지 않은 구현은
사람 분석에서 차단된다.
`--source-level-index`는 합성 `--development-mode` 단일 수준 연결 시험에만
남겨 둔 호환 옵션이며 사람 자료에서 지정하면 실행을 차단한다.

가짜 임의 색쌍과 가짜 개인 C80을 일괄 연결하는 **보조 software baseline** 시험:

```bash
python run_personal_correction.py --pairs ../data/contracts/validation_pairs.example.json --thresholds ../data/contracts/participant_thresholds.example.json --participant P_SYNTHETIC_01 --model D384 --stroke-width-px 3 --original-min 10 --search-step 0.02 --output ../results/correction_results.synthetic.json --development-mode
python validate_contracts.py --contract correction_results --data ../results/correction_results.synthetic.json
```

`--development-mode`은 현재 가짜 자료에서만 쓴다. 실제 자료 형식이 팀 합의로
확정되기 전에는 이 결과를 연구 결과로 해석하지 않는다. 출력의
`approximate_pairwise_not_global`은 색쌍별 근사 탐색이며 전체 화면의 전역
최적화를 보장하지 않는다는 뜻이다.

### 문서

| 파일 | 내용 |
|---|---|
| [docs/experiment-protocol.md](docs/experiment-protocol.md) | 현재 연구 질문·분석·중단 규칙 (최상위 기준) |
| [docs/changmin-runbook.md](docs/changmin-runbook.md) | 이창민이 준비부터 두 참가자 인계까지 실행하는 정확한 순서 |
| [docs/validation-workflow.md](docs/validation-workflow.md) | 64회×3 검증 계획·joint C80 분석·GLS 통합 절차 |
| [docs/research-audit-final.md](docs/research-audit-final.md) | 선행연구·설계·증거·한계 종합 감사 |
| [docs/human-session-record-template.md](docs/human-session-record-template.md) | 동의·장치·중단·노출 사고를 비공개로 기록하는 세션 양식 |
| [data/contracts/human_experiment_config.template.json](data/contracts/human_experiment_config.template.json) | 실제값으로 복사·확정하기 전에는 수집을 차단하는 사람 실험 설정 양식 |
| [docs/software-verification.md](docs/software-verification.md) | 현재 commit의 최종 자동검사·환경·준비도 증거 |
| [docs/project-context.txt](docs/project-context.txt) | 이전 AI 확장안 기록 (비활성) |
| [docs/research-plan.txt](docs/research-plan.txt) | 이전 담당 계획 기록 (비활성) |
| [docs/code-review.md](docs/code-review.md) | 무엇을 왜 고쳤는지, 남은 한계 |
| [schemas/human/](schemas/human/) | 사람 실험용 1.0 계약 |
| [data/contracts/](data/contracts/) | 보조 소프트웨어 합성 연결용 0.1-draft 예제 |
| [AGENTS.md](AGENTS.md) | Codex·Claude Code 공용 작업 지침 |
| [docs/handoff-jeongingyo.md](docs/handoff-jeongingyo.md) | 이전 SLIC 응용 이미지 탐색의 보조 인수인계 (사람 C80 실험 절차 아님) |
| [data/real/README.md](data/real/README.md) | 실제 이미지 수집 규칙 |

---

## 팀원용 — 처음 한 번만 하는 설정

1. **GitHub 계정 만들기** → https://github.com/signup
2. **GitHub Desktop 설치** → https://desktop.github.com (명령어 몰라도 됩니다)
3. 팀장이 보내준 초대 메일 수락
4. GitHub Desktop에서 `File → Clone repository` → 이 저장소 선택 → `Clone`

끝. 이제 내 컴퓨터에 프로젝트 폴더가 생깁니다.

---

## 매일 작업하는 순서 (이것만 지키면 됩니다)

### 1️⃣ 작업 시작 전 — 반드시 먼저!
GitHub Desktop 상단의 **`Fetch origin`** 버튼을 누릅니다.
→ 다른 팀원이 올린 최신 내용을 내 컴퓨터로 받아옵니다.
→ **이걸 건너뛰면 충돌(conflict)이 납니다.**

### 2️⃣ 작업하기
평소처럼 파일을 만들고 수정합니다.

### 3️⃣ 작업 끝나면 — 올리기
1. GitHub Desktop 왼쪽에 바뀐 파일 목록이 뜹니다
2. 아래 칸에 **무엇을 했는지 한 줄** 적습니다
   - 좋은 예: `온도 데이터 그래프 그리는 코드 추가`
   - 나쁜 예: `수정`, `ㅇㅇ`, `asdf`
3. **`Commit to main`** 클릭
4. 상단 **`Push origin`** 클릭 ← 이걸 눌러야 진짜로 올라갑니다

> 💡 하루 작업 다 하고 한 번에 올리지 말고, **의미 있는 단위마다 자주** 올리세요.
> 자주 올릴수록 충돌이 적게 납니다.

---

## ⚠️ 충돌(conflict) 안 나게 하는 규칙

git이 처음이면 충돌이 제일 무섭습니다. 아래만 지키면 거의 안 납니다.

1. **같은 파일을 두 명이 동시에 고치지 않기**
   - 파일 단위로 담당을 나누세요 (아래 담당표 참고)
2. **작업 시작 전 항상 `Fetch origin`**
3. **작업 끝나면 바로 Push** (내 컴퓨터에만 며칠씩 쌓아두지 않기)
4. 급하게 같은 파일을 고쳐야 하면 **단톡방에 먼저 말하기**

### 만약 충돌이 났다면
혼자 해결하려다 파일이 날아갈 수 있습니다.
**아무것도 누르지 말고 팀장에게 알려주세요.**

---

## 담당표

| 파일 / 폴더 | 담당자 |
|---|---|
| `src/machado.py`, `src/correction.py`, 심리측정 분석 | 이창민 |
| 자극·사람 응답 원자료(비공개), 개인 C80 적합 | 한정훈 |
| 실제 표시 조건·응용 이미지 출처와 측정 | 정인교 |
| 연구 질문·검증·보고서 | 세 명 공동 |

---

## ⚠️ 올리면 안 되는 것

- **100MB 넘는 파일** — GitHub이 거부합니다. 큰 데이터는 구글 드라이브에 두고, 링크만 `docs/`에 적어두세요.
- **개인정보** — 설문 응답자 이름, 연락처, 학번 등
- **비밀번호 / API 키**
- 한 번 올리면 지워도 기록에 남습니다. **올리기 전에 확인하세요.**

---

## 파이썬 환경 맞추기 (코드 돌릴 때)

새 패키지를 설치했다면 `requirements.txt`에 추가해주세요.

```bash
cd src
pip install -r ../requirements.txt
```
