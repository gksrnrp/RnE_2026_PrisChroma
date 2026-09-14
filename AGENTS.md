# AI 도구용 프로젝트 지침

이 파일은 Codex와 Claude Code가 **둘 다** 읽는 공용 지침이다.
(Codex는 `AGENTS.md`를, Claude Code는 `CLAUDE.md`를 자동으로 읽는다.
`CLAUDE.md`는 이 파일을 가리키므로 내용은 여기 한 곳에만 둔다.)

한국어로 답한다.

---

## 프로젝트

고정 표시 조건에서 적록색각이상 사례 참가자의 선 굵기별 C80 예측과
제약 기반 쌍 단위 최소 색이동 보정 (R&E 2026).

연구용 보정은 측정한 Lab 중점·방향의 대칭 색쌍만 대상으로, 예측 C80 제약을
만족하는 유한 8-bit 동일경로 후보 중 OKLab 이동 합 최소를 찾는다. 임의 RGB의
A만/B만/둘 다 탐색은 별도 공학 baseline이며 인간 검증 없이 행동 효과로
해석하지 않는다. 보색 치환이나 화면 전체 변환이 아니다.

현재 연구의 최상위 기준은 [docs/experiment-protocol.md](docs/experiment-protocol.md)다.
`project-context.txt`와 `research-plan.txt`는 이전 AI 확장안의 기록이며 현재
가설이나 실험 절차의 근거로 사용하지 않는다. 불확실한 부분은 구현 전에
TODO 또는 실험 변수로 남긴다.

---

## 어디에 뭐가 있나

| 경로 | 내용 |
|---|---|
| `src/` | baseline 구현, 단위 테스트, 탐색 벤치마크 |
| `docs/experiment-protocol.md` | 현재 연구 질문·분석·중단 규칙 (최상위 기준) |
| `docs/project-context.txt` | 이전 AI 확장안 기록 (비활성) |
| `docs/research-plan.txt` | 이전 담당 계획 기록 (비활성) |
| `docs/code-review.md` | **무엇을 왜 고쳤는지, 남은 한계** |
| `results/` | 실험 결과 JSON, 측정 산출물 |
| `docs/handoff-jeongingyo.md` | ②단계 인수인계와 팀원 요청 사항 |

**작업을 이어받으면 `docs/code-review.md`를 먼저 읽는다.** 이미 검토·수정된
내용을 다시 논의하지 않기 위해서다.

---

## 실행

모듈이 서로를 직접 import하므로 **반드시 `src/` 안에서 실행한다.**

```bash
cd src
python -m unittest -q                 # 현재 전체 단위·통합 회귀검사
python two_color_demo.py --help
python benchmark_search.py --pairs 30 --output ../results/search_benchmark.json
```

---

## 이 프로젝트에서 지켜야 할 것

### 1. 색쌍 하나로 일반화하지 않는다

**이 프로젝트에서 두 번 반복된 실수다.** 탐색 안정성, step 의존성, 최소값
수렴 같은 주장을 색쌍 하나로 확인하고 "일반적으로 그렇다"고 쓰면 안 된다.

`src/benchmark_search.py`로 여러 색쌍에서 측정하고, 그 수치를 문서에 적는다.
현재 코드·환경·commit에 결합된 결과 JSON을 `results/`에 보관해야만 근거로
사용한다(seed, 계산식, 실행환경과 구현 provenance 포함). 예전 수치만 남아
있거나 입력·구현 hash가 없으면 역사적 개발 메모일 뿐 현재 성능 근거가 아니다.

### 2. 근거보다 강한 문장을 쓰지 않는다

독스트링과 README는 논문으로 옮겨간다. "step의 영향은 작다" 같은 문장은
반증 가능한 주장이므로, 측정 범위를 함께 적는다.

### 3. 두 보정 경로의 최소 범위를 구분한다

radial grid + 지역 RGB cube + multi-start + 반복 정밀화 방식이다. 과거 고정
30쌍 개발 실행에서는 step 비단조가 관찰되지 않았지만, 그 산출물은 현재
구현 provenance에 결합되지 않아 현재 성능 근거로 재사용하지 않는다. 새
benchmark를 현재 commit에서 재실행해도 전역 최적해를 보장하지 않는다.
임의 RGB baseline은 반드시 `쌍 단위 근사해`로 쓴다.

연구용 동일경로 보정은 명시한 scale grid를 8-bit로 양자화한 유일 후보를 모두
열거하므로 **그 유한 후보 집합 안에서는** 정확한 최소다. 연속 색공간·다른
Lab 방향·전체 팔레트의 전역 최소라고 쓰지 않는다.
자극 생성용 목표 허용오차를 보정 후보의 초과량 상한으로 재사용하지 않는다.
표시 RGB에서 다시 계산한 `Machado 후 ΔE00 >= C80` 후보 전부를 비교한다.

### 4. 재혼동 제약의 의미를 좁혀서 이해한다

`palette` 제약이 보장하는 것은 *"보정이 새 충돌을 만들지 않는다"* 이지
*"장면 전체가 완벽히 구별된다"* 가 아니다. 이동하지 않은 색에는 적용되지 않는다.

### 5. 임계값에 기본 정답값을 두지 않는다

`tau_high` / `tau_low` / `tau_target`은 아직 실험 변수다. 사용자 데이터로
조정해야 하며 코드에 임의의 기본값을 넣지 않는다.

### 6. AI는 이번 연구 범위가 아니다

이번 연구는 규칙 기반 사례연구로 끝낸다. U-Net 등은 참가자와 데이터가
충분해진 뒤의 별도 후속 연구이며 현재 결과에 포함하지 않는다.

---

## 도구를 바꿔 작업할 때

Codex와 Claude Code를 번갈아 쓰는 경우, **세션 자체는 연결되지 않는다.**
저장소가 유일한 인수인계 통로다.

1. 작업을 끝내면 **커밋하고 push** 한다 (안 하면 상대 도구가 못 본다)
2. 무엇을 왜 바꿨는지 `docs/code-review.md`에 남긴다
3. 다른 도구를 시작하면 먼저 `git pull` 한다

커밋 메시지는 한 줄 요약 + 필요하면 이유를 적는다. `수정`, `ㅇㅇ` 같은
메시지는 인수인계가 안 된다.

---

## 보조 이미지 탐지·측정 규칙 — 정인교

현재 주 연구는 통제 선 4AFC이며 실제 이미지 탐지는 선택적 응용 사례다.
필수 파이프라인으로 섞지 않는다. 보조 탐지 코드는 아래 파일들이고, 이창민 소유
파일(`machado.py`, `baseline.py`, `two_color_demo.py`)은 읽거나 호출만
한다.

`slic_extract.py` `measure_pairs.py` `export_tables.py`
`generate_synthetic.py` `apply_baseline.py` `failure_analysis.py`
`param_sweep.py` `report_assets.py`

### 혼동 여부를 판정하지 않는다

②단계 출력에 `is_confused` 같은 0/1 열을 넣지 않는다. 색차·거리·크기
원본 수치만 기록하고 임계값 결정은 ①단계 실험 결과에 맡긴다.
`test_measure_pairs.py`가 이 규칙을 테스트로 고정하고 있다.

큰 색면 세션에서 관측된 0.62는 특정 참가자·세션·과제의 값이다. 선 C80의
기본값이나 자동 환산 기준으로 사용하지 않는다. 보정 함수에는 임계값을
출처와 함께 명시적으로 넘긴다.

### 파라미터 기본값과 근거

`n_segments=800`, `compactness=3`, `merge_thresh=1.0`, `d_max=200`.
아래 값은 `results/sweep_summary.md`에 남은 **과거 합성 개발 실행**의 요약이다.
현재 코드·환경에 결합된 재실행 전에는 현재 파이프라인 성능 근거로 사용하지
않고, 선택적 이미지 응용의 후보 설정을 정하는 역사적 참고로만 사용한다.

- 8 px 선은 n_segments 300부터, 6 px은 500~800부터 잡힌다.
  **3 px 이하는 3000까지 올려도 검출률 0%다.** 안티앨리어싱으로 대표색이
  배경과 섞이는 것이 원인이라 superpixel을 더 잘게 쪼개도 풀리지 않는다.
- `merge_thresh=0`은 쓰지 않는다. 두꺼운 요소가 수십 조각으로 나뉘어
  장면 평균 검출률이 8%까지 떨어진다. 0.5 이상이면 값에 거의 무관하다.
- `d_max=200`에서는 범례와 같은 색 선의 쌍을 하나도 잡지 못했고
  400에서 5쌍을 모두 잡았다. 대신 쌍 수가 30개에서 55개로 늘었다.

### 산출물 위치

| 파일 | 커밋 | 내용 |
|---|---|---|
| `results/conditions.md` | O | 폐기된 0.62 환산안의 역사 기록(현재 입력 금지) |
| `results/sweep_summary.md` | O | 파라미터 민감도 요약 |
| `results/failure_summary.md` | O | 분할 실패 유형별 건수 |
| `results/report_tables.md` | X | 재생성 가능한 보고서용 표(로컬 보관) |
| `results/*_sweep.csv` | O | 민감도 원본 수치 |
| `results/figures/` | X | 재생성 가능한 보고서 그림(로컬 보관) |
| `results/pairs.csv`, `regions.csv` | X | 용량 |
| `results/measurements_*.json` | X | 재생성 가능 |
| `data/synthetic/` | X | `generate_synthetic.py`로 재생성 |
| `data/real/` | 조건부 | 원본은 출처·라이선스·hash가 완결된 경우만 공유 |
