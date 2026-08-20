# AI 도구용 프로젝트 지침

이 파일은 Codex와 Claude Code가 **둘 다** 읽는 공용 지침이다.
(Codex는 `AGENTS.md`를, Claude Code는 `CLAUDE.md`를 자동으로 읽는다.
`CLAUDE.md`는 이 파일을 가리키므로 내용은 여기 한 곳에만 둔다.)

한국어로 답한다.

---

## 프로젝트

적록색약자를 위한 개인화 선택적 AI 색보정 (R&E 2026).

혼동 상대가 있는 색만 골라서, 구별 가능한 **가장 가까운 색**으로 최소 이동시킨다.
보색 치환이 아니다. 화면 전체를 바꾸는 것도 아니다.

연구 합의사항 전체는 [docs/project-context.txt](docs/project-context.txt)에 있다.
**이 문서의 연구 방향을 임의로 바꾸지 말 것.** 불확실한 부분은 구현 전에
TODO 또는 실험 변수로 남긴다.

---

## 어디에 뭐가 있나

| 경로 | 내용 |
|---|---|
| `src/` | baseline 구현, 단위 테스트, 탐색 벤치마크 |
| `docs/project-context.txt` | 연구 합의사항 (최상위 기준) |
| `docs/research-plan.txt` | 담당 파트 계획과 알고리즘 설명 |
| `docs/code-review.md` | **무엇을 왜 고쳤는지, 남은 한계** |
| `results/` | 실험 결과 JSON |

**작업을 이어받으면 `docs/code-review.md`를 먼저 읽는다.** 이미 검토·수정된
내용을 다시 논의하지 않기 위해서다.

---

## 실행

모듈이 서로를 직접 import하므로 **반드시 `src/` 안에서 실행한다.**

```bash
cd src
python -m unittest -q                 # 테스트 37개, 약 30초
python two_color_demo.py --help
python benchmark_search.py --pairs 30 --output ../results/search_benchmark.json
```

---

## 이 프로젝트에서 지켜야 할 것

### 1. 색쌍 하나로 일반화하지 않는다

**이 프로젝트에서 두 번 반복된 실수다.** 탐색 안정성, step 의존성, 최소값
수렴 같은 주장을 색쌍 하나로 확인하고 "일반적으로 그렇다"고 쓰면 안 된다.

`src/benchmark_search.py`로 여러 색쌍에서 측정하고, 그 수치를 문서에 적는다.
결과 JSON은 `results/`에 보관한다 (seed와 계산식이 함께 저장된다).

### 2. 근거보다 강한 문장을 쓰지 않는다

독스트링과 README는 논문으로 옮겨간다. "step의 영향은 작다" 같은 문장은
반증 가능한 주장이므로, 측정 범위를 함께 적는다.

### 3. 탐색 결과는 전역 최적해가 아니다

radial grid + 지역 RGB cube + multi-start 방식이다. 개선됐지만 seed 의존성이
남아 있다. "최소 색이동"으로 보고할 때는 이 점을 밝힌다.

### 4. 재혼동 제약의 의미를 좁혀서 이해한다

`palette` 제약이 보장하는 것은 *"보정이 새 충돌을 만들지 않는다"* 이지
*"장면 전체가 완벽히 구별된다"* 가 아니다. 이동하지 않은 색에는 적용되지 않는다.

### 5. 임계값에 기본 정답값을 두지 않는다

`tau_high` / `tau_low` / `tau_target`은 아직 실험 변수다. 사용자 데이터로
조정해야 하며 코드에 임의의 기본값을 넣지 않는다.

### 6. AI 단계는 baseline 완성 후

규칙 기반 baseline이 검증되기 전에 U-Net 등으로 넘어가지 않는다.

---

## 도구를 바꿔 작업할 때

Codex와 Claude Code를 번갈아 쓰는 경우, **세션 자체는 연결되지 않는다.**
저장소가 유일한 인수인계 통로다.

1. 작업을 끝내면 **커밋하고 push** 한다 (안 하면 상대 도구가 못 본다)
2. 무엇을 왜 바꿨는지 `docs/code-review.md`에 남긴다
3. 다른 도구를 시작하면 먼저 `git pull` 한다

커밋 메시지는 한 줄 요약 + 필요하면 이유를 적는다. `수정`, `ㅇㅇ` 같은
메시지는 인수인계가 안 된다.
