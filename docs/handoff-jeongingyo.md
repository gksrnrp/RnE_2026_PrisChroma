# ②단계 인수인계 · 정인교

갱신 2026-08-20

> **보조 이미지 분석 인수인계:** 이 문서는 SLIC 기반 응용 이미지 탐색 도구의
> 기록이며 현재 사람 C80 연구의 실험 절차가 아니다. 사람 실험의 장치·자극·분석
> 조건은 `docs/experiment-protocol.md`와 `docs/changmin-runbook.md`만 따른다.
> 특히 실험용 시각도는 화면 대각선 추정이 아니라 발광영역 가로 mm, 가로
> 해상도와 관찰거리를 직접 측정해 계산한다.

---

## 이 저장소에 새로 들어온 것

| 파일 | 하는 일 |
|---|---|
| `src/slic_extract.py` | SLIC 분할, 영역 속성, 인접 그래프, 시각도 환산 |
| `src/measure_pairs.py` | 영역 쌍 원본 수치 측정 (판정 없음) |
| `src/export_tables.py` | `regions.csv`, `pairs.csv`, `conditions.md` 생성 |
| `src/generate_synthetic.py` | 정답을 아는 합성 이미지 생성 |
| `src/apply_baseline.py` | 측정 결과를 `run_baseline()`에 연결하는 어댑터 |
| `src/failure_analysis.py` | 분할 실패 유형 분류 |
| `src/param_sweep.py` | 파라미터 민감도 분석 |
| `src/report_assets.py` | 보고서 그림·표 생성 |

테스트는 `test_slic_extract.py`, `test_measure_pairs.py`,
`test_export_tables.py`, `test_failure_analysis.py` 네 개다.

`machado.py`, `baseline.py`, `two_color_demo.py`는 건드리지 않았다.
`cvd_simulation.py`와 `color_utils.py`는 이 저장소에 없었으므로 지울
것이 없었다. 시뮬레이터 구현은 `machado.py` 하나뿐이다.

---

## 이창민에게

### 1. `run_baseline()`은 그대로 쓸 수 있었다

`apply_baseline.correct_measurement()`가 영역 목록을 색쌍으로 바꿔
넘긴다. `baseline.py`는 읽기만 했다.

### 2. 임계값은 설정값으로 유지해 주기 바란다

`apply_baseline.correct_measurement()`에는 더 이상 임계값 기본값이 없다.
호출자가 합성 점검값인지 실제 실험값인지 출처와 함께 명시적으로 넘겨야 한다.
과거의 `original_min=10.0, simulated_max=5.0, target=15.0`은 연구값이 아니며
자동 기본값에서 제거했다.

`simulated_max=3.0`으로는 후보 쌍이 하나도 나오지 않았다. 합성 이미지의
빨강-초록 쌍조차 deutan 시뮬레이션 ΔE00가 4.6이라서다. 3.0이 개발용
임시값이라는 판단이 수치로도 확인된 셈이다.

### 3. 한 영역이 여러 쌍에 걸린다

영역 A가 B와도 C와도 혼동 후보가 되면 `run_baseline`은 쌍마다 다른
보정색을 제안한다. `correct_measurement()`의 반환값에
`region_conflict_count`로 세어 둔다. 영역 단위 전역 최적화는 아직 없으므로
충돌 영역은 `unresolved_region_ids`로 남기고 렌더링에서 원본색을 유지한다.

### 4. 속도

`find_minimum_correction`은 쌍당 5~6초다. 이미지 한 장에서 후보 쌍이
수십 개 나오므로 전체 데이터셋에 그냥 돌리면 오래 걸린다.
`correct_measurement(limit=...)`로 상한을 두고 있다.

---

## 한정훈에게

### 1. 자극 조건은 `docs/experiment-protocol.md`에서만 정한다

`results/conditions.md`는 합성 이미지에서 만든 폐기된 굵기×거리 탐색안이다.
0.62 환산이나 그 표의 조건을 사람 실험에 사용하지 않는다. 현재 연구는
거리·길이·배경을 고정하고 실제 장치에서 세 굵기의 C80을 직접 측정한다.

### 2. 개인 파라미터 형식

`measure_image(cvd_conditions=[("deutan", 0.7)])`처럼 넘기면 그 조건의
시뮬레이션 색차가 열로 추가된다. 열 이름은 `delta_e_sim_deutan_070`이다.
측정된 `type`과 `severity`를 이 형태로 주면 바로 반영된다.

역치는 ②단계에 넣지 않는다. 수치만 넘기고 자르는 일은 ①단계 몫이다.

### 3. 시각도 환산에 필요한 값

화면 대각 인치, 해상도, 관찰 거리 세 가지가 있어야 픽셀을 arcmin으로
바꿀 수 있다. `data/real/sources.csv`에 이미지마다 적는다. 비면 그
이미지는 arcmin 열 없이 픽셀 값만 남는다.

---

## 아직 안 된 것

1. **실제 이미지 20~30장.** 저작권과 출처 확인이 필요해 사람이 직접
   골라야 한다. `data/real/README.md`에 수집 규칙과 실행 명령을 적어
   두었다. 이미지를 채우면 명령 두 줄로 `conditions.md`가 갱신된다.
2. 실제 이미지가 들어온 뒤 `report_assets.build_all()`을 다시 돌려야
   데이터셋 표에 실제 이미지 항목이 채워진다.

---

## 측정에서 드러난 한계 (보고서에 쓸 것)

1. **3 px 이하 선은 어떤 설정으로도 잡히지 않는다.** n_segments를
   3000까지 올려도 검출률 0%다. 안티앨리어싱으로 선 색과 배경색이 섞여
   대표색이 어긋나기 때문이다. superpixel을 잘게 쪼개는 것으로는 해결
   되지 않는다.
2. **굵기별 문턱값이 뚜렷하다.** 8 px 선은 n_segments 300부터, 6 px은
   500~800부터 잡힌다. 4 px은 800 이상에서도 12~25%에 그친다.
3. **merge_thresh는 0이면 안 된다.** 병합을 끄면 두꺼운 요소가 수십
   조각으로 나뉘어 검출률이 8%까지 떨어진다. 0.5 이상이면 값에 거의
   무관하게 평평하다. 색이 같은 배경 superpixel을 합치는 것이 효과의
   대부분이다.
4. **거리로 만든 쌍은 범례를 놓친다.** `legend_lines_w08` 장면에서
   범례와 같은 색 선의 쌍이 d_max에 따라 몇 개나 잡히는지 세었다.
   `results/sweep_summary.md`의 d_max 표에 있다.
5. **경계상자 간격은 실제 거리보다 작다.** 대각으로 뻗은 선에서 특히
   그렇다. 정확한 경계 사이 거리는 쌍이 수만 개일 때 계산 비용이 커서
   근사를 택했다.
