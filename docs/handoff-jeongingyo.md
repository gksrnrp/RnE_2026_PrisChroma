# ②단계 인수인계 · 정인교

갱신 2026-08-20

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

지금 `apply_baseline.DEVELOPMENT_THRESHOLDS`에
`original_min=10.0, simulated_max=5.0, target=15.0`을 두었다. 파이프라인이
끝까지 이어지는지 보려고 고른 개발용 값이고 연구 결과가 아니다.
①단계 실험이 끝나면 통째로 교체한다.

`simulated_max=3.0`으로는 후보 쌍이 하나도 나오지 않았다. 합성 이미지의
빨강-초록 쌍조차 deutan 시뮬레이션 ΔE00가 4.6이라서다. 3.0이 개발용
임시값이라는 판단이 수치로도 확인된 셈이다.

### 3. 한 영역이 여러 쌍에 걸린다

영역 A가 B와도 C와도 혼동 후보가 되면 `run_baseline`은 쌍마다 다른
보정색을 제안한다. `correct_measurement()`의 반환값에
`region_conflict_count`로 세어 두었다. 영역 단위로 색 하나를 정하는
규칙은 아직 없다. ③단계에서 정해야 할 문제로 넘긴다.

### 4. 속도

`find_minimum_correction`은 쌍당 5~6초다. 이미지 한 장에서 후보 쌍이
수십 개 나오므로 전체 데이터셋에 그냥 돌리면 오래 걸린다.
`correct_measurement(limit=...)`로 상한을 두고 있다.

---

## 한정훈에게

### 1. 자극 조건은 `results/conditions.md`에 있다

굵기는 5%와 50%, 거리는 25%와 75% 두 수준씩 잡으면 2×2 네 조건이 된다.
각 조건에서 잰 역치를 0.62로 나누면 환산 비율이 나온다.

**지금 `conditions.md`의 수치는 합성 이미지 기준이다.** 실제 노선도와
그래프를 넣어 다시 돌려야 실험에 쓸 수 있는 값이 된다. 실제 이미지
수집이 남은 일이다.

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
