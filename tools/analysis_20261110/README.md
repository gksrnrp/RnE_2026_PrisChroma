# 2026-11-10 분석 스크립트

`docs/results-20261110/`의 결과를 만든 스크립트. 경로는 운영 노트북 기준으로 고정되어 있다.
원자료(`raw/`, 본실험 jsonl)는 비공개 릴리스 `experiment-data-20261110`에 있다.

Python은 전체 경로로 실행한다: `C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe`

| 파일 | 하는 일 | 결과 |
|---|---|---|
| `v5_compare.py <출력.txt>` | P01·P03 v5 굵기별 정답, 세 모형(굵기 무관·로그 굵기·4 px 포화) AIC. P03 값 재현 확인 포함 | 문서 3-7절 |
| `make_v5_p01_outputs.py` | P01 v5 CSV 3개, 그림 4 | `그림4_P03_v5_굵기별.png` |
| `fig_p01_v5.py` | P01 v5 색차 단계별 그림 | `그림6_P01_v5_굵기별.png` |
| `pooled.py <출력.txt> [bootstrap 횟수]` | 참가자별 전체 실험 합산 굵기 모형(굵기 배율 공유), 합산 C80, 보정량 계산 | 문서 6-2절 |
| `pooled_out.py <pooled 출력.txt>` | 합산 결과로 CSV·그림 5·문서 6-2절 작성 | `보정_계산_전체합산.csv`, `그림5_굵기별_최소보정량.png` |
| `v5_pixel_check.py`, `v6_pixel_check.py` | 화면 캡처로 v5·v6 자극 픽셀 확인 | 실험 전 점검 |

보정 계산은 `RnE-2026-prischroma/src/path_correction.py`의 `find_path_constrained_correction`을 쓴다.
