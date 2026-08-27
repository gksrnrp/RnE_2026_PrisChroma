# 분할 실패 유형별 건수

설정: n_segments=800, compactness=3.0, merge_thresh=1.0

| 유형 | 건수 | 비율 |
|---|---|---|
| `detected` | 111 | 26.2% |
| `merged_with_background` | 297 | 70.2% |
| `fragmented` | 0 | 0.0% |
| `color_shifted` | 10 | 2.4% |
| `not_covered` | 5 | 1.2% |
| 합계 | 423 | |

요소 수를 그대로 합친 값이라 점이 320개인 `dot_plate` 한 장이 전체 비율을 좌우한다. 장면별 표를 함께 본다.

## 요소 굵기별 검출률

굵기를 구간으로 묶었다. `dot_plate`의 점은 지름이 무작위라 값 하나하나로 세면 표가 요소 수만큼 길어진다.

| 굵기 구간 (px) | 요소 수 | detected | 검출률 |
|---|---|---|---|
| 1~2 미만 | 8 | 0 | 0.0% |
| 2~4 미만 | 21 | 0 | 0.0% |
| 4~6 미만 | 12 | 2 | 16.7% |
| 6~8 미만 | 8 | 8 | 100.0% |
| 8~12 미만 | 148 | 18 | 12.2% |
| 12~16 미만 | 155 | 45 | 29.0% |
| 16 이상 | 71 | 38 | 53.5% |

## 장면별

| 장면 | detected | merged_with_background | fragmented | color_shifted | not_covered |
|---|---|---|---|---|---|
| bar_chart | 6 | 0 | 0 | 0 | 0 |
| dot_plate | 56 | 255 | 0 | 4 | 5 |
| legend_lines | 5 | 5 | 0 | 0 | 0 |
| legend_lines_w08 | 10 | 0 | 0 | 0 | 0 |
| lines_w01 | 0 | 8 | 0 | 0 | 0 |
| lines_w02 | 0 | 7 | 0 | 1 | 0 |
| lines_w03 | 0 | 7 | 0 | 1 | 0 |
| lines_w04 | 1 | 5 | 0 | 2 | 0 |
| lines_w06 | 8 | 0 | 0 | 0 | 0 |
| lines_w08 | 8 | 0 | 0 | 0 | 0 |
| lines_w12 | 8 | 0 | 0 | 0 | 0 |
| lines_w16 | 8 | 0 | 0 | 0 | 0 |
| route_map | 1 | 10 | 0 | 2 | 0 |

그림은 `results/failure_cases/`에 장면별로 저장했다. 왼쪽부터 원본, 영역 대표색, 경계선이다.
