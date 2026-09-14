# 실제 화면 이미지 수집 규칙

> **선택적 응용 사례 전용.** 현재 핵심 연구인 고정 선 자극의 C80 측정·예측에
> 이 이미지 자료는 필요하지 않다. 아래 20~30장 목표는 실제 화면 사례까지
> 별도로 분석하기로 팀이 결정했을 때만 적용하며, 옛 지도 결과표를 대신하는
> 근거로 자동 사용하지 않는다.

로컬 작업 폴더에 무시된 `images/daegu_metro.png`가 남아 있을 수 있고, 과거
`results/maps/` 파생표도 존재한다. 둘 다 원출처·허가·원본 해시가 완결되지
않은 격리 자료이며 Git 근거 자료가 아니다. 출처 장부와 재현 가능한 입력을
복구하기 전에는 논문, 발표, 성능 수치에 사용하지 않는다.

평가용 실제 화면 20~30장을 모은다. 이미지는 `images/`에 넣고, 한 장마다
`sources.csv`에 한 줄을 쓴다.

## 목표 구성

| 범주 | 수량 |
|---|---|
| 통계 그래프 | 8~10 |
| 노선도 | 4~5 |
| UI 화면 | 5~6 |
| 색각 검사표 | 3~4 |
| 기타 (지도 범례, 히트맵) | 3~5 |

## 넣을 것과 뺄 것

**넣는다** — 색이 정보를 구분하는 이미지. 범례 없이는 어느 선이 어느
계열인지 알 수 없는 그래프, 노선 색으로 노선을 구분하는 노선도, 상태를
색으로만 알리는 UI.

**뺀다** — 색이 장식인 이미지. 배경 그라데이션, 브랜드 색만 입힌 버튼,
흑백으로 바꿔도 정보가 그대로 읽히는 그림.

## 출처

공공누리, CC 라이선스, 퍼블릭 도메인, 직접 캡처만 쓴다. 출처를 확인할 수
없는 이미지는 넣지 않는다. `sources.csv`의 `license`와 `source_url`을
비워두면 보고서에 실을 수 없다.

## 촬영 조건을 반드시 적는다

`screen_diagonal_inch`, `screen_width_px`, `screen_height_px`,
`viewing_distance_cm` 네 값이 있어야 픽셀 크기를 시각도로 바꿀 수 있다.
이 환산이 안 되면 한정훈의 2차 실험에서 자극 크기를 정할 수 없다.

직접 캡처가 아니라 내려받은 이미지라면 "이 크기로 화면에 띄웠을 때"를
가정하고 그 값을 적는다. 가정한 값이라는 것을 `notes`에 남긴다.

## 측정 실행

```powershell
cd src
python -c "from measure_pairs import measure_directory; measure_directory('../data/real/images','../results/measurements_real.json', n_segments=800, merge_thresh=4.0, sources_csv='../data/real/sources.csv')"
python -c "from export_tables import export_all; export_all('../results/measurements_real.json','../results')"
```

`sources_csv`를 넘기면 이미지마다 그 줄의 화면 조건을 써서 시각도를
환산한다. 파일에 없는 이미지는 `capture_meta` 인자의 값을 쓴다.
