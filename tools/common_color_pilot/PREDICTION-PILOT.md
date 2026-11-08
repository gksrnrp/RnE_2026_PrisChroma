# 굵기별 임시 예측기

`fit_width_prediction.py`는 기존 본실험 분석기가 아니라 v3 예비자료를 이용하는 별도 탐색 도구다.
원자료나 v3 프로그램/설정은 변경하지 않는다.

## 하는 것

- 측정한 두 색 경로에서 표시 RGB를 표준 변환해 명목 원본 sRGB CIEDE2000 색차를 계산한다.
- 굵기 무관 / 로그 굵기 / 4px 이후 포화의 세 모형을 비교한다.
- 4AFC 우연 정답 25%를 반영하고 색 경로별 절편, 공통 색차 기울기/굵기 계수/lapse를 유한 격자로 적합한다.
- 굵기 계수 부호는 강제하지 않는다. lapse는 탐색 격자에서 함께 적합하며 기존 참가자 값을 가져오지 않는다.
- 굵기 하나를 통째로 빼고 정답확률을 예측하여 Brier와 logloss를 집계한다.
- 모델상 80% 지점이 측정 색차 범위 안에 있을 때만 잠정값을 반환한다. 범위 밖은 `null` 및 상태를 반환한다.

## 사용

준비된 Python으로 아래를 실행한다.

```
python fit_width_prediction.py --self-test
python fit_width_prediction.py
python fit_width_prediction.py --width 6 --color COLOR_PATH_ID
```

색 경로 ID는 비공개 계획에 고정된 두 조건을 사용한다. 허용 굵기는 1~8px다. 5~7px는 실제 측정하지 않은 모델 내삽이다.
출력은 Git 제외 `raw/_analysis/width-prediction-pilot.json` 및 `.md`다.

## 하면 안 되는 해석

이 모형은 후향적 탐색이며, 4px 포화 가설도 이미 본 자료에서 나온 것이다.
굵기 제외 점수는 독립적인 새 사람 실험이나 C80 자체의 예측 오차 검증이 아니다.
일부 제외 적합은 격자 상한에 닿아 안정성도 추가 확인이 필요하다.
확률/역치 신뢰구간의 coverage 검증을 하지 않았으므로 보정 적용·효능 주장에 쓰지 않는다.
Machado 변환 후 색차나 기존 동결 연구의 C80과 같은 단위/모델로 혼동하지 않는다.
프로그램은 `validated=false`, `correction_enabled=false`로 출력한다.
