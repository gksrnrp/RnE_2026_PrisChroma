# 공통색 선 굵기 예비실험 도구

**기존 동결 본실험과 별도인 탐색용 프로그램이다.** `docs/experiment-protocol.md`의 연구 승인/검증 게이트를 해제하거나 대체하지 않는다.
임상 진단, 확증 C80, 최소 보정 효과를 자동 선언하지 않는다.

## 설치·공유 범위

Windows, Python 3.11 이상(Tk 포함), numpy/Pillow가 필요하다. 이 폴더 안에서:

```
python -m pip install -r requirements.txt
python -m unittest -v test_public_package
python pilot.py --self-test
```

런처는 시스템의 `py -3` 또는 `python`을 사용한다. 개인 PC의 Python 경로를 요구하지 않는다.
`common_conditions.json`은 공통 설계 자산이므로 사람 응답 없이 v2 시작 화면을 열 수 있다.

**Git에는 사람 응답·개인 설정·모델·검증 계획·개인별 결과를 포함하지 않는다.**
따라서 `git pull`만으로 기존 사람 자료가 생기거나 v3/v4의 개인별 계획이 준비되지는 않는다.
필요한 사적 데이터는 적법한 비공개 인계로 별도 제공해야 하며, 인계 요청 없이 원자료를 업로드하지 않는다.

## 프로그램별 역할

| 프로그램 | 역할 | 사전 입력 |
|---|---|---|
| `pilot.py` / `시작.cmd` | v2: 공통12색×4굵기×6간격=288문항 | 공통 설계 포함 |
| `analyze_pilot.py` | 완료된 v2 한 회차 기술통계 | 비공개 v2 journal |
| `repeat_pilot.py --prepare` | 색차 난도 변화로 2경로 선정·계획 고정 | v2 원자료/분석 |
| `repeat_pilot.py` / `반복비교_시작.cmd` | v3: 1/2/3/4/8px·동일색쌍 반복 240문항 | 비공개 v3 계획 |
| `analyze_repeat_pilot.py` | v3 기술통계·품질 점검 | 비공개 v3 journal |
| `fit_width_prediction.py` | 굵기 무관/로그/4px포화 모델 및 굵기 제외 점검 | 비공개 v3/분석 |
| `validate_width_pilot.py --prepare` | 모델 스냅샷·6px 예측 사전 고정 | 비공개 예측 산출물 |
| `validate_width_pilot.py` / `6px예측확인_시작.cmd` | v4: 새6px·120문항 | 비공개 v4 계획/스냅샷 |
| `evaluate_validation_pilot.py` | 고정 예측과 새 응답 비교, 재적합 없음 | 비공개 v4 journal |
| `wide_pilot.py` / `run_v5.py` | v5: 같은 색쌍을 8개 굵기에서 비교하는 384문항 | 비공개 v3 계획 |
| `lum_pilot.py` | v6: 밝기 단서 세 조건을 비교하는 240문항 | 비공개 v3 계획과 명시한 모형 설정 |

v5 실행은 [선 굵기 후속 실험](WIDE-PILOT.md), v6 실행은 [밝기 단서 실험](LUMINANCE-PILOT.md)을 참고한다.
v6 공개본에는 참가자의 실제 severity가 들어 있지 않다. 준비 시 설정값과 출처를 직접 입력해야 한다.
추가 도구 검사는 `python -m unittest -q test_public_package test_followup_pilots`로 실행한다.

## 데이터/버전 보존

응답은 각 문항 append/flush/fsync, 설정·코드·계획 해시로 결합하며 `raw/` 아래 Git 제외 저장한다.
Esc/창 이탈은 일시정지와 재제시 사건을 남긴다. 연습은 본문 응답에 섞지 않는다. 같은 버전·같은 가명으로 재개한다.
자료가 깨지거나 해시가 다르면 자동 삭제·덮어쓰기하지 않는다. 이미 동의한 참가자에게만 시행하고 실명을 입력하지 않는다.

이식용 코드의 경로/안내 정리로 실행 코드 해시는 원래 로컬 실험 코드와 다르다.
**이미 시작한 사람 회차는 그때의 원본 프로그램으로 재개한다. 기존 config의 해시를 새 코드에 맞춰 고치면 안 된다.**
분석용으로 인계한 원자료의 원설정·기록도 그대로 보존한다. 기존 작업공간의 원본 프로그램/응답은 변경하지 않았다.

`analyze_repeat_pilot.py`는 안경/번짐 진술을 자동 생성하지 않는다. 실제 참가자 진술이 있을 때만
`--same-glasses-reported` / `--color-fringing-reported`를 명시하여 사후 주석을 기록한다.
각 회차의 결과 문구는 입력 자료로 계산하며 이전 참가자의 숫자/우열을 그대로 출력하지 않는다.

## 해석 제한

v2의 하강 색차 순서는 시간/피로와 얽히며, v3/v4의 반복도 독립 참가자 수가 아니다.
색 조건 선별과 4px포화 가설은 탐색적이다. 격자 경계, 희소 반응, 모델 불일치, 재제시를 보고한다.
정답확률 예측 점수는 C80 자체의 예측 오차가 아니다. 측정 범위 밖 역치는 확정 숫자로 내놓지 않는다.
명목 원본 sRGB ΔE00와 기존 Machado 후 ΔE00를 혼동하지 않는다. 광학·안경·거리·색도 검증을 자동 보증하지 않는다.
신뢰구간 coverage/독립 인간 효과 검증 없이 보정 효능을 주장하지 않는다.

## 공통 소스 출처

`source_conditions_v1.json`과 `legacy_color_math.py`는 같은 저장소의
`hanjeonghun-colorvision` commit `95dc87999a50281f3bcbc16eac990da76ae13f4d`에서 가져온 공통 설계/수학 코드다.
사람 CSV/params/config가 아니다. 원본 바이트/주석을 보존했다.

- 원 설계 SHA256: `f47393515811c02a6111a0f4d49642e97e102a2f3a98e5cfccca0f2b25495d6c`
- 색 수학 SHA256: `26a1612cb7560e0222c51479a48b01e5166421f63621dcca08bb31b1b2eb25eb`

부가 설계 설명: `REPEAT-PILOT.md`, `PREDICTION-PILOT.md`, `VALIDATION-PILOT.md`.
