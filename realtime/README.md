# 실시간 보정

화면의 색 팔레트와 선 굵기를 추정하고, C80 조건에 미달하는 색쌍을 보정하는 응용 코드다.
학생 MLP의 출력은 24개 배율로 변환해 8-bit RGB에서 다시 검사한다. 선택한 색 이동은 64³ LUT로 화면에 적용한다.
본실험의 동일경로 보정과는 별도인 공학적 응용이다.

## 설치

저장소 루트의 `src/`를 함께 사용한다. 아래 명령은 이 폴더에서 실행한다.

```bash
python -m pip install -r requirements.txt
python -m unittest -q test_realtime
```

학습에는 `torch>=2.2`가, 화면 캡처 데모에는 `mss>=9.0`과 GUI를 지원하는 OpenCV가 추가로 필요하다.
서버에서 화면 창을 띄우지 않는 검증에는 `opencv-python-headless`를 사용할 수 있다.

## 파일

| 파일 | 내용 |
|---|---|
| `common.py` | 색 변환, Machado 시뮬레이션, 사용자·화면 설정 |
| `student.py` | NumPy MLP 추론과 색쌍 재검사 |
| `train_student.py` | 합성 교사 라벨을 이용한 MLP 학습 |
| `gen_teacher_labels.py` | 기존 근사 탐색기를 호출하는 교사 라벨 생성 |
| `pipeline.py` | 팔레트·혼합색 처리·굵기 추정·LUT 생성과 적용 |
| `base_models.py` | 고전 daltonization과 외부 참조 변환의 LUT 변환 |
| `scene_gen.py`, `evaluate.py`, `width_bins.py` | 합성 선 장면 생성과 계산상 평가 |
| `realtime_demo.py` | 화면·영상·이미지 보정 결과를 창에 표시 |
| `models/student_v1/student.npz` | 제공된 학습 가중치 |

## 실행 설정

`profiles/user.template.json`과 `profiles/display.template.json`을 복사해 실제 설정을 작성한다.
템플릿의 `null`은 직접 채워야 하며, 실행 가능한 기본값이 아니다. 작성한 JSON은 `profiles/` 안에 두면 Git에서 제외된다.

사용자 설정에는 색각 유형(`protan` 또는 `deutan`), severity, 기준 C80, 기준·최소·최대 시각도, 굵기 지수 `beta`, 원본 색차 하한 `original_min`, 값의 출처 `source`가 필요하다.
화면 설정에는 발광 영역 가로 길이(mm), 가로 해상도(px), 관찰 거리(mm)를 입력한다.
C80은 `c80_ref × (시각도 / 기준 시각도)^beta`로 계산하며, 범위 밖은 경계값으로 고정한다.

`<사용자설정>`과 `<화면설정>`은 작성한 JSON 경로다.

```bash
python evaluate.py --profile <사용자설정> --display <화면설정> --student models/student_v1/student.npz --scenes 200 --seed0 50000 --aa --out results/evaluation
python realtime_demo.py --profile <사용자설정> --display <화면설정> --student models/student_v1/student.npz --mode student --region 0 0 1280 720
```

`--base-lut`을 생략하면 비교 기준은 고전 daltonization이다. Ebelin 참조 변환과 비교하려면 별도로 확보한 파일을 `base_models.py`로 변환해 명시해야 한다. 외부 참조 이미지와 LUT는 포함하지 않았다.
데모는 보정된 캡처 영역을 창에 보여 주며 OS 전체 화면을 변환하지 않는다. `s`를 누르면 화면 이미지가 로컬에 저장된다.

굵기별 집계는 `python width_bins.py . models/student_v1/student.npz <사용자설정> <화면설정> 200 <출력.csv>`로 실행한다. 외부 기준 LUT가 있으면 마지막 인자로 추가한다.

학습 명령과 가중치 구성은 [모델 안내](models/student_v1/README.md)에 정리했다.

## 해석 범위

교사는 `src/correction.py`의 임의 RGB 쌍 단위 근사 탐색기다. 전역 최적해나 연구용 동일경로 전수 탐색기의 정확도를 보장하지 않는다.
학생은 주어진 배율 순서에서 처음 통과하는 후보를 선택하며, 모든 색이동 후보의 최소 비용을 찾는 방식이 아니다.

색쌍 재검사는 팔레트의 모든 이웃 관계, LUT 양자화, EMA 중간 프레임의 제약까지 보장하지 않는다.
장면 평가의 해결률은 정답 선 색을 LUT로 변환해 계산한 값이며, 안티앨리어싱된 모든 표시 픽셀이나 사람의 실제 구별을 측정한 값이 아니다.
한 색 경로에서 얻은 C80을 임의 장면에 사용하는 것도 별도의 가정이다.

제공된 가중치는 배포된 원본 파일을 유지했다. 학습 라벨과 당시 실행 환경은 포함하지 않아 원래 학습·보고서 수치의 완전 재현은 확인하지 않았다.
실제 참가자 프로파일·원응답·개인별 평가 결과는 공개하지 않는다.
