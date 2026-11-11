# 학생 MLP

`student.npz`는 제공된 모델의 원본 가중치다. 재학습하거나 수치를 바꾸지 않았다.

- 구조: 입력 12개 → 256 → 256 → 256 → 출력 6개
- 활성함수: LeakyReLU(0.01)
- 출력: 두 색의 OKLab 이동량. 학습 스케일 10을 나누어 사용한다.
- 입력 정규화: 파일의 `mean`, `std`
- 파일 내용: 숫자 배열만 저장한 NumPy NPZ
- SHA-256: `16836208b210715681c48dee3782ff3ec2ead1ef4bb47b2e47735d5920c30b89`

입력 순서는 `student.py`의 `FEATURE_NAMES`에 정의되어 있다. 추론에는 PyTorch가 필요하지 않다.
학습 기록의 성능 수치는 이 공개 환경에서 재검증한 결과로 인용하지 않는다.

## 새 모델 학습

`realtime/`에서 실행한다. 아래 C80 범위는 합성 라벨 생성 범위이며 참가자의 실험 역치가 아니다.

```bash
python -m pip install torch
python gen_teacher_labels.py --n 30000 --out data/teacher --workers 4 --seed 20261105 --c80-min 1 --c80-max 10
python train_student.py --labels data/teacher --out results/student --hidden 256 --depth 3 --epochs 300
```

생성기는 임의 RGB에 대한 근사 탐색기를 사용한다. 라벨 생성의 매개변수나 구현을 바꾸면 별도 출력 폴더를 사용한다.
같은 ID의 라벨을 여러 번 합치지 않는다. 학습·검증·시험 분할은 ID를 기준으로 증강 전에 나눈다.
새로 학습한 모델은 제공된 가중치와 다른 모델이며, 보고서 수치가 그대로 재현된다고 가정하지 않는다.
