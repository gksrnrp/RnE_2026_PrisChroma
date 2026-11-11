# 밝기 단서 예비실험 v6

v3 두 색 경로의 최대 간격 색쌍에서 두 번째 색을 선형 RGB로 조절한다.
지정한 Machado 모형에서 L* 차이를 맞춘 조건, 원래 조건, 차이를 두 배로 만든 조건을 비교한다.
1·2·3·4·8px에서 총 240문항을 제시한다. 밝기를 바꾸면 전체 색차도 함께 달라지므로 순수한 밝기 효과로 단정하지 않는다.

원본 코드에 고정돼 있던 참가자 측정값은 공개본에 포함하지 않았다.
계획 생성 시 `cvd_type`(`protan` 또는 `deutan`), `severity`(0~1), `source`(설정 출처)가 있는 비공개 JSON을 직접 지정한다. 기본값은 없다.
이 파일은 Git 제외 경로인 `raw/private/` 등에 둔다.

이 폴더에서 실행한다. `<설정파일>`은 작성한 비공개 JSON의 경로다.

```bash
python lum_pilot.py --prepare --observer-config <설정파일>
python lum_pilot.py --check
python lum_pilot.py
```

v3 계획은 `raw/_design_v3/design.json`, v6 계획은 `raw/_design_v6/design.json`, 응답은 `raw/v6/`에 둔다.
기존 계획과 설정이 다르면 덮어쓰지 않는다. 프로그램은 3840×2160 해상도를 요구한다.
모형에서 계산한 L* 일치는 참가자의 실제 등휘도 측정이나 색도계 검증을 뜻하지 않는다.

코드 검사는 참가자 자료 없이 실행할 수 있다.

```bash
python -m unittest -q test_public_package test_followup_pilots
```

공개본은 설정 입력 방식과 코드 해시가 원본과 다르다. 이미 시작한 회차를 이 코드로 이어서 수집하거나 원설정의 해시를 바꾸지 않는다.
