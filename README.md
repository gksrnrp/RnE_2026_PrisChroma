# PirsChroma

선 굵기에 따른 색 구별 역치를 측정하고, 필요한 만큼 색을 이동시키는 방법을 연구하는 R&E 2026 프로젝트다.
고정한 화면과 색 경로에서 적록색각이상 사례 참가자의 C80(80% 정답률에 해당하는 색차)을 추정하고, 다른 선 굵기의 역치를 예측한다.

## 구성

| 경로 | 내용 |
|---|---|
| `src/` | 자극 생성, 실험 실행, 역치 분석, 색 보정 및 테스트 |
| `schemas/` | 설정·응답·분석 결과의 데이터 형식 |
| `data/contracts/` | 입력 예제와 실험 설정 양식 |
| `docs/experiment-protocol.md` | 실험 설계와 분석 기준 |
| `docs/validation-workflow.md` | 별도 검증 시행과 결과 분석 절차 |
| `tools/common_color_pilot/` | 본실험과 별도로 사용하는 공통색 예비실험 도구 |

## 실행

Python과 Tk가 필요하다. 프로젝트 루트에서 의존성을 설치한 뒤 `src/` 안에서 실행한다.

```bash
python -m pip install -r requirements.txt
cd src
python -m unittest -q
python two_color_demo.py --help
```

패키지 버전을 고정하려면 `requirements.txt` 대신 `requirements-lock.txt`를 사용한다.
예비실험 도구의 실행 방법은 [별도 안내](tools/common_color_pilot/README.md)에 있다.

## 연구 범위

연구용 보정은 측정한 Lab 중점과 방향을 유지하는 색쌍을 대상으로 한다.
명시한 scale grid를 8-bit RGB로 양자화한 후보 중, 표시 RGB에서 다시 계산한 Machado 후 ΔE00가 예측 C80 이상인 후보를 모두 비교해 OKLab 이동 합이 가장 작은 것을 선택한다.
최소값은 이 유한 후보 집합 안에서만 보장된다. 임의 RGB 색쌍을 다루는 보조 baseline은 쌍 단위 근사해다.

이 저장소에는 코드, 공통 설정과 합성 예제를 공개한다. 참가자별 설정과 응답 원자료, 개인별 분석 결과는 포함하지 않는다.
코드 실행이나 합성 검증만으로 사람의 가독성 향상을 주장하지 않는다. 사람 실험은 [실험 프로토콜](docs/experiment-protocol.md)의 준비 조건을 충족한 뒤 진행한다.
