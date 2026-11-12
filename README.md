<div align="center">

# PrisChroma (팀 작업 저장소)

**선 굵기별 색 구별 역치 예측 기반 최소 색이동 보정**

A Study on Thickness-Based Minimal Recoloring for Red-Green CVD

[소개](#소개) · [현재 상태](#현재-상태) · [브랜치](#브랜치) · [비공개 자료](#비공개-자료) · [실행](#실행) · [문서](#문서) · [팀 규칙](#팀-규칙)

<img src="assets/overview.png" width="760">

</div>

## 소개

적록색각이상 사례 참가자의 **선 굵기별 색 구별 역치(C80)** 를 4지선다 과제로 측정하고, 그 값에 못 미치는 색쌍만 필요한 만큼 옮기는 보정을 연구하는 R&E 2026 프로젝트다. C80은 정답률이 80%가 되는 색차(색각 시뮬레이션 후 ΔE00)이다.

이 저장소는 팀이 실험과 분석에 쓰는 **비공개 작업 저장소**다. 제출용으로 정리한 공개 저장소는 [RnE_2026_PrisChroma](https://github.com/gksrnrp/RnE_2026_PrisChroma)이고, 실시간 보정 코드(`realtime/`)는 그쪽에만 있다.

<div align="center">
<img src="assets/example_scene.png" width="600">

<sub>위: 정상색각으로 본 화면. 아래: protan 시뮬레이션. 기존 방식(가운데)은 모든 선과 배경을 바꾸고, 본 연구(오른쪽)는 혼동되는 선만 조금 옮긴다</sub>
</div>

## 현재 상태

2026-11-10까지의 사람 대상 실험이다.

| 참가자 | 색각 검사 | 본실험 (4, 8, 16px) | 1~16px 동일 색쌍 (v5) | 밝기 단서 (v6) | 보고서 반영 |
|:--|:-:|:-:|:-:|:-:|:--|
| P01 (protan, severity 0.94) | 완료 | 384 / 384 | 384회 | 240회 | 반영 |
| P02 (deutan, severity 0.84) | 완료 | 64 / 384 | - | - | 제외 |
| P03 (deutan, severity 0.88) | 완료 | 384 / 384 | 384회 | - | 탐색적 결과로 반영 |

| 질문 | 결과 |
|:--|:--|
| 4~16px에서 C80이 굵기에 따라 달라지는가 | 확인되지 않았다. 굵기 지수 β는 P01 0.023(95% 구간 −0.338 ~ 0.435), P03 −0.157(−0.512 ~ 0.204) |
| 더 가는 선에서는 어떤가 | 같은 색쌍의 정답률이 3px 이하에서 낮았다. P01 39.6% 대 68.3%, P03 27.1% 대 40.4% |
| 밝기가 단서가 되는가 | P01에서 밝기를 맞추면 정답 수가 58/80에서 46/80으로 줄었다 |
| 보정은 혼동을 해결하는가 | 합성 선 장면의 1~20px 혼동 중 93.0%(P01), 95.1%(P03)를 해결하였다 |

<div align="center">
<img src="assets/human_results.png" width="760">
</div>

두 참가자 모두 사전 점검(신뢰 구간 포함률 90%)을 통과하지 못하였고, P03의 본실험은 절차 편차가 있다. 시스템 평가는 계산한 값이며, 보정한 화면을 사람에게 다시 보여 주는 검증은 하지 않았다. 실험별 표와 그림은 [공개 저장소의 README](https://github.com/gksrnrp/RnE_2026_PrisChroma#실험)에 있다.

## 브랜치

| 브랜치 | 내용 | 상태 |
|:--|:--|:--|
| `main` | 자극 생성, 실험 실행, 심리측정 분석, 색 보정, 예비 실험 도구 | 기준 |
| [`wide-width-pilot`](../../tree/wide-width-pilot) | 1~16px 실험(v5), 밝기 단서 실험(v6), 2026-11-10 분석 스크립트와 결과 정리 | main보다 15 커밋 앞섬 |
| [`os-build-deviation-26300`](../../tree/os-build-deviation-26300) | 실험 PC의 Windows 빌드가 바뀐 뒤 본실험을 이어 가기 위한 예외 기록 | main보다 3 커밋 앞섬 |
| [`feature/hanjeonghun-experiment`](../../tree/feature/hanjeonghun-experiment) | 초기 모형 비교와 검증 굵기 선택 (9월) | 이전 작업 |
| [`hanjeonghun-colorvision`](../../tree/hanjeonghun-colorvision) | 색각 검사 도구 (9월) | 이전 작업 |
| [`related-work`](../../tree/related-work) | 선행 연구 정리와 계획서 검토 (9월) | 이전 작업 |
| `feature/map-measurement` | main에 합쳐짐 | 종료 |

가장 최근의 분석과 결과 정리는 `wide-width-pilot`의 [`docs/results-20261110/`](../../tree/wide-width-pilot/docs/results-20261110)에 있다.

## 비공개 자료

참가자 응답 원자료는 저장소 폴더에 올리지 않고, 이 저장소의 [Releases](../../releases)에 묶음 파일로 둔다.

| 릴리스 | 내용 |
|:--|:--|
| `experiment-data-20261110` | 본실험 자료와 P01의 v5 원자료 |
| `p03-report-handoff-20261110` | P03의 신뢰 구간 계산 결과와 v5 원자료 표 |
| `pilot-measurements-20261109` | 예비 실험 v1~v4 원자료 |
| `experiment-handoff-20261105` | 2026-11-05 실험 인계 자료 |

가상 참가자 자료는 방법 검증에만 쓰며, 사람 대상 결과와 섞지 않는다.

## 실행

모든 명령은 `src/` 안에서 실행한다. 다른 폴더에서 실행하면 `ModuleNotFoundError`가 난다.

```bash
cd src
pip install -r ../requirements.txt
python -m unittest -q
```

결과를 재현할 때는 `requirements.txt` 대신 `requirements-lock.txt`(Python 3.13.15, Tk 8.6.15)를 사용한다.

| 하려는 일 | 명령 |
|:--|:--|
| 두 색의 최소 색이동 보기 | `python two_color_demo.py --help` |
| px를 시각도로 바꿔 굵기 확인 | `python plan_line_widths.py --help` |
| 사람 실험과 결론의 준비도 검사 | `python research_readiness.py --format markdown` |
| 분석 코드가 알려진 값을 되찾는지 점검 | `python parameter_recovery.py --help` |
| 본실험 자료 분석 | `python analyze_calibration.py --help` |

옵션과 주의점을 포함한 전체 설명은 [docs/commands.md](docs/commands.md)에 있다.

| 경로 | 내용 |
|:--|:--|
| `src/` | 자극 생성, 실험 실행, 역치 분석, 색 보정, 테스트 |
| `tools/common_color_pilot/` | 예비 실험 도구 |
| `schemas/`, `data/contracts/` | 설정, 응답, 분석 결과의 데이터 형식과 예제 |
| `data/`, `results/` | 입력 자료와 합성 점검 결과 (참가자 자료는 올리지 않음) |
| `docs/` | 실험 프로토콜, 실행 순서, 검증 절차, 검토 기록 |

## 문서

| 문서 | 내용 |
|:--|:--|
| [docs/experiment-protocol.md](docs/experiment-protocol.md) | 연구 질문, 분석, 중단 규칙 (최상위 기준) |
| [docs/changmin-runbook.md](docs/changmin-runbook.md) | 준비부터 참가자 인계까지의 실행 순서 |
| [docs/validation-workflow.md](docs/validation-workflow.md) | 검증 시행의 계획과 분석 절차 |
| [docs/human-session-record-template.md](docs/human-session-record-template.md) | 동의, 장치, 중단을 기록하는 세션 양식 |
| [docs/research-audit-final.md](docs/research-audit-final.md) | 선행 연구, 설계, 증거, 한계의 종합 검토 |
| [docs/software-verification.md](docs/software-verification.md) | 자동 검사와 실행 환경의 기록 |
| [docs/code-review.md](docs/code-review.md) | 무엇을 왜 고쳤는지와 남은 한계 |
| [docs/commands.md](docs/commands.md) | 명령어 전체 설명 |
| [docs/team-git-guide.md](docs/team-git-guide.md) | GitHub Desktop으로 작업하는 순서와 충돌을 피하는 규칙 |
| [AGENTS.md](AGENTS.md) | Codex와 Claude Code가 함께 쓰는 작업 지침 |

이전 계획과 인계 기록은 `docs/project-context.txt`, `docs/research-plan.txt`, `docs/handoff-jeongingyo.md`에 남아 있다.

## 팀 규칙

| 담당 | 맡은 일 |
|:--|:--|
| 이창민 | `src/machado.py`, `src/correction.py`, 심리측정 분석 |
| 한정훈 | 자극, 사람 응답 원자료(비공개), 개인 C80 적합 |
| 정인교 | 실제 표시 조건, 응용 이미지 출처와 측정 |
| 세 명 공동 | 연구 질문, 검증, 보고서 |

- 작업을 시작하기 전에 항상 최신 내용을 받는다(`Fetch origin`).
- 같은 파일을 두 명이 동시에 고치지 않는다.
- 개인정보, 참가자 원자료, 비밀번호, 100MB가 넘는 파일은 올리지 않는다. 한 번 올리면 지워도 기록에 남는다.
- 충돌이 나면 아무것도 누르지 말고 팀장에게 알린다.

처음 설정하는 방법과 매일 작업하는 순서는 [docs/team-git-guide.md](docs/team-git-guide.md)에 있다.

---

강원과학고등학교 2026 과학영재 창의연구(R&E) · 이창민, 정인교, 한정훈
