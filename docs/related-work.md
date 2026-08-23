# 선행연구 조사와 연구 위치 재설정

작성 2026-08-20 · 한정훈 파트에서 조사

## 왜 이 문서를 쓰는가

실행계획 v4 를 세울 때 선행연구 조사가 부족했다. 뒤늦게 찾아보니 **우리가
새롭다고 생각한 축의 대부분이 이미 발표되어 있다.** 발표 자리에서 지적받기
전에 정리해 두고, 연구의 위치를 다시 잡는다.

결론부터: **개별 요소는 전부 존재한다. 다만 그 요소들을 잇는 자리 하나가
비어 있고, 선행 논문이 그것을 직접 "future work" 이라고 적어두었다.**

---

## 1. 찾은 선행연구

### 1-1. Zhu et al. (2022) — 가장 가까운 선행연구

> Zhenyang Zhu, Masahiro Toyoura, Kentaro Go, Kenji Kashiwagi, Issei Fujishiro,
> Tien-Tsin Wong, Xiaoyang Mao.
> **"Personalized Image Recoloring for Color Vision Deficiency Compensation"**
> *IEEE Transactions on Multimedia*, 24, 1721–1734, 2022.
> doi:10.1109/TMM.2021.3070108

무엇을 했나

- Machado(2009) 모델로 **severity 에 맞춘 개인화 색보정**
- 목적함수 = **대비 향상 + 자연스러움 보존**
- 이색형이 아니라 **이상삼색형(색약)이 주 대상** — 우리와 같은 문제의식
- CVD 피험자 14명 주관 평가, 기존 방법들보다 우수
- 임상 검사(이시하라, Panel D15, 100-hue)와 계산 모델의 severity 를 비교

우리 계획과 겹치는 것

| 우리 실행계획 | Zhu 2022 |
|---|---|
| §2 Machado 시뮬 + severity 개인화 | 동일 |
| §6-4 `L_preserve` (원본 왜곡 최소화) | naturalness preservation |
| §6-4 `L_sep` (구별 조건 달성) | contrast enhancement |
| §2-1 "기존은 이색형만" 이라는 문제의식 | 이 논문의 도입부 그 자체 |

**§2-1 의 논거는 이 논문이 이미 해결한 문제다.** 그대로 두면 안 된다.

이 논문이 못 한 것 (결론에서 직접 인정)

> "Potential future work could apply the proposed method to develop a
> user-friendly system for the **accurate measurement of T**."

`T` 는 Machado 행렬, 즉 유형과 severity 다. 이 논문은 그것을 **측정하지 못했고**,
대신 **1% 단위 스크롤바로 사용자가 직접 고르게** 했다 (논문 Fig. 9).

### 1-2. Chen et al. (2024) — 신경망 버전. 같은 연구실

> Ligeng Chen, Zhenyang Zhu, Wangkang Huang, Kentaro Go, Xiaodiao Chen,
> Xiaoyang Mao.
> **"Image recoloring for color vision deficiency compensation using Swin transformer"**
> *Neural Computing and Applications*, 36(11), 6051–6066, 2024.
> doi:10.1007/s00521-023-09367-2 · 코드 https://github.com/Ligeng-c/CVD_swin

우리 계획의 AI 단계와 겹치는 것

- **신경망 기반** 색보정 (Swin transformer)
- 손실함수 = **자연스러움 + 대비** — 우리 `L_preserve` + `L_sep`
- **비지도 학습** (정답 보정 이미지 없음) — 우리가 "자기지도" 라 부른 것
- **혼동 색쌍 데이터셋을 직접 수집**
- **실시간 처리** — Zhu 2022 의 속도 한계(이미지 1장 3~20초)를 해결

즉 실행계획 §5~§6 의 구상(시뮬레이터를 손실에 넣고 정답 없이 학습)은
이미 구현되고 발표되었다. **"AI 라서 빠르다" 도 우리 기여가 아니다.**

### 1-3. Hue4U (2025) — AR 실시간 개인화

> Jingwen Qin, Semen Checherin, Yue Li, Berend-Jan van der Zwaag,
> Ozlem Durmaz-Incel.
> **"Hue4U: Real-Time Personalized Color Correction in Augmented Reality"**
> arXiv:2509.06776, 2025.

Machado severity + 신경망 + 개인화를 Meta Quest 에서 실시간으로 한다.
피험자 10명. 다만 **개인화가 사용자의 수동 조절**이며 측정이 아니다.

### 1-4. CAD / Cambridge Colour Test — 우리 실험 방법의 원형

한정훈 파트가 만든 절차(혼동선 위 색쌍, 4지선다, 적응형 계단법, 확인용 시행)는
**임상 색각 검사가 수십 년째 쓰는 방법**이다.

- **CAD** (Colour Assessment and Diagnosis, City St George's, London) —
  움직이는 색 사각형 + 동적 휘도 잡음
- **CCT** (Cambridge Colour Test) — 유사등색 패턴 + 정적 휘도 잡음,
  Landolt C 방향 판별, 강제선택 적응 절차

둘 다 **유형과 severity 를 산출**하며 서로 잘 일치한다는 비교 연구도 있다.

**설계 전에 이것을 조사했어야 했다.** 결과적으로 같은 원리에 도달했지만,
"우리가 고안했다" 고 말할 수 있는 방법이 아니다.

우리 구현이 임상 검사와 다른 점 (모두 우리 쪽이 부족한 방향):

- **휘도 잡음을 쓰지 않는다.** 밝기 단서를 완전히 차단하지 못한다.
- 화면 캘리브레이션을 하지 않았다.
- 대신 우리는 **자극 색을 유형 식별력 기준으로 최적화**했고, 이건 임상 검사가
  명시적으로 다루지 않는 부분이다.

### 1-5. ClearColorVision — 논문은 없지만 거의 같은 도구

크롬 확장 프로그램. https://clearcolorvision.org/

혼동축 색 구별 + 적응형 계단법 + 휘도 차단 + 확인용 시행 + protan/deutan 한정 +
연속 severity → 보정 강도. **한정훈 파트와 사실상 같은 구성**이다.

동료평가 논문은 없고 Machado 파라미터로 내는지도 명시되어 있지 않다.

### 1-6. 참고: 분야 개관

> Zhu et al. **"Image recoloring for color vision deficiency compensation:
> a survey"** *The Visual Computer*, 2021. doi:10.1007/s00371-021-02240-0

---

## 2. 그래서 무엇이 남는가

### 겹치는 것 (우리 기여가 아님)

```
Machado severity 기반 개인화 색보정      Zhu 2022
비지도 신경망, 자연스러움+대비 손실       Chen 2024
혼동 색쌍 데이터셋                       Chen 2024
실시간 처리                              Chen 2024
계단법으로 색 구별 역치 측정              CAD / CCT
역치 → 개인화 보정 강도                   ClearColorVision (도구)
```

### 실행계획 §7-4 대조 실험에 대한 정정

「빨강+파랑이면 유지, 빨강+초록이면 이동」은 **픽셀 단위 Daltonization 대비로는
유효한 차별점**이다. 그러나 Zhu 2022 처럼 **이미지에 실제로 존재하는 색들 사이의
대비만 최적화하는 방법**은 빨강과 파랑만 있을 때 손댈 이유가 없어 같은 결과를
낸다. 최신 선행연구 대비 차별점으로 내세우면 안 된다.

### 비어 있는 자리

```
CAD / CCT       유형·severity 를 잰다.  단, 자기 단위(u'v' vector length)로 낸다
                         ↓  연결 없음  ↓
Machado 모델     severity 를 입력으로 받는다.  단, 재는 방법이 없다
```

**임상 측정과 보정 모델 사이에 변환이 없다.** Zhu 2022 가 스크롤바를 붙인 것도,
Hue4U 가 수동 조절을 택한 것도 이 다리가 없기 때문이다.

찾지 못한 것: **측정한 Machado 파라미터를 보정에 넣고, 그 측정이 타당한지
검증한 연구.**

---

## 3. 연구 질문 재설정 제안

### 지금 (선행연구와 겹침)

> 색각이상 사용자의 유형과 severity 를 고려하여 이미지 내 혼동 색의 관계를
> 탐지하고, 구별 가능한 최소한의 색 이동을 학습하는 것이, 기존 고정 색변환
> 방법보다 나은가?

### 제안

> **색 구별 실험으로 측정한 Machado 파라미터가 사용자가 직접 고른 값 및 임상
> 검사 결과와 일치하는가? 그리고 그 값으로 보정했을 때 실제로 더 잘 보이는가?**

이렇게 두면 Zhu 2022 와 Chen 2024 가 **경쟁자가 아니라 재료**가 된다.
우리는 그들이 비워둔 입력단을 채우고, 그들의 방법을 검증 장치로 쓴다.

### 검증 실험 — 세 방법 교차 비교

같은 피험자에게 세 가지로 severity 를 구하고 비교한다.

| 방법 | 출처 | 성격 |
|---|---|---|
| 색쌍 구별 실험 | 본 연구 (한정훈) | 심리물리 측정 |
| singular degree | Zhu 2022 | 보정 이미지 선호 기반 |
| 이시하라 / Panel D15 | 표준 | 임상 |

Zhu 2022 의 **singular degree** 는 여러 severity 로 보정한 이미지를 보여주고
구별력이 유의하게 뛰는 지점을 그 사람의 severity 로 보는 방법이며, 논문에서
임상 severity 와 상관을 보였다.

- 세 값이 일치하면 → 측정 방법의 타당성 입증
- 어긋나면 → 어느 값이 실제 보정 성능을 더 잘 예측하는지 비교

**이 비교 자체가 아직 수행된 바 없다.**

---

## 4. 실행계획 수정할 곳

| 절 | 현재 | 수정 |
|---|---|---|
| §0 연구질문 | AI 색보정이 나은가 | 측정 파라미터의 타당성 + 보정 효과 |
| §1-3 차별점 | Daltonization 대비 | **Zhu 2022 / Chen 2024 대비**로 교체 |
| §2-1 왜 Machado | 그대로 유지 | 근거 여전히 타당 |
| §7-4 대조 실험 | 핵심 차별점 | Daltonization 비교용으로 격하 |
| §9-3 비교 대상 | Daltonization, Yang, Kim | **Zhu 2022, Chen 2024 추가** |
| §12 미해결 | τ 값 등 | 측정법 타당성 검증 추가 |
| §3-3 측정 프로토콜 | 자체 설계처럼 서술 | **CAD/CCT 계열임을 명시하고 인용** |

특히 §3-3 은 반드시 고쳐야 한다. 현재 서술은 우리가 고안한 절차처럼 읽히지만
실제로는 확립된 임상 패러다임의 축소 구현이다.

---

## 5. 우리 쪽이 오히려 나은 부분

방어적으로 쓰는 것이 아니라, 실제로 위 논문들이 보고하지 않는 항목이다.

**교차검증.** 추정에 쓰지 않은 색쌍 712 시행으로 예측 성능을 확인했다
(`colorvision/validate.py`). 개인화 모델이 「평균만 쓰기」 대비 로그가능도
+75.7, 「정상 색각 모델」 대비 +123.3. 색을 갈라 재추정해도 유형과 severity 가
일관되게 나왔다.

**자극 설계 최적화.** protan 과 deutan 의 혼동선이 14.6도밖에 떨어져 있지 않아
색쌍을 임의로 고르면 유형 판정이 뒤집힌다는 것을 발견하고, severity 를 어떻게
맞춰 끼워도 두 유형이 서로를 흉내내지 못하도록 조건을 최적화했다
(`colorvision/design.py`). 적중률 62~100% → 100%.

**측정 조건의 시야각 기록.** 자극이 7.7도였다는 것을 기록한다. 이 값이 없으면
다른 화면·다른 연구의 수치와 비교할 수 없다.

**실험실 역치와 이미지 역치의 괴리를 정량화.** 큰 패치에서 잰 역치를 실제
이미지 조건에 그대로 쓸 수 없음을 수치로 보였다 (혼동 색쌍의 37%만 검출).
이 문제 자체를 다룬 선행연구를 찾지 못했다.

---

## 6. 남은 위험

- **신규성 주장의 폭이 좁다.** 「측정과 검증」으로 좁혀야 하며, 「새로운 색보정
  방법」으로 주장하면 곧바로 반박당한다.
- **휘도 잡음 미적용.** 임상 검사가 쓰는 통제를 우리는 하지 않는다. 밝기 단서가
  섞였을 가능성을 배제하지 못한다.
- **피험자 1명.** 개인화를 주장하려면 최소한 색각이상 2명 + 정상 1명이 필요하다.
- **화면 캘리브레이션 없음.** Zhu 2022 는 X-Rite i1Display 2 로 보정한 23인치
  화면을 썼다.

---

## 참고문헌

1. Zhu, Z., Toyoura, M., Go, K., Kashiwagi, K., Fujishiro, I., Wong, T.-T., Mao, X.
   "Personalized Image Recoloring for Color Vision Deficiency Compensation."
   *IEEE Trans. Multimedia*, 24, 1721–1734, 2022.
2. Chen, L., Zhu, Z., Huang, W., Go, K., Chen, X., Mao, X.
   "Image recoloring for color vision deficiency compensation using Swin transformer."
   *Neural Computing and Applications*, 36(11), 6051–6066, 2024.
3. Qin, J., Checherin, S., Li, Y., van der Zwaag, B.-J., Durmaz-Incel, O.
   "Hue4U: Real-Time Personalized Color Correction in Augmented Reality."
   arXiv:2509.06776, 2025.
4. Zhu, Z., et al. "Image recoloring for color vision deficiency compensation:
   a survey." *The Visual Computer*, 2021.
5. Machado, G. M., Oliveira, M. M., Fernandes, L. A. F.
   "A Physiologically-based Model for Simulation of Color Vision Deficiency."
   *IEEE TVCG*, 15(6), 1291–1298, 2009.
6. Barbur, J. L., et al. Colour Assessment and Diagnosis (CAD) test.
   City St George's, University of London.
7. Regan, B. C., Reffin, J. P., Mollon, J. D. Cambridge Colour Test.
8. ClearColorVision (크롬 확장). https://clearcolorvision.org/ — 동료평가 없음
