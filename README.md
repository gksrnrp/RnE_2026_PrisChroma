# R&E 프로젝트

> 연구 주제 / 한 줄 설명을 여기에 적어주세요.

## 폴더 구조

| 폴더 | 용도 |
|---|---|
| `src/` | 분석·실험 스크립트 (.py) |
| `data/` | 입력 데이터 (큰 파일은 올리지 말 것 — 아래 주의사항 참고) |
| `results/` | 그래프, 표, 출력 결과 |
| `docs/` | 회의록, 참고자료 정리, 보고서 초안 |

---

## 코드 실행

**모든 명령은 `src/` 안에서 실행합니다.** 모듈들이 서로를 직접 import하기 때문에
다른 폴더에서 실행하면 `ModuleNotFoundError`가 납니다.

```bash
cd src
pip install -r ../requirements.txt
```

두 색 보정 실행:

```bash
python two_color_demo.py --color-a 220 50 50 --color-b 60 160 60     --type deutan --severity 0.8     --original-threshold 20 --confusion-threshold 15 --target-threshold 15
```

주변 색과의 재혼동까지 막으려면 `--palette` 에 R G B 를 3개씩 이어서 넣습니다.

자동 테스트 (37개, 약 30초):

```bash
python -m unittest -q
```

탐색 안정성 벤치마크 (수 분 소요, 결과는 `results/`에 보관):

```bash
python benchmark_search.py --pairs 30 --output ../results/search_benchmark.json
```

연구 계획과 알고리즘 설명은 [docs/research-plan.txt](docs/research-plan.txt),
코드 리뷰·검증 기록은 [docs/code-review.md](docs/code-review.md)에 있습니다.

---

## 팀원용 — 처음 한 번만 하는 설정

1. **GitHub 계정 만들기** → https://github.com/signup
2. **GitHub Desktop 설치** → https://desktop.github.com (명령어 몰라도 됩니다)
3. 팀장이 보내준 초대 메일 수락
4. GitHub Desktop에서 `File → Clone repository` → 이 저장소 선택 → `Clone`

끝. 이제 내 컴퓨터에 프로젝트 폴더가 생깁니다.

---

## 매일 작업하는 순서 (이것만 지키면 됩니다)

### 1️⃣ 작업 시작 전 — 반드시 먼저!
GitHub Desktop 상단의 **`Fetch origin`** 버튼을 누릅니다.
→ 다른 팀원이 올린 최신 내용을 내 컴퓨터로 받아옵니다.
→ **이걸 건너뛰면 충돌(conflict)이 납니다.**

### 2️⃣ 작업하기
평소처럼 파일을 만들고 수정합니다.

### 3️⃣ 작업 끝나면 — 올리기
1. GitHub Desktop 왼쪽에 바뀐 파일 목록이 뜹니다
2. 아래 칸에 **무엇을 했는지 한 줄** 적습니다
   - 좋은 예: `온도 데이터 그래프 그리는 코드 추가`
   - 나쁜 예: `수정`, `ㅇㅇ`, `asdf`
3. **`Commit to main`** 클릭
4. 상단 **`Push origin`** 클릭 ← 이걸 눌러야 진짜로 올라갑니다

> 💡 하루 작업 다 하고 한 번에 올리지 말고, **의미 있는 단위마다 자주** 올리세요.
> 자주 올릴수록 충돌이 적게 납니다.

---

## ⚠️ 충돌(conflict) 안 나게 하는 규칙

git이 처음이면 충돌이 제일 무섭습니다. 아래만 지키면 거의 안 납니다.

1. **같은 파일을 두 명이 동시에 고치지 않기**
   - 파일 단위로 담당을 나누세요 (아래 담당표 참고)
2. **작업 시작 전 항상 `Fetch origin`**
3. **작업 끝나면 바로 Push** (내 컴퓨터에만 며칠씩 쌓아두지 않기)
4. 급하게 같은 파일을 고쳐야 하면 **단톡방에 먼저 말하기**

### 만약 충돌이 났다면
혼자 해결하려다 파일이 날아갈 수 있습니다.
**아무것도 누르지 말고 팀장에게 알려주세요.**

---

## 담당표

| 파일 / 폴더 | 담당자 |
|---|---|
| `src/` | |
| `data/` | |
| `results/` | |
| `docs/` | |

---

## ⚠️ 올리면 안 되는 것

- **100MB 넘는 파일** — GitHub이 거부합니다. 큰 데이터는 구글 드라이브에 두고, 링크만 `docs/`에 적어두세요.
- **개인정보** — 설문 응답자 이름, 연락처, 학번 등
- **비밀번호 / API 키**
- 한 번 올리면 지워도 기록에 남습니다. **올리기 전에 확인하세요.**

---

## 파이썬 환경 맞추기 (코드 돌릴 때)

새 패키지를 설치했다면 `requirements.txt`에 추가해주세요.

```bash
pip install -r requirements.txt
```
