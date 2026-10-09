# 공유 Colab에 최신 코드 적용하기

공유된 노트북의 **링크를 그대로 유지한 채** 코드만 바꾸는 방법입니다.
한 번만 해두면, 앞으로는 코드를 고쳐도 **노트북을 다시는 안 건드려도 됩니다.**

---

## 왜 이렇게 바꾸나

지금 노트북은 70KB인데 그중 **65KB(92%)가 `%%writefile` 코드**입니다.
코드를 한 줄 고칠 때마다 노트북 전체를 다시 올려야 하고, 그러면 친구가 고친 부분과 충돌합니다.

코드를 GitHub에서 받아오게 바꾸면 **노트북이 5KB**로 줄고, 이후 갱신은
`git push` 한 번이면 끝납니다. 친구는 런타임만 다시 실행하면 최신 코드가 들어옵니다.

---

## 0단계. 안전장치 (1분)

적용 전에 **기존 노트북 사본을 하나 떠두세요.**
Colab에서 `파일 → Drive에 사본 저장`. 뭔가 잘못되면 되돌릴 수 있습니다.

---

## 1단계. 저장소를 공개로 전환

현재 저장소는 **비공개**라서 Colab이 받아올 수 없습니다.

https://github.com/UT-operator/dalkakcon

웹에서: `Settings` → 맨 아래 `Danger Zone` → `Change repository visibility` → `Public`

> 공개해도 **API 키·보고서·카카오 자료는 올라가 있지 않습니다.**
> `.gitignore`로 `*.docx`, `*.zip`, 생성 결과물을 전부 막아뒀고,
> 코드 안의 실명도 역할명으로 바꿔뒀습니다.

---

## 2단계. 공유 노트북에 셀 1개 추가

공유된 Colab을 열고, **2번 "API 키 등록" 셀 바로 아래**에 코드 셀을 새로 만들어
아래를 그대로 붙여넣으세요.

```python
# ============================================================
#  코드 받아오기 — GitHub에서 최신 .py 파일을 내려받는다.
#  코드가 바뀌면 이 셀만 다시 실행하면 된다. 노트북은 안 건드려도 된다.
# ============================================================
import urllib.request, os

BASE = "https://raw.githubusercontent.com/UT-operator/dalkakcon/main/"
FILES = ["emoticon_tools.py", "geo_motion.py", "geo_effects.py",
         "charsample.py", "app.py", "test_integration.py"]

for f in FILES:
    urllib.request.urlretrieve(BASE + f, f)
    print(f"  {f:<22} {os.path.getsize(f)/1024:6.1f}KB")
print("\n코드 받기 완료")
```

실행하면 파일 6개와 각 용량이 찍힙니다. 그게 보이면 성공입니다.

---

## 3단계. 기존 `%%writefile` 셀 삭제

아래 6개 셀을 지웁니다. 각 셀 오른쪽 위 **휴지통 아이콘**을 누르면 됩니다.

| 셀 | 첫 줄 |
|---|---|
| 3번 항목 | `%%writefile emoticon_tools.py` |
| 4번 항목 | `%%writefile geo_motion.py` |
| 5번 항목 | `%%writefile geo_effects.py` |
| 6번 항목 | `%%writefile charsample.py` |
| 7번 항목 | `%%writefile app.py` |
| 9번 항목 | `%%writefile test_integration.py` |

> 위에 붙은 제목(마크다운) 셀도 같이 지워도 되고, 남겨둬도 동작엔 영향이 없습니다.

**남겨야 하는 셀** — 이건 지우지 마세요.

- `## 8. 자체 점검` 아래의 코드 셀 (API 비용 0원 점검)
- `!python test_integration.py` 가 들어 있는 셀
- `# 결과 확인` 셀
- 서버 실행 / 터널 셀

---

## 4단계. 확인

런타임 → **런타임 다시 시작 및 모두 실행**

1번(설치) → 2번(API 키) → **새 셀(코드 받기)** → 자체 점검 순으로 돌아가면 끝입니다.

---

## 앞으로의 작업 흐름

```
코드 수정 (내 PC)
   │
   ▼  git add . && git commit -m "..." && git push
   │
   ▼  친구는 Colab에서 '코드 받기' 셀만 다시 실행
   │
   ▼  최신 코드 적용 완료
```

**노트북은 더 이상 주고받지 않습니다.**

---

## 충돌 안 나게 하는 규칙

파일이 역할대로 나뉘어 있습니다. 이 선만 지키면 거의 충돌하지 않습니다.

| 파일 | 담당 |
|---|---|
| `app.py` | 프론트엔드 담당 |
| `emoticon_tools.py` · `geo_motion.py` · `geo_effects.py` | 함수 코드 담당 |
| `딸깍콘.ipynb` | 둘 중 한 명만 (이제 거의 안 건드림) |

같은 파일을 동시에 고치면 충돌이 납니다. 그럴 땐 먼저 `git pull` 하고 고치세요.

---

## 공개하기 싫다면

저장소를 비공개로 두고 싶으면, 2단계의 셀 대신 **매번 `.py` 파일 6개를
Colab 왼쪽 파일창에 드래그**해서 올리면 됩니다.
동작은 같지만 런타임이 끊길 때마다 다시 올려야 합니다.
