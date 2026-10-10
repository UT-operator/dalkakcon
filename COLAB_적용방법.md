# 공유 Colab에 최신 코드 적용하기

공유 노트북의 **링크를 유지한 채** 코드만 GitHub에서 받아오도록 바꿉니다.
한 번만 해두면 앞으로는 코드를 고쳐도 **노트북을 다시 안 건드려도 됩니다.**

대상: 원본 상태의 공유 노트북 (셀 11개, `%%writefile app.py` 가 하나 있는 것)

---

## 왜 이렇게 바꾸나

코드를 `%%writefile` 셀에 넣어두면, 한 줄 고칠 때마다 노트북 전체를 주고받아야 하고
그러면 서로 고친 게 충돌합니다. GitHub에서 받아오게 하면 갱신이 `git push` 한 번으로 끝납니다.

---

## 1단계. 저장소를 공개로

https://github.com/UT-operator/dalkakcon
→ `Settings` → 맨 아래 `Change repository visibility` → `Public`

비공개면 Colab이 코드를 못 받아옵니다.

> API 키·보고서·카카오 자료는 `.gitignore` 로 전부 막아뒀고, 코드 안의 실명도
> 역할명으로 바꿔뒀습니다.

---

## 2단계. 설치 셀 교체

첫 코드 셀(`%pip install langchain langchain-openai`)의 내용을 전부 지우고:

```python
%pip install -q langchain langchain-openai openai streamlit pillow
!apt-get install -y -qq fonts-nanum || echo '폰트 설치 실패 - 코드가 자동으로 받아옵니다'
```

`openai`(이미지 생성)와 한글 폰트가 빠져 있어서 추가합니다.
바로 아래 `!pip install streamlit` 셀은 지워도 되고 둬도 됩니다.

> 폰트 설치가 실패해도 괜찮습니다. `geo_effects.py` 가 시스템을 직접 뒤져
> 한글 폰트를 찾고, 그래도 없으면 나눔고딕을 자동으로 내려받습니다.

---

## 3단계. "코드 받기" 셀 추가

API 키 셀(`userdata.get('openai')`) 아래에 `+ 코드` 로 새 셀을 만들고:

```python
import urllib.request, os, sys, importlib

BASE = "https://raw.githubusercontent.com/UT-operator/dalkakcon/main/"
FILES = ["emoticon_tools.py", "geo_motion.py", "geo_effects.py",
         "charsample.py", "app.py", "test_integration.py"]

for f in FILES:
    urllib.request.urlretrieve(BASE + f, f)
    print(f"  {f:<22} {os.path.getsize(f)/1024:6.1f}KB")

# 파이썬은 한 번 불러온 모듈을 '기억'한다. 파일만 새로 받으면 메모리에는
# 옛날 모듈이 그대로 남아 AttributeError 가 난다. 그래서 그 기억을 지운다.
for f in FILES:
    sys.modules.pop(f[:-3], None)
importlib.invalidate_caches()

print("\n코드 받기 완료 (모듈 캐시도 비웠습니다)")
```

**이 셀만 먼저 실행해서** 파일 6개가 찍히는지 확인하세요.
404 가 나면 1단계를 안 한 겁니다.

> 코드를 고친 뒤에는 **이 셀을 다시 실행**하면 바로 반영됩니다.
> 런타임을 다시 시작할 필요 없습니다.

---

## 4단계. 옛날 `app.py` 셀 삭제

`%%writefile app.py` 로 시작하는 셀 **하나만** 지웁니다 (🗑 아이콘).

빈칸이 있어서 실행하면 에러가 나던 옛날 버전입니다. 3단계가 최신 `app.py` 를
받아왔으니 더 필요 없습니다.

---

## 5단계. 점검 (선택, API 비용 0원)

이미지 생성에 돈을 쓰기 전에 Pillow 쪽이 멀쩡한지 확인합니다.
한글 폰트 문제가 여기서 잡힙니다.

```python
import charsample, geo_motion as gm, geo_effects as fx, emoticon_tools as et

print("모션", len(charsample.PLAN), "종 / 효과", len(fx.EFFECTS), "종")
print("한글 폰트:", fx.ensure_korean_font() or "확보 실패")

base = charsample.draw_character()
frames, durs = gm.apply_motion(base, "jump", base_ms=62)
ov = fx.fx_text("고마워", n=len(frames),
                anchors=fx.anchors_from_bbox(et.robust_bbox(base), (360, 360)))
et.save_gif(fx.composite(frames, ov), "/content/_check.gif", durs)

from IPython.display import Image as IPImage, display
display(IPImage(filename="/content/_check.gif"))
```

고양이가 점프하면서 "고마워" 글자가 뜨면 정상입니다.
글자가 네모(□)로 나오면 폰트 문제입니다.

---

## 6단계. 실행

**런타임 → 런타임 다시 시작 및 모두 실행**

맨 아래 `cloudflared` 셀에서 나오는 `https://....trycloudflare.com` 으로 접속합니다.

---

## 코드를 노트북에서 보고 싶을 때

코드가 파일로만 있어서 본문에서 안 보입니다. 보고 싶으면:

```python
from IPython.display import Code, display, Markdown

파일 = "emoticon_tools.py"      # 보고 싶은 파일로 바꾸세요
display(Markdown(f"### `{파일}`"))
display(Code(파일, language="python"))
```

고치면서 보려면 `%load emoticon_tools.py` (그 셀이 코드로 바뀝니다).
단 여기서 고친 건 GitHub에 반영되지 않고, "코드 받기" 셀을 다시 실행하면 덮어써집니다.

> 출력도 노트북에 저장됩니다. 여러 파일을 띄우면 노트북이 다시 무거워지니
> 확인 후 `셀 출력 지우기` 를 눌러주세요.

---

## ⚠️ Colab 파일창에서 직접 고치면 사라집니다

왼쪽 파일창에서 `app.py` 같은 `.py` 파일을 고칠 수는 있지만, **'코드 받기' 셀을
다시 실행하거나 런타임을 재시작하면 GitHub 판으로 덮어써집니다.**

```
Colab 에서 app.py 수정  →  '코드 받기' 셀 실행  →  수정분 사라짐
```

화면 문구나 기능을 바꾸려면 **저장소에서 고치고 push** 한 뒤,
Colab 에서 '코드 받기' 셀을 다시 실행하세요.
Colab 에서의 수정은 "지금 한 번만 확인해보는 용도"로만 쓰세요.

---

## 앞으로의 작업 흐름

```
코드 수정 (내 PC) → git push → 친구는 '코드 받기' 셀만 다시 실행
```

노트북은 더 이상 주고받지 않습니다.

---

## 충돌 안 나게 하는 규칙

| 파일 | 담당 |
|---|---|
| `app.py` | 프론트엔드 담당 |
| `emoticon_tools.py` · `geo_motion.py` · `geo_effects.py` | 함수 코드 담당 |

같은 파일을 동시에 고치면 충돌합니다. 그럴 땐 `git pull` 먼저 하세요.
