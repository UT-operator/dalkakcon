# 딸깍콘 (DdalkakCon)

**AI 기반 움직이는 이모티콘 제작 프로그램**
2026 인공지능 인재양성 프로그램 — 심화역량과정

캐릭터 이미지 1장과 설명을 주면, 메신저 규격에 맞는 **움직이는 이모티콘**을 만들어 줍니다.

---

## 처리 흐름

```
사용자 이미지 + 프롬프트
   │
   ▼  LangChain / gpt-4o-mini   프롬프트 해석, 기획 (tool calling)
   │
   ▼  OpenAI 이미지 API
   ①  기준 원화 1장 확정          ← 세트 전체가 공유. 캐릭터 일관성의 핵심
   ②  종별 대표 포즈 생성         ← ①을 레퍼런스로 병렬 생성
   │
   ▼  Pillow (AI 호출 없음, 비용 0)
   ③  투명 배경 안정화            점 노이즈 제거 + 내부 구멍 메우기
   ④  세트 공통 배율 + 규격       종마다 캐릭터 크기가 같아지게
   ⑤  기하 변형으로 움직임 생성    모션 12종 + 이징 7종
   ⑥  효과·글자 합성             하트·반짝임·눈물·음표 + 한글
   ⑦  저장                       GIF + 번호 붙인 PNG 프레임
```

**AI는 포즈만 그리고, 움직임은 수식이 만듭니다.** 그래서 생성 장수가 절반이면서
프레임 수는 두 배가 됩니다.

---

## 파일

| 파일 | 역할 | 담당 |
|---|---|---|
| `emoticon_tools.py` | 이미지 API 호출, 투명 배경, 규격, 저장, LangChain tool | 함수 코드 |
| `geo_motion.py` | 기하 변형 모션 12종 + 이징 7종 | 함수 코드 |
| `geo_effects.py` | 효과 6종 + 한글 글자 + 다크모드 흰 테두리 | 함수 코드 |
| `charsample.py` | API 없이 점검할 때 쓰는 합성 캐릭터 | 함수 코드 |
| `app.py` | Streamlit 화면, 업로드 전달, tool calling 통합 | 프론트엔드 |
| `test_integration.py` | 실제 생성까지 연결하는 통합 테스트 | 공통 |

---

## 플랫폼 규격

| | 카카오 | 네이버 OGQ | 라인 | 디스코드 |
|---|---|---|---|---|
| 개수 | 24 | 24 | 8/16/24 | 슬롯제 |
| 크기 | 360×360 | 740×640 | 320×270 | 320×320 |
| 형식 | WebP | GIF | APNG | APNG |
| 용량 | 650KB | 1MB | 1MB | 500KB |
| 프레임 | 24 이하 | 100 이하 | 5~20 | — |
| 대표 | **마지막** | 첫=끝 | **첫** | — |

세 플랫폼의 "대표 프레임" 규칙이 서로 달라 보이지만, 루프를
`1-2-3-4-3-2-1` 로 만들어 **대표 포즈로 시작하고 대표 포즈로 끝내면** 전부 충족됩니다.

교집합으로 잡은 안전 목표는 **5~18프레임 / 256KB 이하**입니다.

> 카카오는 WebP를 전용 [WebP Animator](https://emoticonstudio.kakao.com/webp-animator)
> 로만 만들게 합니다. 그래서 이 프로그램은 **번호 붙인 PNG 프레임**을 내보내고,
> 변환만 그 도구에 맡깁니다.

---

## Colab에서 쓰기

```python
# 1) 라이브러리 + 한글 폰트
!pip install -q langchain langchain-openai openai streamlit pillow
!apt-get install -y -qq fonts-nanum

# 2) API 키 (Colab 왼쪽 🔑 보안 비밀에 'openai' 로 저장 후 토글 ON)
import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("openai")

# 3) 코드 받아오기
import urllib.request
BASE = "https://raw.githubusercontent.com/UT-operator/dalkakcon/main/"
for f in ["emoticon_tools.py", "geo_motion.py", "geo_effects.py",
          "charsample.py", "app.py", "test_integration.py"]:
    urllib.request.urlretrieve(BASE + f, f)

# 4) 통합 테스트 (API 4회, 약 $0.04)
!python test_integration.py
```

---

## 참고

폰트는 상업적 사용이 허용된 **나눔고딕**(OFL)을 씁니다. OGQ 심사 반려 사유에
"상업적 사용이 불가능한 폰트"가 포함되기 때문입니다.
