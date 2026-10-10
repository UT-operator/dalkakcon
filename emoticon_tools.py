# -*- coding: utf-8 -*-
"""
딸깍콘(DdalkakCon) - 움직이는 이모티콘 생성 핵심 함수 모듈
담당: 함수 코드 (백엔드)

[파이프라인]
  (1) 기준 원화(base sheet) 1장 확정        : OpenAI 이미지 API
  (2) 프레임 N장 '병렬' 생성 ((1)을 레퍼런스): OpenAI 이미지 API + ThreadPoolExecutor
  (3) 투명 배경 안정화 (opening/closing)     : Pillow
  (4) 프레임 공통 바운딩박스 정렬            : Pillow  <- 떨림 제거, 의도된 모션은 보존
  (5) 규격 맞추기 (360x360 캔버스)           : Pillow
  (6) 무한 루핑 시퀀스 (핑퐁)                : 프레임 재배열
  (7) 저장: GIF(공용 팔레트) + WebP(알파)    : Pillow

[설계 의도 - 보고서/발표에 쓸 포인트]
  - 프레임을 따로따로 생성하면 캐릭터가 프레임마다 달라진다.
    -> (1)에서 '기준 원화' 1장을 먼저 확정하고, 모든 프레임이 그 원화만 참조하게 해서 일관성을 확보한다.
  - 프레임마다 캐릭터를 각각 중앙정렬하면 '점프' 같은 의도된 움직임이 사라진다.
    -> (4)에서 전체 프레임의 '합집합' 바운딩박스 하나를 구해 모든 프레임에 똑같이 적용한다.
  - GIF는 프레임별로 팔레트를 따로 만들면 재생 중 색이 깜빡인다.
    -> (7)에서 전체 프레임으로 공용 팔레트 1개를 만들어 모든 프레임이 공유한다.
"""

from __future__ import annotations

import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Optional, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter
from langchain_core.tools import tool

# ---------------------------------------------------------------- 설정 상수

IMAGE_MODEL = "gpt-image-1.5"   # gpt-image-1은 2026-10-23 은퇴 예정 자료가 있어 1.5를 기본값으로 둔다
GEN_SIZE = "1024x1024"          # 생성 해상도 (360보다 크게 뽑아 축소 -> 외곽선이 깔끔해진다)

CANVAS = 360                    # 카카오 이모티콘 스튜디오 '움직이는 이모티콘' 제안 규격 (px)
MAX_FRAMES = 24                 # 제안 규격: 24프레임 이하
# 용량 한도 (공식 가이드 확인값)
KAKAO_MAX_BYTES = 650 * 1024    # 카카오 '움직이는 이모티콘' WebP 1개당
SAFE_MAX_BYTES = 256 * 1024     # 전 플랫폼 교집합 (디스코드 이모지가 가장 빡셈)
MAX_BYTES = SAFE_MAX_BYTES      # 기본값은 어디든 통과하는 쪽으로 둔다

OUT_DIR = os.environ.get("DDALKAK_OUT_DIR", "/content")
GIF_PATH = os.path.join(OUT_DIR, "generated_emoticon.gif")
SHEET_PATH = os.path.join(OUT_DIR, "generated_sheet.png")   # 스프라이트 시트 원본
WEBP_PATH = os.path.join(OUT_DIR, "generated_emoticon.webp")
BASE_PATH = os.path.join(OUT_DIR, "generated_base.png")
FRAME_PATH_FMT = os.path.join(OUT_DIR, "generated_frame_{:02d}.png")

# 이미지 생성 AI에 공통으로 붙이는 스타일 고정 문구
STYLE_SUFFIX = (
    "2D cute emoticon sticker style, bold clean outlines, flat vivid colors, "
    "fully transparent background, single character centered, "
    "no background elements, no ground shadow, "
    "no checkerboard pattern, no border, no frame. "
    # 효과와 글자는 Pillow가 정확한 위치에 직접 그린다.
    # AI가 자기 멋대로 넣으면 우리 효과와 겹쳐서 두 종류가 섞인다.
    "Draw ONLY the character itself: absolutely no text, no letters, no speech bubbles, "
    "no sparkles, no stars, no hearts, no tears, no sweat drops, no motion lines, "
    "no emotion symbols and no decorative effects of any kind around the character"
)

# ------------------------------------------------- 업로드 이미지 공유 슬롯
# LangChain tool은 LLM이 만든 인자만 받으므로, 사용자가 올린 원본 이미지는
# 앱(app.py)이 아래 슬롯에 넣어두고 tool이 꺼내 쓰는 구조로 전달한다.

_reference_image: Optional[bytes] = None
_reference_mime: str = "image/png"

# 기준 원화 캐시. 대화 중 이모티콘을 여러 개 만들어도 같은 캐릭터가 나오게 한다.
# (없으면 매번 새 캐릭터가 생겨서 세트로 쓸 수 없다)
_base_cache: Optional[Image.Image] = None

# 마지막 생성 결과 메타데이터 (UI 표시용)
last_result: dict = {}


def set_reference_image(data: Optional[bytes], mime: str = "image/png") -> None:
    """app.py가 사용자 업로드 이미지를 등록한다. 바뀌면 기준 원화 캐시를 버린다."""
    global _reference_image, _reference_mime, _base_cache
    if data != _reference_image:
        _base_cache = None            # 다른 캐릭터이므로 원화를 다시 만들어야 한다
    _reference_image = data
    _reference_mime = mime or "image/png"


def has_reference_image() -> bool:
    return _reference_image is not None


# ------------------------------------------------------------ OpenAI 호출

def _client():
    """OpenAI 클라이언트. OPENAI_API_KEY 환경변수를 사용한다."""
    from openai import OpenAI
    return OpenAI()


def _ext_for(mime: str) -> str:
    return {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
    }.get(mime, "png")


def _as_upload(data: bytes, mime: str = "image/png", name: str = "ref") -> Tuple:
    """openai SDK가 받는 (파일명, 바이트, MIME) 형태로 감싼다."""
    return (f"{name}.{_ext_for(mime)}", io.BytesIO(data), mime)


def _supports_input_fidelity(model: str) -> bool:
    # gpt-image-1 / gpt-image-1.5 만 input_fidelity 지원 (gpt-image-2 계열은 미지원)
    return model in ("gpt-image-1", "gpt-image-1.5")


def _call_image_api(prompt: str, refs: Optional[List[Tuple]] = None,
                    quality: str = "low", size: Optional[str] = None) -> bytes:
    """
    이미지 1장 생성. refs가 있으면 edit(레퍼런스 기반), 없으면 generate.
    size 를 주면 그 크기로 뽑는다 (스프라이트 시트는 1536x1024 등을 쓴다).
    """
    client = _client()
    common = dict(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=size or GEN_SIZE,
        quality=quality,
        background="transparent",   # 생성 시점부터 투명하게 뽑는다 (사후 제거보다 외곽선이 안전)
        output_format="png",        # background="transparent"는 png/webp에서만 유효
        n=1,
    )
    if refs:
        if _supports_input_fidelity(IMAGE_MODEL):
            common["input_fidelity"] = "high"   # 레퍼런스 캐릭터를 최대한 유지
        result = client.images.edit(image=refs, **common)
    else:
        result = client.images.generate(**common)
    return base64.b64decode(result.data[0].b64_json)


# -------------------------------------------------- (1) 기준 원화 만들기

def build_base_sheet(character_description: str, quality: str = "low") -> Image.Image:
    """
    모든 프레임이 참조할 '기준 원화' 1장을 만든다.
    업로드 이미지가 있으면 그 캐릭터를 이모티콘 스타일로 정규화하고,
    없으면 설명만으로 새로 생성한다.
    """
    if _reference_image is not None:
        prompt = (
            "Redraw the character in the reference image as a clean emoticon sticker. "
            "Keep its identity, colors, proportions and distinctive features exactly. "
            f"{character_description}. Neutral front-facing idle pose. {STYLE_SUFFIX}"
        )
        refs = [_as_upload(_reference_image, _reference_mime, "character")]
        raw = _call_image_api(prompt, refs=refs, quality=quality)
    else:
        prompt = f"{character_description}. Neutral front-facing idle pose. {STYLE_SUFFIX}"
        raw = _call_image_api(prompt, refs=None, quality=quality)

    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    _save_png(img, BASE_PATH)
    return img


def get_base_sheet(character_description: str, quality: str = "low") -> Image.Image:
    """
    기준 원화를 가져온다. 처음 한 번만 만들고 이후에는 재사용한다.

    대화 중 이모티콘을 여러 개 만들 때 매번 새로 생성하면 캐릭터가 조금씩
    달라져서 세트로 쓸 수 없다. 업로드 이미지가 바뀌면 캐시는 자동으로 버려진다.
    """
    global _base_cache
    if _base_cache is None:
        _base_cache = build_base_sheet(character_description, quality=quality)
    return _base_cache


# ----------------------------------------- (2) 프레임 N장 병렬 생성

def build_pose_prompt(pose: str) -> str:
    """
    기준 원화를 참조해 '포즈와 표정만' 바꾸는 프롬프트.
    프레임 생성과 종별 대표 포즈 생성이 같은 문구를 쓰도록 한 곳에 모았다.

    글자는 넣지 않는다 — Pillow가 정확한 위치에 직접 쓴다 (STYLE_SUFFIX 참고).
    """
    return (
        "Use the reference image as the exact character design. "
        "Keep the same character, same art style, same colors, same line weight, "
        "same camera distance and the same canvas framing. "
        f"Change only the pose and expression to: {pose}. {STYLE_SUFFIX}"
    )


def generate_pose(base_bytes: bytes, pose: str, quality: str = "low",
                  retries: int = 2, name: str = "base") -> Image.Image:
    """
    기준 원화를 레퍼런스로 포즈 1장 생성. 실패하면 잠깐 쉬었다 다시 시도한다.

    이미지 API는 일시적 오류(429/5xx)와 경계선 검열 판정이 섞여 나온다.
    한 번 실패했다고 포기하면 세트 전체가 구멍 난다.
    """
    last = None
    for attempt in range(retries + 1):
        try:
            raw = _call_image_api(build_pose_prompt(pose),
                                  refs=[_as_upload(base_bytes, "image/png", name)],
                                  quality=quality)
            return Image.open(io.BytesIO(raw)).convert("RGBA")
        except Exception as exc:       # noqa: BLE001 - 사유를 그대로 올려보낸다
            last = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise last


# ====================================================== 스프라이트 시트
# 프레임을 한 장씩 따로 생성하면 그릴 때마다 색과 선이 미세하게 달라져서
# 재생할 때 캐릭터가 부글거린다(boiling). 실측으로 색 표준편차가 R 4.8 / B 6.6 이었고,
# 진짜 카카오 이모티콘은 R 1.5 / B 2.0 이었다.
#
# 그래서 '한 번의 생성'으로 여러 프레임을 한 장에 그리게 한다.
# 같은 붓질 안에서 나오므로 색·선·비율이 원리적으로 같아진다.
# 실측 결과 색 표준편차가 R 0.5 / B 0.4 까지 떨어졌다 (진짜 이모티콘보다 안정적).

# 칸 수 -> (가로칸, 세로칸, 생성 크기)
# gpt-image-1.5 는 1024x1024 / 1536x1024 / 1024x1536 만 지원한다.
# gpt-image-2 는 임의 크기를 지원하지만 '투명 배경'을 지원하지 않아 쓸 수 없다.
SHEET_LAYOUTS = {
    4: (2, 2, "1024x1024"),     # 칸 512x512
    6: (3, 2, "1536x1024"),     # 칸 512x512
    8: (4, 2, "1536x1024"),     # 칸 384x512
}


def pick_layout(n_steps: int) -> Tuple[int, int, str, int]:
    """원하는 프레임 수에 가장 가까운 격자를 고른다. (가로칸, 세로칸, 크기, 실제칸수)"""
    n = min(SHEET_LAYOUTS, key=lambda k: (abs(k - n_steps), -k))
    cols, rows, size = SHEET_LAYOUTS[n]
    return cols, rows, size, n


def build_sheet_prompt(character_description: str, steps: List[str],
                       cols: int, rows: int) -> str:
    """
    스프라이트 시트 생성 프롬프트.

    핵심 지시 세 가지
      1. 모든 칸이 '같은 캐릭터, 같은 색, 같은 선 굵기'
      2. 모든 칸에서 캐릭터가 '같은 크기, 칸 안 같은 위치' (카메라 고정)
      3. 칸 구분선·번호·글자 없음
    """
    n = cols * rows
    lines = "\n".join(f"  cell {i + 1}: {s}" for i, s in enumerate(steps[:n]))
    return (
        f"Create ONE image that is a {cols}x{rows} sprite sheet of animation frames "
        f"({cols} columns, {rows} rows, {n} cells of equal size).\n"
        f"The character is: {character_description}.\n"
        "Use the reference image as the EXACT character design. Keep the identical "
        "character, identical colors, identical line weight and identical art style "
        "in every single cell.\n"
        "Every cell must show the character at the SAME size and the SAME position "
        "within its own cell, as if the camera never moves. Only the pose changes.\n"
        "The cells form a strict uniform grid with no visible separators.\n"
        f"Frame order is left to right, then top to bottom:\n{lines}\n"
        f"{STYLE_SUFFIX}"
    )


def generate_sprite_sheet(character_description: str, steps: List[str],
                          quality: str = "high", retries: int = 2
                          ) -> Tuple[Image.Image, int, int, int]:
    """
    스프라이트 시트 1장을 생성한다. (시트, 가로칸, 세로칸, 칸수)
    업로드 이미지가 있으면 그것을 레퍼런스로 쓰고, 없으면 설명만으로 그린다.
    """
    cols, rows, size, n = pick_layout(len(steps))
    prompt = build_sheet_prompt(character_description, steps, cols, rows)

    refs = None
    if _reference_image is not None:
        refs = [_as_upload(_reference_image, _reference_mime, "character")]

    last = None
    for attempt in range(retries + 1):
        try:
            raw = _call_image_api(prompt, refs=refs, quality=quality, size=size)
            sheet = Image.open(io.BytesIO(raw)).convert("RGBA")
            _save_png(sheet, SHEET_PATH)
            return sheet, cols, rows, n
        except Exception as exc:        # noqa: BLE001
            last = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise last


def split_sheet(sheet: Image.Image, cols: int, rows: int,
                min_fill: float = 0.01) -> List[Image.Image]:
    """시트를 칸으로 자른다. 거의 비어 있는 칸은 버린다."""
    import numpy as np

    w, h = sheet.size
    cw, ch = w // cols, h // rows
    out = []
    for r in range(rows):
        for c in range(cols):
            cell = sheet.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
            if (np.asarray(cell.getchannel("A")) > 200).mean() < min_fill:
                continue
            out.append(cell)
    return out


def align_frames(cells: List[Image.Image], band: float = 0.40,
                 limit_ratio: float = 0.35) -> List[Image.Image]:
    """
    칸마다 캐릭터가 좌우로 어긋난 것을 맞춘다.

    모델이 칸 안에서 캐릭터를 늘 같은 자리에 두지는 않는다. 실측으로 132px
    (칸 폭의 34%)까지 어긋났고, 그대로 재생하면 캐릭터가 좌우로 덜컹거린다.

    방법: 아래쪽 band 부분(몸통·다리)의 '열별 픽셀 수' 1차원 프로파일을
          1번 프레임과 상호상관시켜 가장 잘 맞는 가로 이동량을 찾는다.
          팔은 움직이므로 위쪽은 보지 않는다.

    왜 이 방법인가 (8가지를 같은 기준으로 비교한 결과)
      보정 없음 0.409 / bbox 중심 0.567 / 하단 겹침최대 0.739 / 이 방법 0.738
      품질은 겹침최대와 같은데 9배 빠르다 (24ms vs 211ms).
    세로는 실측 1px 밖에 안 어긋나 보정하지 않는다.
    """
    import numpy as np

    if len(cells) < 2:
        return list(cells)

    h = cells[0].size[1]
    y0 = int(h * (1 - band))
    limit = int(cells[0].size[0] * limit_ratio)

    def profile(img):
        a = np.asarray(img.getchannel("A"))[y0:, :] > 128
        p = a.sum(axis=0).astype(float)
        return p - p.mean()

    ref = profile(cells[0])
    out = [cells[0]]
    for cell in cells[1:]:
        cur = profile(cell)
        best, shift = -1e18, 0
        for s in range(-limit, limit + 1):
            v = float((ref * np.roll(cur, s)).sum())
            if v > best:
                best, shift = v, s
        if shift == 0:
            out.append(cell)
            continue
        arr = np.roll(np.asarray(cell), shift, axis=1)
        if shift > 0:
            arr[:, :shift] = 0
        else:
            arr[:, shift:] = 0
        out.append(Image.fromarray(arr, "RGBA"))
    return out


def stabilize_alpha(img: Image.Image, threshold: int = 24) -> Image.Image:
    """
    알파 채널을 정리한다.
      - threshold 미만의 희미한 알파를 완전 투명으로 (반투명 찌꺼기 제거)
      - opening(MinFilter3 -> MaxFilter3) 으로 떠다니는 점 노이즈 제거
      - closing(MaxFilter5 -> MinFilter5) 으로 4px 이하 바늘구멍 메우기
    큰 구멍은 fill_interior_holes()가 따로 처리한다.
    """
    img = img.convert("RGBA")
    alpha = img.getchannel("A")
    alpha = alpha.point(lambda v: 0 if v < threshold else v)
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    alpha = alpha.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    img.putalpha(alpha)
    return img


def robust_bbox(img: Image.Image, frac: float = 0.01
                ) -> Optional[Tuple[int, int, int, int]]:
    """
    점 노이즈에 흔들리지 않는 바운딩박스.

    Pillow의 getbbox()는 '알파가 0이 아닌 픽셀'을 전부 포함한다. 그래서
    캐릭터와 멀리 떨어진 작은 점 하나가 박스를 수백 픽셀 부풀리고,
    그만큼 캐릭터가 쓸데없이 작게 축소된다. 생성 AI 결과에는 그런
    찌꺼기가 흔히 남는다.

    여기서는 행/열별 '알파 질량'을 구해, 최댓값의 frac 배를 넘는 행/열만
    경계로 인정한다. 점 노이즈는 질량이 작아 자동으로 빠지고 캐릭터
    윤곽은 남는다. frac=0.01 에서 실측 오차 1px 수준이었고, 0.005 이하로
    내리면 노이즈가 다시 포함됐다.
    """
    import numpy as np

    a = np.asarray(img.convert("RGBA").getchannel("A"), dtype=np.float64)
    if a.sum() <= 0:
        return None

    def span(v):
        peak = v.max()
        if peak <= 0:
            return None
        idx = np.nonzero(v > peak * frac)[0]
        if idx.size == 0:
            return None
        return int(idx[0]), int(idx[-1]) + 1

    sx = span(a.sum(axis=0))
    sy = span(a.sum(axis=1))
    if sx is None or sy is None:
        return img.convert("RGBA").getchannel("A").getbbox()
    return (sx[0], sy[0], sx[1], sy[1])


def fill_interior_holes(img: Image.Image, cut: int = 128) -> Image.Image:
    """
    '테두리와 연결되지 않은' 투명 영역만 불투명으로 메운다.

    이미지 생성 API가 캐릭터 안쪽의 밝은 영역(눈 흰자, 치아, 밝은 옷 등)을
    배경으로 오인해 뚫어버리는 현상이 보고되어 있다. 형태학적 closing은
    작은 구멍만 닫을 수 있어서, 캔버스 바깥 테두리에서 flood fill을 돌려
    '바깥 배경'을 식별하고 그 밖에 남은 투명 영역을 내부 구멍으로 판정한다.
    캐릭터 겉 실루엣은 건드리지 않는다.
    """
    img = img.convert("RGBA")
    w, h = img.size
    binary = img.getchannel("A").point(lambda v: 255 if v >= cut else 0)

    # 1px 투명 테두리를 둘러 바깥 배경이 항상 (0,0)과 이어지게 만든다
    padded = Image.new("L", (w + 2, h + 2), 0)
    padded.paste(binary, (1, 1))
    ImageDraw.floodfill(padded, (0, 0), 128, thresh=0)   # 바깥 배경만 128로 표시
    outside = padded.crop((1, 1, w + 1, h + 1))

    # 128(바깥)만 투명으로 두고, 나머지(캐릭터 + 내부 구멍)는 불투명으로
    filled = outside.point(lambda v: 0 if v == 128 else 255)

    alpha = img.getchannel("A")
    img.putalpha(ImageChops.lighter(alpha, filled))
    return img


# ------------------- (4)+(5) 공통 바운딩박스 정렬 + 규격 캔버스

def _canvas_wh(canvas) -> Tuple[int, int]:
    """canvas 는 정수(정사각형) 또는 (가로, 세로) 둘 다 받는다."""
    if isinstance(canvas, (tuple, list)):
        return int(canvas[0]), int(canvas[1])
    return int(canvas), int(canvas)


def union_bbox(frames: List[Image.Image]) -> Optional[Tuple[int, int, int, int]]:
    """여러 프레임을 모두 덮는 bbox. 효과 기준점을 '고정'하는 데 쓴다."""
    boxes = [robust_bbox(f) for f in frames]
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _layout(side: int, canvas, margin: float,
            reserve_top: float, reserve_bottom: float):
    """
    배치 계산을 한 곳에 모은 것. 프레임 묶음과 종 세트가 같은 수식을 쓴다.

    reserve_top / reserve_bottom 은 글자나 효과를 올릴 자리를 위/아래에
    비워두는 비율(0~1)이다. 안 비우면 캐릭터가 캔버스를 꽉 채워
    글자가 얼굴을 덮는다.

    돌려주는 값: (캔버스 가로세로, 캐릭터를 그릴 한 변, 좌상단 오프셋)
    """
    cw, ch = _canvas_wh(canvas)
    avail_w = cw * (1 - 2 * margin)
    avail_h = ch * (1 - 2 * margin - reserve_top - reserve_bottom)
    if avail_w <= 1 or avail_h <= 1:
        raise ValueError("여백/예약 공간이 너무 커서 캐릭터를 놓을 자리가 없습니다")
    target = max(1, int(min(avail_w, avail_h)))
    ox = (cw - target) // 2
    oy = int(ch * (margin + reserve_top)) + max(0, (int(avail_h) - target) // 2)
    return (cw, ch), target, (ox, oy)


def _place(img: Image.Image, center, side: int, target: int,
           offset, size) -> Image.Image:
    """center 를 중심으로 한 변 side 인 정사각형을 떠서 캔버스에 붙인다."""
    cx, cy = center
    x0, y0 = cx - side // 2, cy - side // 2
    crop = img.crop((x0, y0, x0 + side, y0 + side))   # 경계를 넘어도 투명으로 채워진다
    sheet = Image.new("RGBA", size, (0, 0, 0, 0))
    sheet.paste(crop.resize((target, target), Image.LANCZOS), offset)
    return sheet


def fit_frames_to_canvas(frames: List[Image.Image], canvas=CANVAS,
                         margin_ratio: float = 0.03,
                         reserve_top: float = 0.0,
                         reserve_bottom: float = 0.0) -> List[Image.Image]:
    """
    '한 애니메이션의 프레임들'을 캔버스에 맞춘다.

    모든 프레임에 똑같은 크롭/스케일을 적용한다. 프레임별로 따로 중앙정렬하면
    점프 같은 의도된 모션이 사라지기 때문이다.
    canvas 는 정수 또는 (가로, 세로) — OGQ 740x640, 라인 320x270 대응.
    """
    if not frames:
        return []
    size = _canvas_wh(canvas)
    box = union_bbox(frames)
    if box is None:                      # 전부 투명이면 축소만 한다
        return [f.resize(size, Image.LANCZOS) for f in frames]

    left, top, right, bottom = box
    side = max(right - left, bottom - top)
    center = ((left + right) // 2, (top + bottom) // 2)
    size, target, offset = _layout(side, canvas, margin_ratio,
                                   reserve_top, reserve_bottom)
    return [_place(f, center, side, target, offset, size) for f in frames]


def fit_art_set(arts: dict, canvas=(CANVAS, CANVAS), margin: float = 0.04,
                reserve_top: float = 0.0, reserve_bottom: float = 0.0) -> dict:
    """
    '서로 다른 종'의 그림을 같은 배율로 배치한다.
    {라벨: 그림} -> {라벨: (배치된 그림, 배치된 캐릭터 bbox)}

    fit_frames_to_canvas() 를 종마다 따로 쓰면 안 되는 이유:
      그건 프레임 간 상대 위치를 보존하려고 만든 함수다. 서로 다른 종에 각각
      적용하면 팔 든 포즈와 움츠린 포즈가 제각각 확대돼서 24종의 캐릭터
      크기가 들쭉날쭉해진다. 그래서 여기서는 가장 큰 종을 기준으로 배율을
      하나 정하고 위치만 종별로 맞춘다.

    돌려주는 bbox 는 '실제로 배치된' 캐릭터 영역이라
    geo_effects.anchors_from_bbox() 에 그대로 넘기면 된다.
    """
    size = _canvas_wh(canvas)
    boxes = {k: robust_bbox(v) for k, v in arts.items()}
    boxes = {k: b for k, b in boxes.items() if b}
    if not boxes:
        return {k: (v.resize(size, Image.LANCZOS), (0, 0) + size)
                for k, v in arts.items()}

    side = max(max(b[2] - b[0], b[3] - b[1]) for b in boxes.values())
    size, target, offset = _layout(side, canvas, margin, reserve_top, reserve_bottom)

    out = {}
    for k, img in arts.items():
        b = boxes.get(k)
        if not b:
            out[k] = (img.resize(size, Image.LANCZOS), (0, 0) + size)
            continue
        center = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
        sheet = _place(img, center, side, target, offset, size)
        out[k] = (sheet, robust_bbox(sheet) or offset + (offset[0] + target,
                                                         offset[1] + target))
    return out


def fit_art_for_motion(art: Image.Image, motion: str, canvas=(CANVAS, CANVAS),
                       margin: float = 0.045, reserve_top: float = 0.0,
                       max_up_ratio: float = 0.22):
    """
    '모션이 움직일 공간'과 '글자·효과 자리'를 먼저 빼고 캐릭터를 배치한다.

    왜 필요한가:
      점프는 캐릭터를 95px 위로 올리고, 낙하는 110px 올린다. 캐릭터를 먼저
      크게 배치해놓고 모션을 걸면 머리가 캔버스 밖으로 잘리거나 글자와 겹친다.

    처리 순서
      1. 모션이 필요한 여백(위/아래/좌우)과 늘어남 배율을 미리 구한다
      2. 위로 가는 양이 캔버스의 max_up_ratio 를 넘으면 '움직임'을 줄인다
         (캐릭터를 더 줄이는 것보다 낫다 - 캐릭터가 너무 작아지면 안 예쁘다)
      3. 남은 공간에 캐릭터를 넣고 바닥 쪽에 붙인다

    돌려주는 값: (배치된 그림, 캐릭터 bbox, 모션에 적용할 배율)
    """
    import math

    import geo_motion as gm

    w, h = _canvas_wh(canvas)
    params = gm.MOTIONS[motion]()
    mx, my = w * margin, h * margin

    # 위로 가는 양을 캔버스의 max_up_ratio 로 제한 (넘치면 움직임을 줄인다)
    up_raw = max(0.0, -min(p.get("dy", 0.0) for p in params))
    mscale = min(1.0, h * max_up_ratio / up_raw) if up_raw > 1e-6 else 1.0

    down = max(0.0, max(p.get("dy", 0.0) for p in params)) * mscale
    bottom = h - my - down                      # 캐릭터 발바닥이 놓일 높이
    top_limit = my + h * reserve_top            # 그 위는 글자·효과 자리
    room = bottom - top_limit

    # 프레임마다 '그 순간' 필요한 크기를 따로 계산해 가장 빡빡한 것에 맞춘다.
    # 방향별 최댓값을 따로 구해서 곱하면(늘어남 최대 x 높이 최대) 실제로는
    # 동시에 일어나지 않는 상황까지 피하느라 캐릭터가 쓸데없이 작아진다.
    side = float(min(room, w - 2 * mx))
    for p in params:
        th = math.radians(p.get("rot", 0.0))
        spread = abs(math.cos(th)) + abs(math.sin(th))
        sy = max(1e-3, p.get("sy", 1.0) * spread)
        sx = max(1e-3, p.get("sx", 1.0) * spread)
        up_f = max(0.0, -p.get("dy", 0.0) * mscale)
        dx_f = abs(p.get("dx", 0.0)) * mscale
        side = min(side, (room - up_f) / sy)                  # 위로 안 잘리게
        side = min(side, 2 * (w / 2 - mx - dx_f) / sx)        # 좌우로 안 잘리게

    target = int(max(32, side))

    box = robust_bbox(art)
    if box is None:
        sheet = art.resize((w, h), Image.LANCZOS)
        return sheet, (0, 0, w, h), mscale

    src_side = max(box[2] - box[0], box[3] - box[1])
    center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    ox = (w - target) // 2
    oy = int(bottom) - target
    sheet = _place(art, center, src_side, target, (ox, oy), (w, h))
    return sheet, (robust_bbox(sheet) or (ox, oy, ox + target, oy + target)), mscale


# ----------------------------------------- (6) 무한 루핑 시퀀스

def build_loop_sequence(frames: List[Image.Image], mode: str = "pingpong") -> List[Image.Image]:
    """
    mode="pingpong" : 1-2-3-4-3-2-1 로 재배열
    mode="cycle"    : 1-2-3-4-1 (동작 자체가 이미 순환하는 경우)

    왜 '대표 프레임(1번)'으로 끝나야 하는가 — 세 플랫폼 규칙이 여기서 동시에 풀린다.
      카카오 : 마지막 프레임이 이모티콘샵/키보드 썸네일로 쓰인다
      라인   : 첫 프레임이 정적 이미지로 노출된다
      OGQ    : 첫 프레임과 마지막 프레임이 같아야 한다
    따라서 대표 포즈를 1번에 두고 1번으로 돌아오면 셋 다 만족한다.
    (예전엔 1-2-3-4-3-2 로 끝나서 어중간한 중간 동작이 썸네일이 됐다)
    """
    if len(frames) < 2:
        return list(frames)

    if mode == "cycle":
        seq = list(frames) + [frames[0]]
    elif len(frames) < 3:
        seq = list(frames) + [frames[0]]
    else:
        seq = list(frames) + list(reversed(frames[1:-1])) + [frames[0]]

    if len(seq) > MAX_FRAMES:       # 제안 규격: 24프레임 이하
        seq = seq[:MAX_FRAMES - 1] + [frames[0]]   # 잘라도 대표로 끝맺는다
    return seq


# ------------------------------------------------------- (7) 저장

def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _save_png(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path, format="PNG")


def _hard_mask(img: Image.Image, cut: int = 128) -> Image.Image:
    """GIF는 1비트 투명만 지원하므로 알파를 0/255로 이진화한다."""
    return img.getchannel("A").point(lambda v: 255 if v >= cut else 0)


def _build_shared_palette(frames: List[Image.Image], colors: int = 255) -> Image.Image:
    """
    전체 프레임을 한 장으로 이어붙여 공용 팔레트를 만든다.
    프레임마다 팔레트를 따로 만들면 재생 중 색이 깜빡이는데, 그걸 막는다.
    colors=255 로 두어 인덱스 255번을 '투명'용으로 비워둔다.
    """
    w, h = frames[0].size
    montage = Image.new("RGB", (w, h * len(frames)), (255, 255, 255))
    for i, f in enumerate(frames):
        flat = Image.new("RGB", (w, h), (255, 255, 255))
        flat.paste(f.convert("RGB"), (0, 0), _hard_mask(f))
        montage.paste(flat, (0, h * i))
    return montage.convert("P", palette=Image.ADAPTIVE, colors=colors)


def _reduce_palette(master: Image.Image, colors: int, sample: int = 6144) -> Image.Image:
    """
    이미 만든 공용 팔레트를 '더 적은 색'으로 줄인다.

    용량이 넘칠 때마다 몽타주(수백만 픽셀)를 다시 양자화하면 매번 ~95ms가 든다.
    여기서는 팔레트 색을 '실제 사용 빈도만큼 반복한' 수천 픽셀짜리 띠를 만들어
    그것만 양자화한다. 빈도를 반영하므로 화질 손해는 작고 속도는 수십 배 빠르다.
    """
    hist = master.histogram()[:256]
    total = sum(hist) or 1
    data = []
    for idx, cnt in enumerate(hist):
        if cnt <= 0:
            continue
        data.extend([idx] * max(1, round(cnt / total * sample)))
    if not data:
        return master

    strip = Image.new("P", (len(data), 1))
    strip.putpalette(master.getpalette())
    strip.putdata(data)
    return strip.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=colors)


def save_gif(frames: List[Image.Image], path: str = GIF_PATH,
             duration_ms=150, colors: int = 255,
             master: Optional[Image.Image] = None) -> int:
    """
    투명 배경 + 무한 루프 GIF 저장. 저장된 바이트 크기를 반환한다.

    duration_ms : 정수(전 프레임 동일) 또는 프레임별 리스트
    master      : 미리 만들어둔 공용 팔레트 (없으면 여기서 만든다)
    """
    if master is None:
        master = _build_shared_palette(frames, colors=colors)
    transparent_idx = 255

    pal_frames = []
    for f in frames:
        mask = _hard_mask(f)
        flat = Image.new("RGB", f.size, (255, 255, 255))
        flat.paste(f.convert("RGB"), (0, 0), mask)
        p = flat.quantize(palette=master, dither=Image.Dither.NONE)
        p.paste(transparent_idx, mask=ImageChops.invert(mask))  # 배경을 투명 인덱스로
        pal_frames.append(p)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pal_frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=pal_frames[1:],
        duration=duration_ms,
        loop=0,                     # 0 = 무한 반복
        transparency=transparent_idx,
        disposal=2,                 # 다음 프레임 전에 지우기 (투명 영역 잔상 방지)
        optimize=False,
    )
    return os.path.getsize(path)


def save_webp(frames: List[Image.Image], path: str = WEBP_PATH,
              duration_ms: int = 150) -> int:
    """
    알파가 온전히 살아있는 애니메이션 WebP 저장.
    카카오는 '승인 후' 단계에서 WebP를 요구하므로 미리 같이 내보낸다.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frames[0].save(
        path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        lossless=True,
    )
    return os.path.getsize(path)


def shrink_until_under_limit(frames: List[Image.Image], path: str,
                             duration_ms=150, limit: int = MAX_BYTES) -> Tuple[int, str]:
    """
    용량 한도를 넘으면 색 수를 단계적으로 줄여 다시 저장한다.
    프레임을 버리면 움직임이 끊기므로 색을 먼저 깎는다.

    공용 팔레트는 '한 번만' 만들고, 재시도에서는 _reduce_palette() 로 줄인다.
    """
    master = _build_shared_palette(frames, colors=255)
    size = 0
    for colors in (255, 128, 64, 32):
        pal = master if colors == 255 else _reduce_palette(master, colors)
        size = save_gif(frames, path=path, duration_ms=duration_ms, master=pal)
        if size <= limit:
            note = (f"용량 한도({limit//1024}KB)를 맞추려고 색을 {colors}색으로 줄였습니다"
                    if colors != 255 else "")
            return size, note
    return size, (f"32색으로 줄여도 {size/1024:.0f}KB "
                  f"(한도 {limit//1024}KB 초과) — 프레임 수를 줄여야 합니다")


# =========================================================== LangChain Tool

@tool(parse_docstring=True)
def create_animated_emoticon(
    character_description: str,
    frame_steps: List[str],
    effect: Literal["none", "heart", "sparkle", "tear",
                    "sweat", "anger", "note"] = "none",
    overlay_text: str = "",
    speed_ms: int = 110,
    hold_last_ms: int = 300,
    quality: Literal["low", "medium", "high"] = "high",
) -> str:
    """움직이는 이모티콘 파일(GIF)을 실제로 만든다.

    사용자가 이모티콘 제작을 명확히 요청했을 때만 호출한다. 기획안만 보여달라고 하면
    호출하지 말고 대화로만 답한다. character_description 과 frame_steps 는 영문으로 쓴다.

    한 번의 생성으로 모든 프레임을 한 장(스프라이트 시트)에 그리기 때문에, 프레임마다
    색과 선이 흔들리지 않는다. 손 흔들기, 꾸벅 졸기처럼 팔다리가 따로 움직이는 동작도
    된다. 같은 대화 안에서 여러 번 호출하면 캐릭터가 유지된다.

    Args:
        character_description: 캐릭터의 외형, 색상, 분위기를 묘사한 영문 문장. 사용자가 이미지를 올렸다면 그 캐릭터를 유지하면서 보강할 설명을 쓴다.
        frame_steps: 프레임별 자세를 묘사한 영문 문장 리스트. 6개 또는 8개를 권장하고 4개도 가능하다. 각 문장은 '한 장의 정지 그림'을 묘사하며 움직임을 설명하지 않는다. 동작이 자연스럽게 이어지도록 순서대로 쓰고, 반복 재생되므로 첫 번째와 마지막은 같은 기본 자세로 둔다. 마지막 자세가 이모티콘샵 썸네일이 되므로 캐릭터를 가장 잘 보여주는 자세로 한다. 효과나 글자는 쓰지 않는다. 예시는 front paw down at its side with a calm smile / front paw lifted to chest height / front paw raised high beside the head with an open smile 처럼 쓴다.
        effect: 덧붙일 효과를 고른다. none 은 효과 없음이다. heart 는 하트가 떠올라 설렘에 쓴다. sparkle 은 반짝임으로 최고나 뿌듯함에 쓴다. tear 는 눈물로 슬픔에 쓴다. sweat 는 땀방울로 당황에 쓴다. anger 는 김이 뿜어져 화남에 쓴다. note 는 음표가 떠올라 신남에 쓴다.
        overlay_text: 이모티콘에 넣을 짧은 한글 글자. 필요 없으면 빈 문자열로 둔다. ㅋㅋㅋ 고마워 미안 같이 짧을수록 좋다. 글자는 폰트로 정확히 찍히므로 그림에 맡기지 않는다.
        speed_ms: 프레임 하나의 재생 시간을 밀리초로 준다. 80에서 150 사이를 권장하고 작을수록 빠르다.
        hold_last_ms: 마지막 프레임에서 더 머무는 시간을 밀리초로 준다. 동작이 끝나고 잠깐 쉬는 느낌을 준다. 0이면 쉬지 않는다.
        quality: 그림 품질이다. high 를 권장한다. 빠르게 확인만 할 때는 low 를 쓴다.
    """
    import geo_effects as gfx

    canvas = (CANVAS, CANVAS)
    if len(frame_steps) < 2:
        return "오류: frame_steps 가 너무 적습니다. 6개 또는 8개를 주세요."

    try:
        # (1) 스프라이트 시트 1장 생성 — AI 호출은 여기 한 번뿐
        sheet, cols, rows, n_cells = generate_sprite_sheet(
            character_description, frame_steps, quality=quality)

        # (2) 칸 분할
        cells = split_sheet(sheet, cols, rows)
        if len(cells) < 2:
            return (f"오류: 격자가 제대로 생성되지 않았습니다 "
                    f"({len(cells)}/{n_cells}칸만 내용이 있음). 다시 시도해 보세요.")

        # (3) 칸마다 어긋난 좌우 위치 맞추기
        cells = align_frames(cells)

        # (4) 투명 배경 정리
        cleaned = [stabilize_alpha(c) for c in cells]

        # (5) 공통 배율로 규격 캔버스에 배치. 글자가 있으면 위를 비운다
        reserve = 0.20 if overlay_text else (0.08 if effect != "none" else 0.0)
        frames = fit_frames_to_canvas(cleaned, canvas=canvas, margin_ratio=0.05,
                                      reserve_top=reserve)
        frames = [fill_interior_holes(f) for f in frames]

        # (6) 효과와 글자 — Pillow가 캐릭터 위치에 맞춰 직접 그린다
        box = union_bbox(frames)
        if box and (effect != "none" or overlay_text):
            anchors = gfx.anchors_from_bbox(box, canvas)
            if effect != "none":
                frames = gfx.composite(frames, gfx.EFFECTS[effect](
                    n=len(frames), anchors=anchors, canvas=canvas))
            if overlay_text:
                frames = gfx.composite(frames, gfx.fx_text(
                    overlay_text, n=len(frames), anchors=anchors, canvas=canvas))

        # (7) 재생 시간 — 마지막 프레임에서 잠깐 쉰다
        durations = [speed_ms] * len(frames)
        durations[-1] = min(2000, speed_ms + max(0, hold_last_ms))

        # (8) 저장 — GIF + 번호 붙인 PNG 프레임(카카오 WebP Animator 용)
        gif_size, note = shrink_until_under_limit(frames, GIF_PATH, durations,
                                                  limit=SAFE_MAX_BYTES)
        webp_size = save_webp(frames, WEBP_PATH, durations)

        frames_dir = os.path.join(OUT_DIR, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, f in enumerate(frames, 1):
            f.save(os.path.join(frames_dir, f"{i:02d}.png"))

    except Exception as exc:   # LLM이 사유를 보고 다시 시도할 수 있게 문자열로 돌려준다
        return f"이모티콘 생성 실패: {type(exc).__name__}: {exc}"

    last_result.update(
        gif_path=GIF_PATH, webp_path=WEBP_PATH, sheet_path=SHEET_PATH,
        frames_dir=frames_dir,
        frame_paths=[SHEET_PATH],          # AI가 그린 원본 = 스프라이트 시트 한 장
        unique_frames=len(frames), total_frames=len(frames),
        loop_mode=f"{cols}x{rows} 시트", effect=effect,
        duration_ms=speed_ms, gif_size=gif_size, webp_size=webp_size,
    )

    total_s = sum(durations) / 1000
    dropped = (f" (요청 {len(frame_steps)}칸 중 {len(frames)}칸 사용)"
               if len(frames) != len(frame_steps) else "")
    return (
        f"이모티콘 생성 완료!\n"
        f"- {cols}x{rows} 스프라이트 시트 1장을 생성해 {len(frames)}프레임으로 분할"
        f"{dropped}\n"
        f"- 재생 {total_s:.1f}초 (프레임당 {speed_ms}ms, 마지막 {durations[-1]}ms)\n"
        + (f"- 효과: {effect}\n" if effect != "none" else "")
        + (f"- 글자: {overlay_text}\n" if overlay_text else "")
        + f"- 규격: {CANVAS}x{CANVAS}px, 투명 배경, GIF {gif_size/1024:.0f}KB"
          f" (한도 {SAFE_MAX_BYTES//1024}KB)\n"
        + (f"- 참고: {note}\n" if note else "")
        + "화면 왼쪽 사이드바에서 미리보기와 다운로드가 가능하다고 사용자에게 안내하세요."
    )


TOOLS = [create_animated_emoticon]
TOOL_DICT = {t.name: t for t in TOOLS}
