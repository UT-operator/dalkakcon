# -*- coding: utf-8 -*-
"""
효과 합성 모듈 — 하트 / 반짝임 / 눈물 / 땀 / 분노 / 음표 / 글자

AI에게 "하트를 그려줘"라고 부탁하면 매번 다르게 나오고 비용도 든다.
단순한 도형과 글자는 Pillow로 직접 그리는 게 정확하고 공짜다.

[루프 이음새 처리]
  각 효과 입자는 자기 주기의 시작과 끝에서 알파 0(완전 투명)이 된다.
  보이지 않는 상태에서 한 바퀴가 닫히므로 반복 재생해도 이음새가 없다.

[다크모드 대응]
  카카오와 OGQ 둘 다 "어두운 배경에서도 보이도록 흰색 테두리"를 요구한다.
  add_white_outline() 이 알파를 부풀려 흰 실루엣을 뒤에 깔아준다.
"""

from __future__ import annotations

import glob
import math
import os
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

CANVAS = 360
LINE = (74, 48, 26, 255)

# ------------------------------------------------------------- 기준점 계산
# 효과를 고정 좌표에 찍으면 캐릭터 생김새가 바뀔 때 전부 틀어진다.
# (실제로 '고마워' 글자가 얼굴 한가운데를 덮는 사고가 있었다)
# 그래서 캐릭터가 실제로 차지한 영역(bbox)에서 기준점을 매번 계산한다.

_DEFAULT_BBOX = (60, 120, 300, 340)   # 기준점을 못 받았을 때의 보수적 추정


def anchors_from_bbox(bbox, canvas=(CANVAS, CANVAS)) -> Dict[str, Tuple[int, int]]:
    """
    캐릭터 bbox -> 효과 기준점들.

    bbox 는 캔버스 위에 '실제로 배치된' 캐릭터의 영역이어야 한다.
    (emoticon_tools.fit_art_set() 이 배치 결과와 함께 돌려준다)
    """
    x0, y0, x1, y1 = bbox
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    cx = (x0 + x1) // 2
    cw, ch = canvas

    def clamp(p):
        return (max(0, min(cw, int(p[0]))), max(0, min(ch, int(p[1]))))

    return {
        # 머리 위 '빈 공간의 한가운데'. 글자가 여기 중앙정렬로 놓이므로
        # 캐릭터 바로 위(y0 - 조금)로 잡으면 글자 아랫부분이 귀를 덮는다.
        "above_head": clamp((cx, max(14, y0 * 0.5))),
        "below":      clamp((cx, y1 + h * 0.10)),
        "chest":      clamp((cx, y0 + h * 0.66)),
        "left_eye":   clamp((x0 + w * 0.30, y0 + h * 0.46)),
        "right_eye":  clamp((x0 + w * 0.70, y0 + h * 0.46)),
        "head_tr":    clamp((x1 - w * 0.06, y0 + h * 0.12)),
        "head_tl":    clamp((x0 + w * 0.06, y0 + h * 0.12)),
        # 캐릭터 바깥 위쪽. 음표처럼 '허공에 떠오르는' 효과용.
        # 실루엣에 겹치면 지저분해서 bbox 모서리 바깥으로 뺀다.
        "float_tr":   clamp((min(cw - 26, x1 + w * 0.02), max(20, y0 + h * 0.04))),
        "float_tl":   clamp((max(26, x0 - w * 0.02), max(20, y0 + h * 0.04))),
        "_bbox":      (x0, y0, x1, y1),
        "_size":      (w, h),
    }


def default_anchors(canvas=(CANVAS, CANVAS)) -> Dict[str, Tuple[int, int]]:
    return anchors_from_bbox(_DEFAULT_BBOX, canvas)


def _pick(anchors, name, canvas=(CANVAS, CANVAS)):
    """앵커 dict에서 기준점을 꺼낸다. 없으면 보수적 기본값."""
    if anchors is None:
        anchors = default_anchors(canvas)
    return anchors[name]


def _scale_of(anchors, canvas=(CANVAS, CANVAS)) -> float:
    """
    효과 크기 배율.

    캐릭터 크기에 비례시키되, '캔버스에 남은 여백'도 함께 본다.
    캐릭터가 화면을 꽉 채우면 효과를 놓을 자리가 없는데, 크기만 보고 키우면
    효과가 캔버스 밖으로 잘린다. (실제로 sparkle·anger·note 가 잘렸다)
    """
    if not anchors or "_size" not in anchors:
        return 1.0
    base = anchors["_size"][1] / 220.0
    x0, y0, x1, y1 = anchors["_bbox"]
    cw, ch = canvas
    room = max(8, min(y0, x0, cw - x1, ch - y1))   # 사방 여백 중 가장 좁은 곳
    return max(0.35, min(1.8, base, room / 26.0))

# ------------------------------------------------------------------- 폰트

# 폰트를 '고정 경로 목록'으로 찾으면 환경이 조금만 달라도 못 찾는다.
# (Colab에서 apt-get 이 조용히 실패해 글자가 전부 네모로 나온 적이 있다)
# 그래서 실제로 디렉터리를 뒤지고, 한글 글리프가 '진짜 있는지'까지 확인한다.

_FONT_DIRS = [
    "/usr/share/fonts", "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"), "C:/Windows/Fonts",
]
# 앞에 있을수록 우선. 나눔고딕은 OFL 라이선스라 상업 사용이 허용된다.
_FONT_PREFER = ["nanumgothicbold", "nanumgothic", "malgunbd", "malgun",
                "notosanskr", "notosanscjk", "applesdgothic", "gulim", "batang"]

# 시스템에 아무것도 없을 때 받아오는 곳 (Google Fonts, OFL)
_FONT_URL = ("https://github.com/google/fonts/raw/main/ofl/"
             "nanumgothic/NanumGothic-Regular.ttf")
_LOCAL_FONT = "NanumGothic.ttf"

_resolved_font = None          # 한 번 찾으면 재사용


def _has_korean(path: str) -> bool:
    """
    그 폰트에 한글 글리프가 실제로 있는지 확인한다.

    Pillow 는 폰트 폴백(다른 폰트에서 빌려오기)을 하지 않는다. 그래서 한글이
    없는 폰트로 찍으면 네모(두부)가 나온다. 그런데 크기는 정상으로 보고되므로
    getbbox() 만으로는 구분할 수 없다.
    여기서는 '가' 와 '쓰이지 않는 문자'의 렌더 결과를 비교한다.
    둘이 같으면 둘 다 네모라는 뜻이다.
    """
    try:
        f = ImageFont.truetype(path, 20)
        return bytes(f.getmask("가")) != bytes(f.getmask("\ue000"))
    except Exception:
        return False


def find_korean_font() -> Optional[str]:
    """시스템에서 한글이 되는 폰트를 찾는다. 없으면 None."""
    global _resolved_font
    if _resolved_font and os.path.exists(_resolved_font):
        return _resolved_font

    if os.path.exists(_LOCAL_FONT) and _has_korean(_LOCAL_FONT):
        _resolved_font = _LOCAL_FONT
        return _resolved_font

    found = []
    for d in _FONT_DIRS:
        if os.path.isdir(d):
            for ext in ("ttf", "ttc", "otf"):
                found += glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True)

    def rank(p):
        name = os.path.basename(p).lower().replace("-", "").replace("_", "")
        for i, key in enumerate(_FONT_PREFER):
            if key in name:
                return i
        return len(_FONT_PREFER)

    # 이름이 익숙한 것부터, 그다음 나머지를 (너무 오래 걸리지 않게) 일부만
    ordered = sorted(found, key=rank)
    for p in ordered[:80]:
        if _has_korean(p):
            _resolved_font = p
            return p
    return None


def ensure_korean_font(download: bool = True) -> Optional[str]:
    """
    한글 폰트를 확보한다. 시스템에 없으면 나눔고딕을 받아온다.
    Colab 에서 apt-get 이 실패해도 이쪽으로 복구된다.
    """
    global _resolved_font
    p = find_korean_font()
    if p or not download:
        return p
    try:
        import urllib.request
        urllib.request.urlretrieve(_FONT_URL, _LOCAL_FONT)
        if _has_korean(_LOCAL_FONT):
            _resolved_font = _LOCAL_FONT
            return _resolved_font
    except Exception:
        pass
    return None


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """한글이 되는 폰트를 불러온다. 처음 쓸 때 없으면 받아온다."""
    path = ensure_korean_font()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def font_available() -> bool:
    """한글 폰트를 쓸 수 있는지. 없으면 받아와서라도 확보한다."""
    return ensure_korean_font() is not None


# --------------------------------------------------- 다크모드 흰 테두리

def add_white_outline(img: Image.Image, width: int = 5) -> Image.Image:
    """
    알파를 부풀려 흰색 실루엣을 뒤에 깔아준다.
    어두운 배경(다크모드 채팅방)에서도 형체가 보이게 하는 필수 처리.
    """
    img = img.convert("RGBA")
    alpha = img.getchannel("A")
    grown = alpha
    for _ in range(max(1, width // 2)):
        grown = grown.filter(ImageFilter.MaxFilter(5))
    grown = grown.point(lambda v: 255 if v > 40 else 0)

    halo = Image.new("RGBA", img.size, (255, 255, 255, 0))
    halo.putalpha(grown)
    return Image.alpha_composite(halo, img)


# ------------------------------------------------------------- 도형 그리기

def _heart_polygon(cx: float, cy: float, r: float) -> List[Tuple[float, float]]:
    """매개변수 하트 곡선. 폴리곤 하나로 그려서 외곽선이 한 번만 나온다."""
    pts = []
    for k in range(40):
        t = 2 * math.pi * k / 40
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + x * r / 17, cy - y * r / 17))
    return pts


def _star_polygon(cx: float, cy: float, r: float, inner: float = 0.30,
                  points: int = 4) -> List[Tuple[float, float]]:
    """반짝임. 4갈래 별."""
    pts = []
    for k in range(points * 2):
        ang = math.pi * k / points - math.pi / 2
        rad = r if k % 2 == 0 else r * inner
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    return pts


def _drop_polygon(cx: float, cy: float, r: float) -> List[Tuple[float, float]]:
    """물방울. 위는 뾰족하고 아래는 둥글다."""
    pts = [(cx, cy - r * 1.6)]
    for k in range(1, 24):
        t = math.pi * k / 23
        pts.append((cx + r * math.sin(t) * (0.25 + 0.75 * math.sin(t) ** 0.5),
                    cy + r * (1 - math.cos(t)) - r * 0.35))
    for k in range(23, 0, -1):
        t = math.pi * k / 23
        pts.append((cx - r * math.sin(t) * (0.25 + 0.75 * math.sin(t) ** 0.5),
                    cy + r * (1 - math.cos(t)) - r * 0.35))
    return pts


def _note_shape(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float,
                fill, outline, w: int) -> None:
    """음표. 머리 + 기둥 + 깃발."""
    d.ellipse([cx - r * 0.62, cy - r * 0.45, cx + r * 0.32, cy + r * 0.45],
              fill=fill, outline=outline, width=w)
    d.line([(cx + r * 0.30, cy + r * 0.18), (cx + r * 0.30, cy - r * 1.45)],
           fill=outline, width=max(3, w + 1))
    d.line([(cx + r * 0.30, cy - r * 1.45), (cx + r * 0.95, cy - r * 1.05)],
           fill=outline, width=max(3, w + 1))


SAFE = 3          # 경계에 딱 붙지 않도록 두는 여유 (px)


def _rise_room(ay: float, half: float, floor: float = 0.0) -> float:
    """
    위로 떠오르는 효과(하트·음표)가 캔버스 밖으로 나가지 않을 상승 높이.

    half 는 그 효과가 중심에서 위로 얼마나 뻗는지(반지름 + 기둥 등)다.
    이걸 빼지 않으면 '중심'은 캔버스 안인데 '그림'은 잘린다.
    floor 를 크게 주면 안전치를 덮어써서 오히려 잘리므로 기본값은 0이다.
    """
    return max(floor, min(ay - half - SAFE, ay))


def _clamp_x(x: float, half: float, cw: int) -> float:
    """좌우로도 잘리지 않게 중심 x 를 캔버스 안으로 밀어 넣는다."""
    return max(half + SAFE, min(cw - half - SAFE, x))


def _fade(p: float) -> float:
    """0에서 시작해 중간에 최대, 1에서 다시 0. 루프 이음새를 없애는 핵심."""
    return math.sin(math.pi * max(0.0, min(1.0, p)))


def _a(color: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int, int]:
    return (color[0], color[1], color[2], int(max(0, min(255, alpha * 255))))


# ================================================================== 효과들
# 각 효과는 프레임 수 n을 받아 n장의 투명 오버레이를 돌려준다.


def _particles(n: int, count: int, canvas, draw) -> List[Image.Image]:
    """
    효과 6종이 공유하는 뼈대. n프레임 × count개 입자를 찍어낸다.

    입자 k는 자기 위상 p = (i/n + k/count) % 1 을 따라 0 -> 1 로 진행하고,
    알파는 _fade() 로 양 끝에서 0이 된다. 보이지 않는 상태에서 한 바퀴가
    닫히므로 반복 재생해도 이음새가 생기지 않는다.

    draw(d, k, p, al) 가 입자 하나를 그린다.
    """
    out = []
    for i in range(n):
        layer = Image.new("RGBA", canvas, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for k in range(count):
            p = (i / n + k / count) % 1.0
            al = _fade(p)
            if al >= 0.02:
                draw(d, k, p, al)
        out.append(layer)
    return out


def fx_heart(n: int = 12, count: int = 3, anchors=None,
             canvas=(CANVAS, CANVAS)) -> List[Image.Image]:
    """
    설렘. 하트가 캐릭터 '양옆'에서 떠올라 흔들리며 사라진다.

    가슴 높이에서 똑바로 올리면 얼굴 한가운데를 가로지른다.
    (머리만 있는 캐릭터면 가슴 좌표 자체가 얼굴 안이다)
    그래서 실루엣 바깥에서 띄운다.
    """
    PINK = (255, 96, 128)
    x0, y0, x1, y1 = (anchors or default_anchors(canvas))["_bbox"]
    sc = _scale_of(anchors, canvas)
    pad = 18 * sc
    r_max = 20 * sc
    drift_max = (10 + 6 * (count // 2)) * sc
    start_y = y0 + (y1 - y0) * 0.55
    rise = _rise_room(start_y, r_max * 1.2, floor=50.0)
    sides = [_clamp_x(x0 - pad, r_max + drift_max, canvas[0]),
             _clamp_x(x1 + pad, r_max + drift_max, canvas[0])]

    def draw(d, k, p, al):
        base_x = sides[k % 2]
        drift = (10 + 6 * (k // 2)) * sc * math.sin(p * 4 + k)
        cx = base_x + drift * (1 if k % 2 else -1)
        d.polygon(_heart_polygon(cx, start_y - rise * p, (12 + 8 * p) * sc),
                  fill=_a(PINK, al), outline=_a(LINE[:3], al),
                  width=max(2, int(3 * sc)))

    return _particles(n, count, canvas, draw)


def fx_sparkle(n: int = 12, count: int = 4, anchors=None,
               canvas=(CANVAS, CANVAS)) -> List[Image.Image]:
    """
    최고 / 뿌듯. 반짝임이 캐릭터 '바깥쪽'에서 번쩍인다.
    (얼굴 위에 찍히지 않도록 bbox 바깥에 배치한다)
    """
    GOLD = (255, 214, 64)
    x0, y0, x1, y1 = (anchors or default_anchors(canvas))["_bbox"]
    sc = _scale_of(anchors, canvas)
    pad = 14 * sc
    r_max = 25 * sc
    cw, chh = canvas
    raw = [(x0 - pad, y0 + (y1 - y0) * 0.18), (x1 + pad, y0 + (y1 - y0) * 0.10),
           (x0 + (x1 - x0) * 0.22, y0 - pad), (x0 + (x1 - x0) * 0.78, y0 - pad * 0.6)]
    spots = [(_clamp_x(sx, r_max, cw),
              max(r_max + SAFE, min(chh - r_max - SAFE, sy)))
             for sx, sy in raw]

    def draw(d, k, p, al):
        cx, cy = spots[k % len(spots)]
        d.polygon(_star_polygon(cx, cy, (9 + 16 * al) * sc),
                  fill=_a(GOLD, al), outline=_a(LINE[:3], al), width=2)

    return _particles(n, min(count, len(spots)), canvas, draw)


def fx_tear(n: int = 12, anchors=None, canvas=(CANVAS, CANVAS)) -> List[Image.Image]:
    """슬픔. 양쪽 눈에서 눈물이 흐른다."""
    BLUE = (120, 196, 255)
    eyes = (_pick(anchors, "left_eye", canvas), _pick(anchors, "right_eye", canvas))
    sc = _scale_of(anchors, canvas)

    def draw(d, k, p, al):
        ex, ey = eyes[k]
        d.polygon(_drop_polygon(ex, ey + 76 * sc * p, (9 + 3 * p) * sc),
                  fill=_a(BLUE, al ** 0.5), outline=_a(LINE[:3], al ** 0.5), width=2)

    return _particles(n, 2, canvas, draw)


def fx_sweat(n: int = 12, anchors=None, canvas=(CANVAS, CANVAS)) -> List[Image.Image]:
    """당황. 땀방울이 머리 양옆에서 튀어 날아간다."""
    BLUE = (150, 210, 255)
    srcs = ((_pick(anchors, "head_tr", canvas), 1),
            (_pick(anchors, "head_tl", canvas), -1))
    sc = _scale_of(anchors, canvas)

    cw, chh = canvas
    r_max = 12 * sc

    def draw(d, k, p, al):
        (sx, sy), sgn = srcs[k]
        cx = _clamp_x(sx + sgn * 44 * sc * p, r_max, cw)
        cy = max(r_max * 1.8, min(chh - r_max, sy - 26 * sc * p + 58 * sc * p * p))
        d.polygon(_drop_polygon(cx, cy, (10 - 2 * p) * sc),
                  fill=_a(BLUE, al), outline=_a(LINE[:3], al), width=2)

    return _particles(n, 2, canvas, draw)


def fx_anger(n: int = 12, anchors=None, canvas=(CANVAS, CANVAS)) -> List[Image.Image]:
    """화남. 김이 머리 위로 뿜어져 올라간다. (양옆 2곳 × 2덩이 = 입자 4개)"""
    GRAY = (228, 228, 236)
    srcs = (_pick(anchors, "head_tl", canvas), _pick(anchors, "head_tr", canvas))
    sc = _scale_of(anchors, canvas)

    cw, chh = canvas
    r_max = 28 * sc

    def draw(d, k, p, al):
        sx, sy = srcs[k % 2]
        r = (11 + 17 * p) * sc
        lift = min(54 * sc, _rise_room(sy, r_max, floor=10.0))
        cx = _clamp_x(sx + (8 if k % 2 else -8) * sc * p, r_max, cw)
        cy = sy - lift * p
        d.ellipse([cx - r, cy - r * 0.78, cx + r, cy + r * 0.78],
                  fill=_a(GRAY, al * 0.92), outline=_a(LINE[:3], al * 0.74), width=2)

    return _particles(n, 4, canvas, draw)


def fx_note(n: int = 12, count: int = 2, anchors=None,
            canvas=(CANVAS, CANVAS)) -> List[Image.Image]:
    """신남. 음표가 캐릭터 오른쪽 위 허공으로 떠오른다."""
    PURPLE = (160, 128, 255)
    ax, ay = _pick(anchors, "float_tr", canvas)
    sc = _scale_of(anchors, canvas) * 1.25                      # 작게 보여서 조금 키운다
    r_max = 19 * sc
    sway = 16 * sc
    # 음표는 머리 위로 기둥과 깃발이 더 올라간다 -> 반지름의 1.7배를 여유로 본다
    top_need = r_max * 1.7 + SAFE
    ay = max(top_need, ay)                       # 시작점부터 안으로
    rise = max(0.0, min(96.0 * sc, _rise_room(ay, r_max * 1.7)))
    ax = _clamp_x(ax, r_max + sway, canvas[0])

    def draw(d, k, p, al):
        _note_shape(d, ax + sway * math.sin(p * 4 + k * 2), ay - rise * p,
                    (15 + 4 * p) * sc, _a(PURPLE, al), _a(LINE[:3], al), 2)

    return _particles(n, count, canvas, draw)


def fx_text(text: str, n: int = 12, anchors=None, canvas=(CANVAS, CANVAS),
            size: int = 0, color=(255, 255, 255), bob: float = 10,
            place: str = "above") -> List[Image.Image]:
    """
    글자 효과. 'ㅋㅋㅋ', 'Zzz', '?', '!', '고마워' 등.

    AI에게 글자를 그리게 하면 획이 깨지지만, 폰트로 직접 쓰면 100% 정확하다.
    다크모드 대응 흰 테두리(stroke)도 여기서 정확히 넣는다.

    size=0 이면 '남은 여백에 들어가는 최대 크기'를 스스로 계산한다.
    place: "above"(머리 위) | "below"(발 아래)
    """
    cw, ch = canvas
    ax, ay = _pick(anchors, "above_head" if place == "above" else "below", canvas)

    # 여백에 맞는 글자 크기를 정한다.
    # 글자는 ay 를 '중심'으로 그려지므로 위아래로 절반씩 퍼진다.
    # 렌더 높이의 절반은 대략 size*0.62 (글자높이 + 외곽선).
    if size <= 0:
        room = ay if place == "above" else (ch - ay)
        size = int(max(20, min(76, (room - 8) / 0.62)))
    # 글자가 작아질 대로 작아져도 자리가 모자라면 중심을 안으로 민다.
    # 위아래로 까딱이는 폭(bob)과 10% 확대까지 더해서 잡아야 안 잘린다.
    half = size * 0.62 * 1.10 + abs(bob) + 4 + SAFE
    ay = max(half, min(ch - half, ay))

    font = load_font(size)
    stroke = max(3, size // 12)

    out = []
    for i in range(n):
        layer = Image.new("RGBA", canvas, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        p = i / n
        s = 1 + 0.10 * math.sin(2 * math.pi * p)
        dy = bob * math.sin(2 * math.pi * p)
        d.text((ax, ay + dy), text, font=font, anchor="mm",
               fill=color + (255,), stroke_width=stroke, stroke_fill=LINE)
        if abs(s - 1) > 1e-6:
            layer = layer.resize((int(cw * s), int(ch * s)), Image.LANCZOS)
            sheet = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
            sheet.paste(layer, ((cw - layer.size[0]) // 2, (ch - layer.size[1]) // 2))
            layer = sheet
        out.append(layer)
    return out


EFFECTS: Dict[str, Callable[..., List[Image.Image]]] = {
    "heart": fx_heart,
    "sparkle": fx_sparkle,
    "tear": fx_tear,
    "sweat": fx_sweat,
    "anger": fx_anger,
    "note": fx_note,
}


# ---------------------------------------------------------------- 합성

def composite(frames: List[Image.Image],
              overlays: List[Image.Image]) -> List[Image.Image]:
    """
    모션이 끝난 프레임 위에 효과를 얹는다.
    효과는 캐릭터와 함께 눌리거나 기울어지면 안 되므로 변형 '이후'에 합성한다.
    프레임 수가 다르면 효과 쪽을 비율로 맞춰 집는다.
    """
    out = []
    m = len(overlays)
    for i, f in enumerate(frames):
        ov = overlays[int(i * m / len(frames))]
        out.append(Image.alpha_composite(f.convert("RGBA"), ov))
    return out
