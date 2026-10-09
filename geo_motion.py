# -*- coding: utf-8 -*-
"""
기하 변형 모션 모듈

AI가 그린 프레임에 '위치 / 기울기 / 눌림·늘어남'을 수식으로 입혀 움직임을 만든다.
신경망 보간을 쓰지 않으므로 선이 깨지지 않고 알파가 안전하다.

핵심 3가지
  (1) 앵커(anchor) : 무엇을 기준으로 변형하는가.
                     통통 튀기는 '발바닥'이 기준이어야 바닥에 붙어 보인다.
  (2) 이징(easing) : 어떤 속도 곡선으로 변하는가. 선형은 로봇처럼 보인다.
  (3) 홀드(hold)   : '멈춤'을 프레임 복제가 아니라 '그 프레임의 재생시간'으로 만든다.
                     카카오 WebP Animator가 프레임별 0.05~2.0초를 지원하므로,
                     정점에서 멈칫하는 연출에 프레임을 낭비하지 않는다.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

from PIL import Image

# ------------------------------------------------------------------ 이징 곡선


def linear(t: float) -> float:
    return t


def ease_in(t: float) -> float:
    """느리게 시작해 가속. 떨어지는 것."""
    return t * t


def ease_out(t: float) -> float:
    """빠르게 시작해 감속. 던져진 것이 멈출 때."""
    return 1 - (1 - t) ** 2


def ease_in_out(t: float) -> float:
    """느리게-빠르게-느리게. 가장 무난한 기본값."""
    return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2


def back_out(t: float, s: float = 1.70158) -> float:
    """목표를 살짝 지나쳤다 돌아옴(오버슈트). '뿅' 등장."""
    return 1 + (s + 1) * (t - 1) ** 3 + s * (t - 1) ** 2


def bounce_out(t: float) -> float:
    """바닥에 닿고 통통 튕김."""
    n, d = 7.5625, 2.75
    if t < 1 / d:
        return n * t * t
    if t < 2 / d:
        t -= 1.5 / d
        return n * t * t + 0.75
    if t < 2.5 / d:
        t -= 2.25 / d
        return n * t * t + 0.9375
    t -= 2.625 / d
    return n * t * t + 0.984375


def elastic_out(t: float) -> float:
    """고무줄처럼 진동하며 멈춤."""
    if t in (0.0, 1.0):
        return t
    c = (2 * math.pi) / 3
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * c) + 1


EASINGS: Dict[str, Callable[[float], float]] = {
    "linear": linear, "ease_in": ease_in, "ease_out": ease_out,
    "ease_in_out": ease_in_out, "back_out": back_out,
    "bounce_out": bounce_out, "elastic_out": elastic_out,
}


# ------------------------------------------------------- 변형 1장 적용 (핵심)

def transform_frame(
    img: Image.Image,
    dx: float = 0.0, dy: float = 0.0,
    sx: float = 1.0, sy: float = 1.0,
    rot: float = 0.0,
    anchor: Tuple[float, float] = (0.5, 1.0),
    **_ignored,
) -> Image.Image:
    """
    이동/확대축소/회전을 한 번에 적용한다. 캔버스 크기는 유지된다.
    Pillow의 AFFINE은 '출력 → 입력' 역방향 매핑을 요구하므로 역행렬을 직접 넣는다.

    anchor : 변형 기준점 (0~1 비율). (0.5, 1.0) = 하단 중앙 = 발바닥
    """
    img = img.convert("RGBA")
    w, h = img.size
    ax, ay = anchor[0] * w, anchor[1] * h

    th = math.radians(rot)
    cos, sin = math.cos(th), math.sin(th)
    sx, sy = max(sx, 1e-6), max(sy, 1e-6)
    tx, ty = ax + dx, ay + dy

    a = cos / sx
    b = sin / sx
    c = (-cos * tx - sin * ty) / sx + ax
    d = -sin / sy
    e = cos / sy
    f = (sin * tx - cos * ty) / sy + ay

    return img.transform((w, h), Image.AFFINE, (a, b, c, d, e, f),
                         resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))


def _squash(q: float) -> Tuple[float, float]:
    """부피 보존 눌림. q>1 세로로 늘어남, q<1 납작. 가로를 반대로 보정한다."""
    return (1.0 / math.sqrt(q), q)


def _kf(keys: List[Tuple[float, float, float]], amp: float) -> List[dict]:
    """(dy비율, 눌림, 재생시간배수) 목록을 변형 파라미터로 바꾼다."""
    out = []
    for r, q, hold in keys:
        sx, sy = _squash(q)
        out.append(dict(dy=amp * r, sx=sx, sy=sy, anchor=(0.5, 1.0), hold=hold))
    return out


# ============================================================ 순환형 모션


def _sq(q: float, **kw) -> dict:
    """눌림 q 를 가로세로 배율로 바꿔 변형 파라미터 dict 로 만든다."""
    sx, sy = _squash(q)
    kw.setdefault("anchor", (0.5, 1.0))
    return dict(sx=sx, sy=sy, **kw)


def _cyc(n: int, fn) -> List[dict]:
    """
    순환형 모션 공통 틀. fn(t) 가 t=0~1 구간의 변형 파라미터를 돌려준다.
    마지막 프레임(t=1)을 '포함하지 않아' 한 바퀴가 그대로 이어진다.
    """
    return [fn(i / n) for i in range(n)]


def m_breathe(n: int = 12, pct: float = 0.045) -> List[dict]:
    """숨쉬기. 미세한 상하 신축. 정지 이모티콘에 생명감만 넣는 용도."""
    return _cyc(n, lambda t: _sq(1 + pct * math.sin(2 * math.pi * t)))


def m_bounce(n: int = 12, amp: float = 40) -> List[dict]:
    """
    통통 튀기. 가장 많이 쓰이는 모션.
    상승 38% / 하강 38% / 바닥 접촉 24% 로 나눠, 마지막 프레임이 바닥에 머물게 한다.
    (예전엔 마지막 프레임이 공중에 떠 있어서 루프 이음새가 튀었다)
    """
    def f(t):
        if t < 0.38:
            h = ease_out(t / 0.38)
        elif t < 0.76:
            h = 1 - ease_in((t - 0.38) / 0.38)
        else:
            # 바닥 접촉: 빠르게 눌렸다 천천히 복원해 루프 지점에서 중립이 되게
            p = (t - 0.76) / 0.24
            return _sq(1 - 0.13 * math.sin(math.pi * p ** 0.55), dy=0)
        return _sq(1 + 0.10 * h, dy=-amp * h)
    return _cyc(n, f)


def m_nod(n: int = 12, amp: float = 20, times: int = 2) -> List[dict]:
    """
    끄덕임. 한 바퀴에 2번 끄덕이고, 아래로 빠르게 위로 천천히 — 실제 리듬이다.
    (예전엔 sin 의 음수 구간을 0으로 잘라서 절반이 완전히 멈춰 있었다)
    """
    def f(t):
        p = (t * times) % 1.0
        d = ease_in(p / 0.38) if p < 0.38 else 1 - ease_out((p - 0.38) / 0.62)
        return _sq(1 - 0.075 * d, dy=amp * d)
    return _cyc(n, f)


def m_rock(n: int = 12, deg: float = 11) -> List[dict]:
    """갸우뚱 좌우. 발바닥을 축으로 기울이고 무게중심을 반대로 살짝 민다."""
    def f(t):
        x = math.sin(2 * math.pi * t)
        return dict(rot=deg * x, dx=-2.2 * x, anchor=(0.5, 1.0))
    return _cyc(n, f)


def m_shake(n: int = 12, amp: float = 9, cycles: int = 3) -> List[dict]:
    """부들부들 떨기. 좌우 진동 + 역위상 미세 회전으로 떨림을 강조한다."""
    def f(t):
        x = math.sin(2 * math.pi * cycles * t)
        return dict(dx=amp * x, rot=-2.5 * x, anchor=(0.5, 1.0))
    return _cyc(n, f)


def m_float(n: int = 16, amp: float = 14) -> List[dict]:
    """
    부유. 상하 1주기 + 좌우 2주기로 8자를 그린다.
    (예전엔 좌우가 반주기만 돌아서 루프 지점에서 방향이 툭 바뀌었다)
    """
    def f(t):
        a = 2 * math.pi * t
        return dict(dy=-amp * math.sin(a), dx=amp * 0.55 * math.sin(2 * a),
                    rot=3.0 * math.cos(a), anchor=(0.5, 0.5))
    return _cyc(n, f)


def m_slide(n: int = 16, dist: float = 40) -> List[dict]:
    """
    좌우 미끄러짐. 기울기를 '위치'가 아니라 '속도'(cos)에 비례시킨다.
    (예전엔 가장 멀리 간 지점에서 가장 기울어 있었다. 거긴 멈추는 곳인데.)
    """
    def f(t):
        a = 2 * math.pi * t
        return dict(dx=dist * math.sin(a), rot=-8 * math.cos(a), anchor=(0.5, 1.0))
    return _cyc(n, f)


def m_spin_wobble(n: int = 16, deg: float = 26) -> List[dict]:
    """크게 갸우뚱. 어리둥절, 멍때림. 들림(dy)을 회전과 같은 주기로 맞췄다."""
    def f(t):
        a = 2 * math.pi * t
        return dict(rot=deg * math.sin(a), dy=-7 * (1 - math.cos(a)) / 2,
                    anchor=(0.5, 1.0))
    return _cyc(n, f)


# ====================================================== 키프레임형 모션
# 동작에 뚜렷한 단계가 있는 것들. '멈춤'은 프레임 복제 대신 hold로 처리한다.


def m_jump(amp: float = 95, **_) -> List[dict]:
    """
    점프. 뜸들임 → 발사 → 체공 → 낙하 → 착지 눌림 → 탄성 복귀.
    정점(hold 2.2)과 안정(hold 1.6)을 재생시간으로 처리해 프레임을 아꼈다.
    """
    return _kf([
        ( 0.00, 1.00, 1.0),   # 중립
        ( 0.00, 0.84, 1.5),   # 웅크림 — 뜸들임
        ( 0.00, 0.93, 1.0),
        (-0.34, 1.27, 1.0),   # 발사 — 길게 늘어남
        (-0.74, 1.13, 1.0),
        (-0.95, 1.03, 1.0),
        (-1.00, 1.00, 2.2),   # 정점 — 멈칫
        (-0.93, 1.05, 1.0),
        (-0.63, 1.15, 1.0),
        (-0.22, 1.21, 1.0),   # 낙하 가속
        ( 0.00, 0.76, 1.0),   # 착지 눌림
        ( 0.00, 1.09, 1.0),   # 반동
        ( 0.00, 0.95, 1.0),
        ( 0.00, 1.00, 1.6),   # 안정
    ], amp)


def m_drop(high: float = 110, **_) -> List[dict]:
    """위에서 뚝 떨어져 튕긴 뒤, 다시 위로 사라지며 루프로 이어진다."""
    return _kf([
        (-1.00, 1.10, 1.0),
        (-0.56, 1.17, 1.0),
        (-0.13, 1.23, 1.0),
        ( 0.00, 0.73, 1.0),   # 착지 눌림
        ( 0.00, 1.13, 1.0),   # 반동
        (-0.17, 1.04, 1.0),
        ( 0.00, 0.89, 1.0),
        ( 0.00, 1.00, 2.6),   # 안정 — 멈춤
        (-0.34, 1.07, 1.0),   # 다시 떠오름
        (-0.78, 1.11, 1.0),
    ], high)


def m_pop(**_) -> List[dict]:
    """뿅 등장 → 유지 → 쏙 사라짐. 순환하면 '뿅…쏙'이 반복된다."""
    out = []
    for s, hold in [(0.16, 1.0), (0.72, 1.0), (1.14, 1.0),
                    (1.00, 3.2), (0.74, 1.0), (0.32, 1.0)]:
        out.append(dict(sx=s, sy=s, anchor=(0.5, 0.95), hold=hold))
    return out


def m_zoom_punch(peak: float = 0.18, **_) -> List[dict]:
    """강조. 확 커졌다 빠르게 복귀하고 숨 고른다. '좋아!', '최고!'."""
    out = []
    for p, hold in [(0.00, 1.0), (0.72, 1.0), (1.00, 1.2), (0.78, 1.0),
                    (0.42, 1.0), (0.16, 1.0), (0.03, 2.6)]:
        s = 1 + peak * p
        out.append(dict(sx=s, sy=s, anchor=(0.5, 0.90), hold=hold))
    return out


MOTIONS: Dict[str, Callable[..., List[dict]]] = {
    "breathe": m_breathe, "bounce": m_bounce, "nod": m_nod, "rock": m_rock,
    "shake": m_shake, "float": m_float, "slide": m_slide,
    "spin_wobble": m_spin_wobble,
    "jump": m_jump, "drop": m_drop, "pop": m_pop, "zoom_punch": m_zoom_punch,
}

# 순환형(루프 전제) / 키프레임형(단계가 있는 동작)
CYCLIC = {"breathe", "bounce", "nod", "rock", "shake", "float",
          "slide", "spin_wobble"}


def apply_motion(base: Image.Image, name: str, base_ms: int = 80,
                 **kwargs) -> Tuple[List[Image.Image], List[int]]:
    """
    그림 1장 + 모션 이름 → (프레임 리스트, 프레임별 재생시간 ms)

    재생시간은 카카오 규격(0.05~2.0초)으로 잘라낸다.
    """
    params = MOTIONS[name](**kwargs)
    frames = [transform_frame(base, **p) for p in params]
    durations = [int(min(2000, max(50, base_ms * p.get("hold", 1.0))))
                 for p in params]
    return frames, durations
