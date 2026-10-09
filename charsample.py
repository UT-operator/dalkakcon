# -*- coding: utf-8 -*-
"""테스트용 합성 캐릭터 (AI가 그릴 자리의 대역)와 모션 계획표."""

from PIL import Image, ImageDraw

CANVAS = 360

# 캐릭터 기준점 (효과를 어디에 붙일지)
ABOVE_HEAD = (180, 104)
LEFT_EYE = (138, 252)
RIGHT_EYE = (222, 252)
HEAD_RIGHT = (296, 150)
HEAD_LEFT = (64, 150)
BELLY = (180, 300)


def draw_character() -> Image.Image:
    """
    위쪽에 점프 여유를 두고 배치한다.
    평평한 색 + 굵은 외곽선이라 실제 이모티콘 화풍의 변형 특성을 그대로 본다.
    """
    img = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    BODY, LINE = (255, 206, 110, 255), (82, 54, 28, 255)
    DARK, BLUSH = (60, 42, 26, 255), (255, 142, 148, 255)

    for cx in (118, 242):                                   # 귀
        d.polygon([(cx - 26, 182), (cx + 26, 182), (cx, 124)],
                  fill=BODY, outline=LINE, width=7)
    d.ellipse([70, 160, 290, 334], fill=BODY, outline=LINE, width=8)   # 몸통
    for cx in (138, 222):                                   # 눈
        d.ellipse([cx - 15, 218, cx + 15, 258], fill=DARK)
        d.ellipse([cx - 4, 226, cx + 6, 238], fill=(255, 255, 255, 255))
    for cx in (104, 256):                                   # 볼
        d.ellipse([cx - 19, 266, cx + 19, 288], fill=BLUSH)
    d.arc([164, 262, 196, 292], start=0, end=180, fill=LINE, width=7)  # 입
    d.arc([276, 258, 330, 320], start=250, end=40, fill=LINE, width=9) # 꼬리
    return img


# (모션, 기준 재생시간 ms)  — 프레임 수는 각 모션이 스스로 정한다
PLAN = [
    ("breathe",     90),
    ("bounce",      70),
    ("nod",         80),
    ("rock",        90),
    ("shake",       55),
    ("float",      100),
    ("zoom_punch",  70),
    ("pop",         70),
    ("drop",        65),
    ("jump",        62),
    ("slide",       80),
    ("spin_wobble", 85),
]
