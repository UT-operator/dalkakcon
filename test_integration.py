# -*- coding: utf-8 -*-
"""
딸깍콘 (다) 통합 테스트 — 실제 OpenAI 이미지 생성 + 기하 변형 + 효과 합성

이 테스트로 확인하는 것
  1. 이 API 키로 어떤 이미지 모델이 열려 있는가        -> 모델 점검 (과금 없음)
  2. 업로드 이미지에서 캐릭터가 유지되는가              -> 기준 원화
  3. 종별 포즈가 '같은 캐릭터'로 나오는가               -> 일관성
  4. 투명 배경이 깨끗한가                              -> 투명%, 메운 구멍
  5. Pillow 효과가 AI 화풍과 어울리는가                -> 효과 침범률 + 눈으로
  6. 플랫폼 규격을 만족하는가                          -> 프레임 수, 용량

API 호출: 기준 원화 1회 + 종 수 (기본 3종 = 4회, 약 $0.04)

사용법
  python test_integration.py                 # 합성 캐릭터 레퍼런스 (준비물 없음)
  python test_integration.py 내캐릭터.png      # 내 이미지 레퍼런스
  python test_integration.py --no-ref         # 설명만으로 생성
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join("/content" if os.path.isdir("/content") else HERE, "integration_out")
os.makedirs(OUT, exist_ok=True)
os.environ["DDALKAK_OUT_DIR"] = OUT      # emoticon_tools 가 import 때 읽는다
sys.path.insert(0, HERE)

import numpy as np
from PIL import Image

import emoticon_tools as et
import geo_effects as fx
import geo_motion as gm

# 다른 셀이 먼저 import 했을 수도 있으니 경로를 명시적으로 덮어쓴다
et.OUT_DIR = OUT
for _n, _f in [("BASE_PATH", "generated_base.png"), ("GIF_PATH", "generated_emoticon.gif"),
               ("WEBP_PATH", "generated_emoticon.webp")]:
    setattr(et, _n, os.path.join(OUT, _f))
et.FRAME_PATH_FMT = os.path.join(OUT, "generated_frame_{:02d}.png")

CANVAS = (et.CANVAS, et.CANVAS)
RESERVE_TOP, MARGIN = 0.18, 0.06     # 효과·글자를 올릴 자리 (안 비우면 얼굴을 덮는다)
MODELS = ["gpt-image-1.5", "gpt-image-2", "gpt-image-1-mini", "gpt-image-1"]

CHARACTER = ("a round chubby cream-yellow cat character with small triangular ears, "
             "big dark round eyes, pink round cheeks, a tiny mouth and a thin curled tail")

# (종, 대표 포즈 영문 묘사, 모션, 효과, 효과인자, 기준 ms)
SPECIES = [
    ("좋아", "giving a confident thumbs up with one paw raised, bright proud smile",
     "zoom_punch", "sparkle", {}, 75),
    ("고마워", "bowing head slightly with both paws together in front of chest, "
     "warm grateful smile, eyes closed softly", "bounce", "text", {"text": "고마워"}, 75),
    ("신남", "cheering with both arms raised high, wide open happy mouth, excited",
     "jump", "note", {}, 62),
]


def step(title):
    print(f"\n{'=' * 66}\n  {title}\n{'=' * 66}")


def die(msg):
    print(f"\n{msg}")
    sys.exit(1)


# ------------------------------------------------------------- 1. 환경 점검
step("1. 환경 점검")
try:
    import openai
except ImportError:
    die("openai 패키지가 없습니다 -> pip install openai")

if not os.environ.get("OPENAI_API_KEY"):
    die("OPENAI_API_KEY 가 없습니다 -> 2번 셀을 실행하세요")

print(f"  openai {openai.__version__} / 한글 폰트 "
      f"{'OK' if fx.font_available() else '없음 (apt-get install -y fonts-nanum)'}"
      f"\n  출력 폴더: {OUT}")

# ------------------------------------------------------------- 2. 모델 점검
step("2. 모델 점검 (과금 없음)")
client = openai.OpenAI()
chosen = None
for name in MODELS:
    try:
        client.models.retrieve(name)
        print(f"  {name:<20} 사용 가능" + ("   <-- 선택" if chosen is None else ""))
        chosen = chosen or name
    except Exception as exc:
        print(f"  {name:<20} 불가 ({type(exc).__name__})")
if chosen is None:
    die("쓸 수 있는 이미지 모델이 없습니다. 기관에 열려 있는 모델명을 확인하세요.")
et.IMAGE_MODEL = chosen

# --------------------------------------------------------- 3. 레퍼런스 선택
step("3. 레퍼런스 이미지")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if "--no-ref" in sys.argv:
    et.set_reference_image(None)
    print("  레퍼런스 없음 — 설명만으로 캐릭터를 새로 생성합니다")
elif args:
    if not os.path.exists(args[0]):
        die(f"파일을 찾을 수 없습니다: {args[0]}")
    data = open(args[0], "rb").read()
    et.set_reference_image(data, "image/png" if args[0].lower().endswith(".png")
                           else "image/jpeg")
    print(f"  업로드 이미지: {args[0]} ({len(data)/1024:.0f}KB)")
else:
    from charsample import draw_character
    p = os.path.join(OUT, "00_reference.png")
    draw_character().save(p)
    et.set_reference_image(open(p, "rb").read(), "image/png")
    print(f"  합성 캐릭터를 레퍼런스로 사용 ({p})")

# ------------------------------------------------------- 4. 기준 원화 생성
step(f"4. 기준 원화 생성 ({chosen}, API 1회)")
t0 = time.time()
try:
    base = et.build_base_sheet(CHARACTER, quality="low")
except Exception as exc:
    die(f"실패: {type(exc).__name__}: {exc}")
print(f"  완료 {time.time()-t0:.1f}초, {base.size}, 저장 {et.BASE_PATH}")

# -------------------------------------------------- 5. 종별 대표 포즈 생성
step(f"5. 종별 대표 포즈 생성 (API {len(SPECIES)}회, 병렬)")
base_bytes = et._to_png_bytes(base)
t0 = time.time()


def make_pose(item):
    label, pose = item[0], item[1]
    try:
        return label, et.generate_pose(base_bytes, pose, name=f"base_{label}"), None
    except Exception as exc:
        return label, None, f"{type(exc).__name__}: {exc}"


poses = {}
with ThreadPoolExecutor(max_workers=len(SPECIES)) as pool:
    for label, img, err in pool.map(make_pose, SPECIES):
        print(f"  {label:<8} " + ("생성 완료" if img is not None else f"실패 — {err}"))
        if img is not None:
            poses[label] = img
print(f"  {len(poses)}/{len(SPECIES)}종, {time.time()-t0:.1f}초 (병렬)")
if not poses:
    die("모든 포즈 생성에 실패했습니다.")

# ------------------------------------------- 6. 투명 배경 품질 + 배치
step("6. 투명 배경 품질 + 배치")
cleaned = {}
print(f"  {'종':<8}{'생성직후 투명%':>14}{'외곽 반투명':>12}")
for label, img in poses.items():
    a = np.asarray(img.getchannel("A"))
    cleaned[label] = et.stabilize_alpha(img)
    print(f"  {label:<8}{(a < 128).mean()*100:>13.1f}%"
          f"{int(((a > 20) & (a < 235)).sum()):>12}")
print("  * 투명% 가 0에 가까우면 배경이 안 날아간 것 (프롬프트 재조정 필요)")

placed = et.fit_art_set(cleaned, canvas=CANVAS, margin=MARGIN, reserve_top=RESERVE_TOP)
print(f"\n  위쪽 {RESERVE_TOP:.0%} 를 효과·글자용으로 비움")
print(f"  {'종':<8}{'배치된 bbox':>24}{'위 여백':>8}{'메운 구멍':>10}")

art, anchors = {}, {}
for label, (img, box) in placed.items():
    before = int((np.asarray(img.getchannel("A")) < 128).sum())
    filled = et.fill_interior_holes(img)
    after = int((np.asarray(filled.getchannel("A")) < 128).sum())
    art[label] = filled
    anchors[label] = fx.anchors_from_bbox(box, CANVAS)
    filled.save(os.path.join(OUT, f"pose_{label}.png"))
    print(f"  {label:<8}{str(box):>24}{box[1]:>7}px{before-after:>10}")
print("  * 위 여백이 충분해야 글자가 얼굴을 안 덮는다")

# ------------------------------------------- 7. 모션 + 효과 + 저장
step("7. 모션 + 효과 적용 (API 호출 없음)")
print(f"  {'종':<8}{'모션':<12}{'효과':<9}{'프레임':>6}{'효과침범':>9}{'용량':>10}  판정")

for label, _pose, motion, effect, kw, ms in SPECIES:
    if label not in art:
        continue
    frames, durs = gm.apply_motion(art[label], motion, base_ms=ms)
    n = len(frames)

    # 효과 기준점은 '실제 배치된 캐릭터'에서 계산한다 (고정 좌표 금지)
    maker = fx.fx_text if effect == "text" else fx.EFFECTS[effect]
    ov = maker(n=n, anchors=anchors[label], canvas=CANVAS, **kw)

    # 효과가 캐릭터 실루엣을 덮은 비율 — 낮을수록 깔끔하다
    ch = np.asarray(art[label].getchannel("A")) > 128
    om = np.asarray(ov[n // 2].getchannel("A")) > 128
    intrude = (ch & om).sum() / max(1, om.sum()) * 100

    # 변환은 노이즈를 만들지 않으므로 프레임마다 stabilize_alpha 를 돌리지 않는다
    # (측정: 14프레임 191ms 낭비 + 반짝임 끝과 글자 획이 깎이는 부작용)
    merged = fx.composite(frames, ov)

    size, note = et.shrink_until_under_limit(
        merged, os.path.join(OUT, f"{label}.gif"), durs, limit=et.SAFE_MAX_BYTES)

    # 카카오 제출용: 번호 붙인 PNG 프레임 (WebP Animator 에 넣을 것)
    fdir = os.path.join(OUT, f"frames_{label}")
    os.makedirs(fdir, exist_ok=True)
    for i, f in enumerate(merged, 1):
        f.save(os.path.join(fdir, f"{i:02d}.png"))

    bad = [x for x in (f"프레임{n}" if n > 24 else "",
                       f"용량{size//1024}KB" if size > et.SAFE_MAX_BYTES else "",
                       note) if x]
    print(f"  {label:<8}{motion:<12}{effect:<9}{n:>6}{intrude:>8.1f}%"
          f"{size/1024:>9.1f}KB  {', '.join(bad) or 'OK'}")

# ------------------------------------------------------------- 8. 결과
step("8. 결과")
print(f"""  생성 폴더: {OUT}

  확인해 주세요
   (1) 00_reference.png / generated_base.png  캐릭터가 유지됐는지
   (2) pose_*.png                             3종이 '같은 캐릭터'인지
   (3) *.gif                                  효과가 AI 화풍과 어울리는지
   (4) frames_*/01.png~                       WebP Animator 에 넣을 프레임

  총 API 호출 {1 + len(poses)}회 (약 ${(1 + len(poses)) * 0.009:.3f})""")
