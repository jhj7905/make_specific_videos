"""고객 소재(사진/영상) 정규화.

완성도의 80% 가 여기서 결정된다. 고객 사진은 EXIF 회전, 저해상도,
가로/세로 혼재, 색감 제각각인 상태로 들어온다.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from engine import ffmpeg, face, atomic
from engine.text import cache_key

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def kind_of(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    raise ValueError(f"지원하지 않는 소재 형식: {path}")


# ── 스마트 크롭 ───────────────────────────────────────────────────────────
def _energy_map(img: Image.Image, grid: int = 64) -> np.ndarray:
    """엣지 에너지 = 피사체가 있을 확률. (얼굴검출을 붙일 자리)"""
    small = img.convert("L").resize((grid, grid))
    a = np.asarray(small, dtype=np.float32)
    gx = np.abs(np.diff(a, axis=1, prepend=a[:, :1]))
    gy = np.abs(np.diff(a, axis=0, prepend=a[:1, :]))
    return gx + gy


def _choose_headroom(default: float, face_h: float,
                     avoid: tuple) -> float:
    """얼굴이 텍스트 띠 아래로 들어가지 않는 headroom 을 고른다.

    템플릿은 텍스트를 고정 좌표에 두고, 얼굴 크롭은 얼굴을 화면 위쪽
    40% 에 둔다. 둘이 구조적으로 같은 자리를 노리기 때문에 인물 사진에서
    글자가 얼굴을 덮는 사고가 난다 (상용 템플릿이 절대 하지 않는 실수다).
    겹침을 비용으로 두고 기본값에서 가장 덜 벗어나는 지점을 찾는다.
    """
    if not avoid:
        return default
    best, best_cost = default, None
    for hr in np.linspace(0.16, 0.74, 59):
        lo, hi = hr - face_h / 2, hr + face_h / 2
        overlap = sum(max(0.0, min(hi, b1) - max(lo, b0)) for b0, b1 in avoid)
        cost = overlap * 12.0 + abs(hr - default)
        if best_cost is None or cost < best_cost:
            best, best_cost = float(hr), cost
    return best


def _face_crop_box(img: Image.Image, faces, cw: int, ch: int,
                   headroom: float, avoid: tuple = ()) -> tuple[int, int, int, int]:
    """얼굴이 잘리지 않고, 인물이 '눈높이 위쪽 1/3' 에 오도록 창을 배치한다.

    avoid 가 주어지면 얼굴이 그 세로 띠(같은 씬의 텍스트) 아래로 들어가지
    않게 한다. 3:4 휴대폰 사진을 9:16 으로 자르면 가로만 잘리고 세로 여유가
    0 이라 위치를 바꿀 자유도가 아예 없다 — 그때는 창을 최대 24% 좁혀서
    여유를 만든다. 약간 타이트해지는 대신 글자가 얼굴을 덮지 않는다.
    """
    W, H = img.size
    biggest = max(f.size for f in faces)
    main = [f for f in faces if f.size >= biggest * 0.35]   # 배경의 작은 얼굴 무시

    ux1 = min(f.box[0] for f in main)
    uy1 = min(f.box[1] for f in main)
    ux2 = max(f.box[2] for f in main)
    uy2 = max(f.box[3] for f in main)
    fx, fy = (ux1 + ux2) / 2, (uy1 + uy2) / 2

    # 얼굴 묶음이 창보다 크면 담을 수 없다 → 가장 큰 얼굴 하나만 기준으로
    if (ux2 - ux1) > cw or (uy2 - uy1) > ch:
        f0 = max(main, key=lambda f: f.size)
        fx, fy = f0.center
        ux1, uy1, ux2, uy2 = f0.box

    def place(cw_: int, ch_: int, hr: float):
        x0 = fx - cw_ * 0.5
        y0 = fy - ch_ * hr
        pad = (uy2 - uy1) * 0.30          # 이마 여백
        x0 = max(min(x0, ux1 - 8), ux2 + 8 - cw_)
        y0 = max(min(y0, uy1 - pad), uy2 + pad - ch_)
        x0 = int(round(min(max(x0, 0), W - cw_)))
        y0 = int(round(min(max(y0, 0), H - ch_)))
        return x0, y0

    if avoid:
        face_px = (uy2 - uy1) * 1.30
        best = None
        for z in (1.0, 1.06, 1.12, 1.18, 1.24):
            cw_, ch_ = int(cw / z), int(ch / z)
            if cw_ < 16 or ch_ < 16:
                break
            hr = _choose_headroom(headroom, min(0.9, face_px / ch_), avoid)
            x0, y0 = place(cw_, ch_, hr)
            lo, hi = (uy1 - y0) / ch_, (uy2 - y0) / ch_
            ov = sum(max(0.0, min(hi, b1) - max(lo, b0)) for b0, b1 in avoid)
            if best is None or ov < best[0] - 1e-4:
                best = (ov, x0, y0, cw_, ch_)
            if ov <= 0.015:               # 충분히 피했으면 더 좁히지 않는다
                break
        _, x0, y0, cw, ch = best
        return (x0, y0, x0 + cw, y0 + ch)

    x0, y0 = place(cw, ch, headroom)
    return (x0, y0, x0 + cw, y0 + ch)


def smart_crop_box(img: Image.Image, ar: float, center_bias: float = 0.55,
                   faces=None, headroom: float = 0.40,
                   use_face: bool = True,
                   avoid: tuple = ()) -> tuple[int, int, int, int]:
    """목표 종횡비(ar = w/h)로 자를 최적 박스.

    1순위: 얼굴 검출 (SCRFD) — 인물 사진에서 얼굴이 잘리는 사고를 막는다
    2순위: 엣지 에너지 (모델이 없거나 얼굴이 없는 풍경/사물 사진)
    """
    W, H = img.size
    if W / H > ar:
        cw, ch = int(H * ar), H
    else:
        cw, ch = W, int(W / ar)
    if cw >= W and ch >= H:
        return (0, 0, W, H)

    if use_face:
        if faces is None:
            faces = face.detect(img)
        if faces:
            return _face_crop_box(img, faces, cw, ch, headroom, avoid)

    e = _energy_map(img)
    g = e.shape[0]
    best, best_score = None, -1.0
    steps = 24
    for i in range(steps + 1):
        if W - cw > 0:
            x = int((W - cw) * i / steps)
            y = (H - ch) // 2
        elif H - ch > 0:
            x = (W - cw) // 2
            y = int((H - ch) * i / steps)
        else:
            x = y = 0
        gx0, gy0 = int(x / W * g), int(y / H * g)
        gx1, gy1 = max(gx0 + 1, int((x + cw) / W * g)), max(gy0 + 1, int((y + ch) / H * g))
        score = float(e[gy0:gy1, gx0:gx1].mean())
        cxn = (x + cw / 2) / W
        cyn = (y + ch / 2) / H
        dist = ((cxn - 0.5) ** 2 + (cyn - 0.42) ** 2) ** 0.5
        score *= (1 - center_bias * dist)
        if score > best_score:
            best, best_score = (x, y, x + cw, y + ch), score
    return best


# ── 톤 보정 ───────────────────────────────────────────────────────────────
# 프리셋 필드
#   temp      색온도 (+ 따뜻 / - 차갑)
#   contrast  S커브 강도 (중간톤만 민다)
#   lift      검정 들어올림 (필름 느낌)
#   sat       채도 (휘도 보존)
#   shoulder  하이라이트 롤오프 강도 (클수록 부드럽게 눌림)
#   split     스플릿 토닝 — 그림자/하이라이트에 각각 다른 색
GRADES = {
    "none":     dict(),
    "warm":     dict(temp=+10, sat=1.18, contrast=1.19, lift=5, shoulder=1.0,
                     split=dict(shadow="#1a2436", highlight="#ffd8a8", amount=0.07)),
    "cool":     dict(temp=-10, sat=1.16, contrast=1.21, lift=3, shoulder=1.0,
                     split=dict(shadow="#16202e", highlight="#cfe2ff", amount=0.07)),
    "filmic":   dict(temp=+4, sat=1.04, contrast=1.27, lift=10, shoulder=1.3,
                     split=dict(shadow="#1c2230", highlight="#ffe7c4", amount=0.09)),
    "neon":     dict(temp=-5, sat=1.36, contrast=1.29, lift=7, shoulder=1.1,
                     split=dict(shadow="#1b1040", highlight="#ffb0e6", amount=0.10)),
    "wedding":  dict(temp=+7, sat=1.12, contrast=1.15, lift=12, shoulder=1.4,
                     split=dict(shadow="#232a38", highlight="#fff0dc", amount=0.08)),
}

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _rgb01(hexstr: str) -> np.ndarray:
    c = hexstr.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def _shoulder(a: np.ndarray, amount: float, knee: float = 0.90) -> np.ndarray:
    """하이라이트 롤오프.

    색온도·대비는 곱셈이라 밝은 픽셀을 1.0 위로 밀어낸다. 거기서 clip 하면
    하늘이나 역광이 흰 덩어리로 뭉치고, 그게 '폰으로 찍은 티' 의 정체다.
    knee 위쪽을 1.0 에 점근시키면 아무것도 잘리지 않고 계조가 남는다.

    ⚠ 채널별로 누르면 안 된다. R/G/B 를 각각 압축하면 셋이 서로 끌려가면서
    하이라이트 채도가 빠지고 화면이 물빠진 회색이 된다. 채널 최댓값 하나를
    기준으로 눌러 그 비율을 RGB 전체에 곱해야 색상과 채도가 보존된다.
    """
    if amount <= 0:
        return a
    span = 1.0 - knee
    peak = a.max(axis=-1)
    m = peak > knee
    if not m.any():
        return a
    scale = np.ones_like(peak)
    over = peak[m] - knee
    scale[m] = (knee + span * (over / (over + span * amount))) / peak[m]
    return a * scale[..., None]


def apply_grade(img: Image.Image, name: str | None) -> Image.Image:
    g = GRADES.get(name or "none", {})
    if not g:
        return img
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    # 1) lift — 검정을 살짝 들어올린다
    lift = g.get("lift", 0) / 255.0
    if lift:
        a = a * (1 - lift) + lift

    # 2) S커브 대비 — smoothstep 을 섞는다.
    #    ImageEnhance.Contrast 는 전 구간 선형이라 양 끝을 그냥 잘라먹었다.
    #    smoothstep 은 중간톤 기울기를 1.5 배로 올리면서 끝은 완만해져
    #    아무것도 클리핑되지 않는다. 클리핑이 없어진 만큼 강도를 더 줄 수 있다.
    w = float(np.clip((g.get("contrast", 1.0) - 1.0) * 2.0, 0.0, 0.85))
    if w > 1e-6:
        a = (1 - w) * a + w * (a * a * (3.0 - 2.0 * a))

    # 3) 색온도
    t = g.get("temp", 0) / 100.0
    if t:
        a[..., 0] *= (1 + t)
        a[..., 2] *= (1 - t)

    # 4) 스플릿 토닝 — 그림자와 하이라이트에 다른 색을 얹는다.
    #    전체를 한 방향으로 미는 것보다 '보정한 티' 가 제대로 난다.
    st = g.get("split")
    if st:
        lum = a @ LUMA
        amt = st.get("amount", 0.12)
        w_s = (np.clip(1.0 - lum * 2.0, 0, 1) * amt)[..., None]
        w_h = (np.clip(lum * 2.0 - 1.0, 0, 1) * amt)[..., None]
        a = a * (1 - w_s) + _rgb01(st["shadow"]) * w_s
        a = a * (1 - w_h) + _rgb01(st["highlight"]) * w_h

    # 5) 채도 — 휘도를 보존하며 (ImageEnhance.Color 보다 색이 덜 탁해진다)
    sat = g.get("sat", 1.0)
    if abs(sat - 1.0) > 1e-6:
        lum = (a @ LUMA)[..., None]
        a = lum + (a - lum) * sat

    # 6) 롤오프 — 위 연산이 1.0 을 넘긴 값을 잘라내지 않고 눌러 담는다
    a = _shoulder(np.maximum(a, 0.0), g.get("shoulder", 1.0))
    return Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8))


def grade_vf(name: str | None) -> str:
    """영상 소재용 근사 그레이딩 필터.

    사진은 numpy 로 정확히 처리하지만 영상은 ffmpeg 필터로 근사한다.
    colorbalance 가 스플릿 토닝에 그대로 대응하므로 예전 eq 단독보다
    사진과의 색 차이가 훨씬 줄어든다 (한 영상 안에서 사진 씬과 영상 씬의
    톤이 따로 노는 게 가장 티나는 아마추어 신호다).
    """
    g = GRADES.get(name or "none", {})
    if not g:
        return ""
    parts = [f"eq=contrast={g.get('contrast', 1):.3f}"
             f":saturation={g.get('sat', 1):.3f}"
             f":brightness={g.get('lift', 0) / 255:.4f}"]
    t = g.get("temp", 0) / 100.0
    st = g.get("split")
    if st:
        amt = st.get("amount", 0.12)
        sh = (_rgb01(st["shadow"]) - 0.5) * 2 * amt
        hl = (_rgb01(st["highlight"]) - 0.5) * 2 * amt
        parts.append(
            f"colorbalance=rs={sh[0] + t:.3f}:gs={sh[1]:.3f}:bs={sh[2] - t:.3f}"
            f":rh={hl[0] + t:.3f}:gh={hl[1]:.3f}:bh={hl[2] - t:.3f}")
    elif t:
        parts.append(f"colorbalance=rm={t:.3f}:bm={-t:.3f}")
    return ",".join(parts)


def auto_white_balance(img: Image.Image, strength: float = 0.6) -> Image.Image:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    means = a.reshape(-1, 3).mean(axis=0)
    target = means.mean()
    gains = np.where(means > 1, target / means, 1.0)
    gains = 1 + (gains - 1) * strength
    a *= gains
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _fit_inside(img: Image.Image, W: int, H: int, margin: float = 1.0) -> Image.Image:
    """W×H 안에 꽉 차게 맞춘다.

    Image.thumbnail() 은 축소만 하고 확대는 하지 않는다. 고객 사진은 대개
    캔버스(초해상도 2배)보다 작으므로 thumbnail 을 쓰면 사진이 원본 크기 그대로
    조그맣게 박힌다. 반드시 명시적으로 resize 해야 한다.
    """
    scale = min(W / img.width, H / img.height) * margin
    return img.resize((max(1, round(img.width * scale)),
                       max(1, round(img.height * scale))), Image.LANCZOS)


# ── 진입점 ────────────────────────────────────────────────────────────────
def prepare_image(src: Path, out_dir: Path, canvas: tuple[int, int],
                  fit: str = "cover", grade: str | None = None,
                  supersample: int = 2, headroom: float = 0.40,
                  use_face: bool = True, fit_margin: float = 1.0,
                  fit_shift: tuple[float, float] = (0.0, 0.0),
                  avoid: tuple = ()) -> Path:
    """zoompan 계단현상을 막기 위해 캔버스의 supersample 배로 만들어 둔다."""
    W, H = canvas[0] * supersample, canvas[1] * supersample
    key = cache_key(src, src.stat().st_mtime, W, H, fit, grade, headroom, use_face,
                    fit_margin, fit_shift, avoid)
    out = out_dir / f"img_{src.stem}_{key}.png"
    if out.exists():
        return out

    img = Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = auto_white_balance(img)

    if fit == "blurpad":
        bg = ImageOps.fit(img, (W, H), method=Image.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(W * 0.03))
        bg = ImageEnhance.Brightness(bg).enhance(0.62)
        fg = _fit_inside(img, W, H, fit_margin)
        bg.paste(fg, ((W - fg.width) // 2 + int(fit_shift[0] * W),
                      (H - fg.height) // 2 + int(fit_shift[1] * H)))
        img = bg
    elif fit == "blur":
        box = smart_crop_box(img, W / H)
        img = img.crop(box).resize((W, H), Image.LANCZOS)
        img = img.filter(ImageFilter.GaussianBlur(W * 0.022))
        img = ImageEnhance.Brightness(img).enhance(0.58)
    elif fit == "contain":
        canvas_img = Image.new("RGB", (W, H), (8, 8, 12))
        fg = _fit_inside(img, W, H, fit_margin)
        canvas_img.paste(fg, ((W - fg.width) // 2 + int(fit_shift[0] * W),
                              (H - fg.height) // 2 + int(fit_shift[1] * H)))
        img = canvas_img
    else:  # cover + smart crop
        box = smart_crop_box(img, W / H, headroom=headroom, use_face=use_face,
                             avoid=avoid)
        img = img.crop(box).resize((W, H), Image.LANCZOS)

    img = apply_grade(img, grade)
    if fit != "blur":
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=45, threshold=3))
    # 같은 사진이 여러 씬에 쓰이면 워커들이 이 경로에 동시에 쓴다 → 원자 교체
    with atomic.produce(out) as tmp:
        if tmp:
            img.save(tmp, "PNG")
    return out


def prepare_video(src: Path, out_dir: Path, canvas: tuple[int, int],
                  dur: float, trim_start: float = 0.0,
                  fit: str = "cover", grade: str | None = None) -> Path:
    """영상 소재를 캔버스 규격 · 필요한 길이로 잘라 정규화."""
    W, H = canvas
    key = cache_key(src, src.stat().st_mtime, W, H, dur, trim_start, fit, grade)
    out = out_dir / f"vid_{src.stem}_{key}.mp4"
    if out.exists():
        return out

    if fit == "blurpad":
        vf = (f"split[a][b];"
              f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H},gblur=sigma=28,eq=brightness=-0.12[bg];"
              f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2")
    else:
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H}")
    gvf = grade_vf(grade)
    if gvf:
        vf += "," + gvf
    vf += ",setsar=1"

    with atomic.produce(out) as tmp:
        if tmp:
            ffmpeg.run(["-ss", str(trim_start), "-t", str(dur), "-i", str(src),
                        "-an", "-vf", vf, "-r", "30",
                        *ffmpeg.video_encode_args(16), str(tmp)])
    return out
