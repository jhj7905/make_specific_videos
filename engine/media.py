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


def _face_crop_box(img: Image.Image, faces, cw: int, ch: int,
                   headroom: float) -> tuple[int, int, int, int]:
    """얼굴이 잘리지 않고, 인물이 '눈높이 위쪽 1/3' 에 오도록 창을 배치한다."""
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

    x0 = fx - cw * 0.5
    y0 = fy - ch * headroom          # 얼굴을 화면 위쪽에 두는 인물사진 관행

    # 얼굴이 창 밖으로 나가지 않도록 밀어넣기 (이마 여백 0.3 얼굴높이 확보)
    pad = (uy2 - uy1) * 0.30
    x0 = min(x0, ux1 - 8)
    x0 = max(x0, ux2 + 8 - cw)
    y0 = min(y0, uy1 - pad)
    y0 = max(y0, uy2 + pad - ch)

    x0 = int(round(min(max(x0, 0), W - cw)))
    y0 = int(round(min(max(y0, 0), H - ch)))
    return (x0, y0, x0 + cw, y0 + ch)


def smart_crop_box(img: Image.Image, ar: float, center_bias: float = 0.55,
                   faces=None, headroom: float = 0.40,
                   use_face: bool = True) -> tuple[int, int, int, int]:
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
            return _face_crop_box(img, faces, cw, ch, headroom)

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
GRADES = {
    "none":     dict(),
    "warm":     dict(temp=+12, sat=1.08, contrast=1.06, lift=4),
    "cool":     dict(temp=-12, sat=1.05, contrast=1.08, lift=2),
    "filmic":   dict(temp=+4, sat=0.94, contrast=1.14, lift=8),
    "neon":     dict(temp=-6, sat=1.22, contrast=1.16, lift=6),
    "wedding":  dict(temp=+8, sat=1.02, contrast=1.04, lift=10),
}


def apply_grade(img: Image.Image, name: str | None) -> Image.Image:
    g = GRADES.get(name or "none", {})
    if not g:
        return img
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    t = g.get("temp", 0) / 100.0
    a[..., 0] *= (1 + t)          # R
    a[..., 2] *= (1 - t)          # B
    lift = g.get("lift", 0)
    a = a * (1 - lift / 255.0) + lift        # 필름처럼 검정을 살짝 들어올림
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if "contrast" in g:
        img = ImageEnhance.Contrast(img).enhance(g["contrast"])
    if "sat" in g:
        img = ImageEnhance.Color(img).enhance(g["sat"])
    return img


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
                  fit_shift: tuple[float, float] = (0.0, 0.0)) -> Path:
    """zoompan 계단현상을 막기 위해 캔버스의 supersample 배로 만들어 둔다."""
    W, H = canvas[0] * supersample, canvas[1] * supersample
    key = cache_key(src, src.stat().st_mtime, W, H, fit, grade, headroom, use_face,
                    fit_margin, fit_shift)
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
        box = smart_crop_box(img, W / H, headroom=headroom, use_face=use_face)
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
    g = GRADES.get(grade or "none", {})
    if g:
        vf += (f",eq=contrast={g.get('contrast',1)}:saturation={g.get('sat',1)}"
               f":brightness={g.get('lift',0)/255:.3f}")
    vf += ",setsar=1"

    with atomic.produce(out) as tmp:
        if tmp:
            ffmpeg.run(["-ss", str(trim_start), "-t", str(dur), "-i", str(src),
                        "-an", "-vf", vf, "-r", "30",
                        *ffmpeg.video_encode_args(16), str(tmp)])
    return out
