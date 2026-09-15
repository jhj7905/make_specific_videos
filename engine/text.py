"""텍스트 레이어를 Pillow 로 RGBA PNG 로 굽는다.

FFmpeg drawtext 는 한글 자간/줄바꿈/그라데이션/네온글로우를 못 다룬다.
그래서 텍스트는 전부 여기서 만들고 ffmpeg 에는 overlay 로만 넘긴다.
"""
from __future__ import annotations
import hashlib, functools
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from fontTools.ttLib import TTFont
import config
from engine.spec import TextStyle

FALLBACKS = ["Pretendard-Regular.otf",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]


_warned_fonts: set[str] = set()


def font_missing(name: str) -> bool:
    p = Path(name)
    return not (p.is_absolute() and p.exists()) and not (config.FONTS_DIR / name).exists()


def _font_path(name: str, required: bool = False) -> Path:
    """없는 폰트는 예외 대신 Pretendard 로 떨어뜨린다.

    템플릿이 손글씨 폰트를 지정했는데 서버에 아직 안 깔렸다고 렌더 전체가
    죽으면 곤란하다. 모양만 달라지고 영상은 나오게 하고, 누락 사실은
    preflight 가 WARN 으로 잡는다.
    """
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    cand = config.FONTS_DIR / name
    if cand.exists():
        return cand
    if required:
        raise FileNotFoundError(f"폰트 없음: {name}")
    if name not in _warned_fonts:
        _warned_fonts.add(name)
        print(f"[warn] 폰트 없음: {name} → Pretendard-Bold.otf 로 대체")
    fallback = config.FONTS_DIR / "Pretendard-Bold.otf"
    if not fallback.exists():
        raise FileNotFoundError(f"폰트 없음: {name} (폴백도 없음)")
    return fallback


@functools.lru_cache(maxsize=32)
def _cmap(path: str) -> frozenset:
    try:
        tt = TTFont(path, fontNumber=0, lazy=True)
        chars = set()
        for table in tt["cmap"].tables:
            chars.update(table.cmap.keys())
        tt.close()
        return frozenset(chars)
    except Exception:
        return frozenset()


@functools.lru_cache(maxsize=64)
def _load(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


class FontChain:
    """주 폰트에 없는 글리프(♥, 이모지 등)는 폴백 폰트로 그린다."""

    def __init__(self, primary: str, size: int):
        paths = [str(_font_path(primary))]
        for fb in FALLBACKS:
            try:
                paths.append(str(_font_path(fb)))
            except FileNotFoundError:
                if Path(fb).exists():
                    paths.append(fb)
        self.paths = paths
        self.size = size
        self.fonts = [_load(p, size) for p in paths]

    def for_char(self, ch: str) -> ImageFont.FreeTypeFont:
        cp = ord(ch)
        for p, f in zip(self.paths, self.fonts):
            if cp in _cmap(p):
                return f
        return self.fonts[0]

    def advance(self, ch: str) -> float:
        return self.for_char(ch).getlength(ch)

    @property
    def ascent(self) -> int:
        return self.fonts[0].getmetrics()[0]


def _measure(chain: FontChain, text: str, spacing: float) -> float:
    if not text:
        return 0.0
    w = sum(chain.advance(c) for c in text)
    return w + spacing * (len(text) - 1)


def _wrap(chain: FontChain, text: str, spacing: float, max_px: float) -> list[str]:
    """한글은 공백이 적으므로 어절 우선 → 넘치면 글자 단위로 자른다."""
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        cur = ""
        for word in para.split(" "):
            trial = f"{cur} {word}".strip()
            if _measure(chain, trial, spacing) <= max_px or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
            while _measure(chain, cur, spacing) > max_px and len(cur) > 1:
                cut = len(cur)
                while cut > 1 and _measure(chain, cur[:cut], spacing) > max_px:
                    cut -= 1
                lines.append(cur[:cut])
                cur = cur[cut:]
        lines.append(cur)
    return lines


def _draw_line(draw: ImageDraw.ImageDraw, x: float, y: float, text: str,
               chain: FontChain, spacing: float, fill, stroke_w: float = 0,
               stroke_fill=None) -> None:
    for ch in text:
        draw.text((x, y), ch, font=chain.for_char(ch), fill=fill,
                  stroke_width=int(stroke_w), stroke_fill=stroke_fill)
        x += chain.advance(ch) + spacing


def _hex(c: str, alpha: int = 255) -> tuple[int, int, int, int]:
    c = c.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) == 8:
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), int(c[6:8], 16))
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), alpha)


def _vgradient(size: tuple[int, int], colors: list[str], box: tuple[int, int, int, int]) -> Image.Image:
    """텍스트 영역(box) 기준 세로 그라데이션 이미지를 만든다."""
    w, h = size
    img = Image.new("RGB", size, _hex(colors[0])[:3])
    top, bot = box[1], max(box[3], box[1] + 1)
    span = bot - top
    px = img.load()
    n = len(colors) - 1
    for y in range(h):
        t = min(1.0, max(0.0, (y - top) / span))
        seg = min(int(t * n), n - 1)
        lt = t * n - seg
        c0, c1 = _hex(colors[seg])[:3], _hex(colors[seg + 1])[:3]
        col = tuple(int(c0[i] + (c1[i] - c0[i]) * lt) for i in range(3))
        for x in range(w):
            px[x, y] = col
    return img


def scaled_style(style: TextStyle, k: float) -> TextStyle:
    """px 단위 값에 배율을 먹인 스타일 사본. (출력 해상도가 바뀔 때)"""
    if abs(k - 1.0) < 1e-6:
        return style
    d = style.model_dump()
    d["size"] = max(8, int(round(style.size * k)))
    d["letter_spacing"] = style.letter_spacing * k
    d["stroke"]["width"] = style.stroke.width * k
    for eff in ("shadow", "glow"):
        if d.get(eff):
            d[eff]["blur"] = d[eff]["blur"] * k
            d[eff]["offset"] = [d[eff]["offset"][0] * k, d[eff]["offset"][1] * k]
    return TextStyle.model_validate(d)


def render_text_png(content: str, style: TextStyle, canvas: tuple[int, int],
                    pos, out: Path, scale: float = 1.0) -> Path | None:
    """전체 캔버스 크기의 RGBA PNG 를 만든다. overlay=0:0 으로 얹으면 된다."""
    if not content.strip():
        return None
    if style.uppercase:
        content = content.upper()
    style = scaled_style(style, scale)

    W, H = canvas
    chain = FontChain(style.font, style.size)
    spacing = style.letter_spacing
    max_px = W * style.max_width
    lines = _wrap(chain, content, spacing, max_px)

    line_h = style.size * style.line_height
    block_h = line_h * len(lines)
    block_w = max((_measure(chain, l, spacing) for l in lines), default=0)

    px_, py_ = pos
    cx = W / 2 if px_ == "center" else float(px_) * W
    cy = float(py_) * H
    top = cy - block_h / 2
    # 폰트 ascent 보정 (Pillow 는 좌상단 기준으로 그린다)
    baseline_pad = (line_h - style.size) / 2

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    def line_x(text: str) -> float:
        w = _measure(chain, text, spacing)
        if style.align == "left":
            return cx - block_w / 2
        if style.align == "right":
            return cx + block_w / 2 - w
        return cx - w / 2

    # 1) 마스크(글자 모양)
    mask = Image.new("L", (W, H), 0)
    mdraw = ImageDraw.Draw(mask)
    for i, ln in enumerate(lines):
        y = top + i * line_h + baseline_pad
        _draw_line(mdraw, line_x(ln), y, ln, chain, spacing, fill=255)

    bbox = mask.getbbox() or (0, 0, W, H)

    # 2) 글로우 / 그림자 (마스크 블러)
    for eff, is_glow in ((style.glow, True), (style.shadow, False)):
        if not eff:
            continue
        dx, dy = eff.offset
        sh = mask.filter(ImageFilter.GaussianBlur(eff.blur))
        if is_glow:      # 네온은 여러 번 겹쳐야 심지가 살아난다
            sh = Image.eval(sh, lambda v: min(255, int(v * 1.8)))
        col = Image.new("RGBA", (W, H), _hex(eff.color, int(255 * eff.opacity)))
        col.putalpha(Image.eval(sh, lambda v, o=eff.opacity: int(v * o)))
        layer.alpha_composite(col, (int(dx), int(dy)))

    # 3) 외곽선
    if style.stroke.width > 0:
        sdraw = ImageDraw.Draw(layer)
        for i, ln in enumerate(lines):
            y = top + i * line_h + baseline_pad
            _draw_line(sdraw, line_x(ln), y, ln, chain, spacing,
                       fill=None, stroke_w=style.stroke.width,
                       stroke_fill=_hex(style.stroke.color))

    # 4) 본체 (단색 또는 그라데이션)
    if style.gradient and len(style.gradient) >= 2:
        fill_img = _vgradient((W, H), style.gradient, bbox).convert("RGBA")
        fill_img.putalpha(mask)
    else:
        fill_img = Image.new("RGBA", (W, H), _hex(style.color))
        fill_img.putalpha(mask)
    layer.alpha_composite(fill_img)

    out.parent.mkdir(parents=True, exist_ok=True)
    layer.save(out)
    return out


def block_bounds(content: str, style: TextStyle, canvas: tuple[int, int],
                 pos, scale: float = 1.0) -> tuple[float, float] | None:
    """텍스트 블록이 차지하는 세로 구간을 정규화 좌표 (y0, y1) 로.

    렌더와 완전히 같은 폰트·자간·줄바꿈으로 계산하므로 실측값이다.
    크롭이 얼굴을 어디에 둘지 정할 때와 preflight 넘침 검사에서 함께 쓴다.
    """
    if not content.strip():
        return None
    st = scaled_style(style, scale)
    if st.uppercase:
        content = content.upper()
    W, H = canvas
    chain = FontChain(st.font, st.size)
    lines = _wrap(chain, content, st.letter_spacing, W * st.max_width)
    block_h = st.size * st.line_height * len(lines)
    cy = float(pos[1]) * H
    return ((cy - block_h / 2) / H, (cy + block_h / 2) / H)


def cache_key(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


def render_scrim_png(kind: str, params: dict, canvas: tuple[int, int],
                     out: Path) -> Path:
    """텍스트 뒤에 까는 '어둠 판'.

    밝은 배경 사진 위에서 흰 글씨가 죽는 걸 막는다.
    상용 템플릿이 항상 쓰는 장치이고, 없으면 아마추어 티가 난다.
    """
    import numpy as np
    W, H = canvas
    strength = float(params.get("strength", 0.55))
    size = float(params.get("size", 0.45))
    center = float(params.get("center", 0.5))
    color = params.get("color", "#000000")
    feather = float(params.get("feather", 1.6))     # 1=선형, 클수록 부드럽게

    y = np.linspace(0, 1, H, dtype=np.float32)
    if kind == "bottom":
        t = np.clip((y - (1 - size)) / max(size, 1e-3), 0, 1)
    elif kind == "top":
        t = np.clip(((size) - y) / max(size, 1e-3), 0, 1)
    elif kind == "center":
        half = max(size / 2, 1e-3)
        t = np.clip(1 - np.abs(y - center) / half, 0, 1)
    elif kind == "both":
        tb = np.clip((y - (1 - size)) / max(size, 1e-3), 0, 1)
        tt = np.clip((size - y) / max(size, 1e-3), 0, 1)
        t = np.maximum(tb, tt)
    else:  # full
        t = np.ones_like(y)
    alpha = (np.power(t, feather) * strength * 255).astype(np.uint8)

    a = np.repeat(alpha[:, None], W, axis=1)
    rgb = _hex(color)[:3]
    img = np.zeros((H, W, 4), dtype=np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = rgb
    img[..., 3] = a
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, "RGBA").save(out)
    return out


# ── 프레임(콜라주/폴라로이드) 장식 ────────────────────────────────────────
def _round_rect_mask(w: int, h: int, radius: float) -> Image.Image:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    r = max(0, min(radius, min(w, h) / 2))
    if r <= 0:
        d.rectangle([0, 0, w - 1, h - 1], fill=255)
    else:
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return m


def render_round_mask(w: int, h: int, radius: float, out: Path) -> Path:
    """alphamerge 용 그레이스케일 마스크 (둥근 모서리)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    _round_rect_mask(w, h, radius).convert("L").save(out)
    return out


def render_frame_shadow(w: int, h: int, radius: float, pad: int, blur: float,
                        opacity: float, color: str, out: Path) -> Path:
    """프레임 아래 깔 드롭섀도. 캔버스는 (w+2*pad, h+2*pad)."""
    W, H = w + pad * 2, h + pad * 2
    sh = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(sh)
    r = max(0, min(radius, min(w, h) / 2))
    d.rounded_rectangle([pad, pad + int(pad * 0.25), pad + w - 1,
                         pad + h - 1 + int(pad * 0.25)], radius=r, fill=255)
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    img = Image.new("RGBA", (W, H), _hex(color)[:3] + (0,))
    img.putalpha(Image.eval(sh, lambda v: int(v * opacity)))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def render_frame_border(w: int, h: int, radius: float, width: float,
                        color: str, out: Path) -> Path:
    """프레임 위에 얹을 테두리 (폴라로이드 흰 테두리 등)."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = max(0, min(radius, min(w, h) / 2))
    d.rounded_rectangle([width / 2, width / 2, w - 1 - width / 2, h - 1 - width / 2],
                        radius=r, outline=_hex(color), width=int(width))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


# ── 스크랩북 장식 ─────────────────────────────────────────────────────────
# 이 스타일의 배경은 사진이 아니라 '종이' 다. 전체화면 사진 위에 글씨를 얹는
# 기존 템플릿과 구성이 근본적으로 다르고, 종이·기울어진 인화지·낙서 세 가지가
# 갖춰져야 비로소 스크랩북으로 읽힌다.

def rot_size(w: float, h: float, deg: float) -> tuple[int, int]:
    """deg 만큼 돌렸을 때 필요한 바운딩 박스 (짝수로 맞춘다)."""
    import math
    r = math.radians(abs(deg))
    c, s = abs(math.cos(r)), abs(math.sin(r))
    rw, rh = w * c + h * s, w * s + h * c
    return (int(math.ceil(rw / 2) * 2), int(math.ceil(rh / 2) * 2))


def render_paper_png(canvas: tuple[int, int], params: dict, out: Path) -> Path:
    """종이 질감 배경.

    에셋 이미지를 쓰지 않고 합성한다 — 판매물에 쓸 텍스처의 출처를 따로
    관리할 필요가 없고, 해상도/비율이 바뀌어도 그대로 따라온다.
    """
    import numpy as np
    W, H = canvas
    base = _hex(params.get("color", "#F4F1EB"))[:3]
    grain = float(params.get("grain", 1.8))      # 세로결 강도
    fiber = float(params.get("fiber", 1.6))      # 섬유 농담
    vig = float(params.get("vignette", 0.07))
    warm = float(params.get("warm", 3.0))        # 위쪽을 살짝 따뜻하게

    rng = np.random.default_rng(int(params.get("seed", 7)))
    a = np.zeros((H, W, 3), np.float32) + np.array(base, np.float32)

    # 세로 결 — 종이는 가로보다 세로 줄무늬가 눈에 띈다
    if grain > 0:
        cols = rng.normal(0, grain, W).astype(np.float32)
        cols = np.convolve(cols, np.ones(3) / 3, mode="same")
        a += cols[None, :, None]

    # 섬유 농담 — 셀이 크면 '구름' 이 되어 종이가 아니라 얼룩으로 읽힌다.
    # 짧은 변의 1/24 정도로 잘게 잡아야 질감으로 보인다.
    if fiber > 0:
        cell = max(2, min(H, W) // 24)
        small = rng.normal(0, fiber, (max(2, H // cell), max(2, W // cell))).astype(np.float32)
        a += np.asarray(Image.fromarray(small, "F").resize((W, H), Image.BILINEAR))[..., None]

    # 위에서 아래로 아주 옅은 온도 기울기 (완전 균일한 면은 인쇄물처럼 보인다)
    if warm:
        t = np.linspace(1.0, 0.0, H, dtype=np.float32)[:, None, None]
        a += t * np.array([warm, warm * 0.55, -warm * 0.35], np.float32)

    a += rng.normal(0, 0.7, (H, W, 1)).astype(np.float32)     # 미세 입자

    if vig > 0:
        yy = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
        xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
        d = np.sqrt(xx ** 2 + yy ** 2) / 1.414
        a *= (1 - vig * np.clip(d - 0.25, 0, 1)[..., None])

    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB").save(out)
    return out


def render_lightleak_png(canvas: tuple[int, int], params: dict, out: Path) -> Path:
    """구석에서 번지는 부드러운 빛. 종이 배경을 밋밋하지 않게 한다."""
    import numpy as np
    W, H = canvas
    cx = float(params.get("x", 0.12)) * W
    cy = float(params.get("y", 0.10)) * H
    radius = float(params.get("radius", 0.42)) * max(W, H)
    strength = float(params.get("strength", 0.55))
    color = _hex(params.get("color", "#FFFFFF"))[:3]

    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(radius, 1.0)
    alpha = np.clip(1.0 - d, 0, 1) ** 2.2 * strength

    img = np.zeros((H, W, 4), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = color
    img[..., 3] = (alpha * 255).astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, "RGBA").save(out)
    return out


def _heart_points(n: int = 240):
    import math
    for i in range(n + 1):
        t = 2 * math.pi * i / n
        yield (16 * math.sin(t) ** 3,
               -(13 * math.cos(t) - 5 * math.cos(2 * t)
                 - 2 * math.cos(3 * t) - math.cos(4 * t)))


def render_doodle_png(canvas: tuple[int, int], params: dict, out: Path) -> Path:
    """손그림 장식 — 하트 윤곽선, 밑줄 곡선, 동그라미."""
    import math
    W, H = canvas
    kind = params.get("shape", "heart")
    cx = float(params.get("x", 0.5)) * W
    cy = float(params.get("y", 0.5)) * H
    size = float(params.get("size", 0.12)) * min(W, H)
    width = max(1, int(round(float(params.get("width", 2.0)) * min(W, H) / 1080)))
    col = _hex(params.get("color", "#B9B2A6"), int(255 * float(params.get("opacity", 0.75))))

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    if kind == "heart":
        pts = [(cx + x * size / 17, cy + y * size / 17) for x, y in _heart_points()]
        d.line(pts, fill=col, width=width, joint="curve")
    elif kind == "swirl":
        pts = []
        for i in range(181):
            t = i / 180
            ang = t * math.pi * 2.2
            r = size * (0.12 + 0.55 * t)
            pts.append((cx + math.cos(ang) * r * 1.7, cy + math.sin(ang) * r * 0.55))
        d.line(pts, fill=col, width=width, joint="curve")
    elif kind == "circle":
        d.ellipse([cx - size, cy - size * 0.62, cx + size, cy + size * 0.62],
                  outline=col, width=width)
    elif kind == "underline":
        pts = [(cx - size + 2 * size * i / 60,
                cy + math.sin(i / 60 * math.pi) * size * 0.14)
               for i in range(61)]
        d.line(pts, fill=col, width=width, joint="curve")

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def render_tile_png(fw: int, fh: int, mat_l: float, mat_t: float, mat_b: float,
                    radius: float, color: str, deg: float,
                    pad: int, blur: float, opacity: float,
                    canvas: tuple[int, int], out: Path) -> Path:
    """인화지(대지) + 드롭섀도를 한 장으로. 이미 기울여서 굽는다.

    사진 스트림은 ffmpeg 의 rotate 로 같은 각도·같은 캔버스에서 돌린다.
    둘 다 캔버스 중심을 기준으로 돌리므로 정확히 겹친다.
    (PIL 의 rotate 는 반시계, ffmpeg 의 rotate 는 시계 방향이라 부호를 뒤집는다)
    """
    EW, EH = canvas
    layer = Image.new("RGBA", (EW, EH), (0, 0, 0, 0))
    x0, y0 = (EW - fw) / 2, (EH - fh) / 2
    r = max(0.0, min(radius, min(fw, fh) / 2))

    if opacity > 0 and pad > 0:
        sh = Image.new("L", (EW, EH), 0)
        ImageDraw.Draw(sh).rounded_rectangle(
            [x0, y0 + pad * 0.22, x0 + fw - 1, y0 + fh - 1 + pad * 0.22],
            radius=r, fill=255)
        sh = sh.filter(ImageFilter.GaussianBlur(blur))
        shadow = Image.new("RGBA", (EW, EH), (0, 0, 0, 0))
        shadow.putalpha(Image.eval(sh, lambda v: int(v * opacity)))
        layer.alpha_composite(shadow)

    mat = Image.new("RGBA", (EW, EH), (0, 0, 0, 0))
    ImageDraw.Draw(mat).rounded_rectangle(
        [x0, y0, x0 + fw - 1, y0 + fh - 1], radius=r, fill=_hex(color))
    layer.alpha_composite(mat)

    if abs(deg) > 1e-6:
        layer = layer.rotate(-deg, resample=Image.BICUBIC, expand=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    layer.save(out)
    return out
