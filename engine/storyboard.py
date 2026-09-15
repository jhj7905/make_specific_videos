"""스토리보드 — 씬별 대표 프레임 1장씩 뽑아 한 장의 컨택트시트로.

템플릿을 만들거나 고칠 때 확인하고 싶은 건 대부분 '레이아웃이 맞나' 다.
텍스트가 사진의 얼굴을 가리는지, scrim 이 글자 뒤에 제대로 깔렸는지,
콜라주 타일이 찌그러지지 않았는지. 이걸 보려고 30초짜리를 매번 인코딩하는
건 과하다. 영상 인코딩 없이 PNG 1장씩만 뽑으면 씬당 1~2초면 끝난다.

주문 검수용으로도 쓸 수 있다 — 고객에게 프리뷰 영상을 보내기 전에
9장짜리 한 장으로 먼저 훑는다.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageDraw

import config
from engine import cache
from engine.scene import render_scene_still
from engine.spec import Template
from engine.text import FontChain

LABEL_H = 34
GAP = 10
BG = (18, 18, 22)
FG = (232, 232, 238)
DIM = (150, 150, 160)


def _thumb_width(n: int, portrait: bool) -> int:
    if portrait:
        return 240 if n > 6 else 300
    return 380 if n > 6 else 460


def build(tpl: Template, resolved: dict[str, str], work: Path, out: Path,
          *, cols: int | None = None, workers: int | None = None) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    n = len(tpl.scenes)
    workers = workers or config.SCENE_WORKERS

    def _one(i: int) -> Path:
        sc = tpl.scenes[i]
        fp = cache.scene_fingerprint(tpl, sc, resolved)
        png = work / f"board_{i:02d}_{fp}.png"
        return render_scene_still(tpl, sc, i, resolved, work, png)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        stills = list(ex.map(_one, range(n)))

    tw = _thumb_width(n, not tpl.is_landscape)
    th = int(round(tpl.height * tw / tpl.width))
    cols = cols or (4 if not tpl.is_landscape else 3)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols

    W = cols * tw + (cols + 1) * GAP
    H = rows * (th + LABEL_H) + (rows + 1) * GAP + 38
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)

    title = FontChain("Pretendard-Bold.otf", 20).fonts[0]
    label = FontChain("Pretendard-Bold.otf", 15).fonts[0]
    small = FontChain("Pretendard-Regular.otf", 13).fonts[0]

    d.text((GAP, 12), f"{tpl.id} · {tpl.name} · {tpl.width}x{tpl.height} · "
                      f"{tpl.total_duration:.1f}초 · 씬 {n}", font=title, fill=FG)

    for i, p in enumerate(stills):
        r, c = divmod(i, cols)
        x = GAP + c * (tw + GAP)
        y = 38 + GAP + r * (th + LABEL_H + GAP)
        try:
            im = Image.open(p).convert("RGB").resize((tw, th), Image.LANCZOS)
        except Exception:
            im = Image.new("RGB", (tw, th), (60, 20, 20))
        sheet.paste(im, (x, y))
        sc = tpl.scenes[i]
        d.text((x, y + th + 7), f"{i:02d}  {sc.name or '-'}", font=label, fill=FG)
        meta = f"{sc.dur:.1f}s"
        if sc.transition:
            meta += f"  →{sc.transition.name} {sc.transition.dur:.1f}s"
        d.text((x + tw, y + th + 8), meta, font=small, fill=DIM, anchor="ra")

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
