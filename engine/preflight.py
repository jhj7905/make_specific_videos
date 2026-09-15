"""주문 사전 검증.

렌더를 다 돌린 뒤에 문제를 발견하면 GPU 시간과 사람 시간이 같이 날아간다.
'5번 사진이 너무 작아 뭉개졌다'를 확인하려면 최소 한 번 렌더하고 눈으로 봐야
하는데, 그건 주문 1건마다 반복되는 비용이다. 여기서 잡으면 1초 안쪽이다.

검사 항목
  · 슬롯      필수 누락 / 파일 없음 / 지원하지 않는 형식
  · 사진      해상도 부족(확대 배율) · 크롭 손실 · 얼굴 검출 실패
  · 텍스트    길이 초과로 잘림 · 줄 수 과다 · 캔버스 밖으로 넘침
  · 음원      플레이스홀더 / 라이선스 표기 누락  ← 납품 사고 방지
  · 구성      선택 슬롯이 비어서 통째로 빠지는 씬
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from PIL import Image, ImageOps

import config
from engine import media, face
from engine.cache import SLOT
from engine.scene import _substitute
from engine.spec import Template, Job, MediaLayer, TextLayer
from engine.text import FontChain, scaled_style, _wrap

# 임계값 — 실사용하며 조정할 자리
UPSCALE_WARN = 1.30      # 이 배율을 넘겨 확대하면 눈에 띄게 흐려진다
UPSCALE_BAD = 2.00
CROP_LOSS_WARN = 0.50    # 원본의 절반 이상이 잘려나가면 구도가 의도와 달라진다
MIN_SHORT_SIDE = 720
MAX_TEXT_LINES = 3

RANK = {"error": 2, "warn": 1, "info": 0}
MARK = {"error": "✖", "warn": "▲", "info": "·"}


@dataclass
class Finding:
    level: str
    where: str
    message: str
    hint: str = ""


# ── 슬롯이 그려질 목표 크기 ───────────────────────────────────────────────
def media_targets(tpl: Template) -> dict[str, list[dict]]:
    """슬롯 → 그 슬롯이 실제로 그려질 픽셀 박스 목록.

    전체화면이면 캔버스 크기, 콜라주 타일이면 타일 크기다. Ken Burns 로
    최대 zoom 까지 당기므로 요구 해상도는 그만큼 더 크다.
    """
    out: dict[str, list[dict]] = {}
    W, H = tpl.width, tpl.height
    for sc in tpl.scenes:
        for l in sc.layers:
            if not isinstance(l, MediaLayer):
                continue
            slots = SLOT.findall(l.src)
            if not slots:
                continue
            if l.frame:
                _, _, fw, fh = l.frame
                bw = fw * W
                bh = bw / l.frame_ar if l.frame_ar else fh * H
            else:
                bw, bh = float(W), float(H)
            zoom = max(l.motion.zoom) if l.motion.kind != "none" else 1.0
            for s in slots:
                out.setdefault(s, []).append({
                    "scene": sc.name or "-",
                    "w": bw * zoom, "h": bh * zoom,
                    "fit": l.fit, "use_face": l.use_face,
                    "fit_margin": l.fit_margin,
                })
    return out


def _fit_metrics(sw: int, sh: int, t: dict) -> tuple[float, float]:
    """(확대 배율, 잘려나가는 면적 비율)"""
    need_w, need_h = t["w"], t["h"]
    ar = need_w / max(need_h, 1e-6)
    if t["fit"] in ("cover", "blur"):
        if sw / sh > ar:
            cw, ch = sh * ar, float(sh)
        else:
            cw, ch = float(sw), sw / ar
        upscale = need_w / max(cw, 1e-6)
        loss = 1 - (cw * ch) / max(sw * sh, 1e-6)
    else:                                   # contain / blurpad — 전체가 들어간다
        upscale = min(need_w / sw, need_h / sh) * t.get("fit_margin", 1.0)
        loss = 0.0
    return upscale, max(0.0, loss)


# ── 개별 검사 ─────────────────────────────────────────────────────────────
def _check_media(slot: str, path: str, targets: list[dict],
                 out: list[Finding], do_face: bool) -> None:
    p = Path(path)
    try:
        kind = media.kind_of(p)
    except ValueError:
        out.append(Finding("error", slot, f"지원하지 않는 형식입니다: {p.name}",
                           "jpg · png · webp · heic · mp4 · mov"))
        return
    if kind == "video":
        out.append(Finding("info", slot, f"영상 소재 {p.name} — 자동 검사 대상 아님"))
        return

    try:
        img = ImageOps.exif_transpose(Image.open(p))
    except Exception as e:
        out.append(Finding("error", slot, f"이미지를 열 수 없습니다: {e}"))
        return

    sw, sh = img.size
    if min(sw, sh) < MIN_SHORT_SIDE:
        out.append(Finding("warn", slot, f"원본이 작습니다 ({sw}x{sh})",
                           f"짧은 변 {MIN_SHORT_SIDE}px 이상을 다시 받는 게 좋습니다"))

    worst = None
    for t in targets:
        up, loss = _fit_metrics(sw, sh, t)
        if worst is None or up > worst[0]:
            worst = (up, loss, t)
    if worst:
        up, loss, t = worst
        if up >= UPSCALE_WARN:
            lvl = "warn"
            need = f"{int(t['w'])}x{int(t['h'])}"
            out.append(Finding(
                lvl, slot,
                f"{sw}x{sh} → '{t['scene']}' 씬에서 {up:.2f}배 확대"
                + (" (많이 뭉개집니다)" if up >= UPSCALE_BAD else " (다소 흐려집니다)"),
                f"이 슬롯의 목표 크기는 {need}px 입니다"))
        if loss >= CROP_LOSS_WARN:
            out.append(Finding(
                "warn", slot,
                f"'{t['scene']}' 씬에서 원본의 {loss*100:.0f}% 가 잘려나갑니다",
                "종횡비가 크게 달라 구도가 의도와 다를 수 있습니다"))

    if do_face and any(t["use_face"] and t["fit"] in ("cover",) for t in targets):
        if not face.detect(img):
            out.append(Finding(
                "info", slot, "얼굴이 검출되지 않아 엣지 에너지 크롭으로 처리됩니다",
                "인물 사진이라면 크롭 위치를 한 번 확인하세요"))


def _check_texts(tpl: Template, resolved: dict[str, str],
                 out: list[Finding]) -> None:
    W, H = tpl.width, tpl.height
    for sc in tpl.scenes:
        for l in sc.layers:
            if not isinstance(l, TextLayer):
                continue
            content = _substitute(l.content, resolved)
            where = f"{sc.name or '-'}/{l.style}"
            if not content.strip():
                if not l.skip_if_empty:
                    out.append(Finding("warn", where, "빈 텍스트 레이어입니다"))
                continue
            try:
                style = scaled_style(tpl.style(l.style), tpl.scale)
            except KeyError as e:
                out.append(Finding("error", where, str(e)))
                continue
            if style.uppercase:
                content = content.upper()
            chain = FontChain(style.font, style.size)
            lines = _wrap(chain, content, style.letter_spacing,
                          W * style.max_width)
            line_h = style.size * style.line_height
            block_h = line_h * len(lines)
            cy = float(l.pos[1]) * H
            top, bot = cy - block_h / 2, cy + block_h / 2
            preview = content.replace("\n", " / ")[:24]
            if top < 0 or bot > H:
                out.append(Finding(
                    "error", where,
                    f"텍스트가 화면 밖으로 넘칩니다 ({len(lines)}줄, {int(block_h)}px)",
                    f"\"{preview}\" — 문구를 줄이거나 style.size 를 낮추세요"))
            elif len(lines) > MAX_TEXT_LINES:
                out.append(Finding(
                    "warn", where,
                    f"{len(lines)}줄로 줄바꿈됩니다 — 한 화면에 깁니다",
                    f"\"{preview}\""))


def _check_slots(tpl: Template, job: Job, out: list[Finding]) -> dict[str, str]:
    """입력 슬롯을 검사하고, 렌더에 쓸 값을 돌려준다."""
    resolved: dict[str, str] = {}
    for spec in tpl.inputs:
        val = (job.inputs.get(spec.id) or "").strip()
        if not val:
            if spec.required and not spec.default:
                out.append(Finding("error", spec.id,
                                   f"필수 입력이 비어 있습니다 ({spec.type})", spec.guide))
                continue
            if not spec.default:
                out.append(Finding("info", spec.id, "비어 있음 — 해당 레이어는 생략됩니다"))
                resolved[spec.id] = ""
                continue
            val = spec.default
            out.append(Finding("info", spec.id, f"기본값 사용: {val[:20]}"))
        if spec.type == "text" and len(val) > spec.max_len:
            out.append(Finding(
                "warn", spec.id,
                f"{len(val)}자 → {spec.max_len}자로 잘립니다",
                f"잘리는 부분: …{val[spec.max_len:][:16]}"))
            val = val[:spec.max_len]
        if spec.type in ("image", "video") and not Path(val).exists():
            out.append(Finding("error", spec.id, f"파일이 없습니다: {val}"))
            continue
        resolved[spec.id] = val
    return resolved


def _check_bgm(tpl: Template, out: list[Finding], sale: bool) -> None:
    b = tpl.bgm
    lvl = "error" if sale else "warn"
    if not b.src and b.procedural:
        out.append(Finding(
            lvl, "bgm", f"플레이스홀더 합성 음원입니다 (procedural={b.procedural})",
            "판매/납품본은 '클라이언트 납품 허용' 상업 라이선스 음원으로 교체하세요"))
        return
    if not b.src and not b.procedural:
        out.append(Finding("warn", "bgm", "음원이 지정되지 않아 무음으로 나갑니다"))
        return
    lic = (b.license or "").strip()
    if not lic or "PLACEHOLDER" in lic.upper():
        out.append(Finding(lvl, "bgm", "음원 라이선스 표기가 비어 있거나 플레이스홀더입니다",
                           "구매 증빙과 함께 license 필드를 채워두세요"))


def _check_pruning(tpl: Template, resolved: dict[str, str],
                   out: list[Finding]) -> None:
    for sc in tpl.scenes:
        alive = 0
        for l in sc.layers:
            if isinstance(l, MediaLayer):
                src = _substitute(l.src, resolved)
                if src.strip() and not src.startswith("{{"):
                    alive += 1
            elif isinstance(l, TextLayer):
                if _substitute(l.content, resolved).strip():
                    alive += 1
        if alive == 0:
            out.append(Finding("info", sc.name or "-",
                               f"소재가 없어 씬이 통째로 빠집니다 (-{sc.dur:.1f}초)"))


# ── 진입점 ────────────────────────────────────────────────────────────────
def run(tpl: Template, job: Job, *, sale: bool = False,
        check_faces: bool = True) -> list[Finding]:
    out: list[Finding] = []
    resolved = _check_slots(tpl, job, out)
    _check_bgm(tpl, out, sale)

    do_face = check_faces and face.model_path() is not None
    if check_faces and not do_face:
        out.append(Finding("info", "face", "얼굴 검출 모델이 없어 크롭 검사는 건너뜁니다"))

    targets = media_targets(tpl)
    for slot, ts in sorted(targets.items()):
        val = resolved.get(slot, "")
        if val and Path(val).exists():
            _check_media(slot, val, ts, out, do_face)

    _check_texts(tpl, resolved, out)
    _check_pruning(tpl, resolved, out)
    out.sort(key=lambda f: (-RANK[f.level], f.where))
    return out


def summarize(findings: list[Finding]) -> dict[str, int]:
    c = {"error": 0, "warn": 0, "info": 0}
    for f in findings:
        c[f.level] += 1
    return c


def report(tpl: Template, job: Job, findings: list[Finding]) -> str:
    c = summarize(findings)
    head = (f"주문 {job.order_id} · {tpl.id} v{tpl.version} · "
            f"{tpl.width}x{tpl.height} · {tpl.total_duration:.1f}초 · 씬 {len(tpl.scenes)}")
    lines = [head, "─" * max(46, len(head))]
    if not findings:
        lines.append("문제 없음")
    for f in findings:
        lines.append(f"{MARK[f.level]} {f.level.upper():<5} {f.where:<14} {f.message}")
        if f.hint:
            lines.append(f"{'':>7}{'':<14}   └ {f.hint}")
    lines.append("─" * max(46, len(head)))
    verdict = "렌더 중단" if c["error"] else ("확인 후 진행" if c["warn"] else "이상 없음")
    lines.append(f"ERROR {c['error']} · WARN {c['warn']} · INFO {c['info']} — {verdict}")
    return "\n".join(lines)


def to_json(tpl: Template, job: Job, findings: list[Finding]) -> str:
    return json.dumps({
        "order_id": job.order_id, "template": tpl.id, "version": tpl.version,
        "resolution": f"{tpl.width}x{tpl.height}",
        "summary": summarize(findings),
        "findings": [asdict(f) for f in findings],
    }, ensure_ascii=False, indent=2)
