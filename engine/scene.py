"""씬 1개 → ffmpeg 명령 생성 → 중간 클립(mp4) 렌더.

씬 단위로 쪼개는 이유:
  1) 실패한 씬만 재렌더할 수 있다
  2) GPU 4장에 씬을 나눠 병렬 처리할 수 있다
  3) 필터그래프가 사람이 읽을 수 있는 크기로 유지된다
"""
from __future__ import annotations
from pathlib import Path
import config
from engine import ffmpeg, media, atomic
from engine.spec import Template, Scene, MediaLayer, FxLayer, TextLayer
from engine.text import (render_text_png, render_scrim_png, render_round_mask,
                         render_frame_shadow, render_frame_border, cache_key,
                         block_bounds, render_paper_png, render_lightleak_png,
                         render_doodle_png, render_tile_png, rot_size)

# 자체 PNG 를 만들어 overlay 하는 fx (필터 체인으로 표현할 수 없는 것들)
PNG_FX = {"scrim", "lightleak", "doodle"}


def _kenburns(m, frames: int, fps: int, W: int, H: int) -> str:
    z0, z1 = m.zoom
    if m.kind == "none":
        return f"scale={W}:{H},setsar=1"
    if m.kind == "pan":
        z = max(z0, 1.06)
        xe = {"right": f"(iw-iw/{z})*on/{frames}",
              "left":  f"(iw-iw/{z})*(1-on/{frames})"}.get(m.direction, f"iw/2-(iw/{z}/2)")
        ye = {"down": f"(ih-ih/{z})*on/{frames}",
              "up":   f"(ih-ih/{z})*(1-on/{frames})"}.get(m.direction, f"ih/2-(ih/{z}/2)")
        zexpr = f"{z}"
    else:  # kenburns
        zexpr = f"{z0}+({z1}-{z0})*on/{frames}"
        anchor_x = {"left": "0", "right": "iw-iw/zoom"}.get(m.anchor, "iw/2-(iw/zoom/2)")
        anchor_y = {"top": "0", "bottom": "ih-ih/zoom"}.get(m.anchor, "ih/2-(ih/zoom/2)")
        xe, ye = anchor_x, anchor_y
    return (f"zoompan=z='{zexpr}':x='{xe}':y='{ye}':"
            f"d={frames}:s={W}x{H}:fps={fps},setsar=1")


def _video_motion(m, frames: int, fps: int, W: int, H: int) -> str:
    if m.kind == "none":
        return f"scale={W}:{H},setsar=1"
    z0, z1 = m.zoom
    return (f"zoompan=z='{z0}+({z1}-{z0})*on/{frames}':x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={fps},setsar=1")


def _fx_chain(fx: FxLayer, W: int, H: int, k: float = 1.0) -> tuple[str, list[str]]:
    """(체인 문자열, 추가 lavfi 입력) 을 돌려준다. 추가 입력이 필요한 fx 는 render 에서 합친다."""
    p = fx.params
    if fx.kind == "vignette":
        return f"vignette=angle=PI/{p.get('angle_div', 4.2)}", []
    if fx.kind == "grain":
        return f"noise=alls={int(p.get('strength', 8))}:allf=t+u", []
    if fx.kind == "letterbox":
        bar = int(H * p.get("ratio", 0.055))
        return (f"drawbox=x=0:y=0:w={W}:h={bar}:color=black@1:t=fill,"
                f"drawbox=x=0:y={H-bar}:w={W}:h={bar}:color=black@1:t=fill"), []
    if fx.kind == "flash":
        at, d = p.get("at", 0.0), p.get("dur", 0.25)
        amt = p.get("amount", 0.7)
        return (f"eq=brightness='if(between(t\\,{at}\\,{at+d})\\,"
                f"{amt}*(1-(t-{at})/{d})\\,0)':eval=frame"), []
    if fx.kind == "bloom":
        s = p.get("sigma", 26) * k
        o = p.get("opacity", 0.28)
        return (f"split[bl_a][bl_b];[bl_b]gblur=sigma={s}[bl_c];"
                f"[bl_a][bl_c]blend=all_mode=screen:all_opacity={o}"), []
    return "null", []


def _media_fade(anim) -> str:
    """미디어 레이어의 등장/퇴장 페이드. kind=none 이고 out_at 이 없으면 빈 문자열
    이라 필터가 아예 안 붙는다 — 기존 템플릿 동작이 바뀌지 않는다."""
    parts = []
    if anim.kind != "none":
        parts.append(f"fade=t=in:st={anim.at:.3f}:d={max(anim.dur, 0.01):.3f}:alpha=1")
    if anim.out_at is not None:
        parts.append(f"fade=t=out:st={anim.out_at:.3f}:"
                     f"d={max(anim.out_dur, 0.01):.3f}:alpha=1")
    return ",".join(parts)


def _text_anim(idx: int, anim, dur: float, W: int, H: int,
               k: float = 1.0) -> tuple[str, str, str]:
    """(전처리 체인, overlay x식, overlay y식)"""
    chain = ["format=rgba"]
    if anim.kind != "none":
        chain.append(f"fade=t=in:st={anim.at:.3f}:d={max(anim.dur,0.01):.3f}:alpha=1")
    if anim.out_at is not None:
        chain.append(f"fade=t=out:st={anim.out_at:.3f}:d={anim.out_dur:.3f}:alpha=1")
    else:
        chain.append(f"fade=t=out:st={max(dur-0.35,0.01):.3f}:d=0.35:alpha=1")

    p = f"min(1\\,max(0\\,(t-{anim.at:.3f})/{max(anim.dur,0.01):.3f}))"
    ease = f"pow(1-{p}\\,3)"          # ease-out cubic 의 잔여량
    d = anim.distance * k
    x, y = "0", "0"
    if anim.kind == "fade_up":
        y = f"'{d}*{ease}'"
    elif anim.kind == "fade_down":
        y = f"'-{d}*{ease}'"
    elif anim.kind == "slide_left":
        x = f"'{d}*{ease}'"
    elif anim.kind == "slide_right":
        x = f"'-{d}*{ease}'"
    return ",".join(chain), x, y


def build_scene_cmd(tpl: Template, scene: Scene, resolved: dict[str, str],
                    work: Path, out: Path, gpu: int | None = None,
                    still_at: float | None = None) -> list[str]:
    """씬 하나를 렌더하는 ffmpeg 인자를 만든다.

    still_at 이 주어지면 mp4 대신 그 시각의 정지 프레임 PNG 하나만 뽑는다.
    (스토리보드용 — 영상 인코딩 없이 레이아웃만 3초 안에 확인한다)
    """
    W, H, fps = tpl.width, tpl.height, tpl.fps
    k = tpl.scale                       # px 단위 값 환산 배율
    frames = max(1, int(round(scene.dur * fps)))
    inputs: list[str] = []
    filters: list[str] = []
    n = 0

    # 이 씬의 텍스트가 차지하는 세로 띠 — 얼굴이 그 아래로 들어가지 않게 한다
    text_bands: list[tuple[float, float]] = []
    for l in scene.layers:
        if not isinstance(l, TextLayer):
            continue
        content = _substitute(l.content, resolved)
        if l.skip_if_empty and not content.strip():
            continue
        try:
            b = block_bounds(content, tpl.style(l.style), (W, H), l.pos, k)
        except KeyError:
            continue
        if b:
            text_bands.append(b)
    bands = tuple(sorted(text_bands))

    base_label = None
    media_layers = [l for l in scene.layers if isinstance(l, MediaLayer)]
    full = [l for l in media_layers if not l.frame]
    framed = [l for l in media_layers if l.frame]

    def _even(v: float) -> int:
        return max(2, int(round(v / 2) * 2))

    # 1) 배경 — 종이(paper) fx 가 있으면 그것이 base 다.
    #    스크랩북 스타일은 배경이 사진이 아니라 종이다. fx 루프는 미디어보다
    #    뒤에 돌기 때문에 거기서 그리면 사진을 덮어버린다.
    paper = next((l for l in scene.layers
                  if isinstance(l, FxLayer) and l.kind == "paper"), None)
    if not full:
        if paper is not None:
            key = cache_key("paper", sorted(paper.params.items()), W, H)
            pp = work / f"paper_{key}.png"
            with atomic.produce(pp) as tmp:
                if tmp:
                    render_paper_png((W, H), paper.params, tmp)
            inputs += ["-loop", "1", "-i", str(pp)]
            filters.append(f"[{n}:v]scale={W}:{H},setsar=1,fps={fps}[paper]")
            base_label = "paper"
        else:
            inputs += ["-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={fps}"]
            base_label = f"{n}:v"
        n += 1
    for i, ml in enumerate(full):
        src = _resolve_src(tpl, ml.src, resolved)
        if media.kind_of(src) == "image":
            prepared = media.prepare_image(Path(src), work, (W, H), ml.fit, ml.grade,
                                           headroom=ml.headroom, use_face=ml.use_face,
                                           fit_margin=ml.fit_margin, fit_shift=ml.fit_shift,
                                           avoid=bands if ml.avoid_text else ())
            inputs += ["-loop", "1", "-i", str(prepared)]
            chain = _kenburns(ml.motion, frames, fps, W, H)
        else:
            prepared = media.prepare_video(Path(src), work, (W, H), scene.dur,
                                           ml.trim_start, ml.fit, ml.grade)
            inputs += ["-i", str(prepared)]
            chain = _video_motion(ml.motion, frames, fps, W, H)
        lbl = f"m{i}"
        filters.append(f"[{n}:v]{chain},format=yuv420p[{lbl}]")
        if base_label is None:
            base_label = lbl
        else:
            filters.append(f"[{base_label}][{lbl}]overlay=0:0:format=auto[mx{i}]")
            base_label = f"mx{i}"
        n += 1

    # 2) 프레임 미디어 (콜라주 / 폴라로이드)
    #
    #    mat 이나 rotate 가 있으면 '타일 캔버스' 경로를 탄다. 인화지+그림자를
    #    Pillow 로 미리 기울여 굽고, 사진 스트림은 같은 캔버스에서 ffmpeg 의
    #    rotate 로 돌린다. 둘 다 캔버스 중심 기준이라 정확히 겹친다.
    #    둘 다 0 이면 기존 경로를 그대로 써서 기존 템플릿이 1픽셀도 안 바뀐다.
    for i, ml in enumerate(framed):
        fxn, fyn, fwn, fhn = ml.frame
        fw = _even(fwn * W)
        fh = _even(fw / ml.frame_ar) if ml.frame_ar else _even(fhn * H)
        px, py = int(round(fxn * W)), int(round(fyn * H))
        radius = ml.radius * k
        border_w = ml.border_width * k
        src = _resolve_src(tpl, ml.src, resolved)
        tilted = ml.mat > 0 or abs(ml.rotate) > 1e-6

        if tilted:
            cxp, cyp = fxn * W + fw / 2, fyn * H + fh / 2
            m = ml.mat * min(fw, fh)
            mat_b = m * ml.mat_bottom
            pw = _even(fw - 2 * m)
            ph = _even(fh - m - mat_b)
            pad = max(14, int(min(fw, fh) * 0.09)) if ml.shadow else 0
            rw, rh = rot_size(fw, fh, ml.rotate)
            EW, EH = rw + 2 * pad, rh + 2 * pad
            ox = int(round((EW - fw) / 2 + m))
            oy = int(round((EH - fh) / 2 + m))
            X, Y = int(round(cxp - EW / 2)), int(round(cyp - EH / 2))

            key = cache_key("tile", fw, fh, m, mat_b, radius, ml.mat_color,
                            ml.rotate, pad, EW, EH, ml.shadow)
            tp = work / f"tile_{key}.png"
            with atomic.produce(tp) as tmp:
                if tmp:
                    render_tile_png(fw, fh, m, m, mat_b, radius, ml.mat_color,
                                    ml.rotate, pad, max(pad * 0.5, 4.0), 0.42,
                                    (EW, EH), tmp)
            inputs += ["-loop", "1", "-i", str(tp)]
            filters.append(f"[{n}:v]format=rgba[ft{i}]")
            n += 1
        else:
            pw, ph = fw, fh
            if ml.shadow:
                pad = max(12, int(min(pw, ph) * 0.10))
                key = cache_key("shadow", pw, ph, radius, pad)
                sp = work / f"fshadow_{key}.png"
                with atomic.produce(sp) as tmp:
                    if tmp:
                        render_frame_shadow(pw, ph, radius, pad, pad * 0.55, 0.55,
                                            "#000000", tmp)
                inputs += ["-loop", "1", "-i", str(sp)]
                _f = _media_fade(ml.anim)
                filters.append(f"[{n}:v]format=rgba{',' + _f if _f else ''}[fs{i}]")
                filters.append(f"[{base_label}][fs{i}]overlay={px-pad}:{py-pad}:"
                               f"format=auto[bs{i}]")
                base_label = f"bs{i}"
                n += 1

        if media.kind_of(src) == "image":
            prepared = media.prepare_image(Path(src), work, (pw, ph), ml.fit, ml.grade,
                                           supersample=2, headroom=ml.headroom,
                                           use_face=ml.use_face)   # 타일은 작아서 회피 불필요
            inputs += ["-loop", "1", "-i", str(prepared)]
            chain = _kenburns(ml.motion, frames, fps, pw, ph)
        else:
            prepared = media.prepare_video(Path(src), work, (pw, ph), scene.dur,
                                           ml.trim_start, ml.fit, ml.grade)
            inputs += ["-i", str(prepared)]
            chain = _video_motion(ml.motion, frames, fps, pw, ph)
        vlbl = f"fv{i}"
        filters.append(f"[{n}:v]{chain},format=rgba[{vlbl}]")
        n += 1

        if radius > 0:
            key = cache_key("mask", pw, ph, radius)
            mp = work / f"fmask_{key}.png"
            with atomic.produce(mp) as tmp:
                if tmp:
                    render_round_mask(pw, ph, radius, tmp)
            inputs += ["-loop", "1", "-i", str(mp)]
            filters.append(f"[{n}:v]format=gray[fm{i}]")
            filters.append(f"[{vlbl}][fm{i}]alphamerge[fa{i}]")
            vlbl = f"fa{i}"
            n += 1

        fade = _media_fade(ml.anim)

        if tilted:
            import math
            rad = math.radians(ml.rotate)
            rot = (f",rotate={rad:.6f}:c=none:ow={EW}:oh={EH}"
                   if abs(ml.rotate) > 1e-6 else "")
            filters.append(f"[{vlbl}]pad={EW}:{EH}:{ox}:{oy}:color=black@0{rot}[fp{i}]")
            # 인화지와 사진을 한 덩어리로 합친 뒤 통째로 페이드해야 같이 나타난다
            filters.append(f"[ft{i}][fp{i}]overlay=0:0:format=auto[tile{i}]")
            lbl = f"tile{i}"
            if fade:
                filters.append(f"[{lbl}]{fade}[tfa{i}]")
                lbl = f"tfa{i}"
            filters.append(f"[{base_label}][{lbl}]overlay={X}:{Y}:format=auto[fo{i}]")
            base_label = f"fo{i}"
            continue

        if fade:
            filters.append(f"[{vlbl}]{fade}[fva{i}]")
            vlbl = f"fva{i}"

        filters.append(f"[{base_label}][{vlbl}]overlay={px}:{py}:format=auto[fo{i}]")
        base_label = f"fo{i}"

        if border_w > 0:
            key = cache_key("border", pw, ph, radius, border_w, ml.border_color)
            bp = work / f"fborder_{key}.png"
            with atomic.produce(bp) as tmp:
                if tmp:
                    render_frame_border(pw, ph, radius, border_w, ml.border_color, tmp)
            inputs += ["-loop", "1", "-i", str(bp)]
            _f = _media_fade(ml.anim)
            filters.append(f"[{n}:v]format=rgba{',' + _f if _f else ''}[fb{i}]")
            filters.append(f"[{base_label}][fb{i}]overlay={px}:{py}:format=auto[fbo{i}]")
            base_label = f"fbo{i}"
            n += 1

    cur = base_label
    # FX (텍스트보다 아래에 깔리는 것들 먼저)
    fx_layers = [l for l in scene.layers
                 if isinstance(l, FxLayer) and l.kind != "paper"]   # paper 는 base
    for j, fx in enumerate(fx_layers):
        if fx.kind in PNG_FX:
            key = cache_key(fx.kind, sorted(fx.params.items()), W, H)
            png = work / f"{fx.kind}_{key}.png"
            with atomic.produce(png) as tmp:
                if tmp:
                    if fx.kind == "scrim":
                        render_scrim_png(fx.params.get("shape", "bottom"),
                                         fx.params, (W, H), tmp)
                    elif fx.kind == "lightleak":
                        render_lightleak_png((W, H), fx.params, tmp)
                    else:
                        render_doodle_png((W, H), fx.params, tmp)
            inputs += ["-loop", "1", "-i", str(png)]
            filters.append(f"[{n}:v]format=rgba[sc{j}]")
            filters.append(f"[{cur}][sc{j}]overlay=0:0:format=auto[fx{j}]")
            n += 1
        else:
            chain, _ = _fx_chain(fx, W, H, k)
            filters.append(f"[{cur}]{chain}[fx{j}]")
        cur = f"fx{j}"

    # 텍스트
    ti = 0
    for l in scene.layers:
        if not isinstance(l, TextLayer):
            continue
        content = _substitute(l.content, resolved)
        if l.skip_if_empty and not content.strip():
            continue
        style = tpl.style(l.style)
        key = cache_key(content, style.model_dump_json(), l.pos, W, H, round(k, 4))
        png = work / f"txt_{tpl.id}_{key}.png"
        with atomic.produce(png) as tmp:
            if tmp:
                render_text_png(content, style, (W, H), l.pos, tmp, scale=k)
        inputs += ["-loop", "1", "-i", str(png)]
        pre, ox, oy = _text_anim(ti, l.anim, scene.dur, W, H, k)
        filters.append(f"[{n}:v]{pre}[t{ti}]")
        filters.append(f"[{cur}][t{ti}]overlay=x={ox}:y={oy}:format=auto[tx{ti}]")
        cur = f"tx{ti}"
        n += 1
        ti += 1

    if still_at is not None:
        # trim 으로 원하는 프레임 하나만 통과시킨다. -vsync/-fps_mode 의
        # ffmpeg 버전별 차이를 피하려고 select 대신 trim 을 쓴다.
        n0 = max(0, min(frames - 1, int(round(still_at * fps))))
        filters.append(f"[{cur}]trim=start_frame={n0}:end_frame={n0+1},"
                       f"setpts=PTS-STARTPTS,format=rgb24[vout]")
        return [*inputs, "-filter_complex", ";".join(filters),
                "-map", "[vout]", "-frames:v", "1", str(out)]

    filters.append(f"[{cur}]format=yuv420p[vout]")
    return [*inputs, "-filter_complex", ";".join(filters),
            "-map", "[vout]", "-t", f"{scene.dur:.3f}", "-r", str(fps), "-an",
            *ffmpeg.video_encode_args(config.SCENE_CQ, gpu=gpu), str(out)]


def _substitute(text: str, resolved: dict[str, str]) -> str:
    for k, v in resolved.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def _resolve_src(tpl: Template, src: str, resolved: dict[str, str]) -> str:
    s = _substitute(src, resolved)
    if s.startswith("{{"):
        raise KeyError(f"입력 슬롯이 채워지지 않았습니다: {src}")
    p = Path(s)
    return str(p if p.exists() else tpl.asset(s))


def render_scene(tpl: Template, scene: Scene, idx: int, resolved: dict[str, str],
                 work: Path, gpu: int | None = None,
                 out: Path | None = None) -> Path:
    out = out or work / f"scene_{idx:02d}.mp4"
    with atomic.produce(out) as tmp:
        if tmp:
            cmd = build_scene_cmd(tpl, scene, resolved, work, tmp, gpu=gpu)
            ffmpeg.run(cmd, log=work / "render.log")
    return out


def render_scene_still(tpl: Template, scene: Scene, idx: int,
                       resolved: dict[str, str], work: Path,
                       out: Path, at: float | None = None) -> Path:
    """씬의 대표 프레임 1장. at 은 씬 시작 기준 초(기본: 씬 중간)."""
    at = scene.dur * 0.55 if at is None else at
    with atomic.produce(out) as tmp:
        if tmp:
            cmd = build_scene_cmd(tpl, scene, resolved, work, tmp, still_at=at)
            ffmpeg.run(cmd, log=work / "render.log")
    return out
