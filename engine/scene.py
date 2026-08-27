"""씬 1개 → ffmpeg 명령 생성 → 중간 클립(mp4) 렌더.

씬 단위로 쪼개는 이유:
  1) 실패한 씬만 재렌더할 수 있다
  2) GPU 4장에 씬을 나눠 병렬 처리할 수 있다
  3) 필터그래프가 사람이 읽을 수 있는 크기로 유지된다
"""
from __future__ import annotations
from pathlib import Path
import config
from engine import ffmpeg, media
from engine.spec import Template, Scene, MediaLayer, FxLayer, TextLayer
from engine.text import (render_text_png, render_scrim_png, render_round_mask,
                         render_frame_shadow, render_frame_border, cache_key)


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
                    work: Path, out: Path, gpu: int | None = None) -> list[str]:
    W, H, fps = tpl.width, tpl.height, tpl.fps
    k = tpl.scale                       # px 단위 값 환산 배율
    frames = max(1, int(round(scene.dur * fps)))
    inputs: list[str] = []
    filters: list[str] = []
    n = 0

    base_label = None
    media_layers = [l for l in scene.layers if isinstance(l, MediaLayer)]
    full = [l for l in media_layers if not l.frame]
    framed = [l for l in media_layers if l.frame]

    def _even(v: float) -> int:
        return max(2, int(round(v / 2) * 2))

    # 1) 전체화면 미디어 (없으면 검정 배경)
    if not full:
        inputs += ["-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={fps}"]
        base_label = f"{n}:v"
        n += 1
    for i, ml in enumerate(full):
        src = _resolve_src(tpl, ml.src, resolved)
        if media.kind_of(src) == "image":
            prepared = media.prepare_image(Path(src), work, (W, H), ml.fit, ml.grade,
                                           headroom=ml.headroom, use_face=ml.use_face,
                                           fit_margin=ml.fit_margin, fit_shift=ml.fit_shift)
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
    for i, ml in enumerate(framed):
        fx, fy, fw, fh = ml.frame
        px, py = int(round(fx * W)), int(round(fy * H))
        pw = _even(fw * W)
        # frame_ar 을 주면 캔버스 비율이 바뀌어도 타일 모양이 유지된다
        ph = _even(pw / ml.frame_ar) if ml.frame_ar else _even(fh * H)
        radius = ml.radius * k
        border_w = ml.border_width * k
        src = _resolve_src(tpl, ml.src, resolved)

        if ml.shadow:
            pad = max(12, int(min(pw, ph) * 0.10))
            key = cache_key("shadow", pw, ph, radius, pad)
            sp = work / f"fshadow_{key}.png"
            if not sp.exists():
                render_frame_shadow(pw, ph, radius, pad, pad * 0.55, 0.55, "#000000", sp)
            inputs += ["-loop", "1", "-i", str(sp)]
            filters.append(f"[{n}:v]format=rgba[fs{i}]")
            filters.append(f"[{base_label}][fs{i}]overlay={px-pad}:{py-pad}:"
                           f"format=auto[bs{i}]")
            base_label = f"bs{i}"
            n += 1

        if media.kind_of(src) == "image":
            prepared = media.prepare_image(Path(src), work, (pw, ph), ml.fit, ml.grade,
                                           supersample=2, headroom=ml.headroom,
                                           use_face=ml.use_face)
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
            if not mp.exists():
                render_round_mask(pw, ph, radius, mp)
            inputs += ["-loop", "1", "-i", str(mp)]
            filters.append(f"[{n}:v]format=gray[fm{i}]")
            filters.append(f"[{vlbl}][fm{i}]alphamerge[fa{i}]")
            vlbl = f"fa{i}"
            n += 1

        filters.append(f"[{base_label}][{vlbl}]overlay={px}:{py}:format=auto[fo{i}]")
        base_label = f"fo{i}"

        if border_w > 0:
            key = cache_key("border", pw, ph, radius, border_w, ml.border_color)
            bp = work / f"fborder_{key}.png"
            if not bp.exists():
                render_frame_border(pw, ph, radius, border_w, ml.border_color, bp)
            inputs += ["-loop", "1", "-i", str(bp)]
            filters.append(f"[{n}:v]format=rgba[fb{i}]")
            filters.append(f"[{base_label}][fb{i}]overlay={px}:{py}:format=auto[fbo{i}]")
            base_label = f"fbo{i}"
            n += 1

    cur = base_label
    # FX (텍스트보다 아래에 깔리는 것들 먼저)
    for j, fx in enumerate([l for l in scene.layers if isinstance(l, FxLayer)]):
        if fx.kind == "scrim":
            skind = fx.params.get("shape", "bottom")
            key = cache_key("scrim", skind, sorted(fx.params.items()), W, H)
            png = work / f"scrim_{key}.png"
            if not png.exists():
                render_scrim_png(skind, fx.params, (W, H), png)
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
        if not png.exists():
            render_text_png(content, style, (W, H), l.pos, png, scale=k)
        inputs += ["-loop", "1", "-i", str(png)]
        pre, ox, oy = _text_anim(ti, l.anim, scene.dur, W, H, k)
        filters.append(f"[{n}:v]{pre}[t{ti}]")
        filters.append(f"[{cur}][t{ti}]overlay=x={ox}:y={oy}:format=auto[tx{ti}]")
        cur = f"tx{ti}"
        n += 1
        ti += 1

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
                 work: Path, gpu: int | None = None) -> Path:
    out = work / f"scene_{idx:02d}.mp4"
    cmd = build_scene_cmd(tpl, scene, resolved, work, out, gpu=gpu)
    ffmpeg.run(cmd, log=work / "render.log")
    return out
