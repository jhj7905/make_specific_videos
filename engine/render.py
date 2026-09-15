"""렌더 오케스트레이터: Job(주문) → 마스터 mp4 + 워터마크 프리뷰."""
from __future__ import annotations
import json, shutil, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageDraw
import config
from engine import ffmpeg, audio, cache
from engine.scene import render_scene, _substitute
from engine.spec import (Template, Job, load_template, parse_size,
                         MediaLayer, TextLayer)
from engine.text import FontChain


# ── 입력 검증/치환 ────────────────────────────────────────────────────────
def resolve_inputs(tpl: Template, job: Job) -> dict[str, str]:
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for spec in tpl.inputs:
        val = job.inputs.get(spec.id, "").strip()
        if not val:
            if spec.required and not spec.default:
                missing.append(f"  - {spec.id} ({spec.type}) {spec.guide}")
                continue
            val = spec.default
        if spec.type == "text" and len(val) > spec.max_len:
            print(f"[warn] {spec.id}: {spec.max_len}자 초과 → 잘림")
            val = val[:spec.max_len]
        if spec.type in ("image", "video") and val and not Path(val).exists():
            missing.append(f"  - {spec.id}: 파일 없음 ({val})")
            continue
        resolved[spec.id] = val
    if missing:
        raise ValueError("주문 입력이 부족합니다:\n" + "\n".join(missing))
    return resolved


def prune_template(tpl: Template, resolved: dict[str, str]) -> Template:
    """비어 있는 선택 슬롯을 정리한다.

    고객이 '선택' 사진/영상을 안 넣는 건 흔한 일이다. 그때 렌더가 죽으면 안 되고,
    해당 레이어(또는 씬 전체)를 조용히 빼고 나머지로 완성해야 한다.
    """
    kept = []
    for sc in tpl.scenes:
        layers = []
        for l in sc.layers:
            if isinstance(l, MediaLayer):
                src = _substitute(l.src, resolved)
                if not src.strip() or src.startswith("{{"):
                    continue
            if isinstance(l, TextLayer) and l.skip_if_empty:
                if not _substitute(l.content, resolved).strip():
                    continue
            layers.append(l)
        if not any(isinstance(l, (MediaLayer, TextLayer)) for l in layers):
            print(f"[skip] 씬 '{sc.name or '-'}' — 채워진 소재가 없어 생략")
            continue
        kept.append(sc.model_copy(update={"layers": layers}))

    if not kept:
        raise ValueError("렌더할 씬이 하나도 남지 않았습니다. 입력을 확인하세요.")
    kept[-1] = kept[-1].model_copy(update={"transition": None})
    out = tpl.model_copy(update={"scenes": kept})
    out.dir = tpl.dir
    return out


# ── 씬 결합 ───────────────────────────────────────────────────────────────
MIN_CUT = 0.034     # '컷' 전환도 xfade 로 처리 (1프레임)


def timeline_duration(tpl: Template, clips: list[Path]) -> float:
    """전환 겹침을 뺀 최종 길이. 결합을 캐시에서 건너뛰어도 값이 같아야 한다."""
    durs = [ffmpeg.duration(c) for c in clips]
    acc = durs[0]
    for i in range(1, len(clips)):
        s = tpl.scenes[i - 1]
        tdur = max(s.transition.dur, MIN_CUT) if s.transition else MIN_CUT
        acc += durs[i] - tdur
    return acc


def concat_scenes(tpl: Template, clips: list[Path], out: Path, work: Path) -> float:
    if len(clips) == 1:
        shutil.copy(clips[0], out)
        return ffmpeg.duration(out)

    durs = [ffmpeg.duration(c) for c in clips]
    trans = []
    for s in tpl.scenes[:-1]:
        if s.transition:
            trans.append((s.transition.name, max(s.transition.dur, MIN_CUT)))
        else:
            trans.append(("fade", MIN_CUT))

    args: list[str] = []
    for c in clips:
        args += ["-i", str(c)]

    filters, cur, acc = [], "0:v", durs[0]
    for i in range(1, len(clips)):
        name, tdur = trans[i - 1]
        off = acc - tdur
        lbl = f"x{i}"
        filters.append(f"[{cur}][{i}:v]xfade=transition={name}:"
                       f"duration={tdur:.3f}:offset={off:.3f}[{lbl}]")
        cur = lbl
        acc = acc + durs[i] - tdur
    filters.append(f"[{cur}]format=yuv420p,fps={tpl.fps}[vout]")

    args += ["-filter_complex", ";".join(filters), "-map", "[vout]", "-an",
             *ffmpeg.video_encode_args(config.MASTER_CQ),
             "-r", str(tpl.fps), str(out)]
    ffmpeg.run(args, log=work / "render.log")
    return acc


# ── 워터마크 ──────────────────────────────────────────────────────────────
def make_watermark(size: tuple[int, int], text: str, out: Path) -> Path:
    W, H = size
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    chain = FontChain("Pretendard-Bold.otf", int(W * 0.055))
    f = chain.fonts[0]
    tile = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    step_x, step_y = int(W * 0.62), int(H * 0.13)
    for gy in range(0, H * 2, step_y):
        for gx in range(0, W * 2, step_x):
            td.text((gx, gy), text, font=f, fill=(255, 255, 255, 46))
    tile = tile.rotate(30, resample=Image.BICUBIC)
    img.alpha_composite(tile, (0, 0), (W // 2, H // 2, W // 2 + W, H // 2 + H))
    d.rectangle([0, H - int(H * 0.055), W, H], fill=(0, 0, 0, 150))
    d.text((W * 0.5, H - int(H * 0.030)), text, font=chain.fonts[0],
           fill=(255, 255, 255, 210), anchor="mm")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def render_preview(master: Path, tpl: Template, out: Path, work: Path,
                   wm_text: str = "PREVIEW") -> Path:
    W = 960 if tpl.is_landscape else 540
    H = int(round(tpl.height * W / tpl.width / 2) * 2)
    wm = make_watermark((W, H), wm_text, work / f"wm_{W}x{H}.png")
    ffmpeg.run(["-i", str(master), "-i", str(wm), "-filter_complex",
                f"[0:v]scale={W}:{H}:flags=lanczos[v];[v][1:v]overlay=0:0,format=yuv420p[o]",
                "-map", "[o]", "-map", "0:a?", "-c:a", "aac", "-b:a", "128k",
                *ffmpeg.video_encode_args(config.PREVIEW_CQ),
                "-movflags", "+faststart", str(out)], log=work / "render.log")
    return out


# ── 메인 ──────────────────────────────────────────────────────────────────
def render_job(job: Job, *, out_dir: Path | None = None, keep_work: bool = True,
               gpus: list[int] | None = None, workers: int | None = None,
               aspect: str | None = None, force: bool = False) -> dict:
    """주문 1건 렌더. 이미 만들어 둔 씬은 재사용한다(force=True 면 전부 다시).

    '프리뷰 → 문구 수정 → 재렌더' 왕복에서 바뀐 씬만 다시 돈다.
    """
    t_all = time.time()
    tpl = load_template(job.template)
    want = aspect or job.aspect
    if want:
        tpl = tpl.for_size(parse_size(want))
    resolved = resolve_inputs(tpl, job)
    tpl = prune_template(tpl, resolved)

    tag = f"{tpl.width}x{tpl.height}"
    work = config.WORK_DIR / f"{job.order_id}_{tpl.id}_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir or config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    gpus = gpus or config.GPU_IDS
    workers = workers or config.SCENE_WORKERS
    state = {} if force else cache.load_state(work)

    print(f"▶ 템플릿 {tpl.id} ({tpl.name}) · 씬 {len(tpl.scenes)}개 · "
          f"{tpl.total_duration:.1f}초 · {tag}@{tpl.fps} "
          f"({'가로' if tpl.is_landscape else '세로'}, px배율 {tpl.scale:.2f})")

    # 1) 씬 렌더 — 지문이 같으면 건너뛴다
    scene_fps = [cache.scene_fingerprint(tpl, sc, resolved) for sc in tpl.scenes]
    clips = [work / f"scene_{i:02d}_{fp}.mp4" for i, fp in enumerate(scene_fps)]
    if force:
        for p in clips:
            p.unlink(missing_ok=True)
    reused = [i for i, p in enumerate(clips) if p.exists()]

    t0 = time.time()

    def _one(i: int) -> Path:
        if clips[i].exists():
            return clips[i]
        gpu = gpus[i % len(gpus)] if gpus else None
        p = render_scene(tpl, tpl.scenes[i], i, resolved, work, gpu=gpu, out=clips[i])
        print(f"  · scene {i:02d} ({tpl.scenes[i].name or '-'}) 완료")
        return p

    if reused:
        print(f"  · 캐시 재사용 {len(reused)}/{len(clips)}씬 "
              f"({', '.join(f'{i:02d}' for i in reused)})")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        clips = list(ex.map(_one, range(len(tpl.scenes))))
    t_scene = time.time() - t0

    # 2) 전환 결합
    t0 = time.time()
    cfp = cache.concat_fingerprint(tpl, scene_fps)
    silent = work / f"silent_{cfp}.mp4"
    dur = timeline_duration(tpl, clips)
    if force or not silent.exists():
        for stale in work.glob("silent_*.mp4"):
            if stale != silent:
                stale.unlink(missing_ok=True)
        concat_scenes(tpl, clips, silent, work)
    else:
        print("  · 결합 캐시 재사용")
    t_concat = time.time() - t0

    # 3) BGM + 마스터
    t0 = time.time()
    mfp = cache.master_fingerprint(tpl, cfp)
    master = out_dir / f"{job.order_id}_{tpl.id}_{tag}_master.mp4"
    if force or state.get("master") != mfp or not master.exists():
        bgm = audio.ensure_bgm(tpl, dur, work)
        audio.mux(silent, bgm, tpl, dur, master, work, cq=config.MASTER_CQ)
        state["master"] = mfp
    else:
        print("  · 마스터 캐시 재사용")
    t_audio = time.time() - t0

    result = {
        "order_id": job.order_id, "template": tpl.id, "template_version": tpl.version,
        "resolution": tag, "duration": round(dur, 2), "master": str(master),
        "cached_scenes": len(reused), "total_scenes": len(clips),
        "timing": {"scenes": round(t_scene, 1), "concat": round(t_concat, 1),
                   "audio": round(t_audio, 1), "total": round(time.time() - t_all, 1)},
    }

    # 4) 프리뷰
    if job.preview:
        prev = out_dir / f"{job.order_id}_{tpl.id}_{tag}_preview.mp4"
        if force or state.get("preview") != mfp or not prev.exists():
            render_preview(master, tpl, prev, work)
            state["preview"] = mfp
        result["preview"] = str(prev)

    cache.save_state(work, state)
    cache.sweep(work, {p.name for p in clips})
    (work / "manifest.json").write_text(
        json.dumps({**result, "inputs": resolved}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)

    print(f"✔ 완료 {result['timing']['total']}초 "
          f"(씬 {t_scene:.1f}s / 결합 {t_concat:.1f}s / 오디오 {t_audio:.1f}s"
          f"{f' · 캐시 {len(reused)}씬' if reused else ''})")
    return result
