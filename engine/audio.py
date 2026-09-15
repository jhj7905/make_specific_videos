"""BGM 처리.

⚠️ 판매용 영상의 음원은 반드시 '클라이언트 납품 허용' 상업 라이선스여야 한다.
유튜브 오디오 라이브러리는 스마트스토어 판매물에 안전하지 않다.
아래 procedural 프리셋은 저작권 이슈 없이 파이프라인을 검증하기 위한
합성 패드일 뿐, 실제 판매에는 라이선스 음원으로 교체할 것.
"""
from __future__ import annotations
from pathlib import Path
import config
from engine import ffmpeg, atomic
from engine.spec import Template

# 화성 (주파수 Hz)
CHORDS = {
    "warm_pad":   [[220.00, 261.63, 329.63], [196.00, 246.94, 293.66],
                   [174.61, 220.00, 261.63], [196.00, 233.08, 293.66]],
    "neon_pulse": [[146.83, 220.00, 293.66], [164.81, 246.94, 329.63],
                   [130.81, 196.00, 261.63], [174.61, 261.63, 349.23]],
}


def _pad_expr(preset: str, dur: float, bar: float = 4.0) -> str:
    """마디마다 코드가 바뀌는 패드를 aevalsrc 수식 하나로 만든다."""
    chords = CHORDS.get(preset, CHORDS["warm_pad"])
    terms = []
    nbars = max(1, int(dur / bar) + 1)
    for b in range(nbars):
        ch = chords[b % len(chords)]
        t0, t1 = b * bar, (b + 1) * bar
        gate = f"between(t\\,{t0}\\,{t1})"
        voices = "+".join(
            f"{0.16/len(ch):.4f}*sin(2*PI*{f:.2f}*t)" for f in ch)
        voices += "+" + "+".join(
            f"{0.05/len(ch):.4f}*sin(2*PI*{f*2:.2f}*t)" for f in ch)
        env = f"min(1\\,(t-{t0})/0.9)*min(1\\,({t1}-t)/1.2)"
        terms.append(f"{gate}*({voices})*{env}")
    body = "+".join(terms)
    shimmer = "(0.85+0.15*sin(2*PI*0.12*t))"
    fade_in = "min(1\\,t/1.5)"
    return f"({body})*{shimmer}*{fade_in}"


def ensure_bgm(tpl: Template, dur: float, work: Path) -> Path | None:
    b = tpl.bgm
    if b.src:
        return tpl.asset(b.src)
    if b.procedural:
        out = work / f"bgm_{b.procedural}_{dur:.1f}.wav"
        with atomic.produce(out) as tmp:
            if tmp:
                expr = _pad_expr(b.procedural, dur)
                ffmpeg.run(["-f", "lavfi", "-i",
                            f"aevalsrc=exprs='{expr}':d={dur+2:.2f}:s=48000",
                            "-af", "lowpass=f=3200,aecho=0.7:0.85:420|930:0.28|0.16",
                            "-c:a", "pcm_s16le", str(tmp)], log=work / "render.log")
        return out
    return None


def mux(video: Path, bgm: Path | None, tpl: Template, dur: float,
        out: Path, work: Path, *, cq: int, extra_vf: str | None = None,
        scale: tuple[int, int] | None = None) -> Path:
    """완성된 무음 영상 + BGM → 최종 mp4 (마스터/프리뷰 공용)."""
    args: list[str] = ["-i", str(video)]
    if bgm:
        args += ["-stream_loop", "-1", "-i", str(bgm)]

    # 영상에 손댈 게 없으면 재인코딩하지 않는다 (속도 + 화질 1세대 절약)
    vf = []
    if scale:
        vf.append(f"scale={scale[0]}:{scale[1]}:flags=lanczos")
    if extra_vf:
        vf.append(extra_vf)
    if vf:
        vf.append("format=yuv420p")
        args += ["-vf", ",".join(vf)]
        vcodec = ffmpeg.video_encode_args(cq)
    else:
        vcodec = ["-c:v", "copy"]

    if bgm:
        b = tpl.bgm
        af = (f"volume={b.gain_db}dB,"
              f"afade=t=in:st=0:d={b.fade_in},"
              f"afade=t=out:st={max(dur - b.fade_out, 0.1):.2f}:d={b.fade_out},"
              f"loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000")
        args += ["-map", "0:v:0", "-map", "1:a:0", "-af", af,
                 "-c:a", "aac", "-b:a", "192k", "-ac", "2"]
    else:
        args += ["-map", "0:v:0", "-an"]

    args += ["-t", f"{dur:.3f}", *vcodec, "-movflags", "+faststart", str(out)]
    ffmpeg.run(args, log=work / "render.log")
    return out
