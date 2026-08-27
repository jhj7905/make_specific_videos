"""FFmpeg 실행 래퍼."""
from __future__ import annotations
import json, subprocess, shlex, time
from pathlib import Path
import config


class FFmpegError(RuntimeError):
    pass


def run(args: list[str], *, log: Path | None = None, quiet: bool = True) -> None:
    cmd = [config.FFMPEG, "-hide_banner", "-nostdin", "-y"] + args
    t0 = time.time()
    proc = subprocess.run(cmd, env=config.ff_env(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.decode("utf-8", "replace")
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}\n{out}\n")
    if proc.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-25:])
        raise FFmpegError(f"ffmpeg 실패 (rc={proc.returncode})\n{tail}")
    if not quiet:
        print(f"  ffmpeg {time.time()-t0:.1f}s")


def probe(path: str | Path) -> dict:
    cmd = [config.FFPROBE, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    proc = subprocess.run(cmd, env=config.ff_env(),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.decode("utf-8", "replace"))
    return json.loads(proc.stdout)


def duration(path: str | Path) -> float:
    info = probe(path)
    return float(info["format"]["duration"])


def video_encode_args(cq: int, *, gpu: int | None = None, preset: str = "p6") -> list[str]:
    """NVENC 우선, 없으면 libx264 폴백."""
    if config.USE_NVENC:
        args = ["-c:v", "h264_nvenc", "-preset", preset, "-tune", "hq",
                "-rc", "vbr", "-cq", str(cq), "-b:v", "0",
                "-profile:v", "high", "-pix_fmt", "yuv420p"]
        if gpu is not None:
            args += ["-gpu", str(gpu)]
        return args
    return ["-c:v", "libx264", "-preset", "slow", "-crf", str(cq),
            "-profile:v", "high", "-pix_fmt", "yuv420p"]
