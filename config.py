"""전역 설정. 서버를 옮기면 여기만 고치면 된다."""
from __future__ import annotations
import os, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── FFmpeg ────────────────────────────────────────────────────────────────
# 1) 환경변수 MV_FFMPEG 가 있으면 그걸 쓰고
# 2) 소스빌드 디렉터리(FFMPEG_SRC)가 있으면 거기서 찾고
# 3) 없으면 PATH 의 ffmpeg
FFMPEG_SRC = Path(os.environ.get("MV_FFMPEG_SRC", "/home/hyunjo/project/ffmpeg"))
VENDOR_LIB = ROOT / "vendor" / "lib"


def _resolve_bin(name: str) -> str:
    env = os.environ.get(f"MV_{name.upper()}")
    if env:
        return env
    cand = FFMPEG_SRC / name
    if cand.exists():
        return str(cand)
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"{name} 를 찾을 수 없습니다. MV_{name.upper()} 환경변수로 경로를 지정하세요.")


def ff_env() -> dict:
    """소스빌드 ffmpeg 는 공유 라이브러리 경로가 필요하다."""
    env = dict(os.environ)
    if FFMPEG_SRC.exists():
        libs = [FFMPEG_SRC / d for d in (
            "libavdevice", "libavcodec", "libavformat",
            "libavfilter", "libavutil", "libswresample", "libswscale")]
        paths = [str(p) for p in libs if p.exists()]
        if VENDOR_LIB.exists():
            paths.append(str(VENDOR_LIB))
        if paths:
            prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(paths + ([prev] if prev else []))
    return env


FFMPEG = _resolve_bin("ffmpeg")
FFPROBE = _resolve_bin("ffprobe")

# ── 렌더 ──────────────────────────────────────────────────────────────────
USE_NVENC = os.environ.get("MV_NVENC", "1") == "1"
GPU_IDS = [int(x) for x in os.environ.get("MV_GPUS", "0").split(",") if x.strip() != ""]
SCENE_WORKERS = int(os.environ.get("MV_WORKERS", "4"))

# 중간 씬 클립은 화질 손실을 막기 위해 고품질로
SCENE_CQ = 16
MASTER_CQ = 19
PREVIEW_CQ = 28

# ── 경로 ──────────────────────────────────────────────────────────────────
TEMPLATES_DIR = ROOT / "templates"
ASSETS_DIR = ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
STORAGE = ROOT / "storage"
WORK_DIR = STORAGE / "work"
OUTPUT_DIR = STORAGE / "output"
UPLOAD_DIR = STORAGE / "uploads"

for _p in (WORK_DIR, OUTPUT_DIR, UPLOAD_DIR):
    _p.mkdir(parents=True, exist_ok=True)
