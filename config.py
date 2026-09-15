"""전역 설정. 서버를 옮기면 여기만 고치면 된다."""
from __future__ import annotations
import os, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── FFmpeg ────────────────────────────────────────────────────────────────
# 1) 환경변수 MV_FFMPEG 가 있으면 그걸 쓰고
# 2) 소스빌드 디렉터리(FFMPEG_SRC)가 있으면 거기서 찾고
# 3) 없으면 PATH 의 ffmpeg
# 소스빌드 ffmpeg 디렉터리. 비워두면(기본) PATH 의 ffmpeg 을 쓴다.
# 개발 머신 경로를 기본값으로 박아두면 다른 서버에서 조용히 엉뚱한 바이너리를
# 찾거나 못 찾는다. 서버마다 MV_FFMPEG_SRC 로 지정할 것.
FFMPEG_SRC = Path(os.environ.get("MV_FFMPEG_SRC", "")) if os.environ.get("MV_FFMPEG_SRC") else None
VENDOR_LIB = ROOT / "vendor" / "lib"


def _resolve_bin(name: str) -> str:
    env = os.environ.get(f"MV_{name.upper()}")
    if env:
        return env
    if FFMPEG_SRC is not None:
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
    if FFMPEG_SRC is not None and FFMPEG_SRC.exists():
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


def __getattr__(name: str):
    """ffmpeg 경로는 처음 쓸 때 찾는다.

    import 시점에 찾아버리면 ffmpeg 이 없는 환경에서 `cli.py templates` 처럼
    렌더와 무관한 명령까지 트레이스백을 내며 죽는다.
    """
    if name in ("FFMPEG", "FFPROBE"):
        val = _resolve_bin(name.lower())
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ── 렌더 ──────────────────────────────────────────────────────────────────
USE_NVENC = os.environ.get("MV_NVENC", "1") == "1"
NVENC_PRESET = os.environ.get("MV_NVENC_PRESET", "p6")
# CPU 폴백 프리셋. 기존 기본값 'slow' 는 1080x1920 한 씬에 수 분이 걸려
# GPU 없는 서버에서는 사실상 못 쓴다. 'medium' 이 실사용 가능한 하한이고,
# 템플릿을 만들며 반복 렌더할 때는 MV_X264_PRESET=veryfast 가 편하다.
X264_PRESET = os.environ.get("MV_X264_PRESET", "medium")
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
