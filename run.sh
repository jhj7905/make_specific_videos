#!/usr/bin/env bash
# make_videos 실행 래퍼. venv + ffmpeg 라이브러리 경로를 잡아준다.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python cli.py "$@"
