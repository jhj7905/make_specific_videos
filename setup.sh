#!/usr/bin/env bash
# 새 서버(클라우드 이전 시) 초기 세팅
set -euo pipefail
cd "$(dirname "$0")"

# 1) Python 환경
python3 -m venv --without-pip .venv 2>/dev/null || python3 -m venv .venv
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  curl -sS -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
  .venv/bin/python /tmp/get-pip.py -q
fi
.venv/bin/pip install -q -r requirements.txt

# 2) 폰트 (Pretendard, OFL 라이선스)
mkdir -p assets/fonts
for w in Bold ExtraBold Regular; do
  [ -f "assets/fonts/Pretendard-$w.otf" ] || curl -sSL -o "assets/fonts/Pretendard-$w.otf" \
    "https://github.com/orioncactus/pretendard/raw/main/packages/pretendard/dist/public/static/Pretendard-$w.otf"
done

# 3) ffmpeg 확인 (NVENC 있으면 GPU, 없으면 libx264 자동 폴백)
.venv/bin/python -c "import config; print('ffmpeg:', config.FFMPEG)"
echo "완료. ./run.sh templates 로 확인하세요."
