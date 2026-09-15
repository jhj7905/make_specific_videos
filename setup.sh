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

# 2) 폰트
#    Pretendard(OFL) — 본문/UI
#    나눔손글씨 펜·붓 — 스크랩북 계열 템플릿의 손글씨.
#    네이버 글꼴은 글꼴 자체를 유료로 판매하는 것만 금지하고, 영상물 자막·
#    영상 광고를 포함한 상업적 사용은 허용된다. 아래는 Google Fonts 미러.
mkdir -p assets/fonts
for w in Bold ExtraBold Regular; do
  [ -f "assets/fonts/Pretendard-$w.otf" ] || curl -sSL -o "assets/fonts/Pretendard-$w.otf" \
    "https://github.com/orioncactus/pretendard/raw/main/packages/pretendard/dist/public/static/Pretendard-$w.otf"
done
for n in NanumPenScript NanumBrushScript; do
  d=$(echo "$n" | tr 'A-Z' 'a-z')
  [ -f "assets/fonts/$n-Regular.ttf" ] || curl -sSL -o "assets/fonts/$n-Regular.ttf" \
    "https://github.com/google/fonts/raw/main/ofl/$d/$n-Regular.ttf"
done

# 3) ffmpeg 확인 (NVENC 있으면 GPU, 없으면 libx264 자동 폴백)
.venv/bin/python -c "import config; print('ffmpeg:', config.FFMPEG)"
echo "완료. ./run.sh templates 로 확인하세요."
