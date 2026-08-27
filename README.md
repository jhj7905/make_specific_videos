# make_videos — 템플릿 기반 영상 자동 제작 엔진

주문 1건(사진 + 문구) → 완성 영상 1편을 **사람 손 없이** 뽑는 렌더 엔진.
스마트스토어 "영상 제작 대행" 상품의 내부 자동화 무기.

**핵심 원칙: 템플릿은 데이터(JSON), 엔진은 고정 코드.**
템플릿을 100종 만들어도 `engine/` 코드는 늘어나지 않는다.

---

## 빠른 시작

```bash
./setup.sh                       # 최초 1회 (venv + 폰트)
./run.sh templates               # 템플릿 목록
./run.sh inputs propose_neon_01  # 입력 슬롯 확인 + job 양식 출력
./run.sh demo propose_neon_01                 # 폴더 사진으로 데모 job 자동 생성
./run.sh render jobs/demo.json --aspect 16:9  # 가로 1920x1080
./run.sh render jobs/demo.json --aspect 16:9,9:16 --gpus 0,1,2,3   # 가로+세로 한 번에
```

결과물 (`<태그>` = `1920x1080` / `1080x1920`):
- `storage/output/<주문번호>_<템플릿>_<태그>_master.mp4` — 납품용
- `storage/output/<주문번호>_<템플릿>_<태그>_preview.mp4` — 워터마크 (결제/확정 전 노출용)
- `storage/work/<주문번호>_<템플릿>_<태그>/manifest.json` — 재현용 기록 (입력 + 템플릿 버전)
- `storage/work/<주문번호>_<템플릿>_<태그>/render.log` — 실행된 ffmpeg 명령 전문

---

## 구조

```
engine/
  spec.py    템플릿 JSON 스키마 (pydantic). 새 기능은 여기부터 정의
  media.py   고객 소재 정규화: EXIF회전·스마트크롭·화이트밸런스·그레이딩
  text.py    Pillow 로 텍스트/스크림 PNG 굽기 (자간·그라데이션·네온글로우)
  scene.py   씬 1개 → filter_complex 생성 → 중간 mp4
  audio.py   BGM 루프·페이드·loudnorm
  render.py  씬 병렬 렌더 → xfade 결합 → 마스터/프리뷰
templates/<id>/template.json + assets/
```

### 왜 씬 단위로 쪼개나
1. 실패한 씬만 재렌더 → 40초 영상에서 1씬 오타를 고치는데 전체를 다시 돌리지 않는다
2. GPU 4장에 씬을 나눠 병렬 처리 (31초 영상이 **약 25초**에 완성)
3. 필터그래프가 사람이 읽을 수 있는 크기로 유지된다

### 왜 얼굴을 검출하나
엣지 에너지만으로 자르면 인물 사진에서 **얼굴을 통째로 놓친다**(실측: 배경의 도표나
소품 쪽으로 크롭됨). SCRFD 로 얼굴을 찾아 창 안에 넣고, 인물사진 관행대로
얼굴을 화면 위쪽 40%(`headroom`) 지점에 배치한다. 사진당 CPU 0.08초.
모델이 없거나(`weight/face/scrfd_10g_bnkps.onnx`) 얼굴이 없는 풍경 사진이면
자동으로 엣지 에너지 크롭으로 폴백한다. `MV_FACE=0` 으로 끌 수 있다.

### 왜 텍스트를 Pillow 로 굽나
FFmpeg `drawtext` 는 한글 자간 / 자동 줄바꿈 / 그라데이션 / 네온 글로우를 못 다룬다.
Pillow 로 전체 캔버스 RGBA PNG 를 만들고 ffmpeg 은 `overlay` 만 시킨다.
표현력이 AE 급으로 올라가고, 애니메이션은 `fade(alpha)` + `overlay` 좌표식으로 붙인다.

---

## 가로 / 세로 — 한 템플릿으로 둘 다

템플릿은 **하나의 레이아웃**만 그리고, 반대 방향은 오버라이드로 보정한다.
가로(TV·프로젝터 상영)와 세로(카톡·릴스 공유)를 옵션 상품으로 따로 팔 수 있다.

```json
{
  "resolution": [1080, 1920],
  "design":     [1080, 1920],     ← 이 레이아웃을 그린 기준 캔버스
  ...
  { "type": "text", "content": "{{msg1}}", "pos": ["center", 0.80],
    "landscape": { "pos": ["center", 0.875] } }     ← 가로일 때만 덮어씀
}
```

동작 원리:
1. **px 값 자동 환산** — 폰트/자간/외곽선/그림자/모서리/이동거리는 `design` 대비
   **짧은 변 비율**로 스케일된다. 1080×1920 → 1920×1080 은 배율 1.0 이라 그대로 쓰인다.
2. **정규화 좌표** — `pos`, `frame`, scrim 은 캔버스 대비 비율이라 자동으로 따라간다.
3. **오버라이드** — 레이어(또는 씬)에 `landscape` / `portrait` 블록을 두면
   출력 방향이 맞을 때 그 필드만 덮어쓴다. 보통 손볼 것은 **텍스트 y좌표, scrim 크기,
   콜라주 타일 배치** 셋뿐이다.
4. **`frame_ar`** — 콜라주 타일의 가로세로비를 고정한다. 없으면 9:16 의 세로 타일이
   16:9 에서 납작한 띠로 찌그러진다 (반드시 지정할 것).
5. **`fit_margin` / `fit_shift`** — `blurpad`/`contain` 에서 사진이 캔버스를 채우는
   비율과 위치. 사진을 살짝 줄여 아래쪽에 자막 자리를 비운다.

지원 규격: `16:9` `9:16` `1:1` `4:5` `4:3` `21:9` 또는 `1920x1080` 처럼 직접 지정.

## 새 템플릿 만들기

`templates/<새id>/template.json` 하나만 추가하면 끝. 코드 수정 없음.

지원 요소:

| 종류 | 값 |
|---|---|
| `media.fit` | `cover`(스마트크롭) · `contain` · `blurpad` · `blur`(배경흐림) |
| `media.motion.kind` | `kenburns` · `pan` · `none` |
| `media.grade` | `warm` · `cool` · `filmic` · `neon` · `wedding` · `none` |
| `fx.kind` | `scrim` · `vignette` · `grain` · `bloom` · `letterbox` · `flash` |
| `text.anim.kind` | `fade` · `fade_up` · `fade_down` · `slide_left` · `slide_right` |
| `transition.name` | ffmpeg xfade 전부 (`fadeblack` `dissolve` `smoothleft` `circleopen` `zoomin` …) |
| `media.frame` | 콜라주/폴라로이드 타일 `[x,y,w,h]` (정규화) + `frame_ar` `radius` `border_width` |
| 텍스트 스타일 | 폰트·크기·자간·행간·세로 그라데이션·외곽선·그림자·네온글로우 |
| 방향 오버라이드 | 모든 레이어/씬에 `landscape` · `portrait` 블록 |

### ⚠️ 반드시 지킬 것: 텍스트 뒤에는 `scrim` 을 깐다
고객 사진이 밝으면 흰 글씨가 그냥 사라진다. 상용 템플릿이 항상 쓰는 장치다.

```json
{ "type": "fx", "kind": "scrim",
  "params": { "shape": "bottom", "size": 0.55, "strength": 0.66 } }
```
`shape`: `bottom` · `top` · `center` · `both` · `full`

---

## 운영 메모

- **환경변수**: `MV_FFMPEG` (ffmpeg 경로) · `MV_FFMPEG_SRC` (소스빌드 디렉터리)
  · `MV_NVENC=0` (GPU 없는 클라우드에서 libx264 폴백) · `MV_GPUS=0,1,2,3` · `MV_WORKERS`
  · `MV_FACE=0` (얼굴검출 끄기) · `MV_FACE_MODEL` (SCRFD onnx 경로) · `MV_FACE_THREADS`
- **클라우드 이전 시**: `setup.sh` 실행 → `MV_NVENC=0` 이면 CPU 인코딩으로 자동 폴백.
  GPU 인스턴스면 nvenc 가 8~10배 빠르므로 렌더 서버는 GPU 인스턴스를 권장.
- 이 서버의 소스빌드 ffmpeg 은 `libx264.so.164` 링크가 깨져 있어 `vendor/lib/` 에
  복사해 두고 `config.ff_env()` 가 `LD_LIBRARY_PATH` 를 자동으로 잡는다.

## 판매 전 반드시 교체할 것

1. **BGM** — 현재 `procedural` 합성 패드는 파이프라인 검증용 플레이스홀더다.
   판매물에는 **'클라이언트 납품 허용' 상업 라이선스** 음원만 쓸 것
   (Artlist / Epidemic Sound / 셀바이뮤직). 유튜브 오디오 라이브러리는 안전하지 않다.
   라이선스 증빙은 `assets/bgm/` 에 음원과 함께 보관.
2. **폰트** — Pretendard(OFL) 은 상업 이용 가능. 다른 폰트 추가 시 '영상 삽입' 허용 여부를 개별 확인.
3. **고객 사진** — 납품 후 자동 삭제 정책(예: 30일)과 포트폴리오 사용 별도 동의 필요.

## 알려진 한계 / 다음 단계

- 씬 컷이 BGM 비트에 동기화되어 있지 않다. `librosa.beat.beat_track` 으로 비트를 뽑아
  씬 길이를 ±0.2초 스냅시키면 체감 퀄리티가 한 단계 올라간다.
- 웹 주문/업로드 페이지, 렌더 큐, 스마트스토어 커머스API 연동은 아직 없다 (2단계).
