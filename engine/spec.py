"""템플릿 스펙(JSON) 정의. 새 템플릿 = 이 스키마를 만족하는 JSON 하나."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field, field_validator
import config


# ── 스타일 ────────────────────────────────────────────────────────────────
class Shadow(BaseModel):
    color: str = "#000000"
    opacity: float = 0.6
    blur: float = 12
    offset: tuple[float, float] = (0, 4)


class Stroke(BaseModel):
    color: str = "#000000"
    width: float = 0


class TextStyle(BaseModel):
    font: str = "Pretendard-Bold.otf"
    size: int = 72
    color: str = "#FFFFFF"
    gradient: list[str] | None = None          # 세로 그라데이션 (2색 이상)
    letter_spacing: float = 0                  # px
    line_height: float = 1.3
    align: Literal["left", "center", "right"] = "center"
    max_width: float = 0.82                    # 캔버스 너비 대비 줄바꿈 폭
    stroke: Stroke = Field(default_factory=Stroke)
    shadow: Shadow | None = None
    glow: Shadow | None = None                 # 네온 효과용 (blur 큰 shadow)
    uppercase: bool = False


# ── 레이어 ────────────────────────────────────────────────────────────────
class Motion(BaseModel):
    kind: Literal["none", "kenburns", "pan"] = "kenburns"
    zoom: tuple[float, float] = (1.0, 1.12)
    anchor: Literal["center", "top", "bottom", "left", "right"] = "center"
    direction: Literal["left", "right", "up", "down"] = "right"


class Anim(BaseModel):
    kind: Literal["none", "fade", "fade_up", "fade_down",
                  "slide_left", "slide_right"] = "fade_up"
    at: float = 0.0        # 씬 시작 기준 등장 시각(초)
    dur: float = 0.7       # 등장 길이
    out_at: float | None = None   # 퇴장 시작 시각(없으면 씬 끝까지 유지)
    out_dur: float = 0.4
    distance: float = 48   # 이동 거리 px


class MediaLayer(BaseModel):
    type: Literal["media"] = "media"
    landscape: dict | None = None   # 가로(16:9 등)일 때 덮어쓸 필드
    portrait: dict | None = None    # 세로(9:16 등)일 때 덮어쓸 필드
    src: str                                    # "{{photo1}}" 또는 에셋 상대경로
    fit: Literal["cover", "contain", "blurpad", "blur"] = "cover"
    motion: Motion = Field(default_factory=Motion)
    grade: str | None = None                    # 컬러 그레이딩 프리셋 이름
    opacity: float = 1.0
    trim_start: float = 0.0                     # 영상 소재일 때
    # blurpad/contain 에서 사진이 캔버스를 채우는 비율과 위치 (자막 자리 확보용)
    fit_margin: float = 1.0
    fit_shift: tuple[float, float] = (0.0, 0.0)
    headroom: float = 0.40                      # 얼굴을 창의 세로 몇 % 지점에 둘지
    use_face: bool = True                       # False 면 엣지에너지 크롭
    # 콜라주/폴라로이드: 캔버스 대비 정규화 좌표 [x, y, w, h]. None 이면 전체화면
    frame: tuple[float, float, float, float] | None = None
    # 타일의 가로세로비(w/h)를 고정하면 캔버스 비율이 바뀌어도 타일이 찌그러지지 않는다
    frame_ar: float | None = None
    radius: float = 0                           # 모서리 둥글기 (px)
    border_width: float = 0
    border_color: str = "#FFFFFF"
    shadow: bool = True                         # frame 일 때 드롭섀도


class FxLayer(BaseModel):
    type: Literal["fx"] = "fx"
    landscape: dict | None = None
    portrait: dict | None = None
    kind: Literal["vignette", "grain", "lightleak", "letterbox",
                  "flash", "bloom", "gradient_wash", "scrim"]
    params: dict = Field(default_factory=dict)


class TextLayer(BaseModel):
    type: Literal["text"] = "text"
    landscape: dict | None = None
    portrait: dict | None = None
    content: str                                 # "{{name_a}} ♥ {{name_b}}"
    style: str = "body"
    pos: tuple[Union[float, Literal["center"]], float] = ("center", 0.5)
    anim: Anim = Field(default_factory=Anim)
    skip_if_empty: bool = True                   # 치환 결과가 비면 레이어 생략


Layer = Annotated[Union[MediaLayer, FxLayer, TextLayer], Field(discriminator="type")]


class Transition(BaseModel):
    name: str = "fade"          # xfade transition 이름
    dur: float = 0.5


class Scene(BaseModel):
    name: str = ""
    landscape: dict | None = None
    portrait: dict | None = None
    dur: float
    layers: list[Layer] = Field(default_factory=list)
    transition: Transition | None = None   # 다음 씬으로 넘어가는 전환

    @field_validator("dur")
    @classmethod
    def _min_dur(cls, v):
        if v < 0.5:
            raise ValueError("씬 길이는 0.5초 이상이어야 합니다")
        return v


# ── 입력 슬롯 ─────────────────────────────────────────────────────────────
class InputSpec(BaseModel):
    id: str
    type: Literal["image", "video", "text"]
    required: bool = True
    guide: str = ""
    default: str = ""
    max_len: int = 40
    max_sec: float = 8.0


class Bgm(BaseModel):
    src: str | None = None
    license: str = ""
    gain_db: float = -16.0
    fade_in: float = 1.0
    fade_out: float = 2.0
    procedural: str | None = None   # 데모용 무저작권 생성 음원 프리셋


# ── 템플릿 ────────────────────────────────────────────────────────────────
class Template(BaseModel):
    id: str
    name: str
    category: str = ""
    version: str = "1"
    aspect: str = "9:16"
    resolution: tuple[int, int] = (1080, 1920)
    # 이 레이아웃을 그린 기준 캔버스. 출력 해상도가 달라지면 px 값(폰트/자간/
    # 모서리/그림자)을 짧은 변 비율로 환산한다. 없으면 resolution 을 기준으로 본다.
    design: tuple[int, int] | None = None
    fps: int = 30
    bgm: Bgm = Field(default_factory=Bgm)
    inputs: list[InputSpec] = Field(default_factory=list)
    styles: dict[str, TextStyle] = Field(default_factory=dict)
    scenes: list[Scene] = Field(default_factory=list)

    # 런타임 주입
    dir: Path = Field(default=Path("."), exclude=True)

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    @property
    def is_landscape(self) -> bool:
        return self.width >= self.height

    @property
    def scale(self) -> float:
        """px 단위 값에 곱할 배율 (짧은 변 기준)."""
        dw, dh = self.design or self.resolution
        return min(self.width, self.height) / max(1, min(dw, dh))

    def for_size(self, size: tuple[int, int]) -> "Template":
        """출력 해상도를 바꾸고 가로/세로 전용 오버라이드를 반영한 사본."""
        W, H = int(size[0]), int(size[1])
        key = "landscape" if W >= H else "portrait"

        def merge(obj):
            ov = getattr(obj, key, None)
            if not ov:
                return obj
            return obj.model_copy(update=ov)

        scenes = []
        for sc in self.scenes:
            sc2 = merge(sc)
            sc2 = sc2.model_copy(update={"layers": [merge(l) for l in sc2.layers]})
            scenes.append(sc2)

        out = self.model_copy(update={
            "resolution": (W, H),
            "design": self.design or self.resolution,
            "aspect": f"{W}:{H}",
            "scenes": scenes,
        })
        out.dir = self.dir
        return out

    @property
    def total_duration(self) -> float:
        total = sum(s.dur for s in self.scenes)
        total -= sum(s.transition.dur for s in self.scenes[:-1] if s.transition)
        return round(total, 3)

    def asset(self, rel: str) -> Path:
        """템플릿 폴더 → 공용 assets 순으로 탐색."""
        for base in (self.dir / "assets", self.dir, config.ASSETS_DIR):
            p = base / rel
            if p.exists():
                return p
        raise FileNotFoundError(f"에셋을 찾을 수 없습니다: {rel} (template={self.id})")

    def style(self, name: str) -> TextStyle:
        if name not in self.styles:
            raise KeyError(f"스타일 '{name}' 이 템플릿 {self.id} 에 없습니다")
        return self.styles[name]


def load_template(template_id: str) -> Template:
    d = config.TEMPLATES_DIR / template_id
    f = d / "template.json"
    if not f.exists():
        raise FileNotFoundError(f"템플릿 없음: {f}")
    tpl = Template.model_validate(json.loads(f.read_text(encoding="utf-8")))
    tpl.dir = d
    return tpl


def list_templates() -> list[Template]:
    out = []
    for d in sorted(config.TEMPLATES_DIR.iterdir()):
        if (d / "template.json").exists():
            try:
                out.append(load_template(d.name))
            except Exception as e:      # 잘못된 템플릿이 전체를 막지 않도록
                print(f"[warn] {d.name}: {e}")
    return out


ASPECTS = {
    "16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080),
    "4:5": (1080, 1350), "4:3": (1440, 1080), "21:9": (2560, 1080),
}


def parse_size(spec: str) -> tuple[int, int]:
    """'16:9' 또는 '1920x1080' → (w, h)"""
    s = spec.strip().lower()
    if s in ASPECTS:
        return ASPECTS[s]
    if "x" in s:
        w, h = s.split("x", 1)
        return (int(w), int(h))
    raise ValueError(f"알 수 없는 출력 규격: {spec} (예: 16:9, 9:16, 1920x1080)")


# ── 주문(Job) ─────────────────────────────────────────────────────────────
class Job(BaseModel):
    template: str
    order_id: str = "local"
    inputs: dict[str, str] = Field(default_factory=dict)
    preview: bool = True        # 워터마크 프리뷰도 같이 뽑을지
    aspect: str | None = None   # "16:9" / "9:16" / "1:1" / "1920x1080" (None=템플릿 기본)

    @staticmethod
    def load(path: str | Path) -> "Job":
        return Job.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
