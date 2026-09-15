"""씬 단위 증분 렌더.

주문 1건은 거의 항상 '프리뷰 전달 → 문구 한 줄 수정 → 재렌더' 를 한두 번 돈다.
그때마다 30초짜리를 통째로 다시 뽑는 건 순수한 낭비다. 상세페이지에 '수정 1회'
를 걸어두는 이상 이 왕복은 원가에 포함된 작업이고, 줄이면 그대로 처리량이 된다.

씬의 결과물을 결정하는 모든 입력 — 템플릿 정의, 그 씬이 참조하는 슬롯의 값,
출력 규격, 인코딩 설정 — 을 해시해서 파일명에 박는다. 문구 하나를 고치면
그 씬의 해시만 바뀌고 나머지 8개는 캐시에서 그대로 나온다.

파일 경로가 같아도 내용이 바뀌면(고객이 사진을 교체) 크기·mtime 이 달라지므로
해시도 달라진다.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import config
from engine.spec import Template, Scene, MediaLayer, TextLayer

SLOT = re.compile(r"\{\{(\w+)\}\}")


def _slots(scene: Scene) -> set[str]:
    """이 씬이 실제로 참조하는 입력 슬롯만 모은다."""
    found: set[str] = set()
    for l in scene.layers:
        if isinstance(l, MediaLayer):
            found |= set(SLOT.findall(l.src))
        elif isinstance(l, TextLayer):
            found |= set(SLOT.findall(l.content))
    return found


def _stamp(value: str) -> str:
    p = Path(value) if value else None
    try:
        if p and p.is_file():
            st = p.stat()
            return f"file:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        pass
    return f"val:{value}"


def _encode_stamp() -> list:
    return [config.SCENE_CQ, config.USE_NVENC, config.X264_PRESET, config.NVENC_PRESET]


def scene_fingerprint(tpl: Template, scene: Scene, resolved: dict[str, str]) -> str:
    payload = {
        "template": tpl.id,
        "version": tpl.version,
        "size": [tpl.width, tpl.height],
        "fps": tpl.fps,
        "scale": round(tpl.scale, 6),
        "encode": _encode_stamp(),
        "scene": json.loads(scene.model_dump_json()),
        "refs": {k: _stamp(resolved.get(k, "")) for k in sorted(_slots(scene))},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def concat_fingerprint(tpl: Template, scene_fps: list[str]) -> str:
    trans = [[s.transition.name, s.transition.dur] if s.transition else None
             for s in tpl.scenes]
    blob = json.dumps({"scenes": scene_fps, "trans": trans, "fps": tpl.fps,
                       "cq": config.MASTER_CQ, "encode": _encode_stamp()},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def master_fingerprint(tpl: Template, concat_fp: str) -> str:
    blob = json.dumps({"concat": concat_fp,
                       "bgm": json.loads(tpl.bgm.model_dump_json()),
                       "cq": config.MASTER_CQ, "encode": _encode_stamp()},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# ── 작업 디렉터리 상태 ────────────────────────────────────────────────────
def load_state(work: Path) -> dict:
    f = work / "cache_state.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(work: Path, state: dict) -> None:
    (work / "cache_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def sweep(work: Path, keep: set[str]) -> int:
    """더 이상 쓰이지 않는 씬 클립을 지운다. 지운 개수를 돌려준다."""
    removed = 0
    for p in work.glob("scene_*.mp4"):
        if p.name not in keep:
            p.unlink(missing_ok=True)
            removed += 1
    return removed
