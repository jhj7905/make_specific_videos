"""캐시 파일을 '한 번만, 완성된 상태로만' 만든다.

같은 사진이 여러 씬에 쓰이면(propose_neon_01 은 photo1 이 opening/outro,
photo6 이 climax/proposal 에 쓰인다) ThreadPoolExecutor 의 여러 워커가
같은 캐시 경로에 동시에 PNG 를 쓴다. PIL 의 save() 는 원자적이지 않아서
다른 워커가 절반만 쓰인 파일을 읽어가는 사고가 간헐적으로 난다.
(재현이 어렵고 '가끔 씬 하나가 깨진다' 로 나타나기 때문에 더 고약하다)

임시 파일에 쓰고 os.replace 로 옮기면 '파일이 존재한다 == 완성되어 있다' 가
보장된다. os.replace 는 같은 파일시스템 안에서 원자적이다.
"""
from __future__ import annotations
import os, threading, uuid
from contextlib import contextmanager
from pathlib import Path

_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _guard:
        return _locks.setdefault(key, threading.Lock())


@contextmanager
def produce(path: str | Path):
    """이미 있으면 None, 없으면 '여기에 쓰라'는 임시 경로를 준다.

        with atomic.produce(png) as tmp:
            if tmp:
                render_something(tmp)

    블록이 정상 종료되면 tmp → path 로 원자 교체한다.
    예외가 나면 임시 파일만 지우고 그대로 올려보낸다(반쯤 쓰인 캐시가 남지 않는다).
    """
    path = Path(path)
    if path.exists():
        yield None
        return

    with _lock_for(str(path)):
        if path.exists():           # 락을 기다리는 동안 다른 워커가 만들었다
            yield None
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # 확장자를 유지해야 ffmpeg 이 컨테이너를 추론한다 (.part 로 끝나면 실패)
        tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex[:8]}.part{path.suffix}")
        try:
            yield tmp
            if not tmp.exists():
                raise FileNotFoundError(f"캐시를 만들지 못했습니다: {path}")
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
