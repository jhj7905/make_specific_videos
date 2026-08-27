"""SCRFD 얼굴 검출 (onnxruntime + numpy 전용).

insightface / opencv 를 끌어오지 않는다. 전처리·후처리를 직접 구현해서
의존성을 onnxruntime 하나로 묶었다 — 클라우드 서버로 옮길 때 이게 훨씬 가볍다.

모델: scrfd_10g_bnkps.onnx (검출 + 5점 랜드마크, 입력 RGB, (x-127.5)/128)
"""
from __future__ import annotations
import os, threading
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import config

MODEL_NAME = "scrfd_10g_bnkps.onnx"
STRIDES = (8, 16, 32)
NUM_ANCHORS = 2
INPUT_SIZE = 640

_session = None
_lock = threading.Lock()
_unavailable_reason: str | None = None


@dataclass
class Face:
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 (원본 좌표)
    score: float
    kps: np.ndarray                          # (5, 2)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def size(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(x2 - x1, y2 - y1)


def model_path() -> Path | None:
    env = os.environ.get("MV_FACE_MODEL")
    if env:
        return Path(env) if Path(env).exists() else None
    for base in (config.ROOT / "weight" / "face",
                 Path("/home/hyunjo/project/AI_Repurpose_Service_inte/weight/face")):
        p = base / MODEL_NAME
        if p.exists():
            return p
    return None


def _get_session():
    """모델이나 onnxruntime 이 없으면 None — 호출부는 엣지에너지 크롭으로 폴백."""
    global _session, _unavailable_reason
    if _session is not None or _unavailable_reason:
        return _session
    with _lock:
        if _session is not None or _unavailable_reason:
            return _session
        if os.environ.get("MV_FACE", "1") != "1":
            _unavailable_reason = "MV_FACE=0 (비활성화)"
            return None
        mp = model_path()
        if mp is None:
            _unavailable_reason = f"{MODEL_NAME} 없음"
            return None
        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(os.environ.get("MV_FACE_THREADS", "4"))
            _session = ort.InferenceSession(str(mp), opts, providers=providers)
        except Exception as e:
            _unavailable_reason = f"onnxruntime 로드 실패: {e}"
            return None
    return _session


def unavailable_reason() -> str | None:
    _get_session()
    return _unavailable_reason


# ── 후처리 ────────────────────────────────────────────────────────────────
def _anchor_centers(h: int, w: int, stride: int) -> np.ndarray:
    ys, xs = np.mgrid[:h, :w]
    centers = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
    centers = centers.reshape(-1, 2)
    if NUM_ANCHORS > 1:
        centers = np.repeat(centers, NUM_ANCHORS, axis=0)
    return centers


def _distance2bbox(points: np.ndarray, dist: np.ndarray) -> np.ndarray:
    return np.stack([points[:, 0] - dist[:, 0], points[:, 1] - dist[:, 1],
                     points[:, 0] + dist[:, 2], points[:, 1] + dist[:, 3]], axis=-1)


def _distance2kps(points: np.ndarray, dist: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, dist.shape[1], 2):
        out.append(points[:, 0] + dist[:, i])
        out.append(points[:, 1] + dist[:, i + 1])
    return np.stack(out, axis=-1)


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float = 0.4) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thresh]
    return keep


def detect(img: Image.Image, thresh: float = 0.5) -> list[Face]:
    sess = _get_session()
    if sess is None:
        return []

    W, H = img.size
    scale = min(INPUT_SIZE / W, INPUT_SIZE / H)
    nw, nh = int(round(W * scale)), int(round(H * scale))
    resized = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (0, 0, 0))
    canvas.paste(resized, (0, 0))

    blob = np.asarray(canvas, dtype=np.float32)
    blob = (blob - 127.5) / 128.0
    blob = blob.transpose(2, 0, 1)[None]           # NCHW

    outs = sess.run(None, {sess.get_inputs()[0].name: blob})
    n = len(STRIDES)
    boxes_all, scores_all, kps_all = [], [], []
    for i, stride in enumerate(STRIDES):
        scores = outs[i].reshape(-1)
        bbox = outs[i + n].reshape(-1, 4) * stride
        kps = outs[i + n * 2].reshape(-1, 10) * stride
        fh = fw = INPUT_SIZE // stride
        centers = _anchor_centers(fh, fw, stride)
        keep = scores >= thresh
        if not keep.any():
            continue
        boxes_all.append(_distance2bbox(centers[keep], bbox[keep]))
        scores_all.append(scores[keep])
        kps_all.append(_distance2kps(centers[keep], kps[keep]).reshape(-1, 5, 2))

    if not boxes_all:
        return []
    boxes = np.concatenate(boxes_all) / scale
    scores = np.concatenate(scores_all)
    kps = np.concatenate(kps_all) / scale

    faces = []
    for i in _nms(boxes, scores):
        x1, y1, x2, y2 = boxes[i]
        faces.append(Face(box=(float(max(0, x1)), float(max(0, y1)),
                               float(min(W, x2)), float(min(H, y2))),
                          score=float(scores[i]), kps=kps[i]))
    faces.sort(key=lambda f: f.size * f.score, reverse=True)
    return faces
