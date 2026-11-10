"""과제 A — YOLO 서비스 핵심 로직.

- detection : yolo26n.pt     → 클래스·점수·박스(xyxy, 원본 픽셀 좌표)
- segmentation : yolo26n-seg.pt → 클래스·점수·박스 + 픽셀 마스크(원본 해상도)

CLI(run_yolo.py), 노트북(yolo_service.ipynb), 웹 화면(app.py)이 모두 이 모듈을 공유한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DET_WEIGHTS = "yolo26n.pt"
SEG_WEIGHTS = "yolo26n-seg.pt"

# 과제 요청문의 고정 설정
IMGSZ = 640
DEVICE = "cpu"
CONF_LEVELS = (0.25, 0.50)

_MODELS: dict[str, YOLO] = {}


def load_model(weights: str) -> YOLO:
    """models/ 아래 가중치를 사용(없으면 Ultralytics가 그 위치로 내려받음)."""
    if weights not in _MODELS:
        MODEL_DIR.mkdir(exist_ok=True)
        _MODELS[weights] = YOLO(str(MODEL_DIR / weights))
    return _MODELS[weights]


def _color(cls_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(cls_id * 7 + 3)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def detect(image_path: str | Path, conf: float) -> dict:
    """박스 탐지. 반환: {'image':..., 'detections':[{class_id,class_name,confidence,box_xyxy}]}"""
    model = load_model(DET_WEIGHTS)
    r = model.predict(str(image_path), imgsz=IMGSZ, conf=conf, device=DEVICE, verbose=False)[0]
    dets = []
    for i, b in enumerate(r.boxes):
        cid = int(b.cls)
        dets.append({
            "id": i,
            "class_id": cid,
            "class_name": r.names[cid],
            "confidence": round(float(b.conf), 4),
            "box_xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
        })
    h, w = r.orig_shape
    return {"task": "detect", "model": DET_WEIGHTS, "conf": conf, "imgsz": IMGSZ,
            "image_size_wh": [w, h], "box_format": "xyxy, 원본 이미지 픽셀 좌표(좌상단 원점)",
            "detections": dets, "_result": r}


def segment(image_path: str | Path, conf: float) -> dict:
    """인스턴스 분할. 마스크는 원본 해상도(H×W)의 bool 배열로 반환."""
    model = load_model(SEG_WEIGHTS)
    r = model.predict(str(image_path), imgsz=IMGSZ, conf=conf, device=DEVICE, verbose=False,
                      retina_masks=True)  # 마스크를 원본 해상도로 복원
    r = r[0]
    h, w = r.orig_shape
    insts, masks = [], []
    if r.masks is not None:
        mdata = r.masks.data.cpu().numpy() > 0.5
        for i, b in enumerate(r.boxes):
            cid = int(b.cls)
            m = mdata[i]
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            masks.append(m)
            insts.append({
                "id": i,
                "class_id": cid,
                "class_name": r.names[cid],
                "confidence": round(float(b.conf), 4),
                "box_xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
                "mask_area_px": int(m.sum()),
                "box_area_px": int((b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])),
            })
            insts[-1]["mask_to_box_ratio"] = round(insts[-1]["mask_area_px"] / max(insts[-1]["box_area_px"], 1), 3)
    return {"task": "segment", "model": SEG_WEIGHTS, "conf": conf, "imgsz": IMGSZ,
            "image_size_wh": [w, h], "instances": insts, "_masks": masks}


def draw_boxes(image_bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    out = image_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box_xyxy"])
        c = _color(d["class_id"])
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = f'#{d["id"]} {d["class_name"]} {d["confidence"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, max(y1, th + 6)), c, -1)
        cv2.putText(out, label, (x1 + 2, max(y1 - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_masks(image_bgr: np.ndarray, insts: list[dict], masks: list[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    """마스크를 반투명 색으로 겹치고 윤곽선 + 라벨을 그린다."""
    out = image_bgr.copy()
    for d, m in zip(insts, masks):
        c = np.array(_color(d["class_id"]), dtype=np.float32)
        out[m] = (out[m] * (1 - alpha) + c * alpha).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, tuple(int(v) for v in c), 2)
    return draw_boxes(out, insts) if insts else out


def side_by_side(*images: np.ndarray, titles: list[str] | None = None) -> np.ndarray:
    h = max(i.shape[0] for i in images)
    panels = []
    for k, im in enumerate(images):
        if im.shape[0] != h:
            im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        im = im.copy()
        if titles:
            cv2.rectangle(im, (0, 0), (im.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(im, titles[k], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(im)
    return np.hstack(panels)


def public(d: dict) -> dict:
    """'_'로 시작하는 내부 객체(결과·마스크 배열)를 뺀 JSON 직렬화용 dict."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
