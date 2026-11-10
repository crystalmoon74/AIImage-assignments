"""과제 A — 배치 실행 스크립트.

사용법 (저장소 루트에서):
    .venv-yolo\\Scripts\\python.exe yolo\\run_yolo.py                       # input/bus.jpg, input/desk-objects.jpg
    .venv-yolo\\Scripts\\python.exe yolo\\run_yolo.py input\\my_photo.jpg    # 내 사진 추가

각 이미지 × conf(0.25, 0.50)에 대해 outputs/yolo/<이미지명>/ 아래에 저장:
    conf_0.25/detect.jpg, detect.json, segment.jpg, segment.json, masks/*.png, mask_union.png
    compare_detect_conf.jpg / compare_segment_conf.jpg  (0.25 vs 0.50 나란히)
    threshold_diff.json   (0.25에서만 남고 0.50에서 사라진 후보)
outputs/yolo/manifest.json  (입력·모델·패키지 버전·실행 위치·설정)
outputs/yolo/code/          (이번 실행에 사용한 코드 사본)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import platform
import shutil
import socket
import sys
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import yolo_core as yc  # noqa: E402

ROOT = yc.ROOT
OUT = ROOT / "outputs" / "yolo"
DEFAULT_INPUTS = [ROOT / "input" / "bus.jpg", ROOT / "input" / "desk-objects.jpg"]

# 입력 자료 출처·공유 조건 (과제: "입력 자료의 출처와 공유 조건을 결과 옆에 적습니다")
SOURCES = {
    "bus.jpg": {"source": "https://ultralytics.com/images/bus.jpg",
                "note": "과제 지정 버스 예제 사진(Ultralytics 공식 예제 이미지)"},
    "desk-objects.jpg": {"source": "https://commons.wikimedia.org/wiki/File:Image-segmentation-example.jpg",
                         "license": "CC0 1.0 (MartinThoma, Wikimedia Commons)",
                         "note": "공개 가능한 두 번째 사진. 사람이 칠한 정답 마스크(Image-segmentation-example-segmented.png, CC0)가 있어 원본 대조에 사용"},
    "a1-street-vehicles.jpg": {"source": "https://commons.wikimedia.org/wiki/File:Moscow_bicycle_parade_2025-05-17_(285).jpg",
                               "license": "CC0 1.0 (Retired electrician, Wikimedia Commons), 1280px 축소본",
                               "note": "테스트 이미지 — 작은 대상이 많아 임계값 차이 관찰용"},
    "a2-delivery-ebike.jpg": {"source": "https://commons.wikimedia.org/wiki/File:Moscow_2025-08-10_bicycle_race,_Samokat_delivery_02.jpg",
                              "license": "CC0 1.0 (Retired electrician, Wikimedia Commons), 1280px 축소본",
                              "note": "테스트 이미지 — 전기자전거(bicycle/motorcycle 경계) 클래스 판단 관찰용"},
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def iou_xyxy(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def dropped(low: list[dict], high: list[dict]) -> list[dict]:
    """낮은 임계값 결과 중 높은 임계값 결과에서 (같은 클래스·IoU≥0.9로) 짝이 없는 후보."""
    out = []
    for d in low:
        if not any(h["class_id"] == d["class_id"] and iou_xyxy(h["box_xyxy"], d["box_xyxy"]) >= 0.9 for h in high):
            out.append({k: d[k] for k in ("id", "class_name", "confidence", "box_xyxy")})
    return out


def process(img_path: Path) -> dict:
    img = cv2.imread(str(img_path))
    odir = OUT / img_path.stem
    summary = {"input": str(img_path.relative_to(ROOT)), "by_conf": {}}
    det_imgs, seg_imgs, det_res, seg_res = [], [], {}, {}

    for conf in yc.CONF_LEVELS:
        cdir = odir / f"conf_{conf:.2f}"
        (cdir / "masks").mkdir(parents=True, exist_ok=True)

        det = yc.detect(img_path, conf)
        det_vis = yc.draw_boxes(img, det["detections"])
        cv2.imwrite(str(cdir / "detect.jpg"), det_vis)
        yc.save_json(yc.public(det) | {"input": summary["input"]}, cdir / "detect.json")

        seg = yc.segment(img_path, conf)
        seg_vis = yc.draw_masks(img, seg["instances"], seg["_masks"])
        cv2.imwrite(str(cdir / "segment.jpg"), seg_vis)
        union = np.zeros(img.shape[:2], np.uint8)
        for inst, m in zip(seg["instances"], seg["_masks"]):
            fn = f'{inst["id"]:02d}_{inst["class_name"].replace(" ", "_")}.png'
            cv2.imwrite(str(cdir / "masks" / fn), m.astype(np.uint8) * 255)
            inst["mask_file"] = f"masks/{fn}"
            union[m] = 255
        cv2.imwrite(str(cdir / "mask_union.png"), union)
        yc.save_json(yc.public(seg) | {"input": summary["input"]}, cdir / "segment.json")

        det_imgs.append(det_vis); seg_imgs.append(seg_vis)
        det_res[conf], seg_res[conf] = det["detections"], seg["instances"]
        summary["by_conf"][f"{conf:.2f}"] = {
            "detect_counts": _count(det["detections"]),
            "segment_counts": _count(seg["instances"]),
        }

    lo, hi = yc.CONF_LEVELS
    cv2.imwrite(str(odir / "compare_detect_conf.jpg"),
                yc.side_by_side(img, *det_imgs, titles=["original", f"detect conf={lo}", f"detect conf={hi}"]))
    cv2.imwrite(str(odir / "compare_segment_conf.jpg"),
                yc.side_by_side(img, *seg_imgs, titles=["original", f"segment conf={lo}", f"segment conf={hi}"]))
    diff = {
        "설명": f"conf={lo}에서는 나왔지만 conf={hi}에서 사라진 후보(=점수가 {lo}~{hi} 사이). 오탐/정탐 여부는 원본 대조로 판단.",
        "detect_dropped": dropped(det_res[lo], det_res[hi]),
        "segment_dropped": dropped(seg_res[lo], seg_res[hi]),
    }
    yc.save_json(diff, odir / "threshold_diff.json")
    summary["threshold_diff"] = {k: len(v) for k, v in diff.items() if k != "설명"}
    return summary


def _count(items: list[dict]) -> dict:
    c: dict[str, int] = {}
    for d in items:
        c[d["class_name"]] = c.get(d["class_name"], 0) + 1
    return c


def main(argv: list[str]) -> None:
    inputs = [Path(a).resolve() for a in argv] or DEFAULT_INPUTS
    OUT.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now().astimezone()
    results = [process(p) for p in inputs]

    code_dir = OUT / "code"
    code_dir.mkdir(exist_ok=True)
    for f in ("yolo_core.py", "run_yolo.py", "eval_human_mask.py", "app.py"):
        src = Path(__file__).parent / f
        if src.exists():
            shutil.copy2(src, code_dir / f)

    manifest = {
        "assignment": "A. YOLO 서비스 만들기",
        "run_started": started.isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "execution_location": {
            "where": "로컬 PC (Windows, CPU 전용)",
            "hostname_hash": hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12],
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "processor": platform.processor(),
            "working_dir": "<repo root>",
        },
        "settings": {"imgsz": yc.IMGSZ, "device": yc.DEVICE, "conf_levels": list(yc.CONF_LEVELS),
                     "iou_nms": "Ultralytics 기본값", "retina_masks": True,
                     "box_format": "xyxy 원본 픽셀 좌표"},
        "models": {
            w: {"path": f"models/{w}", "sha256": sha256(yc.MODEL_DIR / w),
                "source": f"https://github.com/ultralytics/assets/releases (Ultralytics, {w})",
                "trained_on": "COCO 80 classes"}
            for w in (yc.DET_WEIGHTS, yc.SEG_WEIGHTS)
        },
        "packages": {p: version(p) for p in ("ultralytics", "torch", "torchvision", "opencv-python", "numpy")},
        "inputs": [
            {"file": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name,
             "sha256": sha256(p), **SOURCES.get(p.name, {"source": "사용자 제공(공유 동의 필요)"})}
            for p in inputs
        ],
        "results_summary": results,
    }
    yc.save_json(manifest, OUT / "manifest.json")
    print(f"saved -> {OUT}")
    for r in results:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1:])
