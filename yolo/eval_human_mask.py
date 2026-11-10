"""과제 A — 사람이 칠한 정답 마스크와 YOLO 분할 마스크를 픽셀 단위로 대조.

입력: input/desk-objects.jpg + input/desk-objects_human-mask.png (MartinThoma, CC0 1.0)
정답 마스크 색 → 의미:  검정=노트북(laptop), 파랑=유리컵(cup), 노랑=책상(dining table), 빨강=케이블(COCO 클래스 없음)

출력: outputs/yolo/desk-objects/human_mask_eval.json
      outputs/yolo/desk-objects/human_mask_eval_conf_<c>.jpg  (초록=일치, 빨강=모델만 칠함(과분할), 파랑=사람만 칠함(누락))
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GT_COLORS_BGR = {  # 정답 마스크 색(BGR)과 대응 COCO 클래스
    "laptop": (0, 0, 0),
    "cup": (255, 0, 32),          # 파랑 계열 (RGB 32,0,255 근처)
    "dining table": (37, 216, 232),  # 노랑 계열
}
COLOR_TOL = 60


def gt_mask(gt_bgr: np.ndarray, bgr: tuple[int, int, int]) -> np.ndarray:
    d = np.abs(gt_bgr.astype(int) - np.array(bgr)).sum(axis=2)
    return d < COLOR_TOL


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    img = cv2.imread(str(ROOT / "input/desk-objects.jpg"))
    gt = cv2.imread(str(ROOT / "input/desk-objects_human-mask.png"))
    odir = ROOT / "outputs/yolo/desk-objects"
    report = {"정답": "input/desk-objects_human-mask.png (사람이 칠한 마스크, CC0)",
              "지표": "IoU = 교집합/합집합, 과분할 = 모델만 칠한 픽셀, 누락 = 사람만 칠한 픽셀",
              "results": {}}
    for conf in ("0.25", "0.50"):
        seg = json.loads((odir / f"conf_{conf}/segment.json").read_text(encoding="utf-8"))
        vis = img.copy()
        rows = {}
        for cls, color in GT_COLORS_BGR.items():
            g = gt_mask(gt, color)
            p = np.zeros(g.shape, bool)
            for inst in seg["instances"]:
                if inst["class_name"] == cls:
                    p |= cv2.imread(str(odir / f"conf_{conf}" / inst["mask_file"]), 0) > 0
            inter, union = (g & p).sum(), (g | p).sum()
            rows[cls] = {
                "gt_px": int(g.sum()), "pred_px": int(p.sum()),
                "iou": round(float(inter / union), 3) if union else None,
                "over_px(모델만)": int((p & ~g).sum()), "miss_px(사람만)": int((g & ~p).sum()),
                "status": "탐지됨" if p.any() else "모델이 찾지 못함(미탐)",
            }
            vis[g & p] = (vis[g & p] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
            vis[p & ~g] = (vis[p & ~g] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
            vis[g & ~p] = (vis[g & ~p] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
        cv2.putText(vis, f"conf={conf}  green=match  red=model only  blue=human only", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(odir / f"human_mask_eval_conf_{conf}.jpg"), vis)
        report["results"][conf] = rows
    (odir / "human_mask_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["results"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
