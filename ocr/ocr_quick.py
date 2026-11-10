"""과제 B 테스트용 — 임의의 표 이미지에 같은 OCR 설정(PP-OCRv5, 영어)을 적용하고 원시 결과를 저장.

run_ocr.py 는 Census Table 1 전용(열 경계·합계 검사·사람 검토 기록 포함)이고,
이 스크립트는 새 이미지에서 '어떤 오류가 나는지' 먼저 확인하는 1단계 도구다. 표 복원·검토는 하지 않는다.

사용법: .venv-ocr\\Scripts\\python.exe ocr\\ocr_quick.py input\\test\\b1-fraser-table.png [다른 이미지 ...]
출력  : outputs/ocr_test/<이미지명>/raw_ocr.json, ocr_regions.png, lines.txt(행 단위로 묶은 텍스트)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_ocr import OCR_PARAMS, ROOT, SCORE_REVIEW, run_ocr  # noqa: E402

NUM_LIKE = re.compile(r"^[\d,.\s]+$")
SUSPECT = re.compile(r"[OoIlS|]")  # 숫자 칸에 섞이면 0/1/5 오인식 의심


def main(paths: list[str]) -> None:
    for p in map(Path, paths):
        out = ROOT / "outputs" / "ocr_test" / p.stem
        out.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(p))
        items = run_ocr(p)
        for it in items:  # 확인 필요 후보 자동 표시
            t, why = it["text"], []
            if it["score"] < SCORE_REVIEW:
                why.append(f"score<{SCORE_REVIEW}")
            if re.search(r"\d", t) and SUSPECT.search(t) and len(re.sub(r"\D", "", t)) >= len(t) / 2:
                why.append("숫자 사이 문자(O/I/l/S) 의심")
            it["needs_review"] = why
        (out / "raw_ocr.json").write_text(json.dumps(
            {"input": str(p).replace("\\", "/"), "params": {k: str(getattr(v, "value", v)) for k, v in OCR_PARAMS.items()},
             "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
        vis = img.copy()
        for it in items:
            c = (0, 0, 255) if it["needs_review"] else (0, 160, 0)
            cv2.polylines(vis, [np.array(it["polygon"], np.int32)], True, c, 2 if it["needs_review"] else 1)
        cv2.imwrite(str(out / "ocr_regions.png"), vis)
        # y-중심으로 같은 줄 묶기 → 사람이 읽을 수 있는 초안
        items.sort(key=lambda t: (t["bbox_xyxy"][1] + t["bbox_xyxy"][3]) / 2)
        lines, cur, yc0 = [], [], None
        for it in items:
            yc = (it["bbox_xyxy"][1] + it["bbox_xyxy"][3]) / 2
            if cur and abs(yc - yc0) > 10:
                lines.append(cur); cur = []
            if not cur:
                yc0 = yc
            cur.append(it)
        if cur:
            lines.append(cur)
        txt = [" | ".join(f"{t['text']}{'(?)' if t['needs_review'] else ''}"
                          for t in sorted(l, key=lambda t: t["bbox_xyxy"][0])) for l in lines]
        (out / "lines.txt").write_text("\n".join(txt), encoding="utf-8")
        flagged = [it for it in items if it["needs_review"]]
        print(f"{p.name}: 영역 {len(items)}개, 확인 필요 후보 {len(flagged)}개 → {out.relative_to(ROOT)}")
        for it in flagged[:15]:
            print("   ", it["id"], repr(it["text"]), it["score"], it["needs_review"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
