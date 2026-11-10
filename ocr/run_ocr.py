"""과제 B — OCR로 표 이미지를 다시 쓰기 (Census 1910 Table 1).

파이프라인 (교재의 OCR 네 작업에 대응)
  1. 전처리   : 원본 PDF 2쪽에서 표 영역만 200dpi PNG로 추출(extract_table_image.py). 추가 보정 없음 → 원본 보존
  2. 검출     : PP-OCRv5 server det (다국어)            ─┐  RapidOCR(ONNX Runtime)로 실행
  3. 인식     : en_PP-OCRv5 mobile rec (영어 인식 가중치) ─┘  ※ PaddleOCR 공식 PP-OCR 가중치를 ONNX로 변환한 것
  4. 레이아웃 : 세로 괘선 검출로 열 경계, y-중심 군집으로 행을 묶어 표로 조립(이 스크립트의 규칙 기반 파서)
  5. 검증     : 낮은 점수 / 숫자 칸 문자 검사 / 합계 산술 검사로 '확인 필요' 자동 표시
  6. 사람 검토 : review/corrections.json 의 수정(원본 대조 근거 포함)을 적용 → table.csv, document.md

사용법 (저장소 루트에서):
    .venv-ocr\\Scripts\\python.exe ocr\\run_ocr.py
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import platform
import re
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input" / "census-table.png"
OUT = ROOT / "outputs" / "ocr"
CORRECTIONS = Path(__file__).parent / "review" / "corrections.json"

SCORE_REVIEW = 0.90  # 이 점수 미만 인식 결과는 '확인 필요'로 표시
OCR_PARAMS = {
    "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.SERVER, "Det.lang_type": LangDet.MULTI,
    "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE, "Rec.lang_type": LangRec.EN,
    "Global.use_cls": True,
}
SOURCE = {
    "title": "Thirteenth Census of the United States Taken in the Year 1910, Volume I: Population — Table 1",
    "publisher": "U.S. Bureau of the Census (1913)",
    "url": "https://www2.census.gov/library/publications/decennial/1910/volume-1/volume-1-p3.pdf",
    "pdf_page": 2, "printed_page": 22,
    "crop_rect_pt": [235, 283, 625, 463], "dpi": 200,
    "license": "미국 연방정부 발행물 — 퍼블릭 도메인",
}
# 표의 계층(들여쓰기)과 검증용 산술 관계
ROW_LEVEL = {"United States, with outlying possessions": "total",
             "United States, exclusive of outlying possessions": "subtotal",
             "Outlying possessions": "subtotal"}


# ---------------------------------------------------------------- OCR
def run_ocr(img_path: Path) -> list[dict]:
    eng = RapidOCR(params=OCR_PARAMS)
    r = eng(str(img_path))
    items = []
    for i, (box, txt, sc) in enumerate(zip(r.boxes, r.txts, r.scores)):
        pts = [[round(float(x), 1), round(float(y), 1)] for x, y in box]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        items.append({"id": i, "text": txt, "score": round(float(sc), 4), "polygon": pts,
                      "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys)]})
    return items


# ---------------------------------------------------------------- 레이아웃
def vertical_rules(gray: np.ndarray) -> list[int]:
    """세로 괘선(표의 열 구분선) x 좌표 검출."""
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gray.shape[0] // 4))
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    cols = np.where(lines.sum(axis=0) > 0)[0]
    groups, cur = [], []
    for x in cols:
        if cur and x - cur[-1] > 3:
            groups.append(cur); cur = []
        cur.append(int(x))
    if cur:
        groups.append(cur)
    return [int(np.mean(g)) for g in groups]


def build_table(items: list[dict], rules: list[int], img_w: int):
    """열: 괘선 사이 구간. 행: 본문 영역 텍스트를 y-중심으로 군집."""
    inner = [x for x in rules if 0.05 * img_w < x < 0.95 * img_w]  # 바깥 테두리(좌 x≈33 등) 제외
    bounds = [0, *inner, img_w]
    col_of = lambda xc: max(i for i in range(len(bounds) - 1) if xc >= bounds[i])
    header_y = 100   # 열 제목 영역 하단(px) — 이미지에서 '1910' 아래 가로줄
    foot_y = 455     # 각주 시작(px)
    header, body, foot = [], [], []
    for it in items:
        x1, y1, x2, y2 = it["bbox_xyxy"]
        it["col"] = min(col_of((x1 + x2) / 2), 2)
        yc = (y1 + y2) / 2
        (header if yc < header_y else foot if yc > foot_y else body).append(it)
    body.sort(key=lambda t: (t["bbox_xyxy"][1] + t["bbox_xyxy"][3]) / 2)
    rows, tol = [], 11
    for it in body:
        yc = (it["bbox_xyxy"][1] + it["bbox_xyxy"][3]) / 2
        if rows and abs(rows[-1]["yc"] - yc) <= tol:
            rows[-1]["cells"].append(it)
        else:
            rows.append({"yc": yc, "cells": [it]})
    return bounds, header, rows, foot


NUM_RE = re.compile(r"^[\d][\d,\s]*$")


def to_int(s: str) -> int | None:
    t = s.replace(" ", "").replace(",", "")
    return int(t) if t.isdigit() else None


def clean_label(s: str) -> str:
    return re.sub(r"[\s.]+$", "", s).strip()


# ---------------------------------------------------------------- 메인
def main() -> None:
    started = dt.datetime.now().astimezone()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INPUT, OUT / "input_census-table.png")  # 입력 사본
    img = cv2.imread(str(INPUT))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) 원시 OCR
    items = run_ocr(INPUT)
    rules = vertical_rules(gray)
    (OUT / "raw_ocr.json").write_text(json.dumps(
        {"input": "input/census-table.png", "image_size_wh": [img.shape[1], img.shape[0]],
         "coord": "픽셀, 입력 PNG 좌상단 원점. polygon=4점(시계방향), bbox_xyxy=외접 사각형",
         "params": {k: str(v.value if hasattr(v, "value") else v) for k, v in OCR_PARAMS.items()},
         "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(OUT / "ocr_text_coords.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "text", "score", "x1", "y1", "x2", "y2"])
        for it in items:
            w.writerow([it["id"], it["text"], it["score"], *it["bbox_xyxy"]])
    (OUT / "ocr_text.txt").write_text("\n".join(it["text"] for it in items), encoding="utf-8")

    # 2) 영역 시각화 (점수 낮은 것은 빨강)
    vis = img.copy()
    for it in items:
        c = (0, 0, 255) if it["score"] < SCORE_REVIEW else (0, 160, 0)
        cv2.polylines(vis, [np.array(it["polygon"], np.int32)], True, c, 2)
        cv2.putText(vis, str(it["id"]), (int(it["bbox_xyxy"][0]), int(it["bbox_xyxy"][1]) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
    for x in rules:
        cv2.line(vis, (x, 0), (x, img.shape[0]), (255, 160, 0), 1)
    cv2.imwrite(str(OUT / "ocr_regions.png"), vis)

    # 3) 표 조립 (원시)
    bounds, header, rows, foot = build_table(items, rules, img.shape[1])
    cells = []  # 셀 단위 출처 기록
    table_raw = []
    for r_i, row in enumerate(rows):
        rec = {"row": r_i, "label": None, "area": None, "pop": None}
        for it in row["cells"]:
            key = ["label", "area", "pop"][it["col"]]
            rec[key] = it
        table_raw.append(rec)

    # 4) 자동 검증 → 확인 필요 표시
    flags: dict[tuple[int, str], list[str]] = {}
    def flag(r, k, msg): flags.setdefault((r, k), []).append(msg)
    for rec in table_raw:
        for k in ("label", "area", "pop"):
            it = rec[k]
            if it is None:
                continue
            if it["score"] < SCORE_REVIEW:
                flag(rec["row"], k, f"OCR 점수 {it['score']:.3f} < {SCORE_REVIEW}")
            if k != "label" and not NUM_RE.match(it["text"]):
                flag(rec["row"], k, f"숫자 칸에 숫자 외 문자: {it['text']!r}")
    for it in header:
        if it["score"] < SCORE_REVIEW:
            flags.setdefault(("header", it["id"]), []).append(f"제목 OCR 점수 {it['score']:.3f}: {it['text']!r}")

    def val(rec, k):
        return to_int(rec[k]["text"]) if rec[k] else None

    def arithmetic(get) -> list[dict]:
        """합계 관계 검사: 전체 = 본토 + 해외영토, 해외영토 = 하위 8개 합."""
        res = []
        for k, name in (("area", "면적"), ("pop", "인구")):
            t, us, out_ = (get(table_raw[i], k) for i in (0, 1, 2))
            kids = [get(r, k) for r in table_raw[3:]]
            s = sum(v for v in kids if v is not None)
            res.append({"check": f"{name}: 전체 = 본토 + 해외영토", "lhs": t, "rhs": (us or 0) + (out_ or 0),
                        "ok": t == (us or 0) + (out_ or 0)})
            res.append({"check": f"{name}: 해외영토 = 하위 지역 합", "lhs": out_, "rhs": s, "ok": out_ == s})
        return res

    raw_checks = arithmetic(val)
    for c in raw_checks:
        if not c["ok"]:
            k = "area" if c["check"].startswith("면적") else "pop"
            for r in range(2, len(table_raw)):
                flag(r, k, f"합계 검사 불일치({c['check']}: {c['lhs']:,} ≠ {c['rhs']:,}) — 이 열 전체 재확인")

    # 5) 사람 검토 수정 적용
    corr = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
    for c in corr["corrections"]:  # 검토 기록이 이번 OCR 결과와 같은 대상을 가리키는지 확인
        if items[c["ocr_id"]]["text"] != c["expected_raw"]:
            raise SystemExit(f"corrections.json 불일치: OCR#{c['ocr_id']} = {items[c['ocr_id']]['text']!r}, "
                             f"기대값 {c['expected_raw']!r} — OCR 결과가 바뀌었으니 다시 검토하세요.")
    corr_by_ocr_id = {c["ocr_id"]: c for c in corr["corrections"] if c.get("scope") != "header"}
    final_rows = []
    for rec in table_raw:
        row = {}
        for k in ("label", "area", "pop"):
            it = rec[k]
            raw = it["text"] if it else ""
            c = corr_by_ocr_id.get(it["id"]) if it else None
            fixed = c["corrected"] if c else raw
            status = "수정(원본 대조)" if c else ("값 없음(원본 점선)" if it is None else "OCR 그대로(원본 대조 일치)")
            row[k] = {"raw": raw, "value": fixed, "status": status, "footnote": c.get("footnote") if c else None,
                      "bbox": it["bbox_xyxy"] if it else None, "score": it["score"] if it else None,
                      "ocr_id": it["id"] if it else None, "auto_flags": flags.get((rec["row"], k), [])}
        final_rows.append(row)
    # 사람 검토 후에도 확인 필요로 남기는 칸
    unresolved = {(u["row_label"], u["col"]): u for u in corr.get("needs_review", [])}

    def fval(rec, k):
        return to_int(rec[k]["value"]) if rec[k]["value"] else None
    final_checks = arithmetic(lambda rec, k: to_int(final_rows[rec["row"]][k]["value"]) if final_rows[rec["row"]][k]["value"] else None)

    # 6) 결과 파일
    colname = {"label": "area_name", "area": "gross_area_sq_mi", "pop": "population"}
    with open(OUT / "table_raw.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["area_name", "gross_area_sq_mi", "population"])
        for r in final_rows:
            w.writerow([r["label"]["raw"], r["area"]["raw"], r["pop"]["raw"]])
    with open(OUT / "table.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["area_name", "level", "gross_area_sq_mi", "population", "population_note", "review_status"])
        for r in final_rows:
            name = clean_label(r["label"]["value"])
            fn = r["pop"]["footnote"]
            st = "확인 필요" if any((name, k) in unresolved for k in colname.values()) else \
                 ("수정됨" if any(r[k]["status"].startswith("수정") for k in r) else "확인됨")
            w.writerow([name, ROW_LEVEL.get(name, "item"), fval(r, "area") if r["area"]["value"] else "",
                        fval(r, "pop"), fn or "", st])
    with open(OUT / "cells.csv", "w", encoding="utf-8-sig", newline="") as f:  # 셀별 출처(위치) 연결
        w = csv.writer(f)
        w.writerow(["row", "area_name", "column", "ocr_raw", "final_value", "ocr_score", "bbox_x1", "bbox_y1",
                    "bbox_x2", "bbox_y2", "source_page", "status", "auto_flags"])
        for i, r in enumerate(final_rows):
            name = clean_label(r["label"]["value"])
            for k in ("label", "area", "pop"):
                c = r[k]
                bb = c["bbox"] or ["", "", "", ""]
                st = "확인 필요" if (name, colname[k]) in unresolved else c["status"]
                w.writerow([i, name, colname[k], c["raw"], c["value"], c["score"] or "", *bb,
                            f"PDF p.{SOURCE['pdf_page']} / 인쇄 p.{SOURCE['printed_page']}", st, " | ".join(c["auto_flags"])])

    # 검토 오버레이: 초록=OCR 그대로 확인, 주황=수정(번호), 빨강=확인 필요. 수정 내용은 아래 범례 패널에 표기
    rv = img.copy()
    legend = []
    def mark(bbox, color, label=None):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(rv, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color, 1 if label is None else 2)
        if label:
            cv2.circle(rv, (x1 - 12, (y1 + y2) // 2), 10, color, -1)
            cv2.putText(rv, label, (x1 - 17, (y1 + y2) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for r in final_rows:
        name = clean_label(r["label"]["value"])
        for k in ("label", "area", "pop"):
            c = r[k]
            if not c["bbox"]:
                continue
            if (name, colname[k]) in unresolved:
                legend.append(((0, 0, 230), f"CHECK {name} / {colname[k]}"))
                mark(c["bbox"], (0, 0, 230), f"C{len(legend)}")
            elif c["status"].startswith("수정"):
                legend.append(((0, 120, 255), f"FIX  {name} / {colname[k]}: OCR '{c['raw']}' -> '{c['value']}' (superscript footnote)"))
                mark(c["bbox"], (0, 120, 255), str(len(legend)))
            else:
                mark(c["bbox"], (0, 170, 0))
    for c in corr["corrections"]:
        if c.get("scope") == "header":
            it = items[c["ocr_id"]]
            legend.append(((0, 120, 255), f"FIX  title: OCR '{it['text']}' -> '{c['corrected']}'"))
            x1, y1, x2, y2 = map(int, it["bbox_xyxy"])
            mark([x1 + 4, y1, x2, y2], (0, 120, 255), str(len(legend)))
    legend.append(((0, 170, 0), "thin green box = OCR text kept as-is after comparing with original"))
    panel = np.full((28 * len(legend) + 16, rv.shape[1], 3), 255, np.uint8)
    for i, (col, txt) in enumerate(legend):
        txt = f"[{i + 1}] {txt}" if txt.startswith(("FIX", "CHECK")) else txt
        cv2.putText(panel, txt, (10, 26 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
    rv = np.vstack([rv, panel])
    cv2.imwrite(str(OUT / "review_overlay.png"), rv)
    # 수정 칸 확대 증거 이미지
    ev = OUT / "review_evidence"; ev.mkdir(exist_ok=True)
    for c in corr["corrections"]:
        x1, y1, x2, y2 = map(int, items[c["ocr_id"]]["bbox_xyxy"])
        crop = img[max(y1 - 6, 0):y2 + 6, max(x1 - 6, 0):x2 + 6]
        cv2.imwrite(str(ev / f"ocr{c['ocr_id']:02d}.png"), cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))

    report = {"raw_checks": raw_checks, "final_checks": final_checks,
              "auto_flags": [{"where": str(k), "reasons": v} for k, v in flags.items()],
              "corrections": corr["corrections"], "needs_review": corr.get("needs_review", [])}
    (OUT / "review_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_document(final_rows, header, foot, items, raw_checks, final_checks, flags, corr)
    write_manifest(started)
    print("raw checks  :", [(c["check"], c["ok"]) for c in raw_checks])
    print("final checks:", [(c["check"], c["ok"]) for c in final_checks])
    print("auto flags  :", len(flags), "/ corrections:", len(corr["corrections"]), "/ still needs review:", len(unresolved))


def write_document(rows, header, foot, items, raw_checks, final_checks, flags, corr) -> None:
    fmt = lambda v: f"{v:,}" if isinstance(v, int) else v
    L = [f"# Table 1 — Population of the United States and of its Outlying Possessions (1910)", "",
         f"- 출처: {SOURCE['publisher']}, *{SOURCE['title']}*",
         f"- 원문: {SOURCE['url']} — PDF {SOURCE['pdf_page']}쪽 / 인쇄 {SOURCE['printed_page']}쪽",
         f"- 입력 이미지: `input/census-table.png` (PDF 좌표 {SOURCE['crop_rect_pt']} pt 영역, {SOURCE['dpi']} dpi) · 공유 조건: {SOURCE['license']}",
         "- 각 값의 원본 위치(픽셀 bbox)는 `cells.csv`, 시각 확인은 `review_overlay.png` 참고", "",
         "## 복원한 표", "",
         "| AREA | Gross area (land and water) in square miles | Population: 1910 | 검토 |",
         "|---|---:|---:|---|"]
    for r in rows:
        name = clean_label(r["label"]["value"])
        indent = "" if name in ROW_LEVEL else "&nbsp;&nbsp;&nbsp;&nbsp;"
        bold = "**" if ROW_LEVEL.get(name) == "total" else ""
        a = fmt(to_int(r["area"]["value"])) if r["area"]["value"] else "—"
        p = fmt(to_int(r["pop"]["value"]))
        if r["pop"]["footnote"]:
            p = f"{p} <sup>{r['pop']['footnote'][0]}</sup>"
        st = "수정" if any(r[k]["status"].startswith("수정") for k in r) else "확인"
        L.append(f"| {indent}{bold}{name}{bold} | {bold}{a}{bold} | {bold}{p}{bold} | {st} |")
    L += ["", "<sup>1</sup> Population in 1903. &nbsp; <sup>2</sup> Population in 1912. (원본 각주)",
          "", "— : 원본에 점선만 있고 값이 없는 칸(0이 아님).", "",
          "## 검증 1 — 합계 산술 검사", "",
          "| 검사 | OCR 원시값 | 결과 | 사람 검토 후 | 결과 |", "|---|---|---|---|---|"]
    for a, b in zip(raw_checks, final_checks):
        L.append(f"| {a['check']} | {a['lhs']:,} vs {a['rhs']:,} | {'✅' if a['ok'] else '❌'} | "
                 f"{b['lhs']:,} vs {b['rhs']:,} | {'✅' if b['ok'] else '❌'} |")
    L += ["", "## 검증 2 — 자동 '확인 필요' 표시 (사람 검토 전)", "",
          f"규칙: OCR 점수 < {SCORE_REVIEW}, 숫자 칸의 숫자 외 문자, 합계 검사 불일치 열", "",
          "| 위치 | 사유 |", "|---|---|"]
    for k, v in flags.items():
        where = f"제목 영역 OCR#{k[1]}" if k[0] == "header" else \
            f"{clean_label(rows[k[0]]['label']['value'])} / {dict(label='행 이름', area='면적', pop='인구')[k[1]]}"
        L.append(f"| {where} | {'; '.join(v)} |")
    L += ["", "## 수정 내역 (원시 인식 → 원본 대조 후)", "",
          "| OCR# | 위치(행/열) | 원시 인식 | 수정값 | 원본 bbox(px) | 근거 |", "|---|---|---|---|---|---|"]
    for c in corr["corrections"]:
        it = items[c["ocr_id"]]
        L.append(f"| {c['ocr_id']} | {c['where']} | `{it['text']}` ({it['score']:.3f}) | `{c['corrected']}` | "
                 f"{[int(v) for v in it['bbox_xyxy']]} | {c['reason']} |")
    L += ["", "확대 이미지: `review_evidence/ocr<번호>.png`", "", "## 확인 필요로 남긴 항목", ""]
    if corr.get("needs_review"):
        for u in corr["needs_review"]:
            L.append(f"- **{u['row_label']} / {u['col']}**: {u['reason']}")
    else:
        L.append("- 없음 (위 자동 표시 항목은 모두 원본 확대 대조 + 합계 검사로 해소)")
    L += ["", "## 처리 범위와 한계", ""] + [f"- {s}" for s in corr["scope_notes"]]
    (OUT / "document.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_manifest(started) -> None:
    import rapidocr, onnxruntime
    mdir = Path(rapidocr.__file__).parent / "models"
    models = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(mdir.glob("*.onnx"))}
    manifest = {
        "assignment": "B. OCR로 표 이미지를 다시 쓰기 (Census Table 1 선택)",
        "run_started": started.isoformat(timespec="seconds"),
        "command": ".venv-ocr\\Scripts\\python.exe ocr\\run_ocr.py",
        "source": SOURCE,
        "inputs": [{"file": "input/census-table.png", "sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                    "made_by": "ocr/extract_table_image.py (원문 PDF → 200dpi 크롭)"}],
        "execution_location": {"where": "로컬 PC (Windows, CPU 전용)", "os": platform.platform(),
                               "python": sys.version.split()[0]},
        "engine": {
            "runtime": "RapidOCR + ONNX Runtime (CPU)",
            "why_not_paddlepaddle_local": "이 PC의 Windows 스마트 앱 컨트롤이 paddlepaddle의 libpaddle DLL 로드를 차단(WinError 4551). "
                                          "같은 PaddleOCR PP-OCR 가중치의 ONNX 변환본으로 로컬 실행. 원본 PaddleOCR은 ocr/paddleocr_colab.ipynb (Colab) 제공",
            "det_model": "ch_PP-OCRv5_det_server (다국어 텍스트 검출)",
            "cls_model": "PP-LCNet textline orientation (0°/180°)",
            "rec_model": "en_PP-OCRv5_rec_mobile (영어 인식 가중치)",
            "model_files_sha256_16": models,
        },
        "settings": {"params": {k: str(v.value if hasattr(v, "value") else v) for k, v in OCR_PARAMS.items()},
                     "score_review_threshold": SCORE_REVIEW, "preprocess": "없음(원본 200dpi 그대로)",
                     "layout": "세로 괘선 검출(열) + y-중심 군집(행, 허용오차 11px)"},
        "packages": {p: version(p) for p in ("rapidocr", "onnxruntime", "opencv-python-headless", "numpy", "pymupdf",
                                              "paddleocr", "paddlepaddle")},
        "note_packages": "paddleocr·paddlepaddle은 설치됐으나 로컬에서 libpaddle DLL이 차단되어 실행에는 쓰지 않음",
        "outputs": sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in OUT.rglob("*") if p.is_file()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
