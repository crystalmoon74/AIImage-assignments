"""원문 PDF에서 Table 1 영역을 200dpi PNG로 추출 → input/census-table.png

원문: https://www2.census.gov/library/publications/decennial/1910/volume-1/volume-1-p3.pdf
      (PDF 2쪽 = 인쇄 22쪽, 퍼블릭 도메인)
사용법: .venv-ocr\\Scripts\\python.exe ocr\\extract_table_image.py
"""
import urllib.request
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www2.census.gov/library/publications/decennial/1910/volume-1/volume-1-p3.pdf"
PDF = ROOT / "sources" / "census-1910-vol1-p3.pdf"
CLIP = (235, 283, 625, 463)  # PDF 좌표(pt): Table 1 + 각주
DPI = 200

if not PDF.exists():
    PDF.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(URL, PDF)
page = pymupdf.open(PDF)[1]  # 0-based → PDF 2쪽
page.get_pixmap(dpi=DPI, clip=pymupdf.Rect(*CLIP)).save(ROOT / "input" / "census-table.png")
print("saved input/census-table.png")
