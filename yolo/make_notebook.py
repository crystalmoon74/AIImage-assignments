"""yolo_service.ipynb 생성 스크립트 (노트북 셀 내용을 코드로 관리)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# 과제 A — YOLO 서비스 노트북\n"
       "이미지 경로를 바꿔 실행하면 **원본 · 박스(detection) · 마스크(segmentation)** 와 JSON을 확인합니다.\n\n"
       "- 모델: `yolo26n.pt`(박스), `yolo26n-seg.pt`(마스크) — COCO 80 클래스\n"
       "- 설정: `imgsz=640`, `device=cpu`, `conf=0.25 / 0.50`\n"
       "- 실행 위치: 로컬 PC(Windows, CPU). 커널은 저장소의 `.venv-yolo` 를 사용"),
    code("import sys, json\nfrom pathlib import Path\nROOT = Path.cwd() if (Path.cwd()/'yolo').exists() else Path.cwd().parent\n"
         "sys.path.insert(0, str(ROOT/'yolo'))\nimport cv2, matplotlib.pyplot as plt\nimport yolo_core as yc\n"
         "print('관찰 가능한 클래스 수:', len(yc.load_model(yc.DET_WEIGHTS).names))"),
    md("## 1) 입력 사진 지정 — 여기만 바꾸면 됩니다\n관찰 대상 클래스: `person`, `bus`(버스 사진), `laptop`, `cup`, `dining table`(책상 사진)"),
    code("IMAGE = ROOT/'input/bus.jpg'   # 예: ROOT/'input/desk-objects.jpg' 또는 내 사진 경로\nCONFS = (0.25, 0.50)"),
    md("## 2) 원본 · 박스 · 마스크 (두 임계값)"),
    code("img = cv2.imread(str(IMAGE))\nres = {}\n"
         "fig, ax = plt.subplots(len(CONFS), 3, figsize=(15, 6*len(CONFS)))\n"
         "for r, c in enumerate(CONFS):\n"
         "    det, seg = yc.detect(IMAGE, c), yc.segment(IMAGE, c)\n"
         "    res[c] = (det, seg)\n"
         "    for k, (im, t) in enumerate([(img, 'original'), (yc.draw_boxes(img, det['detections']), f'detect conf={c}'),\n"
         "                                 (yc.draw_masks(img, seg['instances'], seg['_masks']), f'segment conf={c}')]):\n"
         "        ax[r, k].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax[r, k].set_title(t); ax[r, k].axis('off')\n"
         "plt.tight_layout(); plt.show()"),
    md("## 3) JSON — 화면의 박스 하나를 좌표와 연결해 보기"),
    code("for c in CONFS:\n    print(f'--- conf={c}')\n"
         "    for d in res[c][0]['detections']:\n        print(d)\n"),
    code("print(json.dumps(yc.public(res[0.25][1])['instances'], ensure_ascii=False, indent=1)[:2000])"),
    md("## 4) 임계값 사이(0.25~0.50)에 있던 후보 — 모델 원출력을 낮은 임계값으로 확인\n"
       "두 결과가 같다면, 점수 0.25~0.50 사이 후보가 없었다는 뜻입니다. conf=0.05로 원출력을 보면 무엇이 걸러졌는지 알 수 있습니다."),
    code("for w in (yc.DET_WEIGHTS, yc.SEG_WEIGHTS):\n"
         "    r = yc.load_model(w).predict(str(IMAGE), imgsz=640, conf=0.05, device='cpu', verbose=False)[0]\n"
         "    print(w, [(r.names[int(b.cls)], round(float(b.conf), 3)) for b in r.boxes])"),
    md("## 5) 저장 — 배치 스크립트로 outputs/yolo/ 에 이미지·JSON·마스크·manifest 저장\n"
       "내 사진을 추가하려면 `yolo/run_yolo.py input/bus.jpg input/desk-objects.jpg input/내사진.jpg` 처럼 경로를 넘깁니다."),
    code("import subprocess\nprint(subprocess.run([sys.executable, str(ROOT/'yolo/run_yolo.py')], cwd=ROOT, capture_output=True, text=True, encoding='utf-8').stdout[-1500:])"),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (.venv-yolo)", "language": "python"}
Path(__file__).with_name("yolo_service.ipynb").write_text(nbf.writes(nb), encoding="utf-8")
print("ok")
