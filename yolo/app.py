"""과제 A — 로컬 웹 화면 (이미지 업로드 + 임계값 슬라이더).

추가 의존성 없이 파이썬 표준 http.server 로 구현했다.
(처음엔 Gradio를 썼으나 이 PC의 Windows 스마트 앱 컨트롤이 pandas DLL 로드를 차단해 교체 — README 참고)

실행 (저장소 루트에서):
    .venv-yolo\\Scripts\\python.exe yolo\\app.py
브라우저에서 http://127.0.0.1:7860 접속. 127.0.0.1 에만 바인딩하므로 외부에서 접속할 수 없다.
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
import yolo_core as yc  # noqa: E402

HOST, PORT = "127.0.0.1", 7860
TMP = Path(tempfile.gettempdir()) / "yolo_service_upload"
TMP.mkdir(exist_ok=True)

PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>YOLO detect + segment</title>
<style>
 body{font-family:system-ui,'Malgun Gothic',sans-serif;margin:16px;background:#f6f7f9;color:#111}
 .row{display:flex;gap:12px;flex-wrap:wrap} .card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;flex:1;min-width:280px}
 img{max-width:100%} pre{max-height:360px;overflow:auto;background:#111;color:#ddd;padding:8px;font-size:12px}
 label{font-weight:600}
</style></head><body>
<h2>YOLO 서비스 — 박스(yolo26n.pt) · 마스크(yolo26n-seg.pt)</h2>
<p>로컬 CPU · imgsz=640. 사진을 고르고 임계값 슬라이더를 움직이면 결과가 갱신됩니다. (과제 비교값 0.25 / 0.50)</p>
<div class="card">
 <input type="file" id="file" accept="image/*">
 <button onclick="useExample('bus.jpg')">예제: bus.jpg</button>
 <button onclick="useExample('desk-objects.jpg')">예제: desk-objects.jpg</button><br><br>
 <label>confidence 임계값: <span id="cv">0.25</span></label>
 <input type="range" id="conf" min="0.05" max="0.95" step="0.05" value="0.25" style="width:300px">
 <span id="status"></span>
</div>
<div class="row" style="margin-top:12px">
 <div class="card"><b>원본</b><br><img id="orig"></div>
 <div class="card"><b>detection: 클래스·점수·박스</b><br><img id="det"></div>
 <div class="card"><b>segmentation: 마스크</b><br><img id="seg"></div>
</div>
<div class="card" style="margin-top:12px"><b>결과 JSON</b> (box_xyxy = 원본 픽셀 좌표)<pre id="json"></pre></div>
<script>
let imgB64=null, example=null;
const conf=document.getElementById('conf');
conf.oninput=()=>document.getElementById('cv').textContent=(+conf.value).toFixed(2);
conf.onchange=run;
document.getElementById('file').onchange=e=>{const r=new FileReader();
  r.onload=()=>{imgB64=r.result.split(',')[1];example=null;document.getElementById('orig').src=r.result;run();};
  r.readAsDataURL(e.target.files[0]);};
// URL 파라미터로 바로 실행: /?example=bus.jpg&conf=0.5 (화면 캡처 재현용)
window.onload=()=>{const q=new URLSearchParams(location.search);
  if(q.get('conf')){conf.value=q.get('conf');conf.oninput();}
  if(q.get('example'))useExample(q.get('example'));};
function useExample(n){example=n;imgB64=null;document.getElementById('orig').src='/example/'+n;run();}
async function run(){ if(!imgB64&&!example)return; document.getElementById('status').textContent='실행 중...';
  const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({image:imgB64,example:example,conf:+conf.value})});
  const j=await res.json(); if(j.error){document.getElementById('status').textContent=j.error;return;}
  document.getElementById('det').src='data:image/jpeg;base64,'+j.det;
  document.getElementById('seg').src='data:image/jpeg;base64,'+j.seg;
  document.getElementById('json').textContent=JSON.stringify(j.result,null,1);
  document.getElementById('status').textContent='완료 (conf='+j.result.conf+')';}
</script></body></html>"""


def _jpg_b64(img) -> str:
    return base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()


def predict(image_path: Path, conf: float) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        return {"error": "이미지를 읽을 수 없습니다."}
    det, seg = yc.detect(image_path, conf), yc.segment(image_path, conf)
    return {
        "det": _jpg_b64(yc.draw_boxes(img, det["detections"])),
        "seg": _jpg_b64(yc.draw_masks(img, seg["instances"], seg["_masks"])),
        "result": {"conf": conf, "imgsz": yc.IMGSZ, "device": yc.DEVICE,
                   "detect": det["detections"], "segment": yc.public(seg)["instances"]},
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path.startswith("/example/"):
            name = Path(self.path).name
            p = yc.ROOT / "input" / name
            if name in ("bus.jpg", "desk-objects.jpg") and p.exists():
                return self._send(200, p.read_bytes(), "image/jpeg")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, b"not found", "text/plain")
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        conf = min(max(float(req.get("conf", 0.25)), 0.01), 0.99)
        if req.get("example") in ("bus.jpg", "desk-objects.jpg"):
            path = yc.ROOT / "input" / req["example"]
        elif req.get("image"):
            path = TMP / "upload.img"
            path.write_bytes(base64.b64decode(req["image"]))
        else:
            return self._send(400, b'{"error":"no image"}', "application/json")
        out = predict(path, conf)
        self._send(200, json.dumps(out, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def log_message(self, fmt, *args):
        sys.stderr.write("[app] " + fmt % args + "\n")


if __name__ == "__main__":
    yc.load_model(yc.DET_WEIGHTS), yc.load_model(yc.SEG_WEIGHTS)
    print(f"YOLO service running: http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
