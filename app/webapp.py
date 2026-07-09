"""Upload/URL video caption tester.

Local web app: paste a video URL or upload a file, get the four ensemble
captions (GPT-5.5 + Gemini-3.1-Pro + Opus-4.5 -> Opus writer). Lets you test
ANY video beyond the three provided samples.

    python -m app.webapp        # serves http://127.0.0.1:8799
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, Response

from app import pipeline as P
from app.ensemble import caption_ensemble, caption_ensemble_frames
from app.models import REQUIRED_STYLES

app = Flask(__name__)
STYLES = list(REQUIRED_STYLES)

PAGE = """<!doctype html><meta charset=utf-8><title>Track 2 - Caption any video</title>
<style>
body{font-family:Inter,system-ui,sans-serif;background:#0e1116;color:#eef3f8;max-width:900px;margin:0 auto;padding:28px}
h1{font-size:24px}h2{color:#4fd1b5;font-size:13px;text-transform:uppercase;letter-spacing:1px;margin:18px 0 6px}
input,button{font:inherit;padding:10px 12px;border-radius:8px;border:1px solid #303b48;background:#171d25;color:#eef3f8}
input[type=text]{width:70%}button{background:#4fd1b5;color:#04211b;font-weight:700;cursor:pointer;border:none}
.card{border:1px solid #303b48;border-radius:10px;padding:14px 16px;margin:10px 0;background:#141a22}
.s{color:#78a6ff;font-weight:700;font-size:12px;text-transform:uppercase}
.row{display:flex;gap:10px;align-items:center;margin:8px 0;flex-wrap:wrap}
#out{margin-top:18px}.muted{color:#a9b4c0;font-size:13px}.err{color:#ff6f6f}
</style>
<h1>Track 2 - Caption any video</h1>
<p class=muted>Ensemble of frontier vision models writes four styled captions. Paste a direct video URL (.mp4) or upload a file.</p>
<div class=row><input id=url type=text placeholder="https://.../clip.mp4"><button onclick=go()>Caption URL</button></div>
<div class=row><input id=file type=file accept="video/*"><button onclick=goFile()>Caption upload</button></div>
<div id=out></div>
<script>
async function render(p){const o=document.getElementById('out');o.innerHTML='<p class=muted>Analyzing with the model ensemble... (~1 min)</p>';
 try{const r=await p;const d=await r.json();if(d.error){o.innerHTML='<p class=err>'+d.error+'</p>';return;}
 o.innerHTML=Object.entries(d.captions).map(([s,c])=>'<div class=card><div class=s>'+s+'</div>'+c+'</div>').join('');}
 catch(e){o.innerHTML='<p class=err>'+e+'</p>';}}
function go(){const u=document.getElementById('url').value;render(fetch('/caption',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:u})}));}
function goFile(){const f=document.getElementById('file').files[0];if(!f)return;const fd=new FormData();fd.append('file',f);render(fetch('/caption',{method:'POST',body:fd}));}
</script>"""


@app.get("/")
def index() -> Response:
    return Response(PAGE, mimetype="text/html")


@app.post("/caption")
def caption():
    try:
        if request.files.get("file"):
            up = request.files["file"]
            with tempfile.TemporaryDirectory() as tmp:
                wd = Path(tmp)
                vp = wd / "upload.mp4"
                up.save(vp)
                frames = P._extract_keyframes(vp, wd, P.NUM_FRAMES, P.FRAME_MAX_EDGE)
                caps = asyncio.run(caption_ensemble_frames(frames, STYLES))
        else:
            url = (request.get_json(silent=True) or {}).get("url", "").strip()
            if not url:
                return jsonify({"error": "provide a video URL or upload a file"}), 400
            caps = asyncio.run(caption_ensemble(url, STYLES))
        return jsonify({"captions": caps})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8799, threaded=True)
