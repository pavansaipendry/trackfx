"""
trackfx studio — interactive color-by-name editor.

Loads a clip + its tracked, labeled objects (from phase3_track_all.py:
masks_all.npz + labels.json), shows one color picker per object, and renders the
recolored video. Detection/tracking/parsing happen offline on a GPU; everything
here is instant and local because the masks are cached.

Run:
    python app.py            # http://localhost:5001
"""

import json
import os
import time

import cv2
import numpy as np
from flask import Flask, redirect, render_template_string, request, url_for

import transforms as T
from video_io import read_video, write_h264

HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
SRC = os.path.join(HERE, "clip_lit.mp4")
os.makedirs(STATIC, exist_ok=True)

app = Flask(__name__, static_folder=STATIC, static_url_path="/static")

_FRAMES, _FPS = read_video(SRC)
_d = np.load(os.path.join(HERE, "masks_all.npz"))
_K, _N, _H, _W = [int(v) for v in _d["shape"]]
_MASKS = np.unpackbits(_d["masks"])[: _K * _N * _H * _W].astype(bool).reshape(_K, _N, _H, _W)
_LABELS = [o["label"] for o in json.load(open(os.path.join(HERE, "labels.json")))]
DEFAULT_COLORS = ["#2e6df5", "#e23b3b", "#26b07a", "#e2a23b", "#9b51e0", "#f55fa0"]

write_h264(_FRAMES, os.path.join(STATIC, "original.mp4"), _FPS)


def hex_to_hsv(hx):
    hx = hx.lstrip("#")
    r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1])


def render(colors, enabled):
    """colors: list[hex] per object; enabled: list[bool]. Returns output filename."""
    out = [fr.copy() for fr in _FRAMES]
    for k in range(_K):
        if not enabled[k]:
            continue
        hue, sat = hex_to_hsv(colors[k])
        sat = max(sat, 90)  # keep it visible on B&W
        for f in range(_N):
            m = _MASKS[k, f]
            if m.any():
                out[f] = T.tint(out[f], m, hue=hue, sat=sat)
    name = f"studio_{int(time.time())}.mp4"
    write_h264(out, os.path.join(STATIC, name), _FPS)
    return name


PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>trackfx studio</title>
<style>
 body{font-family:-apple-system,Segoe UI,sans-serif;max-width:1040px;margin:30px auto;
      padding:0 16px;background:#0f1115;color:#e7e9ee}
 h1{letter-spacing:-.5px;margin-bottom:0} .sub{color:#9aa3b2;margin-top:4px}
 .row{display:flex;gap:20px;flex-wrap:wrap;margin-top:18px}
 .col{flex:1;min-width:320px} video{width:100%;border-radius:10px;background:#000}
 .cap{font-size:13px;color:#9aa3b2;margin:6px 0}
 form{background:#171a21;padding:16px 18px;border-radius:12px;margin-top:16px}
 .obj{display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid #232733}
 .obj:last-child{border-bottom:0}
 .name{font-weight:600;text-transform:capitalize;min-width:90px}
 input[type=color]{width:46px;height:30px;border:1px solid #2a2f3a;border-radius:6px;background:#0f1115}
 button{background:#3b82f6;color:#fff;border:0;border-radius:8px;padding:10px 20px;
        font-weight:600;cursor:pointer;margin-top:12px} button:hover{background:#2f6fe0}
 label.sw{font-size:13px;color:#9aa3b2}
</style></head><body>
 <h1>trackfx studio</h1>
 <p class="sub">Detected & tracked objects — pick a color for each, then render.</p>
 <form method="post" action="/render">
   {% for i in range(labels|length) %}
   <div class="obj">
     <span class="name">{{labels[i]}}</span>
     <input type="color" name="color{{i}}" value="{{colors[i]}}">
     <label class="sw"><input type="checkbox" name="on{{i}}" {{'checked' if enabled[i] else ''}}> recolor</label>
   </div>
   {% endfor %}
   <button type="submit">Render ▶</button>
   {% if took %}<span class="cap">&nbsp;rendered in {{took}}s</span>{% endif %}
 </form>
 <div class="row">
   <div class="col"><div class="cap">original</div>
     <video src="/static/original.mp4" controls loop muted autoplay></video></div>
   <div class="col"><div class="cap">result</div>
     {% if result %}<video src="/static/{{result}}" controls loop muted autoplay></video>
     {% else %}<div class="cap">pick colors → Render →</div>{% endif %}</div>
 </div>
 <p class="cap">Objects auto-detected (YOLO-World) and tracked (SAM 2) offline;
   editing here is instant because masks are cached.</p>
</body></html>
"""


@app.route("/")
def index():
    colors = [request.args.get(f"c{i}", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
              for i in range(_K)]
    enabled = [request.args.get(f"on{i}", "1") == "1" for i in range(_K)]
    return render_template_string(
        PAGE, labels=_LABELS, colors=colors, enabled=enabled,
        result=request.args.get("r"), took=request.args.get("s"))


@app.route("/render", methods=["POST"])
def do_render():
    t0 = time.time()
    colors = [request.form.get(f"color{i}", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
              for i in range(_K)]
    enabled = [request.form.get(f"on{i}") is not None for i in range(_K)]
    name = render(colors, enabled)
    args = {"r": name, "s": round(time.time() - t0, 1)}
    for i in range(_K):
        args[f"c{i}"] = colors[i]
        args[f"on{i}"] = "1" if enabled[i] else "0"
    return redirect(url_for("index", **args))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
