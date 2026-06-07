"""
trackfx web UI — a minimal demo of the engine.

Pick a transform, hit Run, and watch the original vs. the result. Right now the
"where" comes from the local crude_mask stand-in; the same UI will drive SAM 2
(click-to-select an object) once that's wired on the GPU.

Run:
    python app.py            # then open http://localhost:5001
"""

import os

import cv2
from flask import Flask, redirect, render_template_string, request, url_for

import segment
import transforms as T
from video_io import read_video, write_h264

HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
SRC = os.path.join(HERE, "clip_lit.mp4")
MASKS_NPZ = os.path.join(HERE, "masks_lit.npz")
os.makedirs(STATIC, exist_ok=True)

app = Flask(__name__, static_folder=STATIC, static_url_path="/static")

# cache frames, real SAM 2 masks (if present), and a browser copy of the original
_FRAMES, _FPS = read_video(SRC)
_MASKS = segment.load_masks(MASKS_NPZ) if os.path.exists(MASKS_NPZ) else None
write_h264(_FRAMES, os.path.join(STATIC, "original.mp4"), _FPS)


def mask_for(i, fr):
    return _MASKS[i] if _MASKS is not None else segment.crude_mask(fr)


def render(transform, hue=0):
    fn = T.TRANSFORMS[transform]
    kwargs = {"hue": hue} if transform in ("tint", "recolor") else {}
    out = [fn(fr, mask_for(i, fr), **kwargs) for i, fr in enumerate(_FRAMES)]
    name = f"result_{transform}.mp4"
    write_h264(out, os.path.join(STATIC, name), _FPS)
    return name


# pre-render a red-car showcase so the page shows the payoff on first load
if not os.path.exists(os.path.join(STATIC, "result_tint.mp4")):
    render("tint", hue=0)

PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>trackfx</title>
<style>
 body{font-family:-apple-system,Segoe UI,sans-serif;max-width:980px;margin:32px auto;
      padding:0 16px;background:#0f1115;color:#e7e9ee}
 h1{font-weight:700;letter-spacing:-.5px} .sub{color:#9aa3b2;margin-top:-8px}
 .row{display:flex;gap:20px;flex-wrap:wrap} .col{flex:1;min-width:300px}
 video{width:100%;border-radius:10px;background:#000}
 form{background:#171a21;padding:16px;border-radius:10px;margin:18px 0;display:flex;
      gap:14px;align-items:end;flex-wrap:wrap}
 label{display:block;font-size:12px;color:#9aa3b2;margin-bottom:4px}
 select,input{background:#0f1115;color:#e7e9ee;border:1px solid #2a2f3a;border-radius:8px;
      padding:8px 10px}
 button{background:#3b82f6;color:#fff;border:0;border-radius:8px;padding:10px 18px;
      font-weight:600;cursor:pointer} button:hover{background:#2f6fe0}
 .cap{font-size:13px;color:#9aa3b2;margin:6px 0}
</style></head><body>
 <h1>trackfx</h1>
 <p class="sub">Select an object in a video → transform it (tracked across frames).</p>
 <form method="post" action="/run">
   <div><label>Transform</label>
     <select name="transform">
       {% for t in transforms %}<option value="{{t}}" {{'selected' if t==sel else ''}}>{{t}}</option>{% endfor %}
     </select></div>
   <div><label>Hue (tint/recolor)</label>
     <input type="number" name="hue" value="{{hue}}" min="0" max="179" style="width:80px"></div>
   <button type="submit">Run ▶</button>
   {% if took %}<span class="cap">done in {{took}}s</span>{% endif %}
 </form>
 <div class="row">
   <div class="col"><div class="cap">original</div>
     <video src="/static/original.mp4" controls loop muted autoplay></video></div>
   <div class="col"><div class="cap">result{{' — '+sel if result else ''}}</div>
     {% if result %}<video src="/static/{{result}}?v={{ver}}" controls loop muted autoplay></video>
     {% else %}<div class="cap">run a transform →</div>{% endif %}</div>
 </div>
 <p class="cap">Object tracked with <b>SAM 2</b> from a single click on the car —
   the person stays untouched. Tip: try <b>tint</b> hue 0 (red) / 120 (blue), or
   <b>remove</b> to erase it.</p>
</body></html>
"""


@app.route("/")
def index():
    default_r = "result_tint.mp4" if os.path.exists(
        os.path.join(STATIC, "result_tint.mp4")) else None
    return render_template_string(
        PAGE, transforms=list(T.TRANSFORMS), sel=request.args.get("t", "tint"),
        hue=request.args.get("hue", 0), result=request.args.get("r", default_r),
        took=request.args.get("s"), ver=request.args.get("v", "0"))


@app.route("/run", methods=["POST"])
def run():
    import time
    t0 = time.time()
    transform = request.form.get("transform", "tint")
    hue = int(request.form.get("hue", 0))
    name = render(transform, hue)
    took = round(time.time() - t0, 1)
    return redirect(url_for("index", t=transform, hue=hue, r=name, s=took,
                            v=int(time.time())))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
