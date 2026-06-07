"""
Phase 3: detect labeled objects -> track ALL of them across the video.

YOLO-World finds + labels objects on frame 0; each box seeds a SAM 2 video track,
so every named object gets a mask in every frame. Saves per-object masks + labels
(for color-by-name) and a labeled overlay video.

    python phase3_track_all.py --video clip_lit.mp4
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch

VOCAB = ["car", "person", "wheel", "tire", "windshield", "headlight"]


def extract_frames(video, outdir="frames"):
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cv2.imwrite(f"{outdir}/{i:05d}.jpg", fr)
        frames.append(fr)
        i += 1
    cap.release()
    return frames, fps


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--ckpt", default="sam2.1_hiera_base_plus.pt")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--max_objs", type=int, default=6)
    args = ap.parse_args()

    frames, fps = extract_frames(args.video)
    H, W = frames[0].shape[:2]
    N = len(frames)

    # --- detect + label on frame 0 ---
    from ultralytics import YOLO
    yolo = YOLO("yolov8x-worldv2.pt")
    yolo.set_classes(VOCAB)
    res = yolo.predict(frames[0], conf=args.conf, iou=0.5, verbose=False)[0]
    dets = sorted(zip(res.boxes.xyxy.cpu().numpy(),
                      res.boxes.cls.cpu().numpy().astype(int),
                      res.boxes.conf.cpu().numpy()), key=lambda x: -x[2])
    objs = []
    for box, c, cf in dets:
        if any(iou(box, o["box"]) > 0.6 for o in objs):
            continue
        objs.append({"box": box.astype(np.float32), "label": VOCAB[c],
                     "conf": float(cf)})
        if len(objs) >= args.max_objs:
            break
    print(f"{N} frames {W}x{H}; tracking "
          + ", ".join(f"{o['label']}({o['conf']:.2f})" for o in objs))

    # --- track all with SAM 2 video ---
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device="cuda")
    masks = np.zeros((len(objs), N, H, W), dtype=bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path="frames")
        for oi, o in enumerate(objs):
            predictor.add_new_points_or_box(inference_state=state, frame_idx=0,
                                            obj_id=oi, box=o["box"])
        for fidx, obj_ids, mlogits in predictor.propagate_in_video(state):
            for k, oid in enumerate(obj_ids):
                m = (mlogits[k] > 0.0).squeeze().cpu().numpy()
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                masks[oid, fidx] = m

    np.savez_compressed("masks_all.npz", masks=np.packbits(masks),
                        shape=np.array([len(objs), N, H, W]), fps=fps)
    json.dump([{"label": o["label"], "conf": o["conf"]} for o in objs],
              open("labels.json", "w"))

    # --- labeled overlay video ---
    import imageio.v2 as imageio
    cols = [np.random.default_rng(i + 3).integers(70, 255, 3) for i in range(len(objs))]
    wtr = imageio.get_writer("track_all.mp4", fps=fps, codec="libx264",
                             macro_block_size=None, ffmpeg_log_level="error",
                             output_params=["-pix_fmt", "yuv420p"])
    for f in range(N):
        fr = frames[f].astype(np.float32)
        for oi in range(len(objs)):
            m = masks[oi, f]
            if m.any():
                fr[m] = 0.5 * fr[m] + 0.5 * cols[oi]
        fr = fr.clip(0, 255).astype(np.uint8)
        for oi, o in enumerate(objs):
            m = masks[oi, f]
            if m.any():
                ys, xs = np.where(m)
                cv2.putText(fr, o["label"], (int(xs.mean()) - 10, int(ys.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            tuple(int(c) for c in cols[oi]), 2, cv2.LINE_AA)
        wtr.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    wtr.close()
    print("saved masks_all.npz, labels.json, track_all.mp4")


if __name__ == "__main__":
    main()
