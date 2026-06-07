"""
Run SAM 2 video tracking on a clip from a single click, save per-frame masks.

This runs on a CUDA GPU (RunPod A100). Given a video and one (x,y) click on the
first frame, SAM 2 segments that object and tracks it across every frame. We save
the masks (packed) so transforms can be applied later, cheaply, without a GPU.
Also writes overlay.mp4 (mask drawn on the footage) for a visual sanity check.

    python sam2_masks.py --video clip_shot.mp4 --point 240 200
"""

import argparse
import os

import cv2
import numpy as np
import torch


def extract_frames(video, outdir="frames"):
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(outdir, f"{i:05d}.jpg"), fr)
        frames.append(fr)
        i += 1
    cap.release()
    return frames, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--point", nargs=2, type=int, required=True)
    ap.add_argument("--obj", type=int, default=1)
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--ckpt", default="sam2.1_hiera_large.pt")
    ap.add_argument("--out", default="masks.npz")
    args = ap.parse_args()

    frames, fps = extract_frames(args.video)
    H, W = frames[0].shape[:2]
    N = len(frames)
    print(f"{N} frames {W}x{H} @ {fps:.1f}fps; tracking object at {tuple(args.point)}")

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device="cuda")

    masks = np.zeros((N, H, W), dtype=bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path="frames")
        x, y = args.point
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=args.obj,
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32))
        for fidx, obj_ids, mlogits in predictor.propagate_in_video(state):
            m = (mlogits[0] > 0.0).squeeze().cpu().numpy()
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            masks[fidx] = m

    np.savez_compressed(args.out, masks=np.packbits(masks),
                        shape=np.array([N, H, W]), fps=fps)
    print(f"mean mask coverage: {masks.mean():.3f}")

    # overlay video for a visual check
    import imageio.v2 as imageio
    wtr = imageio.get_writer("overlay.mp4", fps=fps, codec="libx264",
                             macro_block_size=None, ffmpeg_log_level="error",
                             output_params=["-pix_fmt", "yuv420p"])
    for i in range(N):
        fr = frames[i].copy()
        m = masks[i]
        fr[m] = (0.45 * fr[m] + np.array([0, 0, 140])).clip(0, 255).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(fr, cnts, -1, (0, 255, 0), 2)
        wtr.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    wtr.close()
    print("saved", args.out, "and overlay.mp4")


if __name__ == "__main__":
    main()
