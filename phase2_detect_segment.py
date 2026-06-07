"""
Phase 2 prototype: RECOGNIZE -> SEGMENT.

An open-vocabulary detector (YOLO-World) finds and *labels* objects from a
vocabulary; each detected box is then handed to SAM 2 to get a clean mask. Output
is a labeled, segmented overlay — the foundation for "color the car / shirt / drum
by name."

    python phase2_detect_segment.py --image feas_busy.png
"""

import argparse

import cv2
import numpy as np
import torch

# open vocabulary — just names; YOLO-World finds them with no retraining
VOCAB = ["person", "head", "hair", "shirt", "pants", "drum", "drum kit",
         "cymbal", "microphone", "guitar", "keyboard", "car", "wall", "floor"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--ckpt", default="sam2.1_hiera_base_plus.pt")
    ap.add_argument("--conf", type=float, default=0.04)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- RECOGNIZE ---
    from ultralytics import YOLO
    yolo = YOLO("yolov8x-worldv2.pt")
    yolo.set_classes(VOCAB)
    res = yolo.predict(img, conf=args.conf, iou=0.5, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    names = [VOCAB[c] for c in cls]
    print(f"detected {len(boxes)} objects: "
          + ", ".join(f"{n}({c:.2f})" for n, c in zip(names, conf)))

    # --- SEGMENT each detection with SAM 2 ---
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2 = build_sam2(args.cfg, args.ckpt, device="cuda")
    pred = SAM2ImagePredictor(sam2)
    pred.set_image(rgb)

    overlay = img.copy().astype(np.float32)
    rng = np.random.default_rng(1)
    order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) \
        if len(boxes) else []
    cols = {}
    for i in order:  # big -> small so small labels stay on top
        box = boxes[i]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, _ = pred.predict(box=box[None, :], multimask_output=False)
        m = masks[0].astype(bool)
        col = rng.integers(60, 255, 3)
        cols[i] = col
        overlay[m] = 0.5 * overlay[m] + 0.5 * col
    for i in order:  # draw boxes + labels on top
        x1, y1, x2, y2 = boxes[i].astype(int)
        col = cols[i].tolist()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)
        cv2.putText(overlay, f"{names[i]} {conf[i]:.2f}", (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    side = np.hstack([img, overlay.clip(0, 255).astype(np.uint8)])
    cv2.imwrite("phase2.png", side)
    print("saved phase2.png")


if __name__ == "__main__":
    main()
