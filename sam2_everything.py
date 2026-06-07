"""
Feasibility test: SAM 2 automatic "segment everything" on one frame.

Question we're answering: does it carve the scene into PARTS (hair, skin, shirt,
pants, car body, tires, windshield) — or just whole objects? We render every
mask in a distinct color so we can eyeball the granularity.

    python sam2_everything.py --image feas_frame.png --pps 64
"""

import argparse

import cv2
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--ckpt", default="sam2.1_hiera_base_plus.pt")
    ap.add_argument("--pps", type=int, default=64, help="points per side (granularity)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    sam2 = build_sam2(args.cfg, args.ckpt, device="cuda", apply_postprocessing=False)
    gen = SAM2AutomaticMaskGenerator(
        sam2, points_per_side=args.pps, pred_iou_thresh=0.7,
        stability_score_thresh=0.88, crop_n_layers=1,
        crop_n_points_downscale_factor=2, min_mask_region_area=60)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks = gen.generate(rgb)
    print("num masks:", len(masks))

    masks = sorted(masks, key=lambda m: -m["area"])
    overlay = img.copy().astype(np.float32)
    rng = np.random.default_rng(0)
    for m in masks:
        overlay[m["segmentation"]] = (0.45 * overlay[m["segmentation"]]
                                      + 0.55 * rng.integers(50, 255, 3))
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 1)
    cv2.putText(overlay, f"{len(masks)} masks", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    side = np.hstack([img, overlay.clip(0, 255).astype(np.uint8)])
    cv2.imwrite("everything.png", side)
    print("saved everything.png")


if __name__ == "__main__":
    main()
