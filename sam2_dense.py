"""
Push SAM 2 automatic segmentation density from sparse -> max and show the climb.
Renders a montage: same frame at 3 density levels with mask counts, so we can SEE
how close "dense everything" gets.

    python sam2_dense.py --image feas_busy.png
"""

import argparse

import cv2
import numpy as np
import torch

LEVELS = [
    ("sparse  (pps 32)", dict(points_per_side=32, crop_n_layers=0,
                              min_mask_region_area=120, pred_iou_thresh=0.7,
                              stability_score_thresh=0.85)),
    ("medium  (pps 64 +crop)", dict(points_per_side=64, crop_n_layers=1,
                                    crop_n_points_downscale_factor=2,
                                    min_mask_region_area=40, pred_iou_thresh=0.6,
                                    stability_score_thresh=0.8)),
    ("MAX  (pps 128 +2 crops)", dict(points_per_side=128, crop_n_layers=2,
                                     crop_n_points_downscale_factor=2,
                                     min_mask_region_area=15, pred_iou_thresh=0.55,
                                     stability_score_thresh=0.72)),
]


def overlay_of(img, masks):
    masks = sorted(masks, key=lambda m: -m["area"])
    ov = img.copy().astype(np.float32)
    rng = np.random.default_rng(0)
    for m in masks:
        ov[m["segmentation"]] = (0.4 * ov[m["segmentation"]]
                                 + 0.6 * rng.integers(40, 255, 3))
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (255, 255, 255), 1)
    return ov.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--ckpt", default="sam2.1_hiera_base_plus.pt")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    sam2 = build_sam2(args.cfg, args.ckpt, device="cuda", apply_postprocessing=False)

    panels = []
    for name, kw in LEVELS:
        gen = SAM2AutomaticMaskGenerator(sam2, **kw)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = gen.generate(rgb)
        print(f"{name}: {len(masks)} masks")
        ov = overlay_of(img, masks)
        cv2.putText(ov, f"{name}: {len(masks)} regions", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        panels.append(ov)

    montage = np.vstack([np.hstack([img, panels[0]]),
                         np.hstack([panels[1], panels[2]])])
    cv2.imwrite("dense.png", montage)
    print("saved dense.png")


if __name__ == "__main__":
    main()
