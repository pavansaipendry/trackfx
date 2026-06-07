"""
Phase 4: human parsing — split a person into PARTS (hair / skin / shirt / pants…)
so each can be colored separately. Uses the SegFormer clothes-parsing model
(mattmdjaga/segformer_b2_clothes, ATR labels). Runs on CPU; no GPU needed.

    python phase4_human_parse.py person_frame.png

Outputs phase4.png: original | colored parts | recolored person.
"""

import sys

import cv2
import numpy as np
import torch
from PIL import Image

import transforms as T

ATR = {0: "bg", 1: "hat", 2: "hair", 3: "sunglasses", 4: "upper-clothes",
       5: "skirt", 6: "pants", 7: "dress", 8: "belt", 9: "l-shoe", 10: "r-shoe",
       11: "face", 12: "l-leg", 13: "r-leg", 14: "l-arm", 15: "r-arm", 16: "bag",
       17: "scarf"}

# group raw labels into the parts a user cares about
GROUPS = {
    "hair": [2],
    "skin": [11, 12, 13, 14, 15],      # face + arms + legs
    "top":  [4, 7, 17],                # upper-clothes / dress / scarf
    "bottom": [5, 6],                  # skirt / pants
    "shoes": [9, 10],
    "hat": [1],
}
# a color (OpenCV hue 0-179) to paint each part for the "recolored" view
PAINT = {"hair": (15, 150), "skin": (12, 90), "top": (0, 170),
         "bottom": (120, 170), "shoes": (30, 120), "hat": (60, 150)}


def parse(image_path):
    # load only the model (the SegformerImageProcessor pulls torchvision, which is
    # broken in this env) and preprocess by hand with PIL/numpy.
    from transformers import AutoModelForSemanticSegmentation
    model = AutoModelForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes").eval()
    img = Image.open(image_path).convert("RGB")
    W0, H0 = img.size
    arr = np.asarray(img.resize((512, 512))).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    arr = (arr - mean) / std
    t = torch.from_numpy(arr.transpose(2, 0, 1)[None]).float()
    with torch.no_grad():
        logits = model(pixel_values=t).logits
    up = torch.nn.functional.interpolate(logits, size=(H0, W0),
                                         mode="bilinear", align_corners=False)
    return up.argmax(1)[0].cpu().numpy()  # HxW class-id map


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "person_frame.png"
    seg = parse(path)
    img = cv2.imread(path)
    present = sorted({ATR[i] for i in np.unique(seg) if i in ATR} - {"bg"})
    print("parts detected:", present)

    # colored-parts overlay (each group a distinct color)
    parts_overlay = img.copy().astype(np.float32)
    rng = np.random.default_rng(2)
    recolored = img.copy()
    for name, ids in GROUPS.items():
        m = np.isin(seg, ids)
        if not m.any():
            continue
        parts_overlay[m] = 0.4 * parts_overlay[m] + 0.6 * rng.integers(60, 255, 3)
        hue, sat = PAINT[name]
        recolored = T.tint(recolored, m, hue=hue, sat=sat)

    def lab(im, t):
        im = im.copy()
        cv2.putText(im, t, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return im

    montage = np.hstack([
        lab(img, "original"),
        lab(parts_overlay.clip(0, 255).astype(np.uint8), "parsed parts"),
        lab(recolored, "recolored by part"),
    ])
    cv2.imwrite("phase4.png", montage)
    print("saved phase4.png")


if __name__ == "__main__":
    main()
