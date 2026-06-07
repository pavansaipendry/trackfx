"""
Segmentation/tracking layer — the "where" of the engine.

`crude_mask` is a zero-dependency stand-in used for local development: on plain
backgrounds it grabs the (darker) subject by luminance. It is NOT the real thing
— it has no notion of distinct objects and won't track identities. It exists only
to validate the video pipeline for free.

`SAM2Tracker` is the real interface, filled in on the GPU: given the frames and a
click/box on the first frame, it returns a per-frame mask for that tracked object.
"""

import cv2
import numpy as np


def crude_mask(frame):
    """Rough subject mask via Otsu on luminance (dark subject on light bg)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return th > 0


def load_masks(path):
    """Load per-frame boolean masks saved by sam2_masks.py (packed)."""
    d = np.load(path)
    n, h, w = [int(v) for v in d["shape"]]
    flat = np.unpackbits(d["masks"])[: n * h * w].astype(bool).reshape(n, h, w)
    return [flat[i] for i in range(n)]


class SAM2Tracker:
    """Real segment+track via SAM 2 (wired in on the GPU). API placeholder so the
    pipeline is identical locally and on the pod."""

    def __init__(self, checkpoint=None, model_cfg=None, device="cuda"):
        raise NotImplementedError(
            "SAM2Tracker runs on the GPU pod; use crude_mask for local dev.")

    def track(self, frames, prompt):
        """frames: list[BGR]; prompt: {'frame':0,'point':(x,y)} or box.
        returns: list of boolean masks, one per frame."""
        ...
