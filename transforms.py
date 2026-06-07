"""
Appearance transforms for the engine.

Each transform takes a BGR frame (uint8 HxWx3) and a boolean/0-1 mask (HxW) of
the selected object, and returns a new frame with the transform applied only
inside the mask. These are the "what to do to the tracked object" layer — the
segmentation/tracking (SAM 2) decides *where*, these decide *what*.

All pure OpenCV/numpy — no model needed (except `colorize`, which is a hook for
a learned colorizer like DDColor, wired in on the GPU).
"""

import cv2
import numpy as np


def _m(mask):
    return mask.astype(bool) if mask.dtype != bool else mask


def recolor(frame, mask, hue, sat_scale=1.0):
    """Change the object's color to `hue` (0-179, OpenCV HSV) while keeping its
    original brightness/shading — e.g. red car -> blue car, recolor a shirt."""
    m = _m(mask)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0][m] = hue
    hsv[..., 1][m] = np.clip(hsv[..., 1][m] * sat_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def tint(frame, mask, hue, sat=150):
    """Inject a color into the object while keeping its brightness — works even on
    grayscale footage (sets saturation), unlike `recolor`. A poor-man's colorize."""
    m = _m(mask)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0][m] = hue
    hsv[..., 1][m] = sat
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def blur(frame, mask, ksize=31):
    """Privacy blur the object (faces, plates). ksize must be odd."""
    m = _m(mask)
    k = ksize | 1
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    out = frame.copy()
    out[m] = blurred[m]
    return out


def remove(frame, mask, radius=3):
    """Erase the object by inpainting from surrounding pixels (magic eraser)."""
    m = (_m(mask).astype(np.uint8)) * 255
    return cv2.inpaint(frame, m, radius, cv2.INPAINT_TELEA)


def spotlight(frame, mask):
    """Desaturate everything except the object (draws the eye to it)."""
    m = _m(mask)
    gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = gray.copy()
    out[m] = frame[m]
    return out


def colorize(frame, mask, colorizer=None):
    """Hook for a learned colorizer (e.g. DDColor) applied within the mask.
    Wired in on the GPU; here it's a no-op passthrough so the API is stable."""
    if colorizer is None:
        return frame
    colored = colorizer(frame)
    m = _m(mask)
    out = frame.copy()
    out[m] = colored[m]
    return out


# registry so the pipeline/CLI can pick a transform by name
TRANSFORMS = {
    "recolor": recolor,
    "tint": tint,
    "blur": blur,
    "remove": remove,
    "spotlight": spotlight,
    "colorize": colorize,
}
