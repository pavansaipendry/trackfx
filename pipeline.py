"""
The engine pipeline: video in -> mask each frame -> apply a transform -> video out.

The masker is pluggable: `crude_mask` (local, free) now; SAM2Tracker (GPU) later.
With SAM2 the masks come from one click and are tracked per-object; the rest of
the pipeline is identical.

CLI:
    python pipeline.py clip.mp4 out.mp4 --transform blur
    python pipeline.py clip.mp4 out.mp4 --transform tint --hue 120
"""

import argparse

import segment
import transforms as T
from video_io import read_video, write_video


def run(video_in, video_out, transform, masks=None, masker=segment.crude_mask,
        **kwargs):
    frames, fps = read_video(video_in)
    fn = T.TRANSFORMS[transform]
    out = []
    for i, fr in enumerate(frames):
        mask = masks[i] if masks is not None else masker(fr)
        out.append(fn(fr, mask, **kwargs))
    write_video(out, video_out, fps)
    return len(out), fps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video_in")
    p.add_argument("video_out")
    p.add_argument("--transform", default="blur", choices=list(T.TRANSFORMS))
    p.add_argument("--hue", type=int, default=120)
    p.add_argument("--ksize", type=int, default=31)
    args = p.parse_args()

    kwargs = {}
    if args.transform in ("recolor", "tint"):
        kwargs["hue"] = args.hue
    if args.transform == "blur":
        kwargs["ksize"] = args.ksize

    n, fps = run(args.video_in, args.video_out, args.transform, **kwargs)
    print(f"wrote {args.video_out}: {n} frames @ {fps:.1f}fps "
          f"(transform={args.transform})")


if __name__ == "__main__":
    main()
