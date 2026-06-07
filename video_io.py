"""Read/write video frames with OpenCV (no ffmpeg binary needed for the demo).

Audio is dropped here; on the GPU run we mux the original audio back with ffmpeg
(available there). For development this is enough to see the transform working.
"""

import cv2


def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, fps


def write_video(frames, path, fps):
    if not frames:
        raise ValueError("no frames to write")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        vw.write(fr)
    vw.release()
    return path


def write_h264(frames, path, fps):
    """Write a browser-playable H.264 mp4 (yuv420p) via the bundled ffmpeg."""
    import imageio.v2 as imageio
    w = imageio.get_writer(path, fps=fps, codec="libx264",
                           macro_block_size=None, ffmpeg_log_level="error",
                           output_params=["-pix_fmt", "yuv420p"])
    for fr in frames:
        w.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    w.close()
    return path
