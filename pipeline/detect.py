"""
detect.py - run the YOLO flag detector on sampled frames of a GoPro video.

Returns one row per detection:
    frame_idx, frame_time, u, v, conf, class, crop, vid_w, vid_h
frame_time uses the telemetry clock (t=0 = video start); (u, v) is the bbox centre
in native-resolution pixels, which is what the camera calibration expects.
A padded crop of every detection is saved for the submission.
"""

import os
import queue
import logging
import threading

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best.pt")
IMGSZ = 1280        # default YOLO input size: the model was trained at 1280. Smaller shrinks a
                    # 2 m flag at 50-100 m AGL toward undetectable.
SAMPLE_FPS = 1.0    # default frames analysed per second of video
CONF = 0.4          # minimum detection confidence
CROP_PAD = 0.15     # crop padding, as a fraction of the box size
COLUMNS = ["frame_idx", "frame_time", "u", "v", "conf", "class", "crop", "vid_w", "vid_h"]


def _load_model(gpu):
    """Return (model, device, fp16 kwargs)."""
    model = YOLO(MODEL)
    if not gpu:
        return model, "cpu", {}
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu was given but PyTorch sees no CUDA GPU. Install the CUDA "
                           "build of torch (see README) or run without --gpu.")
    # FP16 on the GPU (~2x faster, half the VRAM). The kwarg name differs between
    # ultralytics versions, so probe on a dummy frame (this also warms the model up).
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    for kw in ({"quantize": "fp16"}, {"half": True}):
        try:
            model(dummy, imgsz=64, device=0, verbose=False, **kw)
            return model, 0, kw
        except Exception:
            continue
    return model, 0, {}


def _save_crop(frame, x1, y1, x2, y2, path):
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    xa, ya = max(0, int(round(x1 - bw * CROP_PAD))), max(0, int(round(y1 - bh * CROP_PAD)))
    xb, yb = min(w, int(round(x2 + bw * CROP_PAD))), min(h, int(round(y2 + bh * CROP_PAD)))
    cv2.imwrite(path, frame[ya:yb, xa:xb], [cv2.IMWRITE_JPEG_QUALITY, 95])


def detect(video, crops_dir, start=None, end=None, gpu=False, sample_fps=SAMPLE_FPS, imgsz=IMGSZ):
    """Detect flags in `video` between `start` and `end` seconds (None = whole video),
    analysing `sample_fps` frames per second at YOLO input size `imgsz`."""
    model, device, fp16 = _load_model(gpu)
    logging.info("YOLO: device=%s fp16=%s imgsz=%d", device, bool(fp16), imgsz)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not fps or fps <= 0:
        raise RuntimeError(f"Could not open video: {video}")
    vid_w, vid_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(round(fps / sample_fps)))
    logging.info("Video: %.3f fps, %dx%d | analysing every %d frame(s) (~%.1f per second)",
                 fps, vid_w, vid_h, stride, fps / stride)
    os.makedirs(crops_dir, exist_ok=True)

    rows = []
    processed = 0

    def run_frame(frame, fidx, ftime):
        nonlocal processed
        processed += 1
        boxes = model(frame, conf=CONF, imgsz=imgsz, device=device, verbose=False, **fp16)[0].boxes
        for k, b in enumerate(boxes):
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            crop = f"crop_{fidx:06d}_{k}.jpg"
            _save_crop(frame, x1, y1, x2, y2, os.path.join(crops_dir, crop))
            rows.append([fidx, round(ftime, 4), round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2),
                         round(float(b.conf.item()), 4), model.names[int(b.cls.item())], crop,
                         vid_w, vid_h])

    # Decoding (this thread) and inference (worker thread) overlap: both release the GIL.
    # The bounded queue caps RAM (a 5.3K frame is ~48 MB).
    frames = queue.Queue(maxsize=4)
    errors = []

    def worker():
        while (item := frames.get()) is not None:
            try:
                run_frame(*item)
            except Exception as e:      # remember it and stop; the decode loop checks
                errors.append(e)
                return

    def submit(frame, fidx, ftime):
        """Queue a frame for inference; False if the worker has died."""
        while not errors:
            try:
                frames.put((frame, fidx, ftime), timeout=1.0)
                return True
            except queue.Full:
                pass
        return False

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    if stride >= 10:
        # Sparse sampling: seek to each sampled time instead of decoding every frame.
        # GoPro keyframes are ~1 s apart, so each seek decodes about one GOP.
        t = start if start and start > 0 else 0.0
        last_idx = -1
        while end is None or t <= end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ftime = (pos_ms / 1000.0 - 1.0 / fps) if pos_ms and pos_ms > 0 else t
            fidx = int(round(ftime * fps))
            if fidx != last_idx:        # a seek clamped at the end can return the same frame
                if not submit(frame, fidx, ftime):
                    break
                last_idx = fidx
            t += 1.0 / sample_fps
    else:
        # Dense sampling: decode every frame; grab() skips the pixel copy on unused frames.
        if start and start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        anchored = not (start and start > 0)    # frame_idx is known only after the first read
        fidx = 0
        while True:
            if anchored and fidx % stride == 0:
                ok, frame = cap.read()
            else:
                ok, frame = cap.grab(), None
            if not ok:
                break
            if not anchored:
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms and pos_ms > 0:
                    fidx = int(round(pos_ms / 1000.0 * fps))
                anchored = True
                if frame is None and fidx % stride == 0:
                    ok, frame = cap.retrieve()  # the first frame after the seek is analysed
            ftime = fidx / fps
            if end is not None and ftime > end:
                break
            if fidx % stride == 0 and frame is not None:
                if not submit(frame, fidx, ftime):
                    break
            fidx += 1
    cap.release()

    if not errors:
        frames.put(None)
    thread.join()
    if errors:
        raise errors[0]
    if processed == 0:
        raise RuntimeError("No frames could be decoded - check the video file and --start.")

    logging.info("Detection: %d frame(s) analysed, %d detection(s)", processed, len(rows))
    return pd.DataFrame(rows, columns=COLUMNS)
