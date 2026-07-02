"""
detect.py — Run the trained YOLO flag detector on a GoPro video and log every
detection with a timestamp, ready for sync.py.

Output: detections.csv with columns
    frame_idx, frame_time, u, v, conf, class, crop

  frame_time = frame_idx / fps  → same clock as the GPMF telemetry (t=0 at video start)
  (u, v)     = bbox center in pixels (float, sub-pixel — needed by localization)
  crop       = filename of the saved flag crop (empty unless --save-crops)

Usage:
    python detect.py --model best.pt --source test.mp4 --out detections.csv
    python detect.py --model best.pt --source test.mp4 --out detections.csv \
                     --sample-fps 10 --thresh 0.5 --save-overlay overlay/ --save-crops crops/

Notes:
  * Process at NATIVE resolution. Do not resize — the camera calibration (K) must
    match the pixel size of (u, v). Calibrate at the same resolution you run here.
  * Frame sampling (--sample-fps) keeps runtime well under the 10-minute clock.
"""

import os
import sys
import csv
import argparse

import cv2
from ultralytics import YOLO

from logger import logging
from exception import CustomException


def parse_time(s):
    """Accept 'HH:MM:SS(.ms)', 'MM:SS', plain seconds, or None -> seconds (float)."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if ":" in s:
        sec = 0.0
        for p in s.split(":"):
            sec = sec * 60.0 + float(p)
        return sec
    return float(s)


def save_crop(frame, x1, y1, x2, y2, out_dir, frame_idx, k, pad):
    """Crop the (padded) bbox from a CLEAN frame and save it. Returns the filename.

    `pad` adds a margin around the box (0.15 = 15%) so the classifier / a human
    sees some context. Coordinates are clamped to the image so we never go
    out of bounds. Crop from the clean frame, NOT one with overlay boxes drawn.
    """
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    xa = max(0, int(round(x1 - bw * pad)))
    ya = max(0, int(round(y1 - bh * pad)))
    xb = min(w, int(round(x2 + bw * pad)))
    yb = min(h, int(round(y2 + bh * pad)))
    crop = frame[ya:yb, xa:xb]
    name = f"crop_{frame_idx:06d}_{k}.jpg"
    cv2.imwrite(os.path.join(out_dir, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return name


def parse_args():
    p = argparse.ArgumentParser(description="YOLO flag detector -> detections.csv")
    p.add_argument("--model", required=True, help="Path to YOLO weights (e.g. best.pt)")
    p.add_argument("--source", required=True, help="Path to video file (.mp4/.mov/...)")
    p.add_argument("--out", default="detections.csv", help="Output CSV path")
    p.add_argument("--thresh", type=float, default=0.5, help="Min confidence to keep")
    p.add_argument("--sample-fps", type=float, default=10.0,
                   help="Target detections/sec; we process every Nth frame")
    p.add_argument("--fps", type=float, default=None,
                   help="Override video fps (use the ffprobe value if CAP_PROP_FPS is wrong)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference size. Boxes are still returned in full-res pixels. "
                        "Raise to 1280/1920 for TINY flags at altitude; lower for speed.")
    p.add_argument("--device", default=None,
                   help="'0' for GPU (much faster), 'cpu' to force CPU. Default: auto.")
    p.add_argument("--start", default=None,
                   help="Process FROM this time. 'HH:MM:SS(.ms)', 'MM:SS', or seconds. "
                        "Default: video start. Seeks, so frames before it are never decoded.")
    p.add_argument("--end", default=None,
                   help="Process UP TO this time (same formats). Default: video end.")
    p.add_argument("--save-overlay", default=None,
                   help="Optional dir: save annotated frames here for the visual test")
    p.add_argument("--save-crops", default=None,
                   help="Optional dir: save a cropped image of each detected flag here "
                        "(feeds the USB submission). Cropped from the CLEAN frame.")
    p.add_argument("--crop-pad", type=float, default=0.15,
                   help="Fractional padding around the bbox when cropping (0.15 = 15%%).")
    return p.parse_args()


def main():
    args = parse_args()
    logging.info("detect: START source=%s model=%s", args.source, args.model)

    if not os.path.isfile(args.model):
        logging.error("model not found: %s", args.model)
        sys.exit(f"ERROR: model not found: {args.model}")
    if not os.path.isfile(args.source):
        logging.error("video not found: %s", args.source)
        sys.exit(f"ERROR: video not found: {args.source}")

    model = YOLO(args.model, task="detect")
    labels = model.names

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        logging.error("could not open video: %s", args.source)
        sys.exit(f"ERROR: could not open video: {args.source}")

    # fps drives frame_time — the link to telemetry. Trust --fps over the file if given.
    fps = args.fps if args.fps else cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        logging.error("could not read fps for %s", args.source)
        sys.exit("ERROR: could not read fps; pass --fps explicitly (from ffprobe).")

    stride = max(1, round(fps / args.sample_fps))
    logging.info("fps=%.3f stride=%d (~%.1f detections/sec)", fps, stride, fps / stride)

    if args.save_overlay:
        os.makedirs(args.save_overlay, exist_ok=True)
    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)

    # --- optional time window (Stage 2): seek to --start, stop at --end ---
    start_sec = parse_time(args.start) or 0.0
    end_sec = parse_time(args.end)
    if end_sec is not None and end_sec <= start_sec:
        logging.error("--end (%.2fs) must be after --start (%.2fs)", end_sec, start_sec)
        sys.exit(f"ERROR: --end ({end_sec}s) must be after --start ({start_sec}s).")
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    if start_sec > 0 or end_sec is not None:
        logging.info("Window: start=%.2fs end=%s", start_sec,
                     "video end" if end_sec is None else f"{end_sec:.2f}s")

    rows = []
    frame_idx = 0
    anchored = (start_sec <= 0)        # after a seek, re-anchor the true index once
    processed = 0
    overlay_saved = 0
    crops_saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # After seeking, recover the TRUE absolute frame index from the frame's
        # timestamp so frame_time stays on the same clock as the GPMF telemetry
        # (t=0 at video start) — keeps sync correct even when we skip ahead.
        if not anchored:
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms and pos_ms > 0:
                frame_idx = int(round(pos_ms / 1000.0 * fps))
            anchored = True

        frame_time = frame_idx / fps
        if end_sec is not None and frame_time > end_sec:
            break

        if frame_idx % stride == 0:
            processed += 1
            results = model(frame, conf=args.thresh, imgsz=args.imgsz,
                            device=args.device, verbose=False)
            boxes = results[0].boxes

            # Keep a CLEAN copy for cropping so crops never contain overlay boxes.
            clean = frame.copy() if args.save_crops else None
            det_k = 0

            for b in boxes:
                conf = float(b.conf.item())
                if conf < args.thresh:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()  # floats
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0
                cls = labels[int(b.cls.item())]

                crop_name = ""
                if args.save_crops:
                    crop_name = save_crop(clean, x1, y1, x2, y2,
                                          args.save_crops, frame_idx, det_k, args.crop_pad)
                    det_k += 1
                    crops_saved += 1

                rows.append([frame_idx, round(frame_time, 4),
                             round(u, 2), round(v, 2), round(conf, 4), cls, crop_name])

                if args.save_overlay and overlay_saved < 20:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls} {conf:.2f}", (int(x1), int(y1) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if args.save_overlay and overlay_saved < 20 and len(boxes):
                cv2.imwrite(os.path.join(args.save_overlay,
                                         f"frame_{frame_idx:06d}.jpg"), frame)
                overlay_saved += 1

        frame_idx += 1

    cap.release()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "frame_time", "u", "v", "conf", "class", "crop"])
        w.writerows(rows)

    logging.info("detect: DONE frames_read=%d processed=%d detections=%d crops=%d -> %s",
                 frame_idx, processed, len(rows), crops_saved, args.out)
    if args.save_crops:
        logging.info("saved %d flag crops in %s/", crops_saved, args.save_crops)
    if args.save_overlay:
        logging.info("overlay frames in %s/ — open them to verify boxes", args.save_overlay)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise CustomException(e, sys) from e
