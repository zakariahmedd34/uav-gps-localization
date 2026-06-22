"""
detect.py — Run the trained YOLO flag detector on a GoPro video and log every
detection with a timestamp, ready for sync.py.

Output: detections.csv with columns
    frame_idx, frame_time, u, v, conf, class

  frame_time = frame_idx / fps  → same clock as the GPMF telemetry (t=0 at video start)
  (u, v)     = bbox center in pixels (float, sub-pixel — needed by localization)

Usage:
    python detect.py --model best.pt --source test.mp4 --out detections.csv
    python detect.py --model best.pt --source test.mp4 --out detections.csv \
                     --sample-fps 10 --thresh 0.5 --save-overlay overlay/

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
    p.add_argument("--save-overlay", default=None,
                   help="Optional dir: save annotated frames here for the visual test")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"ERROR: model not found: {args.model}")
    if not os.path.isfile(args.source):
        sys.exit(f"ERROR: video not found: {args.source}")

    model = YOLO(args.model, task="detect")
    labels = model.names

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open video: {args.source}")

    # fps drives frame_time — the link to telemetry. Trust --fps over the file if given.
    fps = args.fps if args.fps else cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        sys.exit("ERROR: could not read fps; pass --fps explicitly (from ffprobe).")

    stride = max(1, round(fps / args.sample_fps))
    print(f"fps={fps:.3f}  stride={stride}  (~{fps/stride:.1f} detections/sec)")

    if args.save_overlay:
        os.makedirs(args.save_overlay, exist_ok=True)

    rows = []
    frame_idx = 0
    processed = 0
    overlay_saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            processed += 1
            frame_time = frame_idx / fps
            results = model(frame, conf=args.thresh, imgsz=args.imgsz,
                            device=args.device, verbose=False)
            boxes = results[0].boxes

            for b in boxes:
                conf = float(b.conf.item())
                if conf < args.thresh:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()  # floats
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0
                cls = labels[int(b.cls.item())]
                rows.append([frame_idx, round(frame_time, 4),
                             round(u, 2), round(v, 2), round(conf, 4), cls])

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
        w.writerow(["frame_idx", "frame_time", "u", "v", "conf", "class"])
        w.writerows(rows)

    print(f"Frames read: {frame_idx} | frames processed: {processed} "
          f"| detections: {len(rows)}")
    print(f"Wrote {args.out}")
    if args.save_overlay:
        print(f"Overlay frames in {args.save_overlay}/ — open them to verify boxes.")


if __name__ == "__main__":
    main()
