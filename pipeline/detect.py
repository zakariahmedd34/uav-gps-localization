"""
author: @Zakaria_34

detect.py — Run the trained YOLO flag detector on a GoPro video and log every
detection with a timestamp, ready for sync.py.

Output: detections.csv with columns:
    columns = [frame_idx, frame_time, u, v, conf, class, crop]

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
    """
    Accept 'HH:MM:SS(.ms)', 'MM:SS', plain seconds, 
    or None -> seconds (float).
    
    """

    if s is None:
        return None
    s = str(s).strip()
    if ":" in s:
        """
            1:30:00
            sec = 0*60 + 1.0
            sec = 1*60 + 30.0
            sec = 90*60 + 0.0
        
        """
        sec = 0.0
        for p in s.split(":"):
            sec = sec * 60 + float(p)
        return sec
    
    return float(s)



def save_crop(frame,x1,y1,x2,y2,out_dir,frame_idx,k, pad):
    """
    crop the (padded) bbobx from a clean frema and save it
        frame = 3840 x 2160 (4K)
        
        bounding box:
        (300,250),(100,50)

        x1 = 100
        y1 = 50
        x2 = 300
        y2 = 250

        pad = 0.15

        bw = 200
        bh = 200

        xa = 100 - 200*0.15 = 70
        ya = 50 - 200*0.15 = 20
        xb = 300 + 200*0.15 = 330
        yb = 250 + 200*0.15 = 280

        (100,50)   --> (70,20)
        (300,250) --> (330,280)

        bw_new = 330 - 70 = 260
        bh_new = 280 - 20 = 260

        crop it--

    """

    h, w = frame.shape[:2]
    bw, bh = x2-x1, y2-y1
    xa = max(0, int(round(x1-bw*pad)))
    ya = max(0, int(round(y1-bh*pad)))
    xb = min(w, int(round(x2+bw*pad)))
    yb = min(h, int(round(y2+bh*pad)))

    crop = frame[ya:yb, xa:xb]

    name = f"crop_{frame_idx:06d}_{k}.jpg"
    
    cv2.imwrite(os.path.join(out_dir, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return name

def parse_args():
    p = argparse.ArgumentParser(description="YOLO flag detector -> detections.csv")

    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="detections.csv")

    p.add_argument("--thresh", type=float, default=0.5) 
    p.add_argument("--sample-fps", type=float, default=10.0)

    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None) # --device 0,1,2,3 (gpu) or cpu

    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    p.add_argument("--save-overlay", default=None)
    p.add_argument("--save-crops", default=None)
    p.add_argument("--crop-pad", type=float, default=0.15)

    return p.parse_args()
    

def main():
    args = parse_args()
    logging.info(f"detect.py START | source={args.source} | model={args.model} | out={args.out}")
    # exists (not isfile): exported models (OpenVINO/ONNX) are DIRECTORIES
    if not os.path.exists(args.model):
        logging.error(f"model not found: {args.model}")
        sys.exit(f"Error: model not found: {args.model}")
    if not os.path.isfile(args.source):
        logging.error(f"source not found: {args.source}")
        sys.exit(f"Error: source not found: {args.source}")
    
    model = YOLO(args.model)
    labels = model.names    

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        logging.error(f"could not open video: {args.source}")
        sys.exit(f"Error: could not open video: {args.source}")
    fps = args.fps if args.fps is not None else cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        logging.error(f"could not get FPS from video: {args.source}")
        sys.exit(f"Error: could not get FPS from video: {args.source}")


    stride = max(1, int(round(fps / args.sample_fps)))
    logging.info(f"video FPS={fps:.3f} | stride={stride} (~{fps/stride:.1f} detections/sec)")


    if args.save_overlay is not None:
        os.makedirs(args.save_overlay, exist_ok=True)
    if args.save_crops is not None:
        os.makedirs(args.save_crops, exist_ok=True)

    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)

    if end_sec is not None and start_sec is not None and end_sec <= start_sec:
        logging.error(f"end time must be greater than start time: start={start_sec} end={end_sec}")
        sys.exit(f"Error: end time must be greater than start time: start={start_sec} end={end_sec}")

    if start_sec is not None and start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    if  (start_sec is not None and start_sec > 0) or end_sec is not None:
        start_text = "video start" if start_sec is None else f"{start_sec:.2f}s"
        end_text = "video end" if end_sec is None else f"{end_sec:.2f}s"
        logging.info(f"Window: start={start_text}, end={end_text}")



    rows = []
    frame_idx = 0
    anchored = (start_sec is None or start_sec <= 0)
    processed = 0
    overlay_saved = 0
    crops_saved = 0

    while True:
        # PERF: only fully decode+retrieve frames we will run inference on.
        # cap.grab() advances the stream without the expensive retrieve/convert
        # of a 4K frame — big win when stride skips most frames on CPU.
        process_this = anchored and (frame_idx % stride == 0)
        if process_this:
            ret, frame = cap.read()
        else:
            ret = cap.grab()
            frame = None
        if not ret:
            break
        if not anchored:
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms and pos_ms > 0:
                frame_idx = int(round(pos_ms/1000.*fps))
            anchored = True
            if frame is None and frame_idx % stride == 0:
                ret, frame = cap.retrieve()   # first frame after seek is processed

        frame_time = frame_idx / fps

        if end_sec is not None and frame_time > end_sec:
            logging.info(f"Reached end time: {frame_time:.2f}s > {end_sec:.2f}s")
            break
        if frame_idx % stride == 0 and frame is not None:
            processed += 1

            results = model(frame, conf=args.thresh, imgsz=args.imgsz,
                            device=args.device, verbose=False)
            boxes = results[0].boxes

            clean = frame.copy() if args.save_crops else None
            det_k = 0

            for b in boxes:
                conf = float(b.conf.item())
                if conf < args.thresh:
                    continue

                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()

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

    logging.info(f"detect: DONE frames_read={frame_idx} processed={processed} detections={len(rows)} crops={crops_saved} -> {args.out}")
    
    if args.save_crops:
        logging.info(f"saved {crops_saved} flag crops in {args.save_crops}/")
    if args.save_overlay:
        logging.info(f"overlay frames in {args.save_overlay}/ — open them to verify boxes")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise CustomException(e, sys) from e
        

