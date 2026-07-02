"""
extract_calib_frames.py — pull good calibration frames out of a checkerboard VIDEO.

Why a video? GoPro Photo mode and Video mode have different sensor crops / FOV /
resolution, so a still photo won't match your FLIGHT footage. Record a short
checkerboard clip with the SAME settings you fly (same resolution, Linear lens,
HyperSmooth OFF), then run this to harvest sharp, well-spread frames for
calibrate_camera.py.

What it does
------------
1. Samples the video at --scan-fps.
2. Rejects blurry frames (variance-of-Laplacian below --blur-thresh).
3. (Recommended) keeps only frames where the checkerboard is actually FOUND
   (pass --rows/--cols). This guarantees calibrate_camera.py won't skip them.
4. Picks --target frames spread EVENLY in time (corner/edge coverage comes from
   how you wave the board while filming) and saves FULL-RESOLUTION jpgs.

Usage
-----
  # recommended: verify the checkerboard while extracting
  python extract_calib_frames.py --video calib.MP4 --out ../data/checkerboard \
      --rows 5 --cols 8 --target 25

  # blur-only (no checkerboard check) — faster, but calibrate may skip some
  python extract_calib_frames.py --video calib.MP4 --out ../data/checkerboard \
      --target 25

--rows/--cols are INTERNAL corners (a 9x6-square board = 5x8 internal corners),
the SAME numbers you'll pass to calibrate_camera.py.

Tip: shoot so the board visits the center, all four corners, the edges, and a
few tilts/distances. Edges matter most — that's where lens distortion lives.
"""

import os
import sys
import argparse

import cv2
import numpy as np

from logger import logging
from exception import CustomException


def blur_score(gray):
    """Variance of Laplacian — higher = sharper. <~80 on 1080p is usually blurry."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def find_board(gray, cols, rows, max_w=960):
    """Fast checkerboard presence check on a downscaled copy. Returns bool.

    Detection is done small (for speed); the FULL-res frame is what gets saved.
    """
    h, w = gray.shape
    scale = min(1.0, max_w / float(w))
    small = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale < 1.0 else gray
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
             + cv2.CALIB_CB_FAST_CHECK)
    found, _ = cv2.findChessboardCorners(small, (cols, rows), flags=flags)
    return bool(found)


def main():
    ap = argparse.ArgumentParser(description="Harvest calibration frames from a video")
    ap.add_argument("--video", required=True, help="Checkerboard video (flight settings)")
    ap.add_argument("--out", required=True, help="Output dir (e.g. ../data/checkerboard)")
    ap.add_argument("--target", type=int, default=25, help="How many frames to keep")
    ap.add_argument("--scan-fps", type=float, default=4.0,
                    help="Frames/sec to inspect (lower = faster scan)")
    ap.add_argument("--blur-thresh", type=float, default=80.0,
                    help="Min variance-of-Laplacian to accept (raise to be stricter)")
    ap.add_argument("--rows", type=int, default=None, help="Internal corner ROWS (enables board check)")
    ap.add_argument("--cols", type=int, default=None, help="Internal corner COLS (enables board check)")
    ap.add_argument("--min-gap", type=float, default=0.3,
                    help="Min seconds between two kept frames (avoid near-duplicates)")
    ap.add_argument("--prefix", default="calib", help="Output filename prefix")
    args = ap.parse_args()
    logging.info("extract_calib_frames: START video=%s -> %s", args.video, args.out)

    if not os.path.isfile(args.video):
        logging.error("video not found: %s", args.video)
        sys.exit(f"ERROR: video not found: {args.video}")
    check_board = args.rows is not None and args.cols is not None
    if (args.rows is None) ^ (args.cols is None):
        logging.error("pass BOTH --rows and --cols, or neither")
        sys.exit("ERROR: pass BOTH --rows and --cols, or neither.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logging.error("could not open video: %s", args.video)
        sys.exit(f"ERROR: could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / args.scan_fps))
    logging.info("fps=%.2f scanning every %d frame(s) board_check=%s",
                 fps, stride, "ON" if check_board else "OFF")

    # --- pass 1: collect candidates (time, sharpness) ---
    candidates = []          # (frame_idx, t_sec, blur)
    frames_cache = {}        # frame_idx -> BGR (only for candidates, full-res)
    idx = 0
    scanned = 0
    rejected_blur = 0
    rejected_noboard = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            scanned += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            b = blur_score(gray)
            if b < args.blur_thresh:
                rejected_blur += 1
            elif check_board and not find_board(gray, args.cols, args.rows):
                rejected_noboard += 1
            else:
                t = idx / fps
                candidates.append((idx, t, b))
                frames_cache[idx] = frame.copy()
            if scanned % 50 == 0:
                logging.info("  scanned %d kept %d (blur-rej %d, noboard-rej %d)",
                             scanned, len(candidates), rejected_blur, rejected_noboard)
        idx += 1
    cap.release()

    if not candidates:
        logging.error("no usable frames found")
        sys.exit("No usable frames found. Lower --blur-thresh, check --rows/--cols, "
                 "or confirm the board is actually visible and in focus.")
    logging.info("Candidates: %d (scanned %d, blur-rej %d, noboard-rej %d)",
                 len(candidates), scanned, rejected_blur, rejected_noboard)

    # --- pass 2: pick TARGET frames spread evenly across the timeline ---
    candidates.sort(key=lambda c: c[1])               # by time
    t0, t1 = candidates[0][1], candidates[-1][1]
    want = min(args.target, len(candidates))
    targets_t = np.linspace(t0, t1, want)
    chosen, last_t = [], -1e9
    for tt in targets_t:
        # nearest candidate to this evenly-spaced time slot, respecting min-gap
        best = min(candidates, key=lambda c: abs(c[1] - tt))
        if best in chosen or abs(best[1] - last_t) < args.min_gap:
            pool = [c for c in candidates if c not in chosen
                    and abs(c[1] - last_t) >= args.min_gap]
            if not pool:
                continue
            best = min(pool, key=lambda c: abs(c[1] - tt))
        chosen.append(best)
        last_t = best[1]

    os.makedirs(args.out, exist_ok=True)
    saved = 0
    for (fidx, t, b) in sorted(chosen, key=lambda c: c[0]):
        path = os.path.join(args.out, f"{args.prefix}_{fidx:06d}.jpg")
        cv2.imwrite(path, frames_cache[fidx], [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    logging.info("extract_calib_frames: DONE saved %d frames to %s/ (spread %.1fs -> %.1fs)",
                 saved, args.out, t0, t1)
    logging.info("Next: python calibrate_camera.py --images '%s/%s_*.jpg' "
                 "--rows R --cols C --square-size S --out ../configs/camera_params_hero13.yaml",
                 args.out, args.prefix)
    if not check_board:
        logging.warning("ran WITHOUT board verification — calibrate_camera.py may still "
                        "skip frames where it can't find all corners. Re-run with "
                        "--rows/--cols to pre-filter if many get skipped.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise CustomException(e, sys) from e
